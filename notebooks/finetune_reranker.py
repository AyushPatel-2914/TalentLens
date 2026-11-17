# %% [markdown]
# # TalentLens: fine-tune the cross-encoder re-ranker (Google Colab)
#
# **Runtime → Change runtime type → T4 GPU**, then run the cells in order.
#
# What this does:
# 1. Loads `train.jsonl` / `validation.jsonl` (made by `scripts/05_export_training_data.py`).
# 2. Fine-tunes `BAAI/bge-reranker-base` on graded labels (0 / 1 / 2).
# 3. After every epoch, measures nDCG@10 on VALIDATION and keeps the best checkpoint (early stopping).
# 4. Pushes the best model to the Hugging Face Hub, and saves `training_log.json` for `results/`.
#
# **Choices, and why:**
# - **Loss:** binary cross-entropy with soft targets (0 → 0.0, 1 → 0.5, 2 → 1.0). It keeps the "partial fit"
#   signal instead of reducing labels to yes/no, and it is stable on a small dataset. A listwise loss
#   (LambdaLoss) optimises ranking more directly, but it is left as a separate experiment.
# - **Learning rate 2e-5 with 10% warm-up:** we are adjusting a pre-trained model, not training from scratch.
#   A large rate would erase what it already knows.
# - **At most 4 epochs + early stopping (patience 1):** with ~1.7k pairs the model can memorise quickly;
#   validation nDCG tells us when it starts to overfit.
# - **Hard negatives** are already in the data: pairs found by BM25/dense that were labelled 0.
# - **No leakage:** train and validation contain different jobs, and the test split is never uploaded here.

# %%
# !pip -q install -U transformers huggingface_hub
import json
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

CONFIG = {
    "base_model": "BAAI/bge-reranker-base",
    "lr": 2e-5,
    "epochs": 4,
    "batch_size": 16,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_length": 512,
    "patience": 1,
    "seed": 42,
    "hub_repo": "YOUR_HF_USERNAME/talentlens-reranker",  # <- change this
}
SOFT_TARGET = {0: 0.0, 1: 0.5, 2: 1.0}
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

random.seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])

# %% [markdown]
# ## 1. Upload the data
# Upload `data/training/train.jsonl` and `data/training/validation.jsonl` from your computer.

# %%
try:
    from google.colab import files  # type: ignore
    files.upload()
except ImportError:
    print("Not on Colab: put train.jsonl and validation.jsonl next to this notebook.")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


train_rows, val_rows = load_jsonl("train.jsonl"), load_jsonl("validation.jsonl")
print(f"train: {len(train_rows)} pairs / {len({r['job_id'] for r in train_rows})} jobs")
print(f"validation: {len(val_rows)} pairs / {len({r['job_id'] for r in val_rows})} jobs")
assert not ({r["job_id"] for r in train_rows} & {r["job_id"] for r in val_rows}), "a job is in both splits (leakage)"

# %% [markdown]
# ## 2. Model, metrics and evaluation

# %%
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
model = AutoModelForSequenceClassification.from_pretrained(CONFIG["base_model"]).to(device)


def encode(rows):
    return tokenizer([r["query"] for r in rows], [r["doc"] for r in rows], truncation="longest_first",
                     max_length=CONFIG["max_length"], padding=True, return_tensors="pt")


def dcg(gains):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_10(ranked_labels):
    ideal = dcg(sorted(ranked_labels, reverse=True)[:10])
    return dcg(ranked_labels[:10]) / ideal if ideal > 0 else None


@torch.no_grad()
def evaluate(rows, batch_size=64):
    """Score every labelled pair, rank each job's candidates, average nDCG@10 over jobs."""
    model.eval()
    scores = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        enc = encode(batch).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            scores += model(**enc).logits.view(-1).float().cpu().tolist()
    by_job = defaultdict(list)
    for r, s in zip(rows, scores):
        by_job[r["job_id"]].append((s, r["label"]))
    values = [ndcg_at_10([l for _, l in sorted(v, key=lambda x: -x[0])]) for v in by_job.values()]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else float("nan")


# %% [markdown]
# ## 3. Train with early stopping on validation nDCG@10

# %%
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
steps_per_epoch = math.ceil(len(train_rows) / CONFIG["batch_size"])
scheduler = get_linear_schedule_with_warmup(optimizer, int(CONFIG["warmup_ratio"] * steps_per_epoch * CONFIG["epochs"]),
                                            steps_per_epoch * CONFIG["epochs"])
scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

log = {"config": CONFIG, "epochs": []}
base_ndcg = evaluate(val_rows)
log["validation_ndcg@10_before_training"] = base_ndcg
print(f"epoch 0 (base model): validation nDCG@10 = {base_ndcg:.4f}")

best, bad_epochs = base_ndcg, 0
for epoch in range(1, CONFIG["epochs"] + 1):
    model.train()
    random.shuffle(train_rows)
    t0, total = time.time(), 0.0
    for i in range(0, len(train_rows), CONFIG["batch_size"]):
        batch = train_rows[i:i + CONFIG["batch_size"]]
        enc = encode(batch).to(device)
        target = torch.tensor([SOFT_TARGET[r["label"]] for r in batch], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            loss = loss_fn(model(**enc).logits.view(-1).float(), target)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        total += loss.item() * len(batch)
    val = evaluate(val_rows)
    entry = {"epoch": epoch, "train_loss": total / len(train_rows), "validation_ndcg@10": val,
             "seconds": round(time.time() - t0, 1)}
    log["epochs"].append(entry)
    print(entry)
    if val > best:
        best, bad_epochs = val, 0
        model.save_pretrained("best"); tokenizer.save_pretrained("best")
        log["best_epoch"] = epoch
    else:
        bad_epochs += 1
        if bad_epochs > CONFIG["patience"]:
            print("Early stopping: validation nDCG@10 stopped improving.")
            break

log["best_validation_ndcg@10"] = best
if "best_epoch" not in log:
    print("Fine-tuning never beat the base model on validation. Keep the base model and report that honestly.")
with open("training_log.json", "w") as f:
    json.dump(log, f, indent=2)
print(json.dumps(log, indent=2))

# %% [markdown]
# ## 4. Push the best model to the Hugging Face Hub
# Create a **write** token at https://huggingface.co/settings/tokens. The repo is private by default;
# the API loads it with the `HF_TOKEN` in your `.env`. Then put the repo id in `configs/config.yaml`
# under `rerank.finetuned_model`, and download `training_log.json` into `results/finetune_log.json`.

# %%
if "best_epoch" in log:
    from huggingface_hub import login
    login()  # paste your token when asked
    best_model = AutoModelForSequenceClassification.from_pretrained("best")
    best_model.push_to_hub(CONFIG["hub_repo"], private=True)
    AutoTokenizer.from_pretrained("best").push_to_hub(CONFIG["hub_repo"], private=True)
    print("Pushed to", CONFIG["hub_repo"])
try:
    files.download("training_log.json")
except Exception:
    pass
