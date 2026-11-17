---
title: TalentLens API
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Explainable job matching API with a recruiter agent
---

# TalentLens API

Backend for [TalentLens](https://github.com/AyushPatel-2914/TalentLens): hybrid retrieval (BM25 + bge-small/FAISS + RRF),
cross-encoder re-ranking, verified evidence quotes and a recruiter agent with human approval.

- API docs: `/docs`
- Health: `/health`

The free CPU Space sleeps when idle; the first request after that takes a few minutes to wake it.
