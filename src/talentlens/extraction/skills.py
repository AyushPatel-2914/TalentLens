"""Skill-name normalisation using configs/skill_aliases.yaml.

"JS", "javascript" and "Java Script" should all become "JavaScript", otherwise
a resume that says "JS" looks like it's missing a JavaScript requirement.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


def _key(skill: str) -> str:
    return re.sub(r"\s+", " ", skill.strip().strip(".,;:")).lower()


class SkillNormalizer:
    def __init__(self, canonical_to_aliases: dict[str, list[str]]):
        self.lookup: dict[str, str] = {}
        self._forms: dict[str, list[str]] = {}  # canonical (lowercase) -> every spelling, for text search
        for canonical, aliases in canonical_to_aliases.items():
            self._forms[_key(canonical)] = [_key(n) for n in [canonical, *(aliases or [])]]
            for name in [canonical, *(aliases or [])]:
                k = _key(name)
                if k in self.lookup and self.lookup[k] != canonical:
                    raise ValueError(f"alias {name!r} maps to both {self.lookup[k]!r} and {canonical!r}")
                self.lookup[k] = canonical

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SkillNormalizer":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})

    def normalize(self, skill: str) -> str:
        cleaned = re.sub(r"\s+", " ", skill.strip().strip(".,;:"))
        return self.lookup.get(cleaned.lower(), cleaned)

    def forms(self, skill: str) -> list[str]:
        """All known spellings of a skill (lowercase), e.g. "Microsoft Excel" -> ["microsoft excel", "excel", ...]."""
        k = _key(self.normalize(skill))
        return self._forms.get(k, [k])

    @property
    def canonical_names(self) -> list[str]:
        return sorted(set(self.lookup.values()))

    def normalize_list(self, skills: list[str]) -> list[str]:
        """Normalise and de-duplicate case-insensitively, keeping first-seen order."""
        seen, out = set(), []
        for s in skills:
            n = self.normalize(s)
            if n and n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
        return out


def skill_key(skill: str) -> str:
    """Comparison key for matching: case- and whitespace-insensitive."""
    return _key(skill)
