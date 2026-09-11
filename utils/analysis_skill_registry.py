"""Read trusted product skill files on demand, never arbitrary model paths."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


class AnalysisSkillRegistry:
    def __init__(self, root: Path | None = None):
        self.root = (root or Path(__file__).resolve().parents[1] / "analysis_skills").resolve()

    def _read(self, name: str) -> dict[str, str]:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise ValueError("등록된 스킬 이름을 사용해주세요.")
        path = (self.root / name / "SKILL.md").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("스킬 경로가 허용 범위를 벗어났습니다.")
        if path.stat().st_size > 32_000:
            raise ValueError("스킬 파일 크기가 허용 범위를 초과했습니다.")
        text = path.read_text(encoding="utf-8")
        match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.S)
        if not match:
            raise ValueError("스킬 메타데이터가 올바르지 않습니다.")
        # This product's authored manifest uses two plain single-line fields.
        # Refuse unsupported YAML instead of silently interpreting it incorrectly.
        fields = {}
        for line in match[1].splitlines():
            key, sep, value = line.partition(":")
            if not sep or key not in {"name", "description"} or key in fields:
                raise ValueError("지원하지 않는 스킬 메타데이터입니다.")
            fields[key] = value.strip()
        if fields.get("name") != name or not fields.get("description"):
            raise ValueError("스킬 이름/설명이 올바르지 않습니다.")
        return {**fields, "body": match[2].strip(),
                "version": hashlib.sha256(text.encode()).hexdigest()}

    def list(self) -> list[dict[str, str]]:
        return [{k: v for k, v in self._read(p.parent.name).items() if k != "body"}
                for p in sorted(self.root.glob("*/SKILL.md"))]

    def read(self, name: str) -> dict[str, str]:
        return self._read(name)
