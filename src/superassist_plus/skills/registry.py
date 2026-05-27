from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from superassist_plus.config import PROJECT_ROOT

SKILLS_ROOT = PROJECT_ROOT / "skills"
SKILLS_CONTAINER_ROOT = "/mnt/skills"
SKILL_FILE_NAME = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    skill_dir: Path
    skill_file: Path
    relative_path: Path

    @property
    def virtual_file_path(self) -> str:
        return f"{SKILLS_CONTAINER_ROOT}/{self.relative_path.as_posix()}/{SKILL_FILE_NAME}"


def list_public_skills() -> list[Skill]:
    return _list_public_skills_cached()


@lru_cache(maxsize=1)
def _list_public_skills_cached() -> list[Skill]:
    public_root = SKILLS_ROOT / "public"
    if not public_root.exists():
        return []
    skills: list[Skill] = []
    for skill_file in sorted(public_root.glob(f"*/{SKILL_FILE_NAME}")):
        skill = _parse_skill_file(skill_file, skill_file.parent.relative_to(SKILLS_ROOT))
        if skill is not None:
            skills.append(skill)
    return skills


def build_available_skills_section() -> str:
    skills = list_public_skills()
    if not skills:
        return ""
    skill_items = "\n".join(
        (
            "    <skill>\n"
            f"        <name>{_escape_xml(skill.name)}</name>\n"
            f"        <description>{_escape_xml(skill.description)} [built-in]</description>\n"
            f"        <location>{_escape_xml(skill.virtual_file_path)}</location>\n"
            "    </skill>"
        )
        for skill in skills
    )
    return f"""<skill_system>
You have access to skills that provide optimized workflows for specific tasks.

Rules:
1. When a user query matches a skill's use case, call `read_file` on the skill's location before doing the task.
2. Read and follow the skill's workflow and instructions.
3. Load referenced resources incrementally from the same skill folder if needed.

<available_skills>
{skill_items}
</available_skills>
</skill_system>"""


def build_loaded_skills_section(skill_names: list[str] | tuple[str, ...] | set[str]) -> str:
    if not skill_names:
        return ""
    by_name = {skill.name: skill for skill in list_public_skills()}
    sections: list[str] = []
    for name in sorted(set(skill_names)):
        skill = by_name.get(name)
        if skill is None:
            continue
        try:
            content = skill.skill_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            sections.append(f'<skill name="{_escape_xml(skill.name)}">\n{content}\n</skill>')
    return "\n\n".join(sections)


def resolve_skill_virtual_path(path: str) -> Path | None:
    normalized = _normalize_virtual_path(path)
    prefix = f"{SKILLS_CONTAINER_ROOT}/"
    if not normalized.startswith(prefix):
        return None
    relative = normalized.removeprefix(prefix)
    if not relative:
        return None
    candidate = (SKILLS_ROOT / relative).resolve()
    root = SKILLS_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PermissionError(f"Path is outside the skills directory: {path}") from None
    return candidate


def skill_name_from_virtual_path(path: str) -> str | None:
    normalized = _normalize_virtual_path(path)
    prefix = f"{SKILLS_CONTAINER_ROOT}/public/"
    if not normalized.startswith(prefix):
        return None
    parts = normalized.removeprefix(prefix).split("/")
    if len(parts) >= 2 and parts[1] == SKILL_FILE_NAME:
        return parts[0]
    return None


def _parse_skill_file(skill_file: Path, relative_path: Path) -> Skill | None:
    try:
        content = skill_file.read_text(encoding="utf-8")
    except OSError:
        return None
    metadata = _parse_frontmatter(content)
    name = metadata.get("name") or skill_file.parent.name
    description = metadata.get("description", "")
    if not name:
        return None
    return Skill(
        name=name.strip(),
        description=description.strip(),
        skill_dir=skill_file.parent,
        skill_file=skill_file,
        relative_path=relative_path,
    )


def _parse_frontmatter(content: str) -> dict[str, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    metadata: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata


def _normalize_virtual_path(path: str) -> str:
    return str(path or "").replace("\\", "/").rstrip("/")


def _escape_xml(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
