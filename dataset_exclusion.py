import json
from pathlib import Path
from typing import Iterable, Set


def excluded_samples_path(project_root: Path | None = None) -> Path:
    # excluded_samples.json is expected at MedicalAI/excluded_samples.json
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    return project_root / "excluded_samples.json"


def _normalize_stem(stem_or_filename: str) -> str:
    # Accept either "401" or "401.bmp". If filename has extension, strip it.
    s = (stem_or_filename or "").strip()
    if not s:
        return ""
    return Path(s).stem


def load_excluded_samples(project_root: Path | None = None) -> Set[str]:
    path = excluded_samples_path(project_root)
    if not path.exists():
        return set()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # Fail safe: if config is unreadable, do not exclude anything.
        return set()

    exclude = data.get("exclude", []) if isinstance(data, dict) else []
    if not isinstance(exclude, Iterable):
        return set()

    out: Set[str] = set()
    for item in exclude:
        if isinstance(item, str):
            st = _normalize_stem(item)
            if st:
                out.add(st)
    return out


def is_excluded_sample(stem_or_filename: str, excluded: Set[str]) -> bool:
    st = _normalize_stem(stem_or_filename)
    return st in excluded

