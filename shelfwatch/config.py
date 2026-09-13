from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    cfg["root"] = root
    for key in ("artifacts", "references", "media", "planograms", "database", "outputs"):
        cfg[key] = (root / cfg[key]).resolve()
    for key in ("artifacts", "references", "media", "planograms", "outputs"):
        cfg[key].mkdir(parents=True, exist_ok=True)
    cfg["database"].parent.mkdir(parents=True, exist_ok=True)
    return cfg
