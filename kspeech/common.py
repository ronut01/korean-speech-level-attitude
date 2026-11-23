"""Shared paths, config and small helpers."""
import hashlib, json, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
LABELS = {
    "ynat": ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"],
    "nli": ["함의", "중립", "모순"],
}
LETTERS = "ABCDEFG"
ENDINGS = ["Y", "B"]          # haeyo-che, hae-che
ATTITUDES = ["P", "R"]        # affiliative, rude


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def text_hash(s: str) -> str:
    return hashlib.sha1(norm_text(s).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_stimuli(path=None):
    path = Path(path) if path else ROOT / "stimuli" / "selected.json"
    return json.loads(path.read_text(encoding="utf-8"))
