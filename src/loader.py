import json
import gzip
from pathlib import Path
from typing import Iterator


def load_candidates(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    opener = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, "r", encoding="utf-8")
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_sample(path: str | Path) -> list[dict]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)
