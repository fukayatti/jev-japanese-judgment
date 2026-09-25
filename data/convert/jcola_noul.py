"""JCoLA(日本語の文法的容認性コーパス) -> Jev形式のyes/no(noul)への決定論的変換。

元データ: https://github.com/osekilab/JCoLA (CC BY-SA 4.0), data/jcola-v1.0/*.tsv
  列: uid, source, label(1=容認可能 / 0=容認不可), diacritic, sentence, original
  分割: in_domain_train / in_domain_valid / out_of_domain_valid
  (label=1が約83%と偏っている。必要なら後段のリバランスで調整する)

「この文は日本語として文法的に自然か」を、はい/いいえで答える形にする。
これは事実の正誤ではなく文法性の判定。yes/noという答え方の形式を覚えさせるのが主目的。
"""

import csv
import urllib.request
from pathlib import Path

from data.convert.schema import JevExample

_RAW_URL = "https://raw.githubusercontent.com/osekilab/JCoLA/main/data/jcola-v1.0/{split}-v1.0.tsv"
_CANDIDATES_JA = ["はい", "いいえ"]
SPLITS = ("in_domain_train", "in_domain_valid", "out_of_domain_valid")


def row_to_example(row: dict, split: str) -> JevExample:
    return JevExample(
        id=f"jcola_{split}_{row['uid']}",
        source_dataset="jcola",
        context=row["sentence"].strip(),
        question="この文は、日本語として文法的に自然ですか？",
        candidates=list(_CANDIDATES_JA),
        label=0 if int(row["label"]) == 1 else 1,
        task_type="noul",
    )


def convert(split: str = "in_domain_train", cache_dir: Path = Path("data/raw/jcola")) -> list[JevExample]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    path = cache_dir / f"{split}.tsv"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_RAW_URL.format(split=split), path)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [row_to_example(row, split) for row in rows]


if __name__ == "__main__":
    examples = convert("in_domain_train")
    print(f"converted {len(examples)} examples")
    print(examples[0])
