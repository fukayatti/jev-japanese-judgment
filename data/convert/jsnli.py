"""JSNLI -> Jev形式の決定論的変換。

元データ (京大 黒橋・褚・村脇研, TSV, タブ区切り):
  label, premise, hypothesis

label の正確な表記 (entailment/contradiction/neutral か、日本語表記か) は
実データで要確認 (TODO)。ラベル文字列は _LABEL_MAP で調整すること。
"""

import csv
from pathlib import Path

from data.convert.schema import JevExample

_LABEL_MAP = {
    "entailment": 0,
    "contradiction": 1,
    "neutral": 2,
}
_CANDIDATES_JA = ["成立する", "矛盾する", "無関係"]


def convert(tsv_path: Path) -> list[JevExample]:
    examples = []
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            label_str, premise, hypothesis = row[0], row[1], row[2]
            if label_str not in _LABEL_MAP:
                # TODO: 実データのラベル表記に合わせて _LABEL_MAP を修正
                continue
            examples.append(
                JevExample(
                    id=f"jsnli_{i}",
                    source_dataset="jsnli",
                    context=premise,
                    question="この文から確実に言えることは？",
                    candidates=list(_CANDIDATES_JA),
                    label=_LABEL_MAP[label_str],
                    task_type="nli",
                )
            )
    return examples


if __name__ == "__main__":
    import sys

    examples = convert(Path(sys.argv[1]))
    print(f"converted {len(examples)} examples")
    print(examples[0] if examples else "no examples found")
