"""JSNLI -> Jev形式の決定論的変換。

元データ: https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip
  (京大 黒橋・褚・村脇研, CC BY-SA 4.0)
  train_w_filtering.tsv / train_wo_filtering.tsv / dev.tsv
  TSV, タブ区切り: label, premise, hypothesis
  label: entailment / contradiction / neutral の3値 (実データで確認済み)

注意点 (実データ確認で判明、要修正だった箇所):
  1. テキストは分かち書き済み (単語間に半角スペース区切り)。
     自然な日本語文にするため、変換時にスペースを除去する。
  2. hypothesis を使わないと「前提文から何が言えるか」を判定できないため、
     questionにhypothesisを埋め込む形にする。
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


def _detokenize(text: str) -> str:
    """分かち書き済みテキストのスペースを除去して自然な日本語文に戻す。"""
    return text.replace(" ", "").strip()


def convert(tsv_path: Path) -> list[JevExample]:
    examples = []
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            label_str, premise, hypothesis = row[0], row[1], row[2]
            if label_str not in _LABEL_MAP:
                continue
            premise = _detokenize(premise)
            hypothesis = _detokenize(hypothesis)
            examples.append(
                JevExample(
                    id=f"jsnli_{i}",
                    source_dataset="jsnli",
                    context=premise,
                    question=f"「{hypothesis}」は、この文から確実に言えますか？",
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
