"""chABSA-dataset -> Jev形式の決定論的変換。

元データ (https://github.com/chakki-works/chABSA-dataset, JSON):
  各ファイルが1社の決算短信。
  {
    "sentences": [
      {
        "sentence": str,
        "opinions": [
          {"target": str, "category": str, "polarity": "positive"|"negative"|"neutral", ...},
          ...
        ]
      },
      ...
    ]
  }

1つの sentence に複数の opinion (target) がある場合、target ごとに1つのJevExampleを作る。
polarity の正確な取りうる値・表記は実データで要確認 (TODO)。
"""

import json
from pathlib import Path

from data.convert.schema import JevExample

_POLARITY_LABELS = ["positive", "negative", "neutral"]
_POLARITY_CANDIDATES_JA = ["ポジティブ", "ネガティブ", "中立"]


def convert(json_dir: Path) -> list[JevExample]:
    examples = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for s_idx, sentence in enumerate(data["sentences"]):
            for o_idx, opinion in enumerate(sentence.get("opinions", [])):
                polarity = opinion["polarity"]
                if polarity not in _POLARITY_LABELS:
                    # TODO: 実データで想定外の極性表記が出たら要対応
                    continue
                label = _POLARITY_LABELS.index(polarity)
                examples.append(
                    JevExample(
                        id=f"chabsa_{path.stem}_{s_idx}_{o_idx}",
                        source_dataset="chabsa",
                        context=sentence["sentence"],
                        question=f"「{opinion['target']}」についての評価は？",
                        candidates=list(_POLARITY_CANDIDATES_JA),
                        label=label,
                        task_type="sentiment",
                    )
                )
    return examples


if __name__ == "__main__":
    import sys

    examples = convert(Path(sys.argv[1]))
    print(f"converted {len(examples)} examples")
    print(examples[0] if examples else "no examples found")
