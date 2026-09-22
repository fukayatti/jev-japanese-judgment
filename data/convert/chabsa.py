"""chABSA-dataset -> Jev形式の決定論的変換。

原著作者: TIS株式会社 (https://github.com/chakki-works/chABSA-dataset, CC BY 4.0)

公式配布 (S3上のzip) は確認時点でリンク切れ (NoSuchBucket) だったため、
コミュニティミラー "zenless-lab/chABSA" (Hugging Face, 元データをparquet化したもの、
2572/643件のtrain/test分割あり) 経由で取得する。CC BY 4.0なので
ミラー経由でも原著作者のクレジットを保てば利用可能。

スキーマ (ミラー側でflatten済み、1行=1文):
  document_id, document_name, doc_text, edi_id, security_code,
  category33, category17, sentence_id, sentence,
  opinions: list[{from, to, label, polarity, text}]
    - text: 評価対象のtarget文字列 (元のJSON仕様の "target" に相当)
    - polarity: "positive" | "negative" | "neutral"  (実データで確認済み)

1つのsentenceに複数opinionがある場合、opinionごとに1つのJevExampleを作る。
"""

from datasets import load_dataset

from data.convert.schema import JevExample

MIRROR_DATASET = "zenless-lab/chABSA"

_POLARITY_LABELS = ["positive", "negative", "neutral"]
_POLARITY_CANDIDATES_JA = ["ポジティブ", "ネガティブ", "中立"]


def convert(split: str = "train") -> list[JevExample]:
    ds = load_dataset(MIRROR_DATASET, split=split)
    examples = []
    for row in ds:
        for o_idx, opinion in enumerate(row["opinions"]):
            polarity = opinion["polarity"]
            if polarity not in _POLARITY_LABELS:
                continue
            label = _POLARITY_LABELS.index(polarity)
            examples.append(
                JevExample(
                    id=f"chabsa_{row['document_id']}_{row['sentence_id']}_{o_idx}",
                    source_dataset="chabsa",
                    context=row["sentence"],
                    question=f"「{opinion['text']}」についての評価は？",
                    candidates=list(_POLARITY_CANDIDATES_JA),
                    label=label,
                    task_type="sentiment",
                )
            )
    return examples


if __name__ == "__main__":
    examples = convert("train")
    print(f"converted {len(examples)} examples")
    print(examples[0])
