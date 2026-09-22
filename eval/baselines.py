"""比較用ベースライン: 多数派クラス、ヘッド改造前のプロンプトベースLFM。"""

from collections import Counter

from data.convert.schema import JevExample

MAJORITY_BASELINE_PROMPT = """次の質問に対して、選択肢の番号(0始まり)だけを答えてください。

文脈: {context}
質問: {question}
選択肢:
{candidates_block}
"""


def majority_class_accuracy(examples: list[JevExample]) -> float:
    labels = [ex.label for ex in examples]
    most_common_label, count = Counter(labels).most_common(1)[0]
    return count / len(labels)


def build_prompt_baseline_inputs(examples: list[JevExample]) -> list[str]:
    """ヘッド改造前のLFM2.5-1.2B-JPにそのままchat形式で聞くためのプロンプトを作る。
    実際の推論・パースはColab上でchat_template.jinjaを使って行うこと。
    """
    prompts = []
    for ex in examples:
        candidates_block = "\n".join(f"{i}: {c}" for i, c in enumerate(ex.candidates))
        prompts.append(
            MAJORITY_BASELINE_PROMPT.format(
                context=ex.context, question=ex.question, candidates_block=candidates_block
            )
        )
    return prompts
