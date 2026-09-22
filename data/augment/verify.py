"""合成データの自動フィルタ。

1. フォーマット崩れ・重複・空文字の除外
2. distractor(ダミー選択肢)については、別プロンプトで
   「これは本当に不正解か？」をLLMに再検証させ、疑わしいものを除外
"""

VERIFY_DISTRACTOR_PROMPT = """以下の文脈・質問において、候補は本当に不正解ですか？「はい」か「いいえ」のみで答えてください。

文脈: {context}
質問: {question}
候補: {candidate}
正解: {correct_answer}
"""


def filter_format(results: list[dict]) -> list[dict]:
    seen = set()
    filtered = []
    for r in results:
        text = r["output"].strip()
        if not text:
            continue
        key = (r["example_id"], r["kind"], text)
        if key in seen:
            continue
        seen.add(key)
        filtered.append({**r, "output": text})
    return filtered


def build_distractor_verification_prompts(distractor_results: list[dict], examples_by_id: dict) -> list[dict]:
    jobs = []
    for r in distractor_results:
        ex = examples_by_id[r["example_id"]]
        jobs.append(
            {
                "example_id": r["example_id"],
                "candidate": r["output"],
                "prompt": VERIFY_DISTRACTOR_PROMPT.format(
                    context=ex.context,
                    question=ex.question,
                    candidate=r["output"],
                    correct_answer=ex.candidates[ex.label],
                ),
            }
        )
    return jobs


def apply_verification(distractor_results: list[dict], verification_answers: list[str]) -> list[dict]:
    """verification_answers[i] が "はい" (=本当に不正解) の場合のみ残す。"""
    kept = []
    for r, answer in zip(distractor_results, verification_answers):
        if answer.strip().startswith("はい"):
            kept.append(r)
    return kept
