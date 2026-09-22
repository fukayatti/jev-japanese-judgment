"""LLM拡張結果(paraphrase/rewrite_natural/distractor)を元のJevExampleにマージし、
派生した新しいJevExampleのリストを作る。

vllm_worker.py は structured_outputs で {"text": "..."} 形式のJSON文字列を
出力させているので、まずそれをパースしてから合流させる。
"""

import json

from data.augment.verify import filter_format
from data.convert.schema import JevExample


def _parse_output_text(raw_output: str) -> str | None:
    """{"text": "..."} 形式のJSON文字列から実際のテキストを取り出す。"""
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return None
    text = parsed.get("text", "").strip()
    return text or None


def merge_augmentations(examples: list[JevExample], results: list[dict]) -> list[JevExample]:
    """
    paraphrase/rewrite_natural: 同じcontext/candidates/labelで questionだけ差し替えた
      派生exampleを作る。
    distractor: 元の候補リストに1つ追加した派生exampleを作る
      (labelのインデックスは変わらない。シャッフルは postprocess 側で行う)。
    """
    examples_by_id = {ex.id: ex for ex in examples}

    parsed = []
    for r in results:
        text = _parse_output_text(r["output"])
        if text is not None:
            parsed.append({**r, "output": text})
    parsed = filter_format(parsed)

    augmented: list[JevExample] = []
    dropped = 0
    for r in parsed:
        base = examples_by_id.get(r["example_id"])
        if base is None:
            continue
        text = r["output"]

        if r["kind"] in ("paraphrase", "rewrite_natural"):
            # 元の質問文のコピーはリライトになっていないので捨てる。
            # 元の2.5倍以上長い場合は、無関係な文を書き足すハルシネーションの
            # 疑いが強いので捨てる(実データで「別の質問を捏造する」事例を確認済み)。
            if text == base.question:
                dropped += 1
                continue
            if len(text) > len(base.question) * 1.8:
                dropped += 1
                continue
            augmented.append(base.model_copy(update={"id": f"{base.id}_{r['kind']}", "question": text}))

        elif r["kind"] == "distractor":
            # ダミー選択肢のはずが質問文をまるごと返す事例を実データで確認済み。
            # 「？」を含む(=質問文っぽい)、または既存候補より極端に長い場合は
            # 短い答えの候補になっていないとみなして捨てる。
            if text in base.candidates:
                dropped += 1
                continue
            if "？" in text or "?" in text:
                dropped += 1
                continue
            max_candidate_len = max(len(c) for c in base.candidates)
            if len(text) > max_candidate_len * 2:
                dropped += 1
                continue
            augmented.append(
                base.model_copy(update={"id": f"{base.id}_distractor", "candidates": base.candidates + [text]})
            )

    if dropped:
        print(f"merge_augmentations: dropped {dropped} low-quality results")

    return augmented
