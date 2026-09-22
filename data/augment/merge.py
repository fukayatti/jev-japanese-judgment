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
    for r in parsed:
        base = examples_by_id.get(r["example_id"])
        if base is None:
            continue

        if r["kind"] in ("paraphrase", "rewrite_natural"):
            augmented.append(
                base.model_copy(update={"id": f"{base.id}_{r['kind']}", "question": r["output"]})
            )
        elif r["kind"] == "distractor":
            if r["output"] in base.candidates:
                continue
            augmented.append(
                base.model_copy(
                    update={"id": f"{base.id}_distractor", "candidates": base.candidates + [r["output"]]}
                )
            )

    return augmented
