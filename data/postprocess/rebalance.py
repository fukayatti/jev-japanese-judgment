"""source_dataset間の件数バランスを取る後処理。

JSNLIの正式な学習ファイル(train_w_filtering.tsv)は533,005件あり、
JCommonsenseQA(8,939件)やchABSA(6,150件)と比べて桁違いに大きい。
このまま結合すると実質JSNLI特化のデータセットになってしまうため、
公開前に大きすぎるsource_datasetだけをランダムサンプリングして間引く。

vLLM/GPUは一切使わない(HFからのpull/pushとPythonのリスト操作のみ)ので、
LLM拡張のようなエンジン起動コストは発生しない。
"""

import random
from collections import defaultdict

from data.convert.schema import JevExample


def subsample_by_source(
    examples: list[JevExample], max_per_source: int, seed: int = 42
) -> list[JevExample]:
    """source_datasetごとにmax_per_source件を上限にランダムサンプリングする。
    max_per_source以下のsource_datasetはそのまま全件残す。
    """
    rng = random.Random(seed)
    grouped: dict[str, list[JevExample]] = defaultdict(list)
    for ex in examples:
        grouped[ex.source_dataset].append(ex)

    result: list[JevExample] = []
    for source, group in grouped.items():
        if len(group) > max_per_source:
            result.extend(rng.sample(group, max_per_source))
        else:
            result.extend(group)
    return result
