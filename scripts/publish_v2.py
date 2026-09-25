"""v2(noul/score追加学習版)を、モデルカード・GGUF付きでHugging Faceに公開する。

前提(Colabで先に実行しておく):
  1. python -m scripts.gguf_pipeline setup convert merge ... --repo <v2> --work /content/gguf_v2 --quants Q4_K_M Q8_0
  2. python -m scripts.gguf_pipeline eval --merged --ngl 99 --n 0 --quants Q4_K_M Q8_0 --work /content/gguf_v2 \\
       --eval-file data/clean_eval.jsonl        | tee /content/eval_v2_orig.log
     同様に data/clean_eval_extras.jsonl        | tee /content/eval_v2_extras.log
  3. python -m scripts.publish_v2 --orig-log /content/eval_v2_orig.log --extras-log /content/eval_v2_extras.log [--public]
     (--dry-run でカードを書き出すだけ。--public を付けるとリポジトリを公開にする)
"""

import argparse
import ast
import re
from pathlib import Path

V2_REPO = "fukayatti0/jev-japanese-judgment-v2"
V1_REPO = "fukayatti0/jev-japanese-judgment"
GITHUB = "https://github.com/fukayatti/jev-japanese-judgment"
BASE_MODEL = "LiquidAI/LFM2.5-1.2B-JP-202606"
SIZES = {"Q4_0": "664MB", "Q4_K_M": "698MB", "Q8_0": "1.2GB"}

# --- 実測値(bf16、未使用データ)。出所はREADME/ノートブックの測定ログ ---
V1_ORIG = {"acc": 0.9217, "cqa": 0.9175, "sent": 0.925, "nli": 0.9225, "ece": 0.0145}
V2_ORIG = {"acc": 0.9067, "cqa": 0.89, "sent": 0.9375, "nli": 0.8925, "ece": 0.0258}
V1_EXTRAS = {"jnli": 0.8075, "jcola": 0.5175, "jsts": 0.205, "jsick": 0.28, "all": 0.4525, "ece": 0.1245}
V2_EXTRAS = {"jnli": 0.8575, "jcola": 0.58, "jsts": 0.58, "jsick": 0.7025, "all": 0.68, "ece": 0.0207}

_LINE = re.compile(r"^\[(?:merged-)?(\w+)\] overall=([\d.]+) n=(\d+) (\{.*?\}) (\{.*\})\s*$")


def parse_eval_log(path: Path) -> dict[str, dict]:
    """gguf_pipeline evalの出力行 `[merged-Q4_K_M] overall=0.9 n=1200 {タスク別} {ece,brier,nll}` を読む。"""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _LINE.match(line.strip())
        if m:
            q, overall, n, per_task, cal = m.groups()
            out[q] = {"overall": float(overall), "n": int(n), "tasks": ast.literal_eval(per_task), **ast.literal_eval(cal)}
    return out


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_card(orig: dict, extras: dict) -> str:
    def gguf_rows(res, keys):
        return "\n".join(
            f"| gguf/jev-{q}.gguf | {SIZES.get(q, '?')} | "
            + " | ".join(_pct(m['tasks'][k]) for k in keys)
            + f" | {_pct(m['overall'])} | {m['ece']:.4f} |"
            for q, m in res.items()
        )

    return f"""---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/{BASE_MODEL}/blob/main/LICENSE
base_model: {BASE_MODEL}
language:
- ja
tags:
- jev
- judgment
- classification
- lora
- peft
- gguf
---

# {V2_REPO}

[{V1_REPO}]({'https://huggingface.co/' + V1_REPO}) (v1) を出発点に、**yes/no(noul)と段階評価(score)** の判定を追加学習した版。
{BASE_MODEL} をバックボーンに、文脈と候補から確率を直接返す候補スコアリング型のモデル(生成なし)。
標準の `transformers` では読み込めない。コードは {GITHUB} 。

## v1からの変更

- 追加学習データ(約2.97万件): JNLI(yes/no、はい/いいえを半々に調整) + JCoLA(文法的に自然か、yes/no、半々に調整)
  + JSTS(0〜5段階、段階ごとに上限1500件) + JSICK(1〜5段階、上限800件) + v1の学習データから各ソース最大3400件(忘却対策)。
- 学習: v1(epoch1)のLoRA+ヘッドから、1エポック、学習率1e-4。
- 評価には学習に一度も使っていない分割(JNLI test / JCoLA valid / JSTS validation / JSICK test / JCommonsenseQA validation /
  chABSA test / JSNLI dev)だけを使っている。

## 評価結果 (bf16、未使用データ)

**追加したタスク(1600件、yes/noは半々の200+200なのでチャンスは50%):**

| 評価セット | v1 | v2 |
| --- | --- | --- |
| JNLI (yes/no) | {_pct(V1_EXTRAS['jnli'])} | **{_pct(V2_EXTRAS['jnli'])}** |
| JCoLA (yes/no) | {_pct(V1_EXTRAS['jcola'])} | **{_pct(V2_EXTRAS['jcola'])}** |
| JSTS (0〜5段階、チャンス約17%) | {_pct(V1_EXTRAS['jsts'])} | **{_pct(V2_EXTRAS['jsts'])}** |
| JSICK (1〜5段階、チャンス20%) | {_pct(V1_EXTRAS['jsick'])} | **{_pct(V2_EXTRAS['jsick'])}** |
| 全体 | {_pct(V1_EXTRAS['all'])} | **{_pct(V2_EXTRAS['all'])}** |
| ECE | {V1_EXTRAS['ece']:.4f} | {V2_EXTRAS['ece']:.4f} |

**元の3タスク(1200件):**

| 評価セット | v1 | v2 |
| --- | --- | --- |
| commonsense_qa | {_pct(V1_ORIG['cqa'])} | {_pct(V2_ORIG['cqa'])} |
| sentiment | {_pct(V1_ORIG['sent'])} | {_pct(V2_ORIG['sent'])} |
| nli | {_pct(V1_ORIG['nli'])} | {_pct(V2_ORIG['nli'])} |
| 全体 | {_pct(V1_ORIG['acc'])} | {_pct(V2_ORIG['acc'])} |
| ECE | {V1_ORIG['ece']:.4f} | {V2_ORIG['ece']:.4f} |

## GGUF版 (llama.cpp / CPU・GPU向け、LoRAマージ済み)

`gguf/jev-*.gguf` と `gguf/head.npz`。`llama-embedding --pooling last` のhidden stateに、`head.npz` の小さなMLPヘッドを適用する。
使い方は {GITHUB} の `scripts/ask_gguf_lite.py`(標準ライブラリのみ)。

追加タスク(未使用データ{next(iter(extras.values()))['n']}件):

| ファイル | サイズ | JNLI | JCoLA | JSTS | JSICK | 全体 | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
{gguf_rows(extras, ['jnli_noul', 'jcola_noul', 'jsts_score', 'jsick_score'])}

元の3タスク(未使用データ{next(iter(orig.values()))['n']}件):

| ファイル | サイズ | commonsense_qa | sentiment | nli | 全体 | ECE |
| --- | --- | --- | --- | --- | --- | --- |
{gguf_rows(orig, ['commonsense_qa', 'sentiment', 'nli'])}

## JevBench(英語)での目安

[JevBench](https://github.com/fstandhartinger/jevbench)の公開階層を、GGUF(Q4_K_M)でT4 GPUのllama-serverから測った正解率(範囲外の目安)。

| 階層 | v1 | v2 |
| --- | --- | --- |
| easy (48問) | 87.5% | 93.8% |
| original (72問) | 51.4% | 50.0% |
| hard (111問) | 35.1% | 40.5% |
| 合計 (231問) | 51.1% | 54.5% |

差は8問分で、ノイズの範囲と考えるのが妥当。日本語で+40ポイント伸びたJSTS/JSICKの段階評価は、JevBenchのordinal(両方66.7%)には移っていない。

## 制約

- JCoLA(文法的な自然さ)は58.0%で、チャンス(50%)に近い。文法性の判定は、この規模の追加学習ではほとんど身についていない。
- 元の3タスクは、v1から全体で約1.5ポイント下がり、ECEもやや悪化している(1回の学習の結果で、ばらつきは未測定)。
- 学習した範囲(日本語の常識QA、感情、含意、文のペアの類似度、文法性)の外では、確率の校正がずれる(JevBenchのECEは0.14〜0.21)。
  向きが難しさで逆で、易しい問題(easy)は確率が低すぎ(自信不足、NLL最適の温度T=0.4)、難しい問題(original/hard)は高すぎ
  (自信過剰、最適T=2.1/5.1)なので、温度スケーリング1つでは直せない。日本語の範囲内は最適T≈1.1〜1.2でほぼ校正済みで、
  温度を入れてもECEの改善は評価セットによっては悪化する程度に小さかったため、適用していない。
- chABSAのtestは文単位の分割のため、同じ企業文書の別文が学習に含まれる可能性がある。

## ライセンス

- コード: MIT ({GITHUB})
- ベースモデル: [LFM Open License v1.0](https://huggingface.co/{BASE_MODEL}/blob/main/LICENSE)。GGUF版は量子化したベース重みを含む。
- 学習データ: JCommonsenseQA / JNLI / JSTS / JCoLA / JSNLI(CC BY-SA 4.0)、chABSA(CC BY 4.0)、JSICK(CC BY 4.0またはCC BY-SA 4.0、配布元の表記に食い違いあり)。
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-log", type=Path, required=True)
    ap.add_argument("--extras-log", type=Path, required=True)
    ap.add_argument("--work", default="/content/gguf_v2")
    ap.add_argument("--quants", nargs="+", default=["Q4_K_M", "Q8_0"])
    ap.add_argument("--public", action="store_true", help="アップロード後にリポジトリを公開にする")
    ap.add_argument("--dry-run", action="store_true", help="カードをローカルに書き出すだけ(送信しない)")
    ap.add_argument("--card-only", action="store_true", help="README.mdだけ更新する(GGUFは再アップロードしない)")
    ap.add_argument("--out", type=Path, default=Path("README_v2_card.md"))
    a = ap.parse_args()

    orig = {q: m for q, m in parse_eval_log(a.orig_log).items() if q in a.quants}
    extras = {q: m for q, m in parse_eval_log(a.extras_log).items() if q in a.quants}
    missing = [q for q in a.quants if q not in orig or q not in extras]
    if missing:
        raise SystemExit(f"評価ログに結果が無い量子化: {missing}")
    card = render_card(orig, extras)
    if a.dry_run:
        a.out.write_text(card, encoding="utf-8")
        print(f"wrote {a.out} ({len(card)} chars)")
        return

    from huggingface_hub import HfApi

    from scripts.push_to_hub import _get_hf_token

    api = HfApi(token=_get_hf_token())
    work = Path(a.work)
    if not a.card_only:
        for q in a.quants:
            api.upload_file(path_or_fileobj=str(work / f"merged-{q}.gguf"), path_in_repo=f"gguf/jev-{q}.gguf", repo_id=V2_REPO)
        api.upload_file(path_or_fileobj=str(work / "head.npz"), path_in_repo="gguf/head.npz", repo_id=V2_REPO)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=V2_REPO)
    if a.public:
        api.update_repo_settings(repo_id=V2_REPO, private=False)
    print("uploaded to", V2_REPO, "(public)" if a.public else "(private)")


if __name__ == "__main__":
    main()
