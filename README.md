# jev-japanese-judgment

日本語データセットを「判定（judgment）」形式に変換し、非自己回帰的に確率付きの判定を返す小型モデルを作るプロジェクト。

参考: [Jev (TypeSafe AI)](https://en.wikipedia.org/wiki/Jev_(AI_model)) — 文章を生成せず、型付きの判定と確率を返すAIモデル。

## データセット

| データセット | 元タスク | ライセンス | Jev変換後 |
| --- | --- | --- | --- |
| JCommonsenseQA | 5択常識QA | CC BY-SA 4.0 | 5値判定 |
| chABSA-dataset | 決算短信のアスペクト感情分析 | CC BY 4.0 | 3値判定（ポジ/ネガ/中立） |
| JSNLI | 前提・仮説の含意関係 | CC BY-SA 4.0 | 3値判定（成立する/矛盾する/無関係） |

変換したデータセット全体の公開ライセンスは **CC BY-SA 4.0**（元データセットのうち最も制約が強いライセンスを継承）。

## パイプライン

1. **決定論的変換** (`data/convert/`): Pythonスクリプトで元データをJev形式 `{context, question, candidates, label}` に整形
2. **LLM拡張** (`data/augment/`): 言い換え・自然な質問文へのリライト・ダミー選択肢生成のみLLMに担当させる
3. **自動フィルタ** (`data/augment/verify.py`): フォーマット崩れ・重複除外＋自己検証パスで怪しい合成データを除外
4. **シャッフル** (`data/postprocess/`): 候補の順序をシャッフルし、labelインデックスを再計算

## モデル

- ベース: [LiquidAI/LFM2.5-1.2B-JP](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606)（`Lfm2ForCausalLM`、hidden_size=2048、16層）
- `lm_head` を外し、`Lfm2Model` バックボーンに候補スコアリングヘッドを追加（`model/head.py`）
- 候補ごとに `[context][question][candidate_i]` をエンコードし、最終トークンのhidden_stateをスコア化 → softmaxで確率分布を直接出力（テキスト生成なし）
- 学習: listwise cross-entropy + LoRA（`train.py`）
- キャリブレーション: 温度スケーリングによる後処理（`calibrate.py`）。RLベースのキャリブレーションは現時点では見送り

## 評価 (`eval/`)

- `accuracy.py`: タスク別 accuracy、混同行列
- `calibration.py`: ECE、Brier score、NLL、reliability diagram
- `robustness.py`: 候補順序シャッフルへの一貫性、未知の候補数Kへの汎化
- `baselines.py`: 多数派クラス、ヘッド改造前のプロンプトベースライン
- `speed.py`: 推論レイテンシ（このヘッド方式 vs 自己回帰JSON生成）

## 公開物

- データセット / モデル（LoRA + ヘッド、GGUF、量子化版）: [fukayatti0/jev-japanese-judgment](https://huggingface.co/fukayatti0/jev-japanese-judgment)（Hugging Face Hub）
- v2（noul/score追加学習版、LoRA+ヘッド、GGUF Q4_K_M/Q8_0）: [fukayatti0/jev-japanese-judgment-v2](https://huggingface.co/fukayatti0/jev-japanese-judgment-v2)

## 結果

学習に一度も使っていない分割（JCommonsenseQA validation / chABSA test / JSNLI dev、各400件、計1200件）での評価。
`scripts/build_clean_eval.py` で作成（評価セットはHFの `eval/clean_eval.jsonl`）。chABSAのtestは文単位の分割のため、
同じ企業文書の別文が学習に含まれる可能性がある。

| 項目 | 値 |
| --- | --- |
| Accuracy（epoch1、bf16） | 92.2% |
| ECE（キャリブレーション誤差） | 1.45% |
| タスク別 | commonsense_qa 91.8% / nli 92.3% / sentiment 92.5% |

以前は、学習に渡した26,150件の中から抜き出したサンプル（`train_val_split`）で評価しており、学習データと重なっていた
（accuracy 93.8% / ECE 1.14%）。未使用データでは約1.6ポイント低く、特にJCommonsenseQAで差が大きい（98.4% → 91.8%）。

## ローカルPCで動かす（GPU不要）

量子化方式ごとの精度（未使用データ1200件、`scripts/gguf_pipeline.py` で測定）:

| 方式 | サイズ | Accuracy | ECE |
| --- | --- | --- | --- |
| bf16（基準、GPU） | 約2.3GB | 92.2% | 0.0145 |
| **GGUF Q8_0**（LoRAマージ済み、llama.cpp） | 1.2GB | 92.1% | 0.0154 |
| **GGUF Q4_K_M**（LoRAマージ済み） | 698MB | 91.5% | 0.0183 |
| **GGUF Q4_0**（LoRAマージ済み） | 664MB | 91.8% | 0.0210 |

n=1200の標準誤差は約0.8ポイントで、Q4_0とQ4_K_Mの優劣は判別できない。Q8_0はbf16とほぼ同じ精度で、Q4系は0.4〜0.7ポイント下がる程度。

PyTorchの動的int8（per-channel、CPU）は、学習データと重なる200件（絶対値は過大）での比較で、bf16の97.0%に対して89.5%と大きく落ちた。
同じ200件でのGGUFは95.5〜97.0%だったため、ローカル実行にはGGUF版を推奨する。選択的量子化・QAT・ONNXも試したが、
精度悪化または変換不可で不採用（`qat.py`、`scripts/ask.py` のdocstringに経緯を記載）。

### GGUF版（推奨）

LoRAをマージした1ファイルのGGUFをllama.cppで動かし（`--pooling last` で最終トークンのhidden stateを取得）、自作ヘッド（小さなMLP）はPython側で計算する。llama.cppのC++側の改造は不要。

```bash
# 1. llama.cppをビルド（初回のみ）
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build -DLLAMA_CURL=OFF && cmake --build build -j2 --target llama-embedding && cd ..

# 2. 質問する（Python標準ライブラリのみ、モデルは初回に自動ダウンロード）
python3 scripts/ask_gguf_lite.py --llama-embedding llama.cpp/build/bin/llama-embedding \
  --question "日本の首都はどこ？" --candidates "大阪,東京,京都,名古屋"
```

毎回モデルを読み込むので1回あたり1.2〜1.6秒かかる。対話的に何度も質問するなら、`llama-server`を内部で常駐させる`--repl`が速い（2問目以降は1問0.5〜0.7秒程度、4スレッドの実測）:

```bash
python3 scripts/ask_gguf_lite.py --llama-embedding llama.cpp/build/bin/llama-embedding --repl
# llama-serverも同じディレクトリにビルドしておく: cmake --build build -j2 --target llama-server
```

`--serve`でサーバーだけ起動しておき、別プロセスから`--server-url http://127.0.0.1:8089 --question ... --candidates ...`で質問することもできる。

LoRAをマージしていない別ファイル版（`gguf/lfm2-base-*.gguf` + `gguf/jev-lora-f16.gguf`）もHFにあり、`--model` / `--lora` で指定できる（精度は同等）。`scripts/ask_gguf.py` はnumpy + huggingface_hubを使う同等版。`--quant Q8_0` などで量子化タイプを選べる（初回のみ追加ダウンロード）。GGUF変換・LoRAマージ・評価・公開は `python -m scripts.gguf_pipeline`（Colab想定）。

### PyTorch版（GPUあり / 量子化CPU）

```bash
python -m scripts.ask --question "日本の首都はどこ？" --candidates "大阪,東京,京都,名古屋"
python -m scripts.ask --device cpu --quantize ...   # 動的int8（精度は下がる）
```

## JevBench（英語）での目安

[JevBench](https://github.com/fstandhartinger/jevbench)（英語のみの、型付き判断ベンチマーク）の公開階層を測った目安。
このモデルは日本語の3タスクで学習しているので、これは範囲外での位置を見るための値で、日本語専門モデルの成績ではない。
`integrations/jevbench/` のアダプタで、GGUF（Q4_K_M、LoRAマージ済み）を llama-server（T4 GPU）で動かして測定。
測ったのは公開階層（easy 48 / original 72 / hard 111問）の正解率で、非公開の階層と総合スコアは測っていない。

| 階層 | v1 | v2（noul/score追加学習） |
| --- | --- | --- |
| easy | 87.5%（42/48） | 93.8%（45/48） |
| original | 51.4%（37/72） | 50.0%（36/72） |
| hard | 35.1%（39/111） | 40.5%（45/111） |
| 3階層の合計 | 51.1%（118/231） | 54.5%（126/231） |

- v2は、JNLI/JCoLA（yes/no）とJSTS/JSICK（段階評価）を、既存3タスクを混ぜてv1から追加学習したもの。日本語の未使用データでは、
  JSTS 20.5%→58.0%、JSICK 28.0%→70.3%と大きく伸びたが、JevBenchでは差は8問分で、ノイズの範囲と考えるのが妥当。
  JevBenchのordinal（段階評価）は両方66.7%で変わらず、日本語での伸びは移っていない。
- 元の3タスク（未使用データ）は、v1の92.2%からv2で90.7%に下がった。
- 確率の校正は、難しさで向きが逆にずれる。JevBenchでは、NLL最適の温度がeasyで0.4（自信不足）、originalで2.1、hardで5.1（自信過剰）。
  日本語の範囲内は、最適T≈1.1〜1.2でほぼ校正済み。温度スケーリング（`scripts/temperature.py`）は、範囲内のECEの改善が
  ノイズ程度（評価セットによっては悪化）で、範囲外は向きが逆のため、1つの温度では直せず、適用していない。
- 遅延（p50）は0.21〜0.33秒（T4、ネットワークなし）。JevBench公式の遅延（ネットワーク込み）とは条件が違うので比べられない。
- easy/hard の失敗は、当初 llama-server のコンテキスト共有が原因で6問落ちていたため、コンテキストを16384に上げ、
  長い問題は候補を分けて送る形に直して測り直した（失敗0件）。

## Colabでの実行

`notebooks/colab_launcher.ipynb` はこのリポジトリを clone して依存をインストールするだけの薄いランチャー。実際のロジックは全てこのリポジトリのモジュールに置く。学習済みチェックポイントや生成データはGoogle Drive/Hugging Face Hubに保存し、gitにはコードのみを置く。

## ライセンス

- コード: MIT (`LICENSE`)
- 変換済みデータセット: CC BY-SA 4.0（Hugging Face Hub公開時に別途表示）
- モデル: ベースの [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606/blob/main/LICENSE) に従う（GGUF版は量子化したベース重みを含む）
