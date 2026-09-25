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

## 結果

held-out（学習に使っていないデータ）での評価。

| 項目 | 値 |
| --- | --- |
| Accuracy（epoch1、held-out） | 93.8% |
| ECE（キャリブレーション誤差） | 1.14% |

## ローカルPCで動かす（GPU不要）

量子化方式ごとの精度（held-out 200件、`scripts/ask.py` の `load()` と `scripts/gguf_pipeline.py` で測定）:

| 方式 | サイズ | Accuracy |
| --- | --- | --- |
| bf16（基準、GPU） | 約2.3GB | 97.0% |
| **GGUF Q8_0**（LoRAマージ済み、llama.cpp） | 1.2GB | 97.0% |
| **GGUF Q4_0**（LoRAマージ済み） | 664MB | 95.5% |
| **GGUF Q4_K_M**（LoRAマージ済み） | 698MB | 96.0% |
| PyTorch 動的int8（per-channel、CPU） | - | 89.5% |

n=200なので誤差は約±1.3ポイント（Q4_0とQ4_K_Mの優劣は判別できない）。PyTorchの動的int8は精度が大きく落ちるため、ローカル実行にはGGUF版を推奨する。選択的量子化・QAT・ONNXも試したが、精度悪化または変換不可で不採用（`qat.py`、`scripts/ask.py` のdocstringに経緯を記載）。

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

LoRAをマージしていない別ファイル版（`gguf/lfm2-base-*.gguf` + `gguf/jev-lora-f16.gguf`）もHFにあり、`--model` / `--lora` で指定できる（精度は同等）。`scripts/ask_gguf.py` はnumpy + huggingface_hubを使う同等版。`--quant Q8_0` などで量子化タイプを選べる（初回のみ追加ダウンロード）。GGUF変換・LoRAマージ・評価・公開は `python -m scripts.gguf_pipeline`（Colab想定）。

### PyTorch版（GPUあり / 量子化CPU）

```bash
python -m scripts.ask --question "日本の首都はどこ？" --candidates "大阪,東京,京都,名古屋"
python -m scripts.ask --device cpu --quantize ...   # 動的int8（精度は下がる）
```

## Colabでの実行

`notebooks/colab_launcher.ipynb` はこのリポジトリを clone して依存をインストールするだけの薄いランチャー。実際のロジックは全てこのリポジトリのモジュールに置く。学習済みチェックポイントや生成データはGoogle Drive/Hugging Face Hubに保存し、gitにはコードのみを置く。

## ライセンス

- コード: MIT (`LICENSE`)
- 変換済みデータセット: CC BY-SA 4.0（Hugging Face Hub公開時に別途表示）
- モデル: ベースの [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606/blob/main/LICENSE) に従う（GGUF版は量子化したベース重みを含む）
