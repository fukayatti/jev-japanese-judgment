# jev-local

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

## Colabでの実行

`notebooks/colab_launcher.ipynb` はこのリポジトリを clone して依存をインストールするだけの薄いランチャー。実際のロジックは全てこのリポジトリのモジュールに置く。学習済みチェックポイントや生成データはGoogle Drive/Hugging Face Hubに保存し、gitにはコードのみを置く。

## ライセンス

- コード: MIT (`LICENSE`)
- 変換済みデータセット: CC BY-SA 4.0（Hugging Face Hub公開時に別途表示）
