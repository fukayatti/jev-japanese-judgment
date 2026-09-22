"""LLM拡張で使うプロンプトテンプレート集。

拡張の対象は3種類のみ (data/augment/generate.py から呼ばれる):
  - paraphrase: question の言い換え
  - rewrite_natural: 機械的な question を自然な質問文にリライト
  - distractor: candidates に追加するダミー選択肢の生成
"""

PARAPHRASE_PROMPT = """以下の質問文を、意味を変えずに別の言い回しで書き換えてください。
出力は言い換えた質問文のみ。

質問文: {question}
"""

REWRITE_NATURAL_PROMPT = """以下の質問文を、人間が実際に話すような自然な日本語の質問文に書き換えてください。
意味は変えないでください。出力は書き換えた質問文のみ。

質問文: {question}
"""

DISTRACTOR_PROMPT = """以下の文脈と質問に対して、正解ではない、しかし紛らわしいダミーの選択肢を1つ考えてください。
既存の選択肢とは異なるものにしてください。出力はダミー選択肢のテキストのみ。

文脈: {context}
質問: {question}
正解: {correct_answer}
既存の選択肢: {existing_candidates}
"""
