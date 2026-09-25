"""JevBench用アダプタ: このプロジェクトのGGUF版Jevモデル(llama-server常駐 + numpyなしのヘッド)。

使い方は integrations/jevbench/install.py を参照。--endpoint にはGGUFファイルのパスを渡す。
環境変数:
  JEV_REPO              このリポジトリのパス(scripts/ask_gguf_lite.py を読み込む)
  JEV_LLAMA_SERVER      llama-server バイナリのパス
  JEV_HEAD              head.npz のパス
  JEV_THREADS           スレッド数(既定4)
  JEV_OPTION_MODE       label(既定) | criteria : 候補文にラベルそのものを使うか、採点基準の説明文を使うか
"""
from __future__ import annotations

import atexit
import json
import math
import os
import socket
import subprocess
import sys
import time

from .base import DecisionResult


class JevGgufLocalAdapter:
    name = "jev_gguf_local"
    cost_basis = "self_hosted_local_cpu_compute_excluded"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, revision=None):
        self.endpoint = endpoint
        self.model = model or "jev-japanese-judgment-gguf"
        self.price_input_per_m = 0.0
        self.price_output_per_m = 0.0
        self.revision = revision
        self.option_mode = os.environ.get("JEV_OPTION_MODE", "label")
        self._ready = False
        self._proc = None

    def load(self):
        if self._ready:
            return
        repo = os.environ["JEV_REPO"]
        sys.path.insert(0, repo)
        import scripts.ask_gguf_lite as lite

        self._lite = lite
        self._head = lite.load_head(os.environ["JEV_HEAD"])
        # 前回の中断で古いサーバーが残っていても混ざらないよう、空きポートを使う
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(os.environ.get("JEV_PORT") or sock.getsockname()[1])
        cmd = [os.environ["JEV_LLAMA_SERVER"], "-m", self.endpoint, "--embeddings", "--pooling", "last",
               "-t", os.environ.get("JEV_THREADS", "4"), "-c", "4096", "-b", "4096", "-ub", "4096",
               "-ngl", os.environ.get("JEV_NGL", "0"), "--no-warmup", "--host", "127.0.0.1", "--port", str(port)]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)
        self._url = f"http://127.0.0.1:{port}"
        lite.wait_ready(self._url, self._proc)
        if self._proc.poll() is not None:
            raise RuntimeError("llama-server が起動直後に終了した(ポート競合やGPUメモリ不足の可能性)")
        self._ready = True

    def close(self):
        if self._proc is not None:
            self._proc.terminate()

    @staticmethod
    def build_request(task, option_mode="label"):
        state = task.state if isinstance(task.state, str) else json.dumps(
            task.state, ensure_ascii=False, sort_keys=True)
        criteria = task.question.get("criteria") or {}
        qtype = task.question["type"]
        if qtype == "noul":
            labels = ["no", "yes"]
            rubric = {"no": criteria.get("false", "No"), "yes": criteria.get("true", "Yes")}
            options = labels
        elif qtype == "score":
            labels = list(task.labels)
            rubric = {label: criteria[int(label)] for label in labels}
            options = [rubric[label] for label in labels]
        else:
            labels = list(task.labels)
            rubric = {label: criteria.get(label) or label for label in labels}
            options = labels
        if option_mode == "criteria" and qtype == "choice":
            options = [rubric[label] for label in labels]
        question = task.question["instructions"] + "\nAllowed answers and rubric: " + json.dumps(rubric, ensure_ascii=False)
        if qtype == "noul":
            question += "\nAnswer with yes or no."
        return {"state": state, "question": question, "labels": labels, "options": options, "qtype": qtype}

    def run(self, task):
        result = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        request = self.build_request(task, self.option_mode)
        result.request_body = request
        started = time.perf_counter()
        try:
            self.load()
            started = time.perf_counter()
            texts = [f"{request['state']}\n{request['question']}\n{o}" for o in request["options"]]
            embeddings = self._lite.embed_server(self._url, texts)
            scores = [self._lite.head_score(self._head, x) for x in embeddings]
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            total = sum(exps)
            result.latency_s = time.perf_counter() - started
            result.probs = {label: e / total for label, e in zip(request["labels"], exps)}
            result.usage = {"input_tokens": sum(len(t) for t in texts) // 3, "output_tokens": 0}
            result.raw = {"runtime": {"checkpoint": self.model, "probability_origin": "native-candidate-scorer-softmax",
                                      "option_mode": self.option_mode, "generated_tokens": 0}}
            result.ok = True
        except Exception as exc:  # noqa: BLE001
            result.latency_s = time.perf_counter() - started
            result.error = f"{type(exc).__name__}: {str(exc)[:300]}"
            self._errors_shown = getattr(self, "_errors_shown", 0) + 1
            if self._errors_shown <= 3:
                print(f"[jev_gguf_local] {task.id}: {result.error}", file=sys.stderr, flush=True)
        return result

    def reserve_estimate(self, task):
        return 0.0
