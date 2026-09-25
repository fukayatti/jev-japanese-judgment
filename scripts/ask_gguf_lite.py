"""GGUF版Jevモデルに質問を投げる、Python標準ライブラリだけ版(numpy/torch不要)。

必要なもの: ビルド済みの llama.cpp の `llama-embedding`。モデルファイルは
--model/--lora/--head で直接指定するか、省略してHugging Face Hubから取得する
(取得時のみ huggingface_hub が必要)。

使い方:
  python scripts/ask_gguf_lite.py --llama-embedding /path/to/llama-embedding \\
    --question "日本の首都はどこ？" --candidates "大阪,東京,京都,名古屋"
"""

import argparse
import ast
import json
import math
import operator
import shutil
import struct
import subprocess
import tempfile
import zipfile
from array import array
from pathlib import Path

REPO_ID = "fukayatti0/jev-japanese-judgment"
SEP = "<#sep#>"


def _read_npy(raw: bytes) -> tuple[tuple, array]:
    major = raw[6]
    hlen, off = (struct.unpack("<H", raw[8:10])[0], 10) if major == 1 else (struct.unpack("<I", raw[8:12])[0], 12)
    header = ast.literal_eval(raw[off : off + hlen].decode("latin1"))
    if header["descr"] != "<f4" or header["fortran_order"]:
        raise ValueError(f"想定外のnpy形式: {header}")
    data = array("f")
    data.frombytes(raw[off + hlen :])
    return header["shape"], data


def load_head(npz_path: Path) -> dict[str, tuple[tuple, array]]:
    with zipfile.ZipFile(npz_path) as z:
        return {n[: -len(".npy")]: _read_npy(z.read(n)) for n in z.namelist()}


def head_score(w: dict, x: list[float]) -> float:
    (rows, cols), w0 = w["mlp.0.weight"]
    b0 = w["mlp.0.bias"][1]
    hidden = []
    for r in range(rows):
        h = sum(map(operator.mul, w0[r * cols : (r + 1) * cols], x)) + b0[r]
        hidden.append(0.5 * h * (1.0 + math.erf(h / math.sqrt(2.0))))
    return sum(map(operator.mul, w["mlp.2.weight"][1], hidden)) + w["mlp.2.bias"][1][0]


def embed(binary: str, model: Path, lora: Path, texts: list[str], threads: int) -> list[list[float]]:
    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "prompts.txt"
        pf.write_text(SEP.join(texts), encoding="utf-8")
        cmd = [
            binary, "-m", model, "--lora", lora, "-f", pf, "--embd-separator", SEP,
            "--pooling", "last", "--embd-normalize", "-1", "--embd-output-format", "json",
            "-t", threads, "-c", 2048, "-b", 2048, "-ub", 2048, "-ngl", 0, "--no-warmup",
        ]
        out = subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True).stdout
    return [d["embedding"] for d in json.loads(out)["data"]]


def _main() -> None:
    ap = argparse.ArgumentParser(description="GGUF版Jevモデルに質問を投げる(標準ライブラリのみ)")
    ap.add_argument("--question", required=True)
    ap.add_argument("--candidates", required=True, help="カンマ区切りの候補一覧")
    ap.add_argument("--context", default="")
    ap.add_argument("--llama-embedding", default=shutil.which("llama-embedding"))
    ap.add_argument("--quant", default="Q4_K_M", choices=["Q4_0", "Q4_K_M", "Q8_0"])
    ap.add_argument("--model", type=Path, help="ベースGGUFのパス(省略時はHubから取得)")
    ap.add_argument("--lora", type=Path, help="LoRA GGUFのパス")
    ap.add_argument("--head", type=Path, help="head.npzのパス")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if not a.llama_embedding:
        raise SystemExit("llama-embedding が見つからない。--llama-embedding で指定すること")

    if not (a.model and a.lora and a.head):
        from huggingface_hub import hf_hub_download

        fetch = lambda name: Path(hf_hub_download(REPO_ID, f"gguf/{name}"))
        a.model = a.model or fetch(f"lfm2-base-{a.quant}.gguf")
        a.lora = a.lora or fetch("jev-lora-f16.gguf")
        a.head = a.head or fetch("head.npz")

    candidates = [c.strip() for c in a.candidates.split(",") if c.strip()]
    texts = [f"{a.context}\n{a.question}\n{c}" for c in candidates]
    w = load_head(a.head)
    scores = [head_score(w, x) for x in embed(a.llama_embedding, a.model, a.lora, texts, a.threads)]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    probs = [e / sum(exps) for e in exps]

    print(f"question: {a.question}")
    print(f"predicted: {candidates[scores.index(m)]}")
    for c, p in sorted(zip(candidates, probs), key=lambda cp: -cp[1]):
        print(f"  {c}: {p:.4f}")


if __name__ == "__main__":
    _main()
