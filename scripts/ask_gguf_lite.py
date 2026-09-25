"""GGUF版Jevモデルに質問を投げる、Python標準ライブラリだけ版(numpy/torch不要)。

必要なもの: ビルド済みの llama.cpp の `llama-embedding` だけ。モデルファイル(LoRAマージ済みの
1ファイル)は --model/--head で直接指定するか、省略するとHugging Face Hubから
~/.cache/jev-gguf/ にダウンロードする(標準ライブラリのurllibを使うので追加インストール不要)。

使い方:
  # 1回だけ質問(毎回モデルを読み込むので1.5秒前後)
  python scripts/ask_gguf_lite.py --llama-embedding /path/to/llama-embedding \\
    --question "日本の首都はどこ？" --candidates "大阪,東京,京都,名古屋"

  # 対話モード(llama-serverを内部で起動して常駐、2問目以降は0.7秒前後)
  python scripts/ask_gguf_lite.py --llama-embedding /path/to/llama-embedding --repl

  # サーバーを別に起動しておき、そこへ質問する
  python scripts/ask_gguf_lite.py --llama-embedding /path/to/llama-embedding --serve
  python scripts/ask_gguf_lite.py --server-url http://127.0.0.1:8089 --question ... --candidates ...
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
import time
import urllib.request
import zipfile
from array import array
from pathlib import Path

REPO_ID = "fukayatti0/jev-japanese-judgment"
SEP = "<#sep#>"


def fetch(name: str, repo: str = REPO_ID) -> Path:
    # v1(既定)は従来のキャッシュ場所のまま。別リポジトリ(v2など)は、同名ファイルが混ざらないようサブディレクトリに置く
    cache = Path.home() / ".cache" / "jev-gguf"
    dest = (cache if repo == REPO_ID else cache / repo.replace("/", "__")) / name
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/gguf/{name}"
    print(f"downloading {url}", flush=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as out:
        shutil.copyfileobj(r, out, 1 << 20)
    part.rename(dest)
    return dest


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


def embed(binary: str, model: Path, lora: Path | None, texts: list[str], threads: int) -> list[list[float]]:
    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "prompts.txt"
        pf.write_text(SEP.join(texts), encoding="utf-8")
        cmd = [
            binary, "-m", model, *(["--lora", lora] if lora else []), "-f", pf, "--embd-separator", SEP,
            "--pooling", "last", "--embd-normalize", "-1", "--embd-output-format", "json",
            "-t", threads, "-c", 2048, "-b", 2048, "-ub", 2048, "-ngl", 0, "--no-warmup",
        ]
        out = subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True).stdout
    return [d["embedding"] for d in json.loads(out)["data"]]


def start_server(binary: str, model: Path, lora: Path | None, threads: int, port: int) -> subprocess.Popen:
    cmd = [
        binary, "-m", model, *(["--lora", lora] if lora else []), "--embeddings", "--pooling", "last",
        "-t", threads, "-c", 2048, "-b", 2048, "-ub", 2048, "-ngl", 0, "--no-warmup",
        "--host", "127.0.0.1", "--port", port,
    ]
    return subprocess.Popen([str(c) for c in cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_ready(url: str, proc: subprocess.Popen | None = None, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise SystemExit("llama-server が起動に失敗した")
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2).read()
            return
        except OSError:
            time.sleep(0.3)
    raise SystemExit("llama-server の起動待ちがタイムアウトした")


def embed_server(url: str, texts: list[str]) -> list[list[float]]:
    req = urllib.request.Request(
        f"{url}/embedding",
        data=json.dumps({"content": texts, "embd_normalize": -1}).encode(),
        headers={"Content-Type": "application/json"},
    )
    items = sorted(json.load(urllib.request.urlopen(req)), key=lambda d: d["index"])
    return [d["embedding"][0] if isinstance(d["embedding"][0], list) else d["embedding"] for d in items]


def judge(w: dict, embeddings: list[list[float]], candidates: list[str]) -> list[tuple[str, float]]:
    scores = [head_score(w, x) for x in embeddings]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    return sorted(zip(candidates, (e / sum(exps) for e in exps)), key=lambda cp: -cp[1])


def show(question: str, ranked: list[tuple[str, float]]) -> None:
    print(f"question: {question}")
    print(f"predicted: {ranked[0][0]}")
    for c, p in ranked:
        print(f"  {c}: {p:.4f}")


def _main() -> None:
    ap = argparse.ArgumentParser(description="GGUF版Jevモデルに質問を投げる(標準ライブラリのみ)")
    ap.add_argument("--question")
    ap.add_argument("--candidates", help="カンマ区切りの候補一覧")
    ap.add_argument("--context", default="")
    ap.add_argument("--llama-embedding", default=shutil.which("llama-embedding"))
    ap.add_argument("--llama-server", help="llama-serverのパス(省略時は--llama-embeddingと同じディレクトリ)")
    ap.add_argument("--quant", default="Q4_K_M", choices=["Q4_0", "Q4_K_M", "Q8_0"])
    ap.add_argument("--repo", default=REPO_ID, help="GGUFを取得するHFモデルリポジトリ(v2: fukayatti0/jev-japanese-judgment-v2)")
    ap.add_argument("--model", type=Path, help="LoRAマージ済みGGUFのパス(省略時はHubから取得)")
    ap.add_argument("--lora", type=Path, help="別ファイル版を使う場合のみ: ベースGGUFを--modelに、LoRA GGUFをここに指定")
    ap.add_argument("--head", type=Path, help="head.npzのパス")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repl", action="store_true", help="サーバーを内部で起動して対話的に質問する")
    ap.add_argument("--serve", action="store_true", help="llama-serverを起動したままにする(別プロセスから質問する用)")
    ap.add_argument("--server-url", help="起動済みのllama-serverのURL(例: http://127.0.0.1:8089)")
    ap.add_argument("--port", type=int, default=8089)
    a = ap.parse_args()

    need_binary = not a.server_url
    if need_binary and not a.llama_embedding:
        raise SystemExit("llama-embedding が見つからない。--llama-embedding で指定すること")
    if not (a.serve or a.repl) and not (a.question and a.candidates):
        raise SystemExit("--question と --candidates を指定するか、--repl / --serve を使うこと")

    a.head = a.head or fetch("head.npz", a.repo)
    w = load_head(a.head)
    proc = None
    url = a.server_url

    if a.serve or a.repl:
        a.model = a.model or fetch(f"jev-{a.quant}.gguf", a.repo)
        server_bin = a.llama_server or str(Path(a.llama_embedding).with_name("llama-server"))
        proc = start_server(server_bin, a.model, a.lora, a.threads, a.port)
        url = f"http://127.0.0.1:{a.port}"
        try:
            wait_ready(url, proc)
        except SystemExit:
            proc.terminate()
            raise
        print(f"llama-server ready: {url}", flush=True)

    try:
        if a.serve:
            proc.wait()
        elif a.repl:
            print("質問と候補を入力(空の質問で終了)。文脈は最初に '文脈 || 質問' の形式でも指定できる。")
            while True:
                q = input("質問> ").strip()
                if not q:
                    break
                context, _, question = q.rpartition("||")
                candidates = [c.strip() for c in input("候補(カンマ区切り)> ").split(",") if c.strip()]
                texts = [f"{context.strip()}\n{question.strip()}\n{c}" for c in candidates]
                show(question.strip(), judge(w, embed_server(url, texts), candidates))
        else:
            candidates = [c.strip() for c in a.candidates.split(",") if c.strip()]
            texts = [f"{a.context}\n{a.question}\n{c}" for c in candidates]
            if url:
                embeddings = embed_server(url, texts)
            else:
                a.model = a.model or fetch(f"jev-{a.quant}.gguf", a.repo)
                embeddings = embed(a.llama_embedding, a.model, a.lora, texts, a.threads)
            show(a.question, judge(w, embeddings, candidates))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    _main()
