"""JevモデルをGGUF(llama.cpp)で動かすためのパイプライン。

方針: バックボーン(+LoRA)だけをllama.cppで動かし、`--pooling last`で最終トークンの
hidden stateを取り出す。自作スコアリングヘッド(小さなMLP)はnumpyで計算するので、
llama.cppのC++側を改造する必要はない。

ステージ(Colabで上から順に実行する想定):
  setup    llama.cppをclone・ビルド
  convert  ベース→f16 GGUF→量子化、LoRA→GGUF(キー名に"model."を補う)
  verify   PyTorch(GPU)版とpooled vector/スコアを比較
  eval     held-outでaccuracyを測る

例: python -m scripts.gguf_pipeline setup convert verify eval --quants Q4_0 Q4_K_M
"""

import argparse
import json
import subprocess
import tempfile
from math import erf
from pathlib import Path

import numpy as np

from data.convert.schema import JevExample
from model.head import BASE_MODEL_NAME

REPO_ID = "fukayatti0/jev-japanese-judgment"
SEP = "<#sep#>"


class Paths:
    def __init__(self, work: str):
        self.work = Path(work)
        self.llama = self.work / "llama.cpp"
        self.bin = self.llama / "build" / "bin"
        self.f16 = self.work / "lfm2-base-f16.gguf"
        self.lora = self.work / "jev-lora-f16.gguf"
        self.hub = self.work / "jev_repo"

    def quant(self, qtype: str) -> Path:
        return self.work / f"lfm2-base-{qtype}.gguf"


def _run(cmd: list, **kw) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def setup(p: Paths) -> None:
    p.work.mkdir(parents=True, exist_ok=True)
    if not p.llama.exists():
        _run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", p.llama])
    _run(["cmake", "-B", "build", "-DGGML_NATIVE=ON", "-DLLAMA_CURL=OFF"], cwd=p.llama)
    _run(["cmake", "--build", "build", "-j", "--target", "llama-quantize", "llama-embedding"], cwd=p.llama)
    _run(["pip", "install", "-q", "-e", p.llama / "gguf-py", "sentencepiece", "protobuf", "safetensors"])


def convert(p: Paths, quants: list[str]) -> None:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    base_dir = snapshot_download(BASE_MODEL_NAME)
    hub_dir = snapshot_download(REPO_ID, local_dir=str(p.hub))
    conv = p.llama / "convert_hf_to_gguf.py"
    lora_conv = p.llama / "convert_lora_to_gguf.py"
    py = "python"

    if not p.f16.exists():
        _run([py, conv, base_dir, "--outfile", p.f16, "--outtype", "f16"])
    for q in quants:
        if not p.quant(q).exists():
            _run([p.bin / "llama-quantize", p.f16, p.quant(q), q])

    # AutoModel(Lfm2Model)で学習したアダプタのキーは "base_model.model.layers.*" で、
    # llama.cppのコンバータが期待するCausalLM形式 "base_model.model.model.layers.*" と
    # 1段ずれているため、補ったコピーを作ってから変換する。
    fixed = p.work / "lora_fixed"
    fixed.mkdir(exist_ok=True)
    (fixed / "adapter_config.json").write_text((Path(hub_dir) / "lora" / "adapter_config.json").read_text())
    sd = load_file(str(Path(hub_dir) / "lora" / "adapter_model.safetensors"))
    sd = {k.replace("base_model.model.layers.", "base_model.model.model.layers."): v.contiguous() for k, v in sd.items()}
    save_file(sd, str(fixed / "adapter_model.safetensors"), metadata={"format": "pt"})
    _run([py, lora_conv, "--base", base_dir, "--outfile", p.lora, "--outtype", "f16", fixed])


class GGUFScorer:
    """llama-embeddingでpooled vectorを取り、numpyでヘッドを計算する。"""

    def __init__(self, p: Paths, qtype: str, threads: int = 4):
        import torch

        self.p = p
        self.model = p.f16 if qtype == "f16" else p.quant(qtype)
        self.threads = threads
        sd = torch.load(p.hub / "head.pt", map_location="cpu")
        self.w = {k: v.float().numpy() for k, v in sd.items()}

    def embed(self, texts: list[str]) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            pf = Path(tmp) / "prompts.txt"
            pf.write_text(SEP.join(texts), encoding="utf-8")
            cmd = [
                self.p.bin / "llama-embedding", "-m", self.model, "--lora", self.p.lora,
                "-f", pf, "--embd-separator", SEP, "--pooling", "last", "--embd-normalize", "-1",
                "--embd-output-format", "json", "-t", self.threads, "-c", 2048, "-b", 2048, "-ub", 2048,
                "-ngl", 0, "--no-warmup",
            ]
            out = subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True).stdout
        return np.array([d["embedding"] for d in json.loads(out)["data"]], dtype=np.float32)

    def head(self, pooled: np.ndarray) -> np.ndarray:
        h = pooled @ self.w["mlp.0.weight"].T + self.w["mlp.0.bias"]
        h = 0.5 * h * (1 + np.vectorize(erf)(h / np.sqrt(2)))
        return (h @ self.w["mlp.2.weight"].T + self.w["mlp.2.bias"]).reshape(-1)

    def scores(self, examples: list[JevExample], chunk: int = 8) -> list[np.ndarray]:
        result = []
        for i in range(0, len(examples), chunk):
            group = examples[i : i + chunk]
            texts = [f"{ex.context}\n{ex.question}\n{c}" for ex in group for c in ex.candidates]
            s = self.head(self.embed(texts))
            pos = 0
            for ex in group:
                result.append(s[pos : pos + len(ex.candidates)])
                pos += len(ex.candidates)
        return result


def verify(p: Paths, qtypes: list[str]) -> None:
    import torch
    from eval.run_eval import load_model_from_hub
    from transformers import AutoTokenizer

    texts = [f"\n日本の首都はどこ？\n{c}" for c in ["東京", "大阪", "京都", "名古屋"]]
    model = load_model_from_hub(REPO_ID, device="cuda").eval()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    ref_pooled, ref_scores = [], []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt").to("cuda")
            h = model.backbone(**enc).last_hidden_state[0, -1]
            ref_pooled.append(h.float().cpu().numpy())
            ref_scores.append(model.head(h[None]).float().item())
    ref_pooled = np.stack(ref_pooled)
    print("PyTorch scores:", np.round(ref_scores, 2))
    for q in qtypes:
        sc = GGUFScorer(p, q)
        g = sc.embed(texts)
        cos = (g * ref_pooled).sum(1) / np.linalg.norm(g, axis=1) / np.linalg.norm(ref_pooled, axis=1)
        print(f"[{q}] cos={np.round(cos, 4)} scores={np.round(sc.head(g), 2)}")


def eval_heldout(p: Paths, qtypes: list[str], n: int) -> None:
    from collections import defaultdict

    from data.postprocess.split import train_val_split
    from scripts.push_to_hub import pull

    _, val = train_val_split(pull(repo_id=REPO_ID), val_size=200, seed=0)
    val = val[:n]
    for q in qtypes:
        sc = GGUFScorer(p, q)
        preds = sc.scores(val)
        ok, tot = defaultdict(int), defaultdict(int)
        for ex, s in zip(val, preds):
            ok[ex.task_type] += int(np.argmax(s) == ex.label)
            tot[ex.task_type] += 1
        overall = sum(ok.values()) / len(val)
        print(f"[{q}] overall={overall:.3f} n={len(val)}", {t: round(ok[t] / tot[t], 3) for t in tot})


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["setup", "convert", "verify", "eval"])
    ap.add_argument("--work", default="/content/gguf_work")
    ap.add_argument("--quants", nargs="+", default=["Q4_0", "Q4_K_M", "Q8_0"])
    ap.add_argument("--n", type=int, default=200, help="evalで使う件数")
    a = ap.parse_args()
    p = Paths(a.work)
    if "setup" in a.stages:
        setup(p)
    if "convert" in a.stages:
        convert(p, a.quants)
    if "verify" in a.stages:
        verify(p, a.quants)
    if "eval" in a.stages:
        eval_heldout(p, a.quants, a.n)


if __name__ == "__main__":
    _main()
