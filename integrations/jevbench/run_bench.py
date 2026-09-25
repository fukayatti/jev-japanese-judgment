"""JevBenchの公開階層(easy / original / hard)を、jev_gguf_localアダプタで流して結果を1つの表にする。

前提: install.py でアダプタを組み込み済みであること。llama-serverはCUDAビルド(JEV_NGL=99)でもCPUでもよい。
例:
  python integrations/jevbench/run_bench.py --jevbench /content/jevbench --gguf /content/gguf_v2/merged-Q4_K_M.gguf \\
    --head /content/gguf_v2/head.npz --server /content/gguf_v2/llama.cpp/build/bin/llama-server --label v2-q4km --ngl 99
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

TIERS = ["easy", "original", "hard"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevbench", required=True, type=Path)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ngl", default="0")
    ap.add_argument("--threads", default="4")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--out", type=Path, default=Path("jevbench_out"))
    ap.add_argument("--option-mode", default="label", choices=["label", "criteria"])
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "JEV_REPO": str(repo), "JEV_LLAMA_SERVER": a.server, "JEV_HEAD": a.head,
           "JEV_NGL": a.ngl, "JEV_THREADS": a.threads, "JEV_OPTION_MODE": a.option_mode}
    out = (a.out / a.label).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for tier in a.tiers.split(","):
        tasks = f"datasets/public/{tier}.jsonl"
        results = out / f"{tier}.jsonl"
        results.unlink(missing_ok=True)
        ledger = out / f"{tier}_ledger.jsonl"
        ledger.unlink(missing_ok=True)
        run = [sys.executable, "-m", "jevbench.cli", "run", "--adapter", "jev_gguf_local", "--endpoint", a.gguf,
               "--model", a.label, "--tasks", tasks, "--results", str(results), "--ledger", str(ledger),
               "--raw-dir", str(out / f"{tier}_raw")]
        subprocess.run(run, cwd=a.jevbench, env=env, check=True)
        summ = subprocess.run([sys.executable, "-m", "jevbench.cli", "summarize", "--tasks", tasks, "--results", str(results)],
                              cwd=a.jevbench, env=env, check=True, capture_output=True, text=True).stdout
        s = json.loads(summ)
        fam = {k: round(v["accuracy"], 3) for k, v in s["per_family"].items() if v.get("accuracy") is not None}
        rows.append({"tier": tier, "n": s["n_scorable"], "accuracy": round(s["accuracy"], 4), "ece": round(s["ece"]["ece"], 4),
                     "brier": round(s["brier_mean"], 4), "p50_s": round(s["latency"]["p50_s"], 3), "families": fam})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\n== {a.label} ==")
    for r in rows:
        print(f"{r['tier']:9s} n={r['n']:3d} acc={r['accuracy']:.3f} ece={r['ece']:.3f} brier={r['brier']:.3f} p50={r['p50_s']}s  {r['families']}")


if __name__ == "__main__":
    main()
