"""温度スケーリング: スコアを1つの温度Tで割り(softmaxの前)、確率の自信過剰/過小を直す。

  gguf      GGUF版で、校正用データからTを決め、評価セットでECE/Brier/NLLを前後で比べる
  jevbench  JevBenchの結果jsonl(確率は保存済み)に、色々なTを当てて、ECEがどう動くかを見る(診断用)

softmax(s/T)で、T>1なら確率が平らに(自信を下げる)、T<1なら尖る。正解率は変わらない(順位が同じ)。
確率をlogにすれば softmax(log p / T) と同じなので、保存済みの確率にも後から温度を当てられる。
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.gguf_pipeline import calibration_from_scores


def _log_softmax(s: np.ndarray, T: float) -> np.ndarray:
    z = s / T
    z = z - z.max()
    return z - math.log(np.exp(z).sum())


def nll_at(scores: list[np.ndarray], labels: list[int], T: float) -> float:
    return float(np.mean([-_log_softmax(s, T)[y] for s, y in zip(scores, labels)]))


def fit_temperature(scores: list[np.ndarray], labels: list[int]) -> float:
    """NLLが最小になるTを、粗い格子→細かい格子の順で探す。"""
    coarse = np.arange(0.2, 5.01, 0.1)
    best = min(coarse, key=lambda T: nll_at(scores, labels, float(T)))
    fine = np.arange(max(0.1, best - 0.1), best + 0.1, 0.01)
    return float(min(fine, key=lambda T: nll_at(scores, labels, float(T))))


def report(name: str, scores: list[np.ndarray], labels: list[int], T: float) -> None:
    before = calibration_from_scores(scores, labels)
    after = calibration_from_scores([s / T for s in scores], labels)
    print(f"  {name:22s} ECE {before['ece']:.4f} -> {after['ece']:.4f} | Brier {before['brier']:.4f} -> {after['brier']:.4f} "
          f"| NLL {before['nll']:.4f} -> {after['nll']:.4f}")


def cmd_gguf(a) -> None:
    from scripts.build_clean_eval import load
    from scripts.gguf_pipeline import GGUFScorer, Paths

    p = Paths(a.work, a.repo)
    calib = load(a.calib)
    evals = {path.name: load(path) for path in a.evals}
    for q in a.quants:
        sc = GGUFScorer.from_paths(p, q, merged=True, ngl=a.ngl)
        cs = sc.scores(calib)
        labels = [ex.label for ex in calib]
        T = fit_temperature(cs, labels)
        print(f"\n[{q}] 校正用{len(calib)}件でのT = {T:.2f}")
        by_task: dict[str, list[int]] = {}
        for i, ex in enumerate(calib):
            by_task.setdefault(ex.task_type, []).append(i)
        print("  タスクごとの最適T(参考): " + ", ".join(
            f"{t}={fit_temperature([cs[i] for i in idx], [labels[i] for i in idx]):.2f}" for t, idx in by_task.items()))
        for name, exs in evals.items():
            report(name, sc.scores(exs), [ex.label for ex in exs], T)


def cmd_jevbench(a) -> None:
    for tier in a.tiers:
        tasks = {}
        for line in (a.jevbench / "datasets" / "public" / f"{tier}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            tasks[r["id"]] = str(r["expected"])
        scores, labels = [], []
        for line in (a.results / f"{tier}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r.get("ok") or not r.get("probs") or r["task_id"] not in tasks:
                continue
            keys = list(r["probs"].keys())
            if tasks[r["task_id"]] not in keys:
                continue
            scores.append(np.log(np.maximum(np.array([r["probs"][k] for k in keys]), 1e-12)))
            labels.append(keys.index(tasks[r["task_id"]]))
        best = fit_temperature(scores, labels)
        print(f"\n[{tier}] n={len(scores)}  この階層のNLL最適T(診断用、適用はしない) = {best:.2f}")
        for T in a.temps:
            m = calibration_from_scores([s / T for s in scores], labels)
            print(f"  T={T:<4} ECE {m['ece']:.4f}  Brier {m['brier']:.4f}  NLL {m['nll']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gguf")
    g.add_argument("--work", required=True)
    g.add_argument("--repo", default="fukayatti0/jev-japanese-judgment")
    g.add_argument("--quants", nargs="+", default=["Q4_K_M", "Q8_0"])
    g.add_argument("--ngl", type=int, default=0)
    g.add_argument("--calib", type=Path, required=True)
    g.add_argument("--evals", type=Path, nargs="+", required=True)
    g.set_defaults(fn=cmd_gguf)
    j = sub.add_parser("jevbench")
    j.add_argument("--jevbench", type=Path, required=True)
    j.add_argument("--results", type=Path, required=True, help="run_benchのラベル別出力ディレクトリ(easy.jsonl等がある所)")
    j.add_argument("--tiers", nargs="+", default=["easy", "original", "hard"])
    j.add_argument("--temps", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    j.set_defaults(fn=cmd_jevbench)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
