"""Run one independent single-GPU training config per torchrun rank.

    uv run run_distributed.py --nodes 1 --script training/sweep.py ... --configs training/sweeps/x.json

The JSON file is ``{"name": ["--learning-rate", "5e-5", ...], ...}``. Rank i
takes the i-th config, trains it on GPU i (no DDP), then benchmarks the saved
model on the held-out WordNet split with the real inference path. Output goes
to ``<output-root>/<name>/`` including a ``train.log`` and ``result.json``.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from wsd.env import detach_from_torchrun

# Each rank is a plain single-GPU process; must run before torch is imported.
_RANK, _ = detach_from_torchrun()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/nfs-1/amit/wsd/runs"))
    parser.add_argument("--eval-raganato", type=Path, help="also benchmark on this Raganato-format set")
    parser.add_argument("--sense-index", type=Path, help="WordNet 3.0 index.sense (with --eval-raganato)")
    parser.add_argument("--nodes", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    configs = json.loads(args.configs.read_text())
    names = list(configs)
    if _RANK >= len(names):
        print(f"rank {_RANK}: no config, exiting")
        return
    name = names[_RANK]
    out_dir = args.output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(out_dir / "train.log", "a", buffering=1)
    print(f"rank {_RANK} -> {name}: {configs[name]}")

    from training.train import main as train_main

    try:
        final = train_main([*configs[name], "--output-dir", str(out_dir), "--run-name", name])
    except Exception:  # noqa: BLE001 - one failing config must not make torchrun kill the other ranks
        traceback.print_exc()
        (out_dir / "result.json").write_text(json.dumps({"name": name, "error": traceback.format_exc()}))
        return

    os.environ["WSD_MODEL"] = str(final)
    from training.wn_data import split
    from wsd.benchmark import evaluate

    eval_examples, _ = split(n_eval=5000, seed=42)
    correct, total, seconds = evaluate(eval_examples, batch_size=64, failures_path=str(out_dir / "failures.jsonl"))
    result = {"name": name, "args": configs[name], "accuracy": correct / total, "n": total,
              "seconds": seconds, "model": str(final)}
    if args.eval_raganato:
        from training.semcor import load_raganato, load_sense_index

        exs = load_raganato(args.eval_raganato, load_sense_index(args.sense_index))
        c, t, _ = evaluate(exs, batch_size=64, failures_path=str(out_dir / f"failures-{args.eval_raganato.name}.jsonl"))
        result[f"accuracy_{args.eval_raganato.name}"] = c / t
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
