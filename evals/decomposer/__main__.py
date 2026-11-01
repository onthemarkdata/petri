"""Run the offline report, or an explicitly enabled configured-model report."""

import argparse
import os
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.decomposer.archive import load_archive, retrieval_report
from evals.decomposer.dataset import build_dataset, live_task, scripted_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure decomposition quality; offline by default.")
    parser.add_argument("--live", action="store_true", help="Call the explicitly configured model.")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Bound each response in a live native-provider evaluation.")
    parser.add_argument("--archive", type=Path, help="Validate and search a separately supplied 88-node archive.")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive.")
    if args.archive:
        graph = load_archive(args.archive)
        with TemporaryDirectory(prefix="petri-archive-eval-") as directory:
            print(retrieval_report(graph, Path(directory) / "search.sqlite"))
    task = scripted_task
    if args.live:
        if os.environ.get("PETRI_LIVE_EVALS") != "1" or not os.environ.get("PETRI_EVAL_MODEL"):
            parser.error("Live calls require PETRI_LIVE_EVALS=1 and PETRI_EVAL_MODEL.")
        task = partial(live_task, model_name=os.environ["PETRI_EVAL_MODEL"], max_tokens=args.max_tokens)
    report = build_dataset().evaluate_sync(task, max_concurrency=1, progress=False)
    report.print(include_input=False, include_output=False)


if __name__ == "__main__":
    main()
