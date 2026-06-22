"""Pipeline orchestrator — runs all medallion stages in dependency order.

Usage:
    python src/pipeline.py            # Run full pipeline
    python src/pipeline.py --stage bronze   # Run only bronze stages
    python src/pipeline.py --from silver    # Run from silver onward

Without DVC: this script calls each stage's main() in sequence.
With DVC: use 'dvc repro' instead for caching and incremental runs.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import setup_logging

logger = setup_logging(__name__)

STAGES = {
    "bronze": [
        ("ingest_phap_dien", "src.bronze.phap_dien"),
        ("ingest_vbqppl", "src.bronze.vbqppl"),
    ],
    "silver": [
        ("clean_phap_dien", "src.silver.phap_dien"),
        ("split_vbqppl", "src.silver.vbqppl"),
        ("quality_checks", "src.silver.quality"),
    ],
    "gold": [
        ("build_documents", "src.gold.documents"),
        ("chunk_documents", "src.gold.chunks"),
    ],
}

STAGE_ORDER = ["bronze", "silver", "gold"]


def run_stage(name: str, module_path: str) -> None:
    import importlib

    logger.info("--- Stage: %s ---", name)
    mod = importlib.import_module(module_path)
    mod.main()
    logger.info("--- Stage %s complete ---", name)


def run_pipeline(from_stage: str | None = None, stage_only: str | None = None) -> None:
    if stage_only:
        if stage_only not in STAGES:
            logger.error("Unknown stage: %s. Choose from: %s", stage_only, list(STAGES))
            sys.exit(1)
        for name, module_path in STAGES[stage_only]:
            run_stage(name, module_path)
        return

    start_idx = STAGE_ORDER.index(from_stage) if from_stage else 0
    for layer in STAGE_ORDER[start_idx:]:
        for name, module_path in STAGES[layer]:
            run_stage(name, module_path)

    logger.info("=== Full pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Law Crawler Medallion Pipeline")
    parser.add_argument("--stage", choices=list(STAGES), help="Run only this layer")
    parser.add_argument("--from", dest="from_stage", choices=STAGE_ORDER,
                        help="Run from this layer onward")
    args = parser.parse_args()

    run_pipeline(from_stage=args.from_stage, stage_only=args.stage)


if __name__ == "__main__":
    main()
