"""Command line entry point: ``hpp <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from house_price.config import data_config, training_config
from house_price.data.ingest import ingest_data, save_splits, split_data
from house_price.data.validate import validate_training_data
from house_price.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def prepare_data(force: bool = False) -> None:
    frame = ingest_data(force=force)
    validate_training_data(frame)
    train_df, test_df = split_data(frame)
    train_path, test_path = save_splits(train_df, test_df)
    print(f"train={train_path} ({len(train_df)} rows)")
    print(f"test={test_path} ({len(test_df)} rows)")


def cmd_prepare(args: argparse.Namespace) -> int:
    prepare_data(force=args.force)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from house_price.models.compare import run_comparison

    if not data_config().path("paths.processed_dir").exists():
        prepare_data()
    models = args.models.split(",") if args.models else None
    run_comparison(models=models, n_trials=args.trials)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from house_price.evaluation.report import run_evaluation

    run_evaluation(with_shap=not args.no_shap)
    return 0


def cmd_ablation(_: argparse.Namespace) -> int:
    from house_price.features.ablation import run_ablation

    run_ablation()
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    from house_price.monitoring.drift import (
        compute_drift,
        load_reference_stats,
        write_html_report,
    )
    from house_price.monitoring.prediction_log import PredictionLog

    cfg = data_config()
    reference_path = cfg.path("paths.reference_stats_file")
    if not reference_path.exists():
        print(f"Reference stats not found at {reference_path}. Run `hpp train` first.")
        return 1

    reference = load_reference_stats(reference_path)
    current = PredictionLog().recent_features(limit=args.limit)
    if current.empty:
        print("No prediction logs yet. Send requests to /predict first.")
        return 0

    thresholds = training_config().serving.drift_thresholds
    result = compute_drift(
        reference,
        current,
        psi_stable=float(thresholds.psi_stable),
        psi_moderate=float(thresholds.psi_moderate),
    )
    reports_dir = cfg.path("paths.reports_dir")
    json_path = reports_dir / "drift_report.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = write_html_report(result, reports_dir / "drift_report.html")

    print(
        json.dumps(
            {k: result[k] for k in ("status", "max_psi", "n_checked", "n_drifted")}, indent=2
        )
    )
    print(f"Saved drift report: {json_path}")
    print(f"Saved drift HTML: {html_path}")
    return 1 if result["status"] == "drift" else 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "house_price.serving.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    prepare_data(force=False)
    cmd_train(args)
    cmd_evaluate(args)
    cmd_ablation(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hpp", description="House price MLOps pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="download, validate and split the dataset")
    prepare.add_argument("--force", action="store_true", help="re-download the raw dataset")
    prepare.set_defaults(func=cmd_prepare)

    train = subparsers.add_parser("train", help="tune, compare and register models")
    train.add_argument(
        "--models", default=None, help="comma separated subset, e.g. lightgbm,xgboost"
    )
    train.add_argument("--trials", type=int, default=None, help="Optuna trials per model")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="report metrics, errors and SHAP")
    evaluate.add_argument("--no-shap", action="store_true", help="skip SHAP explanations")
    evaluate.set_defaults(func=cmd_evaluate)

    ablation = subparsers.add_parser("ablation", help="feature engineering ablation study")
    ablation.set_defaults(func=cmd_ablation)

    drift = subparsers.add_parser("drift", help="compare logged requests with training data")
    drift.add_argument("--limit", type=int, default=1000)
    drift.set_defaults(func=cmd_drift)

    serve = subparsers.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    run_all = subparsers.add_parser("all", help="prepare, train, evaluate and ablate")
    run_all.add_argument("--models", default=None)
    run_all.add_argument("--trials", type=int, default=None)
    run_all.add_argument("--no-shap", action="store_true")
    run_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
