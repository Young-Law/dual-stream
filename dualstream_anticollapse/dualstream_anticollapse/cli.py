
import argparse, os, json, sys
import pandas as pd
from .config import Config, Thresholds, RetrainPolicy
from .retrain import build_model, fit_model, predict
from .metrics import classification_metrics
from .governance import save_model, load_model, ModelRegistry, RegistryItem
from .monitor import ModelMonitor

def _load_csv(path):
    return pd.read_csv(path)

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def cmd_train(args):
    df = _load_csv(args.train_csv)
    features = args.features.split(",") if args.features else None
    cfg = Config(target=args.target, id_column=args.id_column, features=features,
                 model_type=args.model_type, output_dir=args.artifacts)
    X = df[cfg.features] if cfg.features else df.drop(columns=[c for c in [cfg.target, cfg.id_column] if c])
    y = df[cfg.target].astype(int)
    model = build_model(cfg.model_type)
    model = fit_model(model, X, y)
    y_pred, y_proba = predict(model, X)
    base_metrics = classification_metrics(y, y_pred, y_proba)
    os.makedirs(cfg.output_dir, exist_ok=True)
    record = save_model(model, os.path.join(cfg.output_dir, "model.joblib"), {"stage":"baseline", "metrics": base_metrics})
    _save_json(os.path.join(cfg.output_dir, "baseline.json"), {"metrics": base_metrics, "feature_summary": df.describe(include='all').to_dict()})
    reg = ModelRegistry(os.path.join(cfg.output_dir, "registry"))
    reg.add(RegistryItem(version="v0.1.0", path=record["path"], sha256=record["sha256"], created_at=record["saved_at"], metrics=base_metrics))
    print(json.dumps({"status":"trained", "metrics": base_metrics}))

def cmd_monitor(args):
    import pandas as pd, json, os
    cfg = Config(target=args.target, id_column=args.id_column, features=args.features.split(",") if args.features else None,
                 model_type=args.model_type, output_dir=args.artifacts)
    baseline = json.load(open(os.path.join(cfg.output_dir, "baseline.json")))
    mon = ModelMonitor(cfg, baseline, state_path=os.path.join(cfg.output_dir, "state.json"), alert_sink="stdout")
    ref = pd.read_csv(args.reference_csv)
    cur = pd.read_csv(args.current_csv)
    drift = mon.check_drift(ref, cur)
    outliers_trig, outliers = mon.check_outliers(cur)
    # Simulate eval with labels in current_csv
    X = cur[cfg.features] if cfg.features else cur.drop(columns=[c for c in [cfg.target, cfg.id_column] if c])
    y = cur[cfg.target].astype(int)
    from .governance import load_model
    model = load_model(os.path.join(cfg.output_dir, "model.joblib"))
    from .retrain import predict
    y_pred, y_proba = predict(model, X)
    metrics = mon.check_performance(classification_metrics(y, y_pred, y_proba))
    print(json.dumps({"drift_triggered": drift, "outliers_triggered": outliers_trig, "outliers": outliers, "performance_triggered": metrics}))

def cmd_audit_dual(args):
    cfg = Config(target=args.target, id_column=args.id_column, features=None, output_dir=args.artifacts)
    baseline = {"metrics": {}}
    mon = ModelMonitor(cfg, baseline, state_path=os.path.join(cfg.output_dir, "state.json"))
    records = [json.loads(line) for line in open(args.dual_jsonl)]
    results = mon.audit_dual_streams(records)
    out_path = os.path.join(cfg.output_dir, "coherence_report.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"report": out_path, "violations": sum(1 for r in results if not r["coherent"])}))

def build_parser():
    p = argparse.ArgumentParser(prog="ds-anticollapse", description="Dual-Stream anticollapse toolkit")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--train_csv", required=True)
    t.add_argument("--target", required=True)
    t.add_argument("--id_column", default=None)
    t.add_argument("--features", default=None)
    t.add_argument("--model_type", default="sgd_classifier", choices=["sgd_classifier","random_forest"])
    t.add_argument("--artifacts", default="artifacts")
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("monitor")
    m.add_argument("--reference_csv", required=True)
    m.add_argument("--current_csv", required=True)
    m.add_argument("--target", required=True)
    m.add_argument("--id_column", default=None)
    m.add_argument("--features", default=None)
    m.add_argument("--model_type", default="sgd_classifier")
    m.add_argument("--artifacts", default="artifacts")
    m.set_defaults(func=cmd_monitor)

    a = sub.add_parser("audit-dual")
    a.add_argument("--dual_jsonl", required=True, help="Path to JSONL with {answer, monologue, logits_topk?}")
    a.add_argument("--target", default="y")
    a.add_argument("--id_column", default=None)
    a.add_argument("--artifacts", default="artifacts")
    a.set_defaults(func=cmd_audit_dual)

    return p

def main(argv=None):
    argv = argv or sys.argv[1:]
    p = build_parser()
    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help(); sys.exit(2)
    return args.func(args)

if __name__ == "__main__":
    main()
