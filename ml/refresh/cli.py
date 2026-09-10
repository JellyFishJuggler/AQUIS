"""refresh/cli.py — command-line interface for the refresh pipeline.

Run from the ml/ directory (matching the other pipeline scripts):

    python -m refresh.cli validate-config
    python -m refresh.cli dry-run --no-fetch
    python -m refresh.cli forecast            # default: fetch + features + inference
    python -m refresh.cli retrain [--smoke]
    python -m refresh.cli status
    python -m refresh.cli schedule [--once] [--poll 60]

Exit codes: 0 = ok, 1 = cycle failed / gate rejected, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from refresh import publish, scheduler
from refresh.config import config_path, load_config
from refresh.state import RefreshState

ROOT = Path(__file__).resolve().parent.parent


def _config(args):
    cfg = load_config(path=Path(args.config) if getattr(args, "config", None) else config_path())
    return cfg


def _cmd_validate(args) -> int:
    from refresh.config import ConfigError
    try:
        cfg = _config(args)
        cfg.validate()
        print(f"config OK @ {config_path()}")
        print(json.dumps(
            {k: v for k, v in cfg.__dict__.items()}, indent=2, default=str))
        return 0
    except ConfigError as exc:
        print(f"[error] {exc}")
        return 2


def _cmd_status(args) -> int:
    cfg = _config(args)
    state = RefreshState()
    inf = _status(cfg, state)
    print(json.dumps(inf, indent=2, default=str))
    return 0


def _status(cfg, state):
    from refresh import inference
    sched = scheduler.next_due_human(cfg, state)
    out = inference.status()
    out["schedule"] = sched
    out["feature_mode"] = (publish.read_runtime() or {}).get("feature_mode")
    out["last_retrain"] = state.get("last_retrain_ts")
    out["last_retrain_ok"] = state.get("last_retrain_ok")
    out["last_retrain_error"] = state.get("last_retrain_error")
    return out


def _cmd_dry_run(args) -> int:
    cfg = _config(args)
    from refresh import pipeline
    rep = pipeline.run_forecast_cycle(
        cfg=cfg, dry_run=True, do_fetch=not args.no_fetch,
        do_features=not args.no_features, do_inference=not args.no_inference,
        max_stations=args.stations or 0)
    print(json.dumps(rep, indent=2, default=str))
    return 0


def _cmd_forecast(args) -> int:
    cfg = _config(args)
    from refresh import pipeline
    rep = pipeline.run_forecast_cycle(
        cfg=cfg, dry_run=False, do_fetch=not args.no_fetch,
        do_features=not args.no_features, do_inference=not args.no_inference,
        max_stations=args.stations or 0, skip_unchanged=not args.force)
    ok = bool(rep.get("inference", {}).get("published"))
    print(json.dumps(rep, indent=2, default=str))
    return 0 if ok else 1


def _cmd_retrain(args) -> int:
    cfg = _config(args)
    from refresh import pipeline
    rep = pipeline.run_retrain_cycle(cfg=cfg, smoke=args.smoke,
                                     feature_mode=args.feature_mode or "flat")
    decision = (rep.get("gate") or {}).get("decision", "?")
    print(json.dumps(rep, indent=2, default=str))
    return 0 if rep.get("promoted") else 1


def _cmd_schedule(args) -> int:
    cfg = _config(args)
    scheduler.scheduler_loop(cfg=cfg, once=args.once, poll_s=args.poll,
                             do_fetch=not args.no_fetch,
                             do_features=not args.no_features,
                             do_inference=True,
                             smoke_retrain=args.smoke)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m refresh.cli")
    ap.add_argument("--config", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate-config")
    p.set_defaults(fn=_cmd_validate)

    p = sub.add_parser("status")
    p.set_defaults(fn=_cmd_status)

    p = sub.add_parser("dry-run")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--no-features", action="store_true")
    p.add_argument("--no-inference", action="store_true")
    p.add_argument("--stations", type=int, default=0)
    p.set_defaults(fn=_cmd_dry_run)

    p = sub.add_parser("forecast")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--no-features", action="store_true")
    p.add_argument("--no-inference", action="store_true")
    p.add_argument("--stations", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="recompute even stations whose anchor has not changed")
    p.set_defaults(fn=_cmd_forecast)

    p = sub.add_parser("retrain")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--feature-mode", choices=["flat", "future"], default="flat")
    p.set_defaults(fn=_cmd_retrain)

    p = sub.add_parser("schedule")
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--no-features", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(fn=_cmd_schedule)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())