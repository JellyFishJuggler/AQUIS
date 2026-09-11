"""refresh/config.py — runtime configuration for the refresh pipeline.

Defaults live in code; an editable JSON copy is kept at
``data/refresh/refresh_config.json`` so operators can tune cadences without
touching code. A few keys can also be overridden with environment variables
(``AQUIS_REFRESH_INTERVAL_HOURS``, ``AQUIS_RETRAIN_CADENCE_HOURS``,
``AQUIS_STALE_AFTER_HOURS``, ``AQUIS_FEATURE_MODE``).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFRESH_DIR = ROOT / "data" / "refresh"

FEATURE_MODES = ("flat", "future")


class ConfigError(ValueError):
    pass


@dataclass
class RefreshConfig:
    # ---- cadences (independent) -------------------------------------------
    refresh_interval_hours: float = 6.0      # data + forecast every 6 h
    refresh_at_hours: list[int] = field(default_factory=lambda: [1, 7, 13, 19])  # fixed local-time slots
    retrain_cadence_hours: float = 24.0      # model update independently configurable
    retrain_at_hours: list[int] = field(default_factory=lambda: [2, 14])  # allowed windows
    # ---- freshness / retry -------------------------------------------------
    stale_after_hours: float = 30.0          # data older than this is "stale"
    retry_max: int = 3
    retry_backoff_s: float = 10.0
    fetch_workers: int = 8
    openmeteo_forecast_days: int = 16        # preserves forecast->climatology cap
    # ---- model update / promotion ------------------------------------------
    promote_tolerance: float = 0.0           # candidate must beat incumbent on same window
    candidate_holdout_days: int = 90         # early-stopping val = [now-(h+w), now-w)
    candidate_backtest_window_days: int = 90  # backtest anchors in [now-w, now)
    backtest_stations: int = 60              # candidate/incumbent backtest sample (0 = all)
    anchor_spacing_days: int = 14
    coverage_target: float = 0.90
    # memory budgets for the shared-box candidate build. The long-frame expands
    # every station x horizon; without caps a 600-station retrain holds several
    # GB of float rows and OOMs the box that also runs streamlit + the daemon.
    # Rows are stride-sampled down to these budgets before parquet, so peak RAM
    # stays bounded while the window/station spread is preserved.
    candidate_train_row_cap: int = 1_200_000
    candidate_val_row_cap: int = 300_000
    # ---- inference environment ---------------------------------------------
    feature_mode: str = "flat"               # "future" only after backtest-validated
    min_fresh_stations: int = 20
    max_stations: int = 0                    # 0 = all stations in the aligned table

    def validate(self) -> None:
        if self.refresh_interval_hours <= 0:
            raise ConfigError("refresh_interval_hours must be > 0")
        if self.retrain_cadence_hours <= 0:
            raise ConfigError("retrain_cadence_hours must be > 0")
        if self.stale_after_hours <= 0:
            raise ConfigError("stale_after_hours must be > 0")
        if self.feature_mode not in FEATURE_MODES:
            raise ConfigError(f"feature_mode must be one of {FEATURE_MODES}")
        if not (0.0 < self.coverage_target <= 1.0):
            raise ConfigError("coverage_target must be in (0, 1]")
        if self.candidate_holdout_days <= 0 or self.candidate_backtest_window_days <= 0:
            raise ConfigError("holdout / backtest windows must be > 0 days")

    def ensure_dirs(self) -> None:
        REFRESH_DIR.mkdir(parents=True, exist_ok=True)


def config_path() -> Path:
    return REFRESH_DIR / "refresh_config.json"


def load_config(path: Path | None = None, write_if_missing: bool = True) -> RefreshConfig:
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"corrupt refresh config {p}: {exc}") from exc
    cfg = RefreshConfig(**{k: v for k, v in data.items()
                           if k in RefreshConfig.__dataclass_fields__})
    _apply_env(cfg)

    # the runtime feature_mode can only be flip/flat via the validated pipeline;
    # env override is a debug-only escape hatch.
    _mode = os.environ.get("AQUIS_FEATURE_MODE")
    if _mode:
        cfg.feature_mode = _mode
    cfg.validate()
    if write_if_missing and not p.exists():
        p.write_text(json.dumps(asdict(cfg), indent=2))
    return cfg


def _apply_env(cfg: RefreshConfig) -> None:
    env = {
        "refresh_interval_hours": "AQUIS_REFRESH_INTERVAL_HOURS",
        "retrain_cadence_hours": "AQUIS_RETRAIN_CADENCE_HOURS",
        "stale_after_hours": "AQUIS_STALE_AFTER_HOURS",
        "feature_mode": "AQUIS_FEATURE_MODE",
    }
    for attr, key in env.items():
        v = os.environ.get(key)
        if not v:
            continue
        current = getattr(cfg, attr)
        if isinstance(current, str):
            setattr(cfg, attr, v)
        else:
            setattr(cfg, attr, float(v))