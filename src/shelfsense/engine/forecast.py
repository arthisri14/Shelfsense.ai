"""Demand forecasting: weekday-seasonal exponential smoothing with an uncertainty range.

Deliberately simple and explainable. Known limits (handled in later phases): it does not
know about festivals, promotions or stock-outs that hid demand.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

Z80 = 1.2816          # z-score for an 80% interval
DEFAULT_ALPHA = 0.05  # smoothing speed, chosen with rolling backtests (scripts/tune_forecast.py)
SD_INFLATION = 1.25   # raw intervals covered ~70% instead of 80% in backtests, so widen them


@dataclass(frozen=True)
class Forecast:
    product_id: int
    horizon_days: int
    daily_mean: tuple[float, ...]   # day 1 = as_of + 1
    total_mean: float
    total_sd: float
    low: float                      # 80% interval for the horizon total
    high: float
    history_days: int
    method: str = "weekday-seasonal exponential smoothing"


def daily_units(conn: sqlite3.Connection, store_id: int, product_id: int, as_of: date,
                lookback_days: int = 112) -> np.ndarray:
    """Units sold per day for the last lookback_days days ending at as_of (zeros filled)."""
    start = as_of - timedelta(days=lookback_days - 1)
    rows = conn.execute(
        "SELECT date(s.sold_at) AS d, SUM(i.qty) AS units "
        "FROM sale_items i JOIN sales s ON s.id = i.sale_id "
        "WHERE s.store_id = ? AND i.product_id = ? AND date(s.sold_at) BETWEEN ? AND ? "
        "GROUP BY d",
        (store_id, product_id, start.isoformat(), as_of.isoformat())).fetchall()
    y = np.zeros(lookback_days)
    for r in rows:
        y[(date.fromisoformat(r["d"]) - start).days] = r["units"]
    return y


def forecast_demand(conn: sqlite3.Connection, store_id: int, product_id: int, as_of: date,
                    horizon_days: int = 7, lookback_days: int = 112,
                    alpha: float = DEFAULT_ALPHA,
                    sd_inflation: float = SD_INFLATION) -> Forecast:
    """Forecast units demanded on days as_of+1 ... as_of+horizon_days."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    y = daily_units(conn, store_id, product_id, as_of, lookback_days)
    start = as_of - timedelta(days=lookback_days - 1)
    dows = np.array([(start + timedelta(days=i)).weekday() for i in range(lookback_days)])
    overall = float(y.mean())
    if overall <= 0:
        return Forecast(product_id, horizon_days, tuple([0.0] * horizon_days),
                        0.0, 0.0, 0.0, 0.0, lookback_days)

    # Weekday index, shrunk towards 1 so sparse items do not get wild patterns.
    k = 4.0
    idx = np.ones(7)
    for d in range(7):
        mask = dows == d
        n = int(mask.sum())
        if n:
            idx[d] = (y[mask].sum() + k * overall) / ((n + k) * overall)
    idx = idx / idx.mean()

    # Exponential smoothing on the de-seasonalised series; record one-step errors.
    z = y / idx[dows]
    warm = min(14, lookback_days)
    level = float(z[:warm].mean())
    errors = []
    for t in range(lookback_days):
        pred = level * idx[dows[t]]
        if t >= warm:
            errors.append(y[t] - pred)
        level = alpha * z[t] + (1 - alpha) * level

    sd_daily = float(np.std(errors, ddof=1)) if len(errors) > 2 else math.sqrt(max(level, 0.0))
    sd_daily = max(sd_daily, 0.5 * math.sqrt(max(level, 0.0)))   # avoid overconfidence

    daily = [max(0.0, float(level * idx[(as_of + timedelta(days=h)).weekday()]))
             for h in range(1, horizon_days + 1)]
    total = float(sum(daily))
    total_sd = sd_daily * math.sqrt(horizon_days) * sd_inflation   # independent daily errors
    return Forecast(
        product_id=product_id, horizon_days=horizon_days,
        daily_mean=tuple(round(v, 3) for v in daily),
        total_mean=round(total, 2), total_sd=round(total_sd, 2),
        low=round(max(0.0, total - Z80 * total_sd), 2),
        high=round(total + Z80 * total_sd, 2),
        history_days=lookback_days)


def _backtest_sums(conn: sqlite3.Connection, store_id: int, as_of: date,
                   holdout_days: int, lookback_days: int, alpha: float):
    """Forecast the holdout_days ending at as_of using only data before them.

    Returns (model_abs_error, naive_abs_error, total_actual, intervals_hit, products).
    """
    cutoff = as_of - timedelta(days=holdout_days)
    pids = [r[0] for r in conn.execute(
        "SELECT product_id FROM store_products WHERE store_id = ? AND active = 1", (store_id,))]
    err_model = err_naive = total_actual = 0.0
    inside = 0
    for pid in pids:
        y = daily_units(conn, store_id, pid, as_of, lookback_days + holdout_days)
        actual = float(y[-holdout_days:].sum())
        naive = float(y[-(holdout_days + 28):-holdout_days].mean()) * holdout_days
        fc = forecast_demand(conn, store_id, pid, cutoff, horizon_days=holdout_days,
                             lookback_days=lookback_days, alpha=alpha)
        err_model += abs(fc.total_mean - actual)
        err_naive += abs(naive - actual)
        total_actual += actual
        inside += int(fc.low <= actual <= fc.high)
    return err_model, err_naive, total_actual, inside, len(pids)


def _summary(err_model: float, err_naive: float, total_actual: float,
             inside: int, n: int, **extra) -> dict:
    if total_actual == 0 or n == 0:
        nan = float("nan")
        return {**extra, "wape_model": nan, "wape_naive": nan, "interval_coverage": nan}
    return {**extra,
            "wape_model": round(err_model / total_actual, 4),
            "wape_naive": round(err_naive / total_actual, 4),
            "interval_coverage": round(inside / n, 3)}


def evaluate_forecasts(conn: sqlite3.Connection, store_id: int, as_of: date,
                       holdout_days: int = 14, lookback_days: int = 112,
                       alpha: float = DEFAULT_ALPHA) -> dict:
    """Single backtest: forecast the last holdout_days and compare with actuals.

    WAPE = sum(|error|) / sum(actual), versus a naive baseline (28-day average).
    """
    e_m, e_n, act, inside, n = _backtest_sums(conn, store_id, as_of, holdout_days,
                                              lookback_days, alpha)
    return _summary(e_m, e_n, act, inside, n, products=n, holdout_days=holdout_days)


def rolling_backtest(conn: sqlite3.Connection, store_id: int, last_origin: date,
                     n_origins: int = 6, step_days: int = 14, holdout_days: int = 14,
                     lookback_days: int = 112, alpha: float = DEFAULT_ALPHA) -> dict:
    """Repeat the backtest at several origins and pool the errors (fairer than one window)."""
    tot_m = tot_n = tot_a = 0.0
    inside = n = 0
    for k in range(n_origins):
        origin = last_origin - timedelta(days=step_days * k)
        e_m, e_n, act, ins, cnt = _backtest_sums(conn, store_id, origin, holdout_days,
                                                 lookback_days, alpha)
        tot_m += e_m
        tot_n += e_n
        tot_a += act
        inside += ins
        n += cnt
    return _summary(tot_m, tot_n, tot_a, inside, n, origins=n_origins,
                    holdout_days=holdout_days, alpha=alpha)