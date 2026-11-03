"""Compare forecast smoothing speeds with rolling backtests. Read-only."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from shelfsense.db.connection import get_connection
from shelfsense.engine.common import data_as_of
from shelfsense.engine.forecast import rolling_backtest

ALPHAS = [0.02, 0.03, 0.05, 0.08, 0.15]
# label, holdout_days, step_days, n_origins
SETUPS = [
    ("14-day totals", 14, 14, 6),
    ("5-day totals (typical reorder horizon)", 5, 5, 12),
]


def main() -> None:
    load_dotenv()
    conn = get_connection(os.getenv("DB_PATH", "data/shelfsense.db"))
    as_of = data_as_of(conn)

    for label, holdout, step, origins in SETUPS:
        for store in (1, 2):
            print(f"\n=== {label} | store {store} | {origins} origins ===")
            for a in ALPHAS:
                r = rolling_backtest(conn, store, as_of, n_origins=origins, step_days=step,
                                     holdout_days=holdout, alpha=a)
                ratio = r["wape_model"] / r["wape_naive"]
                print(f"alpha={a:<5} model={r['wape_model']:.4f}  naive={r['wape_naive']:.4f}  "
                      f"model/naive={ratio:.2f}  coverage={r['interval_coverage']:.2f}")


if __name__ == "__main__":
    main()