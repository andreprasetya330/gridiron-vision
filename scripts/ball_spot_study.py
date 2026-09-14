"""Which snap-time landmark best recovers the ball's lateral position?

The pipeline has to place an origin on the field before any cue means anything,
and every candidate landmark is biased in a different way. This measures them
against synthetic ground truth, where the ball sits at y=0 by construction.
"""

from __future__ import annotations

import numpy as np

from gridiron.tracking.schema import TIME_GRID
from gridiron.tracking.synthetic import SyntheticConfig, generate_season

SNAP = int(np.argmin(np.abs(TIME_GRID - 0.0)))


def tight_window(ys: np.ndarray, xs: np.ndarray, k: int = 5, by: str = "x") -> np.ndarray:
    order = np.argsort(ys)
    best, best_span = order[:k], np.inf
    for start in range(0, len(order) - k + 1):
        window = order[start : start + k]
        span = (
            xs[window].max() - xs[window].min()
            if by == "x"
            else ys[window].max() - ys[window].min()
        )
        if span < best_span:
            best_span, best = span, window
    return best


def main() -> None:
    names = [
        "all 22 median",
        "all offense median",
        "on-line tight-y median",
        "on-line tight-y midpoint",
        "defensive front median",
        "both fronts median",
        "interior 3 median",
    ]
    est: dict[str, list[float]] = {n: [] for n in names}

    cfg = SyntheticConfig(weeks=3, plays_per_game=20, seed=3)
    for play in generate_season(cfg):
        ox, oy, dx, dy = [], [], [], []
        for p in play.players:
            x, y = float(p.x[SNAP]), float(p.y[SNAP])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if p.side == "offense":
                ox.append(x)
                oy.append(y)
            else:
                dx.append(x)
                dy.append(y)
        ox, oy = np.array(ox), np.array(oy)
        dx, dy = np.array(dx), np.array(dy)
        if len(ox) < 7 or len(dx) < 7:
            continue

        est["all 22 median"].append(float(np.median(np.concatenate([oy, dy]))))
        est["all offense median"].append(float(np.median(oy)))

        los_x = float(np.median(ox[tight_window(oy, ox)]))
        on_line = np.abs(ox - los_x) <= 1.5
        front = np.abs(dx - los_x) <= 3.0

        if on_line.sum() >= 5:
            w = tight_window(oy[on_line], ox[on_line], by="y")
            line_y = np.sort(oy[on_line][w])
            est["on-line tight-y median"].append(float(np.median(line_y)))
            est["on-line tight-y midpoint"].append(float((line_y[0] + line_y[-1]) / 2))
            # The guard, center and guard: drop the outermost man on each side,
            # which is where a tight end or an unbalanced set does its damage.
            est["interior 3 median"].append(float(np.median(line_y[1:-1])))
        if front.sum() >= 3:
            est["defensive front median"].append(float(np.median(dy[front])))
        both = np.concatenate([oy[on_line], dy[front]])
        if both.size >= 6:
            est["both fronts median"].append(float(np.median(both)))

    header = f"{'estimator':26s} {'bias':>7s} {'MAE':>7s} {'p90 abs':>8s}   n"
    print(header)
    print("-" * len(header))
    for name in names:
        a = np.array(est[name])
        if a.size == 0:
            continue
        print(
            f"{name:26s} {a.mean():+7.3f} {np.abs(a).mean():7.3f} "
            f"{np.percentile(np.abs(a), 90):8.3f}   {a.size}"
        )


if __name__ == "__main__":
    main()
