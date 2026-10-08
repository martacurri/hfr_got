"""WP3: Isolation Forest anomaly score on the sliding-window kernel features.

Reads data/processed/kernels/kernel_responses_<product>[_3x3].nc (written by
kernel_features.py); one model per product per window size (--window, default 4 3). One row = one complete 4x4 window at one timestamp;
features = its 8 (Total) or 4 (radial) kernel values. Window time and
position are kept alongside but are NOT features, so location and time play
no part in the anomaly score.

Per product: z-score -> Isolation Forest for a per-window anomaly score
(higher = more anomalous), fitted on a random sample of <=500k windows, then
applied to every window. (A k-means grouping of the same features was tried
and dropped: the windows form one continuous cloud, so the clusters added
nothing beyond the kernel maps and this score.)
The top 1% by score is a DISPLAY cut for maps/lists, not a QC threshold.

Outputs:
  data/processed/kernels/kernel_models_<product>.npz   (anomaly-score cache)
  outputs/reports/kernel_anomalies_top_<product>.csv

Run: python scripts/kernel_clustering.py
"""
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import IsolationForest

from kernel_features import FEATURE_DIR, PRODUCTS, PROJECT_ROOT, WINDOW, feature_path, size_suffix

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
RANDOM_STATE = 42
FIT_SAMPLE = 500_000
IF_TREES = 200
TOP_FRAC = 0.01
PREDICT_CHUNK = 1_000_000
REVIEW_SIZES = [4, 3]  # window sizes scored and shown side by side


def models_path(product, n=WINDOW):
    return FEATURE_DIR / f"kernel_models_{product}{size_suffix(n)}.npz"


def load_models(product, n=WINDOW):
    with np.load(models_path(product, n)) as z:
        return {k: z[k] for k in z.files}


def load_feature_rows(product, n=WINDOW):
    ds = xr.open_dataset(feature_path(product, n))
    names = list(ds.data_vars)
    valid = np.isfinite(ds[names[0]].values)  # all features share one NaN pattern
    t_idx, wi, wj = np.nonzero(valid)
    X = np.empty((len(t_idx), len(names)), dtype=np.float32)
    for f, name in enumerate(names):
        X[:, f] = ds[name].values[valid]
    out = {
        "names": names, "X": X,
        "t_idx": t_idx.astype(np.int32), "wi": wi.astype(np.int16), "wj": wj.astype(np.int16),
        "times": ds["time"].values, "win_lat": ds["win_lat"].values, "win_lon": ds["win_lon"].values,
    }
    ds.close()
    return out


def _sample(n, size, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(size, n), replace=False))


def _scaler(X_sample):
    mean = X_sample.mean(axis=0)
    std = X_sample.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def _apply_chunked(fn, X, mean, std):
    parts = [fn((X[s:s + PREDICT_CHUNK] - mean) / std) for s in range(0, len(X), PREDICT_CHUNK)]
    return np.concatenate(parts)


def fit_isolation_forest(X, seed=RANDOM_STATE, sample_size=FIT_SAMPLE):
    Xs = X[_sample(len(X), sample_size, seed)]
    mean, std = _scaler(Xs)
    iso = IsolationForest(n_estimators=IF_TREES, random_state=seed, n_jobs=-1).fit((Xs - mean) / std)
    return (-_apply_chunked(iso.score_samples, X, mean, std)).astype(np.float32)


def top_threshold(score, frac=TOP_FRAC):
    return float(np.quantile(score, 1.0 - frac))


def _best_per_timestamp(score, t_idx):
    # Ties (Isolation Forest scores saturate: e.g. 16 AURI windows share the
    # maximum) are broken by a seeded random order, not archive order, so tied
    # hours from any year can be picked; the seed keeps the list reproducible.
    tiebreak = np.random.default_rng(RANDOM_STATE).permutation(len(score))
    order = np.lexsort((tiebreak, -score))
    _, first = np.unique(t_idx[order], return_index=True)
    return order[np.sort(first)]  # best window of each timestamp, in descending score


def pick_review_events(per_size, times, n_each, min_gap_days):
    """Greedy review list: for each (label, score, t_idx) in order, up to
    n_each timestamps by descending best-window score, each at least
    min_gap_days from every timestamp already picked (for any size).
    Returns [(time_index, label)] in pick order."""
    gap = np.timedelta64(int(min_gap_days * 24 * 60), "m")
    picked = []
    for label, score, t_idx in per_size:
        k = 0
        for r in _best_per_timestamp(score, t_idx):
            ti = int(t_idx[r])
            if all(abs(times[ti] - times[p]) >= gap for p, _ in picked):
                picked.append((ti, label))
                k += 1
                if k == n_each:
                    break
    return picked


def pick_top_distinct(score, t_idx, n, times=None, min_gap_days=0):
    """Best window of each timestamp, in descending score, up to n. With
    `times` (the archive time axis) and min_gap_days > 0, a timestamp is also
    skipped if it lies within min_gap_days of one already picked, so each
    pick is a different event rather than another hour of the same one."""
    best = _best_per_timestamp(score, t_idx)
    if times is None or min_gap_days <= 0:
        return best[:n]
    gap = np.timedelta64(int(min_gap_days * 24 * 60), "m")
    picked, picked_t = [], []
    for r in best:
        t = times[t_idx[r]]
        if all(abs(t - p) >= gap for p in picked_t):
            picked.append(r)
            picked_t.append(t)
            if len(picked) == n:
                break
    return np.array(picked, dtype=best.dtype)


def write_tables(product, rows, score, n=WINDOW):
    thr = top_threshold(score)
    is_top = score > thr
    names = rows["names"]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    idx = np.nonzero(is_top)[0]
    idx = idx[np.argsort(-score[idx])]
    top = pd.DataFrame({
        "rank": np.arange(1, len(idx) + 1),
        "time": rows["times"][rows["t_idx"][idx]],
        "win_lat": rows["win_lat"][rows["wi"][idx]], "win_lon": rows["win_lon"][rows["wj"][idx]],
        "row": rows["wi"][idx], "col": rows["wj"][idx], "anomaly_score": score[idx],
    })
    for f, name in enumerate(names):
        top[name] = rows["X"][idx, f]
    p = REPORTS_DIR / f"kernel_anomalies_top_{product}{size_suffix(n)}.csv"
    top.to_csv(p, index=False)
    print(f"Saved: {p}  ({len(top):,} windows above score {thr:.4f})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, nargs="+", default=REVIEW_SIZES)
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS))
    args = ap.parse_args()
    for n in args.window:
        for product in args.products:
            print(f"== {product} {n}x{n} ==")
            rows = load_feature_rows(product, n)
            print(f"  {len(rows['X']):,} complete windows, features {rows['names']}")
            score = fit_isolation_forest(rows["X"])
            np.savez(models_path(product, n), t_idx=rows["t_idx"], wi=rows["wi"], wj=rows["wj"], score=score)
            print(f"Saved: {models_path(product, n)}")
            write_tables(product, rows, score, n)


if __name__ == "__main__":
    main()
