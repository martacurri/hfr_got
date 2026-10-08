"""WP3: kernel-based QC flag for reversals and outbursts (3x3 windows).

Step 1 find: a complete 3x3 window is flagged if its largest |kernel value| (A)
or its Isolation Forest score (B) is in the product's top `top_frac`, with both
thresholds taken from the whole-archive distributions.
Step 2 sort: reversal if some neighbouring cell pair inside the window points
in opposite directions (angle > theta; for radials the angle is 0 or 180, i.e.
opposite RDVA sign), both cells >= s_min and |difference| >= dv_min; otherwise
outburst.
Step 3 (radial reversals): the other radial stations at the same time, window
+-pad cells: >= 2 show a jump -> confirmed real (feature flag, code 1); none
shows a jump and >= 1 is smooth -> 4; otherwise (one jump, or no data) -> 3.
Step 4 (Total reversals): always 4; the radials showing a jump are recorded as
a bit mask (KREV_SRC).
Cells: all 9 cells of a flagged window get its result, worst wins.

WP3 runs this on the validation set only (kernel_qc_report.py); run() works
on any time selection, so it can also be applied to the whole archive.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

N = 3
RADIALS = ["auri", "pira", "tri1", "izol"]
SRC_BITS = {"auri": 1, "pira": 2, "tri1": 4, "izol": 8}
NOT_TESTED, GOOD, PROB_BAD, BAD = 0, 1, 3, 4


@dataclass(frozen=True)
class Params:
    top_frac: float = 0.005
    s_min_total: float = 0.10
    s_min_radial: float = 0.05
    theta: float = 90.0
    dv_min: float = 0.3
    jump: float = 0.3
    min_cells: int = 5
    pad: int = 1

    def s_min(self, product):
        return self.s_min_total if product == "total" else self.s_min_radial


def _kth_largest(x, k):
    return float(np.partition(x, len(x) - k)[len(x) - k])


def thresholds(X, score, top_frac):
    """A threshold (m/s, on each window's largest |kernel value|) and B threshold
    (Isolation Forest score), each the top_frac quantile of all windows given."""
    k = max(1, int(round(top_frac * len(score))))
    return _kth_largest(np.abs(X).max(axis=1), k), _kth_largest(np.asarray(score), k)


def flag_windows(X, score, thr_a, thr_b):
    """(picked by A, picked by B); ties at a threshold are all picked."""
    return np.abs(X).max(axis=1) >= thr_a, np.asarray(score) >= thr_b


# the 20 neighbouring cell pairs (horizontal, vertical, both diagonals) inside a 3x3 window
PAIRS = np.array([(a, b, a + da, b + db) for a in range(N) for b in range(N)
                  for da, db in [(0, 1), (1, 0), (1, 1), (1, -1)]
                  if 0 <= a + da < N and 0 <= b + db < N])


def window_values(fields, chans, t_idx, wi, wj):
    """Cell values of the listed windows: (M, 3, 3, len(chans))."""
    off = np.arange(N)
    t = np.asarray(t_idx)[:, None, None]
    r = np.asarray(wi)[:, None, None] + off[None, :, None]
    c = np.asarray(wj)[:, None, None] + off[None, None, :]
    return np.stack([fields[ch][t, r, c] for ch in chans], axis=-1)


def classify_windows(V, s_min, theta, dv_min):
    """True = reversal, False = outburst. V: (M, 3, 3, nch); nch = 2 (u, v) for
    Total, 1 (RDVA) for radials. Windows are complete, so V has no NaN."""
    if len(V) == 0:
        return np.zeros(0, dtype=bool)
    v1 = V[:, PAIRS[:, 0], PAIRS[:, 1], :]
    v2 = V[:, PAIRS[:, 2], PAIRS[:, 3], :]
    s1, s2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
    cos = (v1 * v2).sum(axis=-1) / np.maximum(s1 * s2, 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    hit = (ang > theta) & (np.minimum(s1, s2) >= s_min) & (np.linalg.norm(v1 - v2, axis=-1) >= dv_min)
    return hit.any(axis=1)


def max_jump_map(field):
    """(T, ny, nx): per cell, the largest |difference| to any valid 8-neighbour
    (NaN where the cell or all its neighbours are missing)."""
    _, ny, nx = field.shape
    pad = np.pad(field, ((0, 0), (1, 1), (1, 1)), constant_values=np.nan)
    out = np.full(field.shape, np.nan, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nb = pad[:, 1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
                out = np.fmax(out, np.abs(field - nb))
    return out


def station_status(field_t, wi, wj, p):
    """One other station at one timestamp, around the 3x3 window at (wi, wj)
    enlarged by p.pad cells (clipped to the grid): 'jump' if some neighbour pair
    with both cells inside that region differs by >= p.jump, 'smooth' if >=
    p.min_cells good cells and no jump, else 'none'. Returns (status, largest
    jump in m/s or NaN)."""
    ny, nx = field_t.shape
    r0, r1 = max(0, wi - p.pad), min(ny, wi + N + p.pad)
    c0, c1 = max(0, wj - p.pad), min(nx, wj + N + p.pad)
    reg = max_jump_map(field_t[None, r0:r1, c0:c1])[0]  # pairs inside the region only
    mj = float(np.nanmax(reg)) if np.isfinite(reg).any() else np.nan
    if np.isfinite(mj) and mj >= p.jump:
        return "jump", mj
    n_good = int(np.isfinite(field_t[r0:r1, c0:c1]).sum())
    return ("smooth" if n_good >= p.min_cells else "none"), mj


def cross_station(others, time, wi, wj, fields, tindex, p):
    """{station: (status, max jump)} for each other radial at this timestamp;
    'none' where the station has no record at that time."""
    out = {}
    for s in others:
        j = tindex[s].get(time)
        out[s] = ("none", np.nan) if j is None else station_status(fields[s]["rdva"][j], wi, wj, p)
    return out


def final_code(product, is_reversal, statuses):
    """-> (KREV_QC code, KOUT_FLAG, KREV_SRC bit mask, verdict text)."""
    if not is_reversal:
        return GOOD, True, 0, "outburst"
    jumps = [s for s, (st, _) in statuses.items() if st == "jump"]
    smooth = any(st == "smooth" for st, _ in statuses.values())
    if product == "total":
        src = sum(SRC_BITS[s] for s in jumps)
        if jumps:
            verdict = "source " + "+".join(s.upper() for s in jumps)
        else:
            verdict = "no radial jump" if smooth else "no radial data"
        return BAD, False, src, verdict
    if len(jumps) >= 2:
        return GOOD, True, 0, "confirmed"
    if len(jumps) == 1:
        return PROB_BAD, False, 0, f"one other ({jumps[0].upper()})"
    if smooth:
        return BAD, False, 0, "contradicted"
    return PROB_BAD, False, 0, "unchecked"


_RANK = np.zeros(5, dtype=np.int8)
_RANK[[NOT_TESTED, GOOD, PROB_BAD, BAD]] = [0, 1, 2, 3]  # worst wins: 4 > 3 > 1 > 0
_CODE_OF_RANK = np.array([NOT_TESTED, GOOD, PROB_BAD, BAD], dtype=np.uint8)


def windows_to_cells(shape, complete, flagged):
    """complete = (t, wi, wj) of all complete windows (local time index): their
    cells are tested (all QC-good by construction) -> 1. flagged = (t, wi, wj,
    code, feature, src): all 9 cells get the window's result, worst code wins,
    feature and source mask are OR-ed."""
    rank = np.zeros(shape, dtype=np.int8)
    kout = np.zeros(shape, dtype=np.uint8)
    ksrc = np.zeros(shape, dtype=np.uint8)
    t, wi, wj = (np.asarray(a, dtype=np.int64) for a in complete)
    ft, fi, fj = (np.asarray(a, dtype=np.int64) for a in flagged[:3])
    code, feat, src = (np.asarray(a) for a in flagged[3:])
    for a in range(N):
        for b in range(N):
            rank[t, wi + a, wj + b] = 1
    for a in range(N):  # flagged windows only after all tested cells are marked, so 1 never overwrites 3/4
        for b in range(N):
            if len(ft):
                np.maximum.at(rank, (ft, fi + a, fj + b), _RANK[code.astype(np.int64)])
                np.maximum.at(kout, (ft, fi + a, fj + b), feat.astype(np.uint8))
                np.bitwise_or.at(ksrc, (ft, fi + a, fj + b), src.astype(np.uint8))
    return {"KREV_QC": _CODE_OF_RANK[rank], "KOUT_FLAG": kout, "KREV_SRC": ksrc}


@dataclass
class Inputs:
    rows: dict    # product -> load_feature_rows(product, 3)
    score: dict   # product -> Isolation Forest score per row (same order)
    fields: dict  # product -> load_good_fields(product)
    tindex: dict  # radial -> {time: index into its fields}


def load_inputs():
    from kernel_clustering import load_feature_rows, load_models
    from kernel_features import load_good_fields
    rows, score, fields = {}, {}, {}
    for product in ["total"] + RADIALS:
        rows[product] = load_feature_rows(product, N)
        m = load_models(product, N)
        assert np.array_equal(m["t_idx"], rows[product]["t_idx"]) and np.array_equal(m["wi"], rows[product]["wi"])
        score[product] = m["score"]
        fields[product] = load_good_fields(product)
        assert np.array_equal(fields[product]["time"], rows[product]["times"])
    tindex = {s: {t: i for i, t in enumerate(fields[s]["time"])} for s in RADIALS}
    return Inputs(rows=rows, score=score, fields=fields, tindex=tindex)


WINDOW_COLUMNS = ["time", "win_lat", "win_lon", "wi", "wj", "picked_a", "picked_b", "max_abs_kernel", "score",
                  "cls", "verdict", "code", "feature", "src"]


def run(product, inp, time_sel, p=Params()):
    """Flag the timestamps where time_sel (bool over the product's times) is True.
    Thresholds always come from all of the product's windows (whole archive)."""
    rows, score = inp.rows[product], inp.score[product]
    thr_a, thr_b = thresholds(rows["X"], score, p.top_frac)
    pick_a, pick_b = flag_windows(rows["X"], score, thr_a, thr_b)
    in_sel = np.asarray(time_sel)[rows["t_idx"]]
    idx = np.nonzero(in_sel & (pick_a | pick_b))[0]
    chans = ["u", "v"] if product == "total" else ["rdva"]
    others = RADIALS if product == "total" else [s for s in RADIALS if s != product]
    V = window_values(inp.fields[product], chans, rows["t_idx"][idx], rows["wi"][idx], rows["wj"][idx])
    rev = classify_windows(V, p.s_min(product), p.theta, p.dv_min)
    sel_t = np.nonzero(time_sel)[0]
    local = np.full(len(time_sel), -1, dtype=np.int64)
    local[sel_t] = np.arange(len(sel_t))
    recs = []
    for k, i in enumerate(idx):
        t = rows["times"][rows["t_idx"][i]]
        wi, wj = int(rows["wi"][i]), int(rows["wj"][i])
        st = cross_station(others, t, wi, wj, inp.fields, inp.tindex, p) if rev[k] else {}
        code, feat, src, verdict = final_code(product, bool(rev[k]), st)
        rec = {"time": t, "win_lat": float(rows["win_lat"][wi]), "win_lon": float(rows["win_lon"][wj]),
               "wi": wi, "wj": wj, "picked_a": bool(pick_a[i]), "picked_b": bool(pick_b[i]),
               "max_abs_kernel": float(np.abs(rows["X"][i]).max()), "score": float(score[i]),
               "cls": "reversal" if rev[k] else "outburst", "verdict": verdict, "code": code,
               "feature": feat, "src": src}
        rec.update({name: float(rows["X"][i, f]) for f, name in enumerate(rows["names"])})
        for s in others:
            s_st, s_mj = st.get(s, ("", np.nan))
            rec[f"status_{s}"], rec[f"jump_{s}"] = s_st, s_mj
        recs.append(rec)
    table = pd.DataFrame(recs, columns=WINDOW_COLUMNS + list(rows["names"])
                         + [c for s in others for c in (f"status_{s}", f"jump_{s}")])
    ny, nx = inp.fields[product][chans[0]].shape[1:]
    comp = np.nonzero(in_sel)[0]
    complete = (local[rows["t_idx"][comp]], rows["wi"][comp], rows["wj"][comp])
    flagged = (local[rows["t_idx"][idx]], rows["wi"][idx], rows["wj"][idx],
               table["code"].to_numpy(dtype=np.int64), table["feature"].to_numpy(dtype=bool),
               table["src"].to_numpy(dtype=np.int64))
    cells = windows_to_cells((len(sel_t), ny, nx), complete, flagged)
    return {"table": table, "cells": cells, "times": rows["times"][sel_t], "thr_a": thr_a, "thr_b": thr_b}
