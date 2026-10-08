"""WP3: figures for the sliding-window kernel analysis.

Per product and window size (4x4: no suffix, 3x3: "_3x3"):
  kernel_anomaly_map_<product>[_3x3].png      % of complete windows covering each
                                              cell that are in the top 1%
  kernel_anomaly_monthly_<product>[_3x3].png  monthly top-1% rate
  kernel_frequency_<product>[_3x3].png        how often each kernel fires strongly, whole archive
Per product:
  kernel_featuremaps_<product>_<time>.png  field + kernel maps of BOTH sizes for one hour, for the
                                        list of the most anomalous events (5 best 4x4 + 5 best
                                        3x3, >= 3 days apart) and the example cases

Run: python scripts/kernel_plots.py [--top | --frequency |
     --featuremap PRODUCT TIME]
"""
import argparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle

from cluster_regimes import RADIAL_SCALE, SPEED_CMAP, TOTAL_VMAX
from kernel_clustering import REVIEW_SIZES, load_feature_rows, load_models, pick_review_events, top_threshold
from kernel_features import (KERNEL_NAMES, PRODUCTS, PROJECT_ROOT, WINDOW, anomaly_rate_map, apply_good_mask,
                             feature_path, load_good_fields, load_good_snapshot, make_kernels, size_suffix)
from plot_qc_flagged_field import RDVA_CMAP, _cell_edges

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RDVA_VMAX = 0.27        # radials' typical speed ceiling, same value behind RADIAL_SCALE
N_EACH = 5            # review list: this many best events per window size
TOP_MIN_GAP_DAYS = 3  # review picks at least this far apart, so each is a different event
BOX_COLOURS = {4: "#ff1493", 3: "#00bcd4"}  # most anomalous window: pink = 4x4, cyan = 3x3
MAP_MIN_WINDOWS = 1000  # cells covered by fewer complete windows are left blank (rate too noisy)
FIELD_SCALE = 8.0       # = cluster_regimes.TOTAL_SCALE, full-domain quiver
LAND = cfeature.NaturalEarthFeature("physical", "land", "10m")
FEATUREMAP_TIMES_TOTAL = [
    ("2025-11-08T07:00:00", "uncaught divergence example"),
    ("2025-12-25T11:00:00", "Bora, Christmas 2025"),
]


def _capped(u, v):
    s = np.hypot(u, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(s > TOTAL_VMAX, TOTAL_VMAX / s, 1.0)
    return u * f, v * f, s


def _map_axes(fig, pos, lat, lon, labels=True, left_labels=True):
    args = pos if isinstance(pos, tuple) else (pos,)
    ax = fig.add_subplot(*args, projection=ccrs.PlateCarree())
    ax.set_extent([lon.min() - 0.02, lon.max() + 0.02, lat.min() - 0.02, lat.max() + 0.02], crs=ccrs.PlateCarree())
    ax.add_feature(LAND, facecolor="#d8d4c8", edgecolor="#5a5a5a", linewidth=0.6, zorder=2)
    gl = ax.gridlines(draw_labels=labels, linewidth=0.4, color="#c9c7bf", linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.left_labels = labels and left_labels
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    return ax


def _draw_field(ax, fields, product, t):
    lat, lon = fields["lat"], fields["lon"]
    if product == "total":
        u, v, s = _capped(fields["u"][t], fields["v"][t])
        lon2d, lat2d = np.meshgrid(lon, lat)
        q = ax.quiver(lon2d, lat2d, u, v, s, transform=ccrs.PlateCarree(), cmap=SPEED_CMAP, scale=FIELD_SCALE,
                      width=0.004, headwidth=2.5, headlength=4, headaxislength=3.5, zorder=3)
        q.set_clim(0, TOTAL_VMAX)
        return q, "Speed (m/s)"
    m = ax.pcolormesh(_cell_edges(lon), _cell_edges(lat), np.ma.masked_invalid(fields["rdva"][t]),
                      transform=ccrs.PlateCarree(), cmap=RDVA_CMAP, vmin=-RDVA_VMAX, vmax=RDVA_VMAX,
                      edgecolors="#c9c7bf", linewidth=0.15, zorder=2.5)
    return m, "RDVA (m/s)  (+ away from station, − toward)"


def _radial_uv(product, time_value):
    """QC-good EWCT/NSCT of a radial at one time (RDVA split along each
    cell's bearing), for the velocity panel next to the RDVA map. Selected
    by time, not index: the IZOL start cut shifts the feature-file index."""
    ds = xr.open_dataset(PRODUCTS[product][0])
    s = ds.sel(time=time_value).squeeze("depth")
    qc = s["QCflag"].values
    u, v = apply_good_mask(s["EWCT"].values, qc), apply_good_mask(s["NSCT"].values, qc)
    ds.close()
    return u, v


def _draw_radial_vectors(ax, lat, lon, u, v):
    """Same convention as the cluster plots (cluster_regimes): length ~ speed,
    RADIAL_SCALE, length capped at TOTAL_VMAX, plasma_r over 0-TOTAL_VMAX."""
    u, v, s = _capped(np.ma.masked_invalid(u), np.ma.masked_invalid(v))
    lon2d, lat2d = np.meshgrid(lon, lat)
    q = ax.quiver(lon2d, lat2d, u, v, s, transform=ccrs.PlateCarree(), cmap=SPEED_CMAP, scale=RADIAL_SCALE,
                  width=0.004, headwidth=2.5, headlength=4, headaxislength=3.5, zorder=3)
    q.set_clim(0, TOTAL_VMAX)
    ax.quiverkey(q, X=0.87, Y=0.06, U=TOTAL_VMAX, label=f"{TOTAL_VMAX:.2f} m/s", labelpos="N",
                 coordinates="axes", fontproperties={"size": 8})
    return q


def plot_anomaly_map(product, rows, models, fields, n=WINDOW):
    wy, wx = len(rows["win_lat"]), len(rows["win_lon"])
    pos = models["wi"].astype(int) * wx + models["wj"].astype(int)
    is_top = models["score"] > top_threshold(models["score"])
    all_c = np.bincount(pos, minlength=wy * wx).reshape(wy, wx)
    top_c = np.bincount(pos[is_top], minlength=wy * wx).reshape(wy, wx)
    rate, denom = anomaly_rate_map(all_c, top_c, n, MAP_MIN_WINDOWS)
    n_hidden = int(((denom > 0) & (denom < MAP_MIN_WINDOWS)).sum())
    lat, lon = fields["lat"], fields["lon"]
    fig = plt.figure(figsize=(7, 6), dpi=150)
    ax = _map_axes(fig, 111, lat, lon)
    vmax = float(np.nanpercentile(rate, 98)) if np.isfinite(rate).any() else 1.0
    m = ax.pcolormesh(_cell_edges(lon), _cell_edges(lat), np.ma.masked_invalid(rate), transform=ccrs.PlateCarree(),
                      cmap="Purples", vmin=0, vmax=max(vmax, 1.0), zorder=1.5)
    fig.colorbar(m, ax=ax, shrink=0.75, label="% of complete windows covering cell in top 1%")
    ax.set_title(f"{product.upper()} {n}×{n}: where the top-1% anomalous windows occur (chance level = 1%)",
                 fontsize=9, loc="left")
    fig.text(0.02, 0.01, f"Blank: cells covered by < {MAP_MIN_WINDOWS:,} complete windows "
             f"({n_hidden} cells hidden, rate too noisy).", fontsize=7, color="#444")
    out = FIGURES_DIR / f"kernel_anomaly_map_{product}{size_suffix(n)}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_anomaly_monthly(product, rows, models, n=WINDOW):
    is_top = models["score"] > top_threshold(models["score"])
    month = pd.DatetimeIndex(rows["times"][models["t_idx"]]).to_period("M")
    df = pd.DataFrame({"month": month, "top": is_top}).groupby("month")["top"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    ax.plot(df.index.to_timestamp(), 100.0 * df["mean"], color="#256abf", linewidth=1.4)
    ax.axhline(1.0, color="#888", linewidth=0.8, linestyle="--")
    ax.text(df.index.to_timestamp()[0], 1.05, "overall 1%", fontsize=7, color="#666", va="bottom")
    ax.set_ylabel("% of month's windows in top 1%")
    ax.set_title(f"{product.upper()} {n}×{n}: monthly anomalous-window rate", fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    out = FIGURES_DIR / f"kernel_anomaly_monthly_{product}{size_suffix(n)}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _outline(ax, fields, wi, wj, n=WINDOW, colour="#ff1493"):
    le, ae = _cell_edges(fields["lon"]), _cell_edges(fields["lat"])
    ax.add_patch(Rectangle((le[wj], ae[wi]), le[wj + n] - le[wj], ae[wi + n] - ae[wi],
                           fill=False, edgecolor=colour, linewidth=2.0, transform=ccrs.PlateCarree(), zorder=4))


FEATURE_VMAX = 0.4  # m/s, fixed so colours mean the same at every timestamp
KERNEL_TITLES = {
    "TB": "TB  top/bottom:\nnorth half − south half",
    "LR": "LR  left/right:\neast half − west half",
    "DR": "DR  diagonal ↘:\nNE − SW triangle",
    "UR": "UR  diagonal ↗:\nNW − SE triangle",
}
CHANNEL_TITLES = {"u": "applied to u\n(east–west speed)", "v": "applied to v\n(north–south speed)",
                  "rdva": "applied to RDVA\n(radial speed)"}


def _draw_kernel_pattern(ax, w):
    """The kernel's +1/-1 layout, north up (row 0 = south at the bottom)."""
    s = np.sign(w)
    ax.imshow(s, origin="lower", cmap="RdBu_r", vmin=-1.6, vmax=1.6)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            ax.text(j, i, {1: "+", -1: "−", 0: "0"}[int(s[i, j])], ha="center", va="center", fontsize=9,
                    color="#222")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("W        E", fontsize=7)
    ax.set_ylabel("S        N", fontsize=7)


def plot_feature_maps(product, time_iso, label, models_by_size=None):
    """Field map(s) + every kernel's value at every window position for one
    timestamp, for BOTH window sizes (4x4 then 3x3). One row per kernel; per
    size one column with its +/- pattern and one column per component.
    models_by_size: {n: models} used for the boxes/ranks (optional)."""
    snap = load_good_snapshot(product, time_iso)
    if snap is None:
        print(f"  {product}: {time_iso} not in archive (or before the product's start date), skipped")
        return
    ti = snap["t_archive"]
    channels = list(PRODUCTS[product][1])
    lat, lon = snap["lat"], snap["lon"]
    radial = product != "total"
    pc = 2 if radial else 1  # number of field-map columns (radials: RDVA + velocity)
    ncol = len(channels)
    widths = [5.2] * pc
    for _ in REVIEW_SIZES:
        widths += [0.9] + [2.6] * ncol
    fig = plt.figure(figsize=(9 + 6.5 * (pc - 1) + len(REVIEW_SIZES) * (1.2 + 3.6 * ncol), 14), dpi=130)
    gs = fig.add_gridspec(5, len(widths), width_ratios=widths, height_ratios=[0.25, 1, 1, 1, 1],
                          wspace=0.08, hspace=0.25)
    ax0 = _map_axes(fig, gs[1:3, 0], lat, lon)
    h, lab = _draw_field(ax0, snap, product, 0)
    fig.colorbar(h, ax=ax0, orientation="horizontal", shrink=0.8, pad=0.06, label=lab)
    ax0.set_title("RDVA (QC-good cells)" if radial else "current field (QC-good cells)", fontsize=10, loc="left")
    map_axes = [ax0]
    if radial:
        axv = _map_axes(fig, gs[1:3, 1], lat, lon, left_labels=False)
        q = _draw_radial_vectors(axv, lat, lon, *_radial_uv(product, snap["time"][0]))
        fig.colorbar(q, ax=axv, orientation="horizontal", shrink=0.8, pad=0.06, label="Speed (m/s)")
        axv.set_title("velocity (QC-good cells)", fontsize=10, loc="left")
        map_axes.append(axv)
    caption = []
    for n in REVIEW_SIZES:
        models = (models_by_size or {}).get(n)
        if models is None:
            continue
        m = models["t_idx"] == ti
        if not m.any():
            caption.append(f"{n}×{n}: no complete windows at this hour")
            continue
        r = np.nonzero(m)[0][np.argmax(models["score"][m])]
        for ax in map_axes:
            _outline(ax, snap, int(models["wi"][r]), int(models["wj"][r]), n, BOX_COLOURS[n])
        rank = int((models["score"] > models["score"][r]).sum()) + 1
        colour_name = "pink" if n == 4 else "cyan"
        caption.append(f"{colour_name} box: most anomalous {n}×{n} window this hour, score {models['score'][r]:.3f}, "
                       f"rank {rank:,} of {len(models['score']):,}")
    if caption:
        ax0.text(0.0, -0.32, "\n".join(caption) + "\n(rank 1 = most anomalous window of its size, whole archive)",
                 transform=ax0.transAxes, fontsize=9, va="top")
    mesh = None
    col = pc
    for n in REVIEW_SIZES:
        f = xr.open_dataset(feature_path(product, n)).isel(time=ti)
        we, ae = _cell_edges(f["win_lon"].values), _cell_edges(f["win_lat"].values)
        kernels = make_kernels(n)
        for c, ch in enumerate(channels):
            hax = fig.add_subplot(gs[0, col + 1 + c])
            hax.axis("off")
            hax.text(0.5, 0.0, f"{n}×{n}\n" + CHANNEL_TITLES[ch], ha="center", va="bottom", fontsize=10,
                     weight="bold", color=BOX_COLOURS[n])
        for r, k in enumerate(KERNEL_NAMES):
            pax = fig.add_subplot(gs[1 + r, col])
            _draw_kernel_pattern(pax, kernels[k])
            pax.set_title(KERNEL_TITLES[k], fontsize=7.5, loc="left")
            for c, ch in enumerate(channels):
                ax = _map_axes(fig, gs[1 + r, col + 1 + c], lat, lon, labels=False)
                mesh = ax.pcolormesh(we, ae, np.ma.masked_invalid(f[f"{k}_{ch}"].values),
                                     transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-FEATURE_VMAX,
                                     vmax=FEATURE_VMAX, zorder=1.5)
        f.close()
        col += 1 + ncol
    leg = gs[3:5, 0].get_position(fig)
    cax = fig.add_axes([leg.x0, leg.y0 + 0.62 * leg.height, 0.85 * leg.width, 0.012])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("kernel value = mean(+ half) − mean(− half), m/s", fontsize=9)
    fig.text(leg.x0, leg.y0 + 0.5 * leg.height,
             "Each coloured square = one window (4×4 or 3×3, placed at its centre). White: the two halves agree.\n"
             "3×3: the middle row / column / diagonal has weight 0.\n"
             "Red: the + half is faster toward east (u) / north (v) /\naway from the station (RDVA) than the − half; "
             "blue: the opposite.\n"
             + ("Divergence: TB on v and LR on u are red (halves moving apart);\nconvergence: they are blue."
                if product == "total" else
                "Dark red or blue = neighbouring cells with very different radial speed."),
             fontsize=9, va="top")
    fig.suptitle(f"{product.upper()} — kernels over the whole map at {time_iso[:16]} UTC ({label})",
                 fontsize=12, x=0.05, ha="left", y=0.93)
    out = FIGURES_DIR / f"kernel_featuremaps_{product}_{time_iso[:16].replace(':', '')}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


STRONG_FRAC = 0.01  # "strong" = the most extreme 1% of a feature's |values| over the whole archive


def feature_frequency(values, pos, n_pos, frac=STRONG_FRAC, min_windows=MAP_MIN_WINDOWS):
    """For one feature over the whole archive: per window position, the % of
    that position's complete windows with a strong positive value (> T) and a
    strong negative value (< -T), T = (1-frac) quantile of |values|. NaN where
    the position has fewer than min_windows complete windows."""
    thr = float(np.quantile(np.abs(values), 1.0 - frac))
    n_all = np.bincount(pos, minlength=n_pos).astype(float)
    n_pos_strong = np.bincount(pos[values > thr], minlength=n_pos)
    n_neg_strong = np.bincount(pos[values < -thr], minlength=n_pos)
    ok = n_all >= min_windows
    with np.errstate(invalid="ignore", divide="ignore"):
        f_pos = np.where(ok, 100.0 * n_pos_strong / n_all, np.nan)
        f_neg = np.where(ok, 100.0 * n_neg_strong / n_all, np.nan)
    return f_pos, f_neg, thr, n_all


def plot_feature_frequency(product, rows, n=WINDOW):
    """Whole-archive version of the kernel-map figure: for every kernel and
    component, how often each window position gave a strong positive (red)
    or strong negative (blue) value, 2021-2026."""
    channels = list(PRODUCTS[product][1])
    kernels = make_kernels(n)
    wy, wx = len(rows["win_lat"]), len(rows["win_lon"])
    pos = rows["wi"].astype(int) * wx + rows["wj"].astype(int)
    we, ae = _cell_edges(rows["win_lon"]), _cell_edges(rows["win_lat"])
    ds_src = xr.open_dataset(PRODUCTS[product][0])
    lat, lon = ds_src["latitude"].values, ds_src["longitude"].values
    t0, t1 = str(ds_src["time"].values[0])[:10], str(ds_src["time"].values[-1])[:10]
    ds_src.close()

    freqs, thrs = {}, {}
    n_all = None
    for k in KERNEL_NAMES:
        for ch in channels:
            name = f"{k}_{ch}"
            f_pos, f_neg, thr, n_all = feature_frequency(rows["X"][:, rows["names"].index(name)], pos, wy * wx)
            freqs[name] = (f_pos.reshape(wy, wx), f_neg.reshape(wy, wx))
            thrs[name] = thr
    allv = np.concatenate([np.concatenate([a.ravel(), b.ravel()]) for a, b in freqs.values()])
    vmax = float(np.nanpercentile(allv, 98)) if np.isfinite(allv).any() else 1.0

    ncol = 2 * len(channels)
    fig = plt.figure(figsize=(8 + 2.7 * ncol, 14), dpi=130)
    gs = fig.add_gridspec(5, 2 + ncol, width_ratios=[5.2, 0.9] + [2.1] * ncol,
                          height_ratios=[0.3, 1, 1, 1, 1], wspace=0.08, hspace=0.3)
    ax0 = _map_axes(fig, gs[1:3, 0], lat, lon)
    n_grid = np.where(n_all > 0, n_all, np.nan).reshape(wy, wx)
    m0 = ax0.pcolormesh(we, ae, np.ma.masked_invalid(n_grid), transform=ccrs.PlateCarree(), cmap="Greys",
                        vmin=0, zorder=1.5)
    fig.colorbar(m0, ax=ax0, orientation="horizontal", shrink=0.8, pad=0.06,
                 label=f"number of complete {n}×{n} windows at this position")
    ax0.set_title(f"data behind each square ({t0} to {t1})", fontsize=10, loc="left")
    comp_titles = {"u": "u (east–west)", "v": "v (north–south)", "rdva": "RDVA (radial)"}
    for c, ch in enumerate(channels):
        for sgn, (lab, col) in enumerate([("strong +  (red)", "#b2182b"), ("strong −  (blue)", "#2166ac")]):
            hax = fig.add_subplot(gs[0, 2 + 2 * c + sgn])
            hax.axis("off")
            hax.text(0.5, 0.0, f"on {comp_titles[ch]}\n{lab}", ha="center", va="bottom", fontsize=9.5,
                     weight="bold", color=col)
    mesh_r = mesh_b = None
    for r, k in enumerate(KERNEL_NAMES):
        pax = fig.add_subplot(gs[1 + r, 1])
        _draw_kernel_pattern(pax, kernels[k])
        cut = ", ".join(f"{ch}: {thrs[f'{k}_{ch}']:.2f}" for ch in channels)
        pax.set_title(KERNEL_TITLES[k] + f"\nstrong: |value| >\n{cut} m/s", fontsize=7.5, loc="left")
        for c, ch in enumerate(channels):
            f_pos, f_neg = freqs[f"{k}_{ch}"]
            for sgn, (arr, cmap) in enumerate([(f_pos, "Reds"), (f_neg, "Blues")]):
                ax = _map_axes(fig, gs[1 + r, 2 + 2 * c + sgn], lat, lon, labels=False)
                mm = ax.pcolormesh(we, ae, np.ma.masked_invalid(arr), transform=ccrs.PlateCarree(), cmap=cmap,
                                   vmin=0, vmax=vmax, zorder=1.5)
                if sgn == 0:
                    mesh_r = mm
                else:
                    mesh_b = mm
    leg = gs[3:5, 0].get_position(fig)
    for i, (mm, lab) in enumerate([(mesh_r, "strong positive"), (mesh_b, "strong negative")]):
        cax = fig.add_axes([leg.x0, leg.y0 + (0.78 - 0.16 * i) * leg.height, 0.85 * leg.width, 0.012])
        cb = fig.colorbar(mm, cax=cax, orientation="horizontal", extend="max")
        cb.set_label(f"% of this position's windows with a {lab} value", fontsize=9)
    fig.text(leg.x0, leg.y0 + 0.42 * leg.height,
             f"Each square = one {n}×{n} window position, counted over the whole archive.\n"
             f"'Strong' = the most extreme {100 * STRONG_FRAC:.0f}% of that kernel's values (cut-off on each row),\n"
             "so averaged over the map, red + blue together = 1%.\n"
             "Squares well above that are places where this kernel fires unusually often.\n"
             + ("LR on u and TB on v: red = divergence, blue = convergence.\n"
                "LR on v and TB on u: red and blue = the two senses of shear."
                if product == "total" else
                "Red/blue = the + half much more away from / toward the station than the − half.")
             + f"\nBlank: fewer than {MAP_MIN_WINDOWS:,} complete windows at that position.",
             fontsize=9, va="top")
    fig.suptitle(f"{product.upper()} {n}×{n} — how often each kernel fires strongly, whole archive {t0} to {t1}",
                 fontsize=12, x=0.05, ha="left", y=0.93)
    out = FIGURES_DIR / f"kernel_frequency_{product}{size_suffix(n)}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def review_events(product):
    """Review list: the N_EACH best 4x4 events, then the N_EACH best 3x3
    events, all >= TOP_MIN_GAP_DAYS apart. [(iso minute, label)]."""
    with xr.open_dataset(feature_path(product, REVIEW_SIZES[0])) as f:
        times = f["time"].values
    per_size = []
    for n in REVIEW_SIZES:
        m = load_models(product, n)
        per_size.append((f"{n}x{n} top", m["score"], m["t_idx"]))
    picks = pick_review_events(per_size, times, N_EACH, TOP_MIN_GAP_DAYS)
    return [(str(np.datetime_as_string(times[ti], unit="m")), lab) for ti, lab in picks]


def plot_review_figures():
    """Only the review-list kernel-map figures (both sizes), every product."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for product in PRODUCTS:
        models = {n: load_models(product, n) for n in REVIEW_SIZES}
        for iso, lab in review_events(product):
            plot_feature_maps(product, iso, lab, models)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for product in PRODUCTS:
        fields = load_good_fields(product)
        for n in REVIEW_SIZES:
            print(f"== {product} {n}x{n} ==")
            rows = load_feature_rows(product, n)
            models = load_models(product, n)
            plot_anomaly_map(product, rows, models, fields, n)
            plot_anomaly_monthly(product, rows, models, n)
            plot_feature_frequency(product, rows, n)
            del rows
        del fields
    plot_review_figures()
    models = {n: load_models("total", n) for n in REVIEW_SIZES}
    for iso, lab in FEATUREMAP_TIMES_TOTAL:
        plot_feature_maps("total", iso, lab, models)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--featuremap", nargs=2, metavar=("PRODUCT", "TIME"),
                    help="only draw the kernel-map figure for one product (total/auri/pira/tri1/izol) "
                         "and one time, e.g. --featuremap total 2024-10-22T08:00")
    ap.add_argument("--frequency", action="store_true",
                    help="only draw the whole-archive kernel-frequency figures (both sizes) for every product")
    ap.add_argument("--top", action="store_true",
                    help="only draw the review-list kernel-map figures (5 best 4x4 + 5 best 3x3 events)")
    args = ap.parse_args()
    if args.top:
        plot_review_figures()
    elif args.frequency:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        for prod in PRODUCTS:
            for n in REVIEW_SIZES:
                plot_feature_frequency(prod, load_feature_rows(prod, n), n)
    elif args.featuremap:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        prod, when = args.featuremap
        prod = prod.lower()
        plot_feature_maps(prod, when, "chosen time", {n: load_models(prod, n) for n in REVIEW_SIZES})
    else:
        main()
