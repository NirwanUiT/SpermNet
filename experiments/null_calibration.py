"""T0.2 (referee B2) -- Calibration panel for the continuum null.

The OU null OVERSHOOTS the observed statistics (g2 1.7x, dwell-law dAIC up to
7x). Two readings: (1) no switching biology (current manuscript), (2) the null
is mis-matched -- real velocity has structure (candidate: the flagellar beat,
~10-13 Hz, 5-6 cycles per classifier window) that suppresses spurious
threshold crossings relative to an AR(1) surrogate. This panel decides.

Compares GT vs null on quantities the null SHOULD match if it is calibrated:
  - velocity ACF at lags 1-50 (the null is fit to lag 1 only)
  - velocity power spectrum: peak in 2-24 Hz, band-power fraction 5-20 Hz
    (direct test of the beat hypothesis: AR(1) is monotone by construction)
  - window-level VCL / LIN marginal quantiles
  - state occupancy fractions
  - switch rate per second, episodes per track, mean episode duration
    (if these differ, every episode-level GT-vs-null comparison in s3.2 is
    between cohorts with different episode structure)

Output: outputs/markov/null_calibration.json
Usage:  python -m experiments.null_calibration
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import compute_frame_states  # noqa: E402

GT_DIR = ROOT / "outputs" / "tracks_gt"
NULL_DIR = ROOT / "outputs" / "tracks_continuum_null"
OUT = config.MARKOV_OUT / "null_calibration.json"

MAX_LAG = 50
ACF_MIN_LEN = 120          # frames required for a per-track ACF
PSD_MIN_LEN = 256          # frames required for a per-track periodogram
BEAT_BAND = (5.0, 20.0)    # Hz
SEARCH_BAND = (2.0, 24.0)  # Hz, peak search range (excludes DC leakage)
N_BOOT = 2000
WINDOW = 25                # frames, matches the classifier window
WVCL_MAX_LAG = 100         # lags for the windowed-VCL series ACF
WVCL_SEARCH = (0.1, 2.0)   # Hz, peak search for windowed-VCL PSD (boxcar
                           # first null at FPS/(WINDOW-1) ~ 2.08 Hz)
THRESHOLDS = (config.VCL_IMMOTILE_MAX, config.VCL_PROGRESSIVE_MIN)  # 5, 25 um/s


def iter_tracks(d: Path):
    for tf in sorted(d.glob("*_tracks.csv")):
        vid = tf.stem.replace("_tracks", "")
        df = pd.read_csv(tf)
        for tid, tr in df.groupby("track_id"):
            tr = tr.sort_values("frame")
            yield vid, tid, tr


def track_acf(v: np.ndarray, max_lag: int) -> np.ndarray | None:
    """Normalised ACF of a 2-D velocity series, pooled over components."""
    c = v - v.mean(axis=0)
    den = float((c * c).sum())
    if den <= 0:
        return None
    out = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        out[k - 1] = float((c[k:] * c[:-k]).sum()) / den
    return out


def track_psd(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean periodogram over velocity components (Hann window)."""
    n = len(v)
    w = np.hanning(n)
    ps = 0.0
    for j in range(v.shape[1]):
        c = v[:, j] - v[:, j].mean()
        spec = np.abs(np.fft.rfft(c * w)) ** 2
        ps = ps + spec
    freqs = np.fft.rfftfreq(n, d=1.0 / config.FPS)
    ps = ps / max(ps.sum(), 1e-12)  # normalise so tracks weigh equally
    return freqs, ps


def sliding_wvcl(xs: np.ndarray, ys: np.ndarray) -> np.ndarray | None:
    """Sliding-window VCL (um/s), stride 1 -- the classifier's actual input."""
    steps_um = np.hypot(np.diff(xs), np.diff(ys)) / config.PIXELS_PER_MICRON
    k = WINDOW - 1
    if len(steps_um) < k:
        return None
    kern = np.ones(k) / k
    return np.convolve(steps_um, kern, mode="valid") * config.FPS


def window_kinematics(xs: np.ndarray, ys: np.ndarray) -> list[tuple[float, float]]:
    """(VCL um/s, LIN) on non-overlapping WINDOW-frame windows."""
    out = []
    dt = 1.0 / config.FPS
    ppm = config.PIXELS_PER_MICRON
    for lo in range(0, len(xs) - WINDOW + 1, WINDOW):
        wx, wy = xs[lo:lo + WINDOW], ys[lo:lo + WINDOW]
        steps = np.hypot(np.diff(wx), np.diff(wy))
        path = float(steps.sum())
        vcl = (path / ppm) / ((WINDOW - 1) * dt)
        net = float(np.hypot(wx[-1] - wx[0], wy[-1] - wy[0]))
        lin = net / path if path > 0 else 0.0
        out.append((vcl, lin))
    return out


def analyse(d: Path, label: str) -> dict:
    acfs, psd_grid, psd_sum, psd_n = [], None, None, 0
    vcl, lin = [], []
    per_video: dict[str, dict] = {}
    peak_freqs = []
    wv_acfs, wv_psd_sum, wv_psd_n, wv_peaks = [], None, 0, []
    wv_grid = np.arange(0.05, 10.01, 0.05)
    cross_intervals: dict[float, list[float]] = {t: [] for t in THRESHOLDS}
    cross_rates: dict[float, list[float]] = {t: [] for t in THRESHOLDS}
    for vid, tid, tr in iter_tracks(d):
        xs = tr["cx"].to_numpy(float)
        ys = tr["cy"].to_numpy(float)
        n = len(xs)
        if n < config.MIN_TRACK_LENGTH:
            continue
        v = np.column_stack([np.diff(xs), np.diff(ys)])
        if n >= ACF_MIN_LEN:
            a = track_acf(v, MAX_LAG)
            if a is not None:
                acfs.append(a)
        if n >= PSD_MIN_LEN:
            freqs, ps = track_psd(v)
            # common grid: interpolate onto 0-25 Hz, 0.25 Hz steps
            grid = np.arange(0.25, 25.01, 0.25)
            psd_i = np.interp(grid, freqs, ps)
            psd_sum = psd_i if psd_sum is None else psd_sum + psd_i
            psd_grid = grid
            psd_n += 1
            m = (freqs >= SEARCH_BAND[0]) & (freqs <= SEARCH_BAND[1])
            if m.any():
                peak_freqs.append(float(freqs[m][np.argmax(ps[m])]))
        for vc, li in window_kinematics(xs, ys):
            vcl.append(vc)
            lin.append(li)
        wv = sliding_wvcl(xs, ys)
        if wv is not None and len(wv) >= 3:
            if len(wv) >= ACF_MIN_LEN:
                a = track_acf(wv[:, None], WVCL_MAX_LAG)
                if a is not None:
                    wv_acfs.append(a)
            if len(wv) >= PSD_MIN_LEN:
                freqs, ps = track_psd(wv[:, None])
                wv_psd_i = np.interp(wv_grid, freqs, ps)
                wv_psd_sum = wv_psd_i if wv_psd_sum is None else wv_psd_sum + wv_psd_i
                wv_psd_n += 1
                m = (freqs >= WVCL_SEARCH[0]) & (freqs <= WVCL_SEARCH[1])
                if m.any():
                    wv_peaks.append(float(freqs[m][np.argmax(ps[m])]))
            dur = len(wv) / config.FPS
            for thr in THRESHOLDS:
                above = wv > thr
                idx = np.flatnonzero(np.diff(above.astype(int)) != 0)
                cross_rates[thr].append(len(idx) / dur)
                if len(idx) >= 2:
                    cross_intervals[thr].extend((np.diff(idx) / config.FPS).tolist())
        states = compute_frame_states(tr)
        sw = int(sum(1 for i in range(1, len(states)) if states[i] != states[i - 1]))
        dur_s = len(states) / config.FPS
        pv = per_video.setdefault(vid, {"switches": 0, "frames": 0, "tracks": 0,
                                        "occ": {"P": 0, "NP": 0, "I": 0},
                                        "rates": []})
        pv["switches"] += sw
        pv["frames"] += len(states)
        pv["tracks"] += 1
        pv["rates"].append(sw / dur_s)
        for s in states:
            key = {"Progressive": "P", "Non-progressive": "NP",
                   "Immotile": "I"}[str(s)]
            pv["occ"][key] = pv["occ"].get(key, 0) + 1

    acf_mean = np.mean(np.vstack(acfs), axis=0) if acfs else None
    total_frames = sum(pv["frames"] for pv in per_video.values())
    occ = {k: sum(pv["occ"].get(k, 0) for pv in per_video.values()) / total_frames
           for k in ("P", "NP", "I")}
    rates_all = np.concatenate([pv["rates"] for pv in per_video.values()])
    q = [5, 25, 50, 75, 95]
    res = {
        "label": label,
        "n_tracks_used": int(sum(pv["tracks"] for pv in per_video.values())),
        "acf": {"lags": list(range(1, MAX_LAG + 1)),
                "mean": [float(x) for x in acf_mean] if acf_mean is not None else None,
                "n_tracks": len(acfs)},
        "psd": None,
        "vcl_quantiles_um_s": {str(p): float(np.percentile(vcl, p)) for p in q},
        "lin_quantiles": {str(p): float(np.percentile(lin, p)) for p in q},
        "occupancy": occ,
        "switch_rate_per_s": {"mean": float(rates_all.mean()),
                              "median": float(np.median(rates_all))},
        "per_video": {vid: {"rate_mean": float(np.mean(pv["rates"])),
                            "n_tracks": pv["tracks"]}
                      for vid, pv in per_video.items()},
    }
    if psd_n:
        psd_mean = psd_sum / psd_n
        band = (psd_grid >= BEAT_BAND[0]) & (psd_grid <= BEAT_BAND[1])
        srch = (psd_grid >= SEARCH_BAND[0]) & (psd_grid <= SEARCH_BAND[1])
        res["psd"] = {
            "grid_hz": [float(x) for x in psd_grid],
            "mean": [float(x) for x in psd_mean],
            "n_tracks": psd_n,
            "band_power_5_20hz": float(psd_mean[band].sum() / psd_mean.sum()),
            "peak_hz_of_mean": float(psd_grid[srch][np.argmax(psd_mean[srch])]),
            "per_track_peak_hz_median": float(np.median(peak_freqs)),
            "per_track_peak_hz_iqr": [float(np.percentile(peak_freqs, 25)),
                                      float(np.percentile(peak_freqs, 75))],
        }
    # windowed-VCL series: the surface the classifier actually thresholds
    res["wvcl_acf"] = None
    if wv_acfs:
        wa = np.mean(np.vstack(wv_acfs), axis=0)
        res["wvcl_acf"] = {"lags": list(range(1, WVCL_MAX_LAG + 1)),
                           "mean": [float(x) for x in wa],
                           "n_tracks": len(wv_acfs)}
    res["wvcl_psd"] = None
    if wv_psd_n:
        wm = wv_psd_sum / wv_psd_n
        srch = (wv_grid >= WVCL_SEARCH[0]) & (wv_grid <= WVCL_SEARCH[1])
        res["wvcl_psd"] = {
            "grid_hz": [float(x) for x in wv_grid],
            "mean": [float(x) for x in wm],
            "n_tracks": wv_psd_n,
            "peak_hz_of_mean": float(wv_grid[srch][np.argmax(wm[srch])]),
            "per_track_peak_hz_median": float(np.median(wv_peaks)) if wv_peaks else None,
        }
    res["threshold_crossings"] = {}
    for thr in THRESHOLDS:
        iv = np.array(cross_intervals[thr])
        rt = np.array(cross_rates[thr])
        res["threshold_crossings"][f"{thr:g}um_s"] = {
            "rate_per_s_mean": float(rt.mean()) if len(rt) else None,
            "rate_per_s_median": float(np.median(rt)) if len(rt) else None,
            "interval_s_quantiles": ({str(p): float(np.percentile(iv, p))
                                      for p in (5, 25, 50, 75, 95)}
                                     if len(iv) else None),
            "n_intervals": int(len(iv)),
            "interval_cv": (float(iv.std() / iv.mean())
                            if len(iv) > 1 and iv.mean() > 0 else None),
        }
    return res


def cluster_ci(per_video: dict, rng: np.random.Generator) -> list[float]:
    vids = list(per_video)
    means = []
    for _ in range(N_BOOT):
        pick = rng.choice(vids, size=len(vids), replace=True)
        w = [per_video[v]["rate_mean"] for v in pick]
        means.append(np.mean(w))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    rng = np.random.default_rng(0)
    print("GT cohort ...", flush=True)
    gt = analyse(GT_DIR, "ground truth")
    print("null cohort ...", flush=True)
    nl = analyse(NULL_DIR, "continuum null (per-track OU)")

    gt["switch_rate_per_s"]["ci95_cluster"] = cluster_ci(gt["per_video"], rng)
    nl["switch_rate_per_s"]["ci95_cluster"] = cluster_ci(nl["per_video"], rng)

    acf_gap = None
    if gt["acf"]["mean"] and nl["acf"]["mean"]:
        g, n = np.array(gt["acf"]["mean"]), np.array(nl["acf"]["mean"])
        acf_gap = {f"lag{k}": {"gt": float(g[k - 1]), "null": float(n[k - 1])}
                   for k in (1, 2, 3, 5, 10, 25, 50)}

    wvcl_acf_gap = None
    if gt.get("wvcl_acf") and nl.get("wvcl_acf"):
        g, n = np.array(gt["wvcl_acf"]["mean"]), np.array(nl["wvcl_acf"]["mean"])
        wvcl_acf_gap = {f"lag{k}": {"gt": float(g[k - 1]), "null": float(n[k - 1])}
                        for k in (5, 12, 25, 50, 100)}

    verdict = {
        "switch_rate_ratio_null_over_gt":
            nl["switch_rate_per_s"]["mean"] / gt["switch_rate_per_s"]["mean"],
        "occupancy_gt": gt["occupancy"], "occupancy_null": nl["occupancy"],
        "acf_key_lags": acf_gap,
        "beat_band_power_gt": gt["psd"]["band_power_5_20hz"] if gt["psd"] else None,
        "beat_band_power_null": nl["psd"]["band_power_5_20hz"] if nl["psd"] else None,
        "beat_peak_gt_hz": gt["psd"]["peak_hz_of_mean"] if gt["psd"] else None,
        "beat_peak_null_hz": nl["psd"]["peak_hz_of_mean"] if nl["psd"] else None,
        "wvcl_acf_key_lags": wvcl_acf_gap,
        "threshold_crossings_gt": gt["threshold_crossings"],
        "threshold_crossings_null": nl["threshold_crossings"],
        "note": ("calibration surfaces that matter downstream: windowed-VCL "
                 "ACF/PSD and threshold-crossing intervals (classifier input), "
                 "plus switch rate. Raw-velocity beat-band peaks are NOT "
                 "evidence for/against the null (prereg 7f30aff)."),
    }

    out = {"gt": gt, "null": nl, "verdict": verdict}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    print(f"\nswitch rate /s : GT {gt['switch_rate_per_s']['mean']:.3f} "
          f"{gt['switch_rate_per_s']['ci95_cluster']}  "
          f"null {nl['switch_rate_per_s']['mean']:.3f} "
          f"{nl['switch_rate_per_s']['ci95_cluster']}  "
          f"ratio {verdict['switch_rate_ratio_null_over_gt']:.2f}")
    print(f"occupancy      : GT {gt['occupancy']}  null {nl['occupancy']}")
    if acf_gap:
        for k, d in acf_gap.items():
            print(f"ACF {k:>6}: GT {d['gt']:+.3f}  null {d['null']:+.3f}")
    if gt["psd"] and nl["psd"]:
        print(f"beat band 5-20 Hz power: GT {verdict['beat_band_power_gt']:.3f} "
              f"(peak {verdict['beat_peak_gt_hz']:.1f} Hz, per-track median "
              f"{gt['psd']['per_track_peak_hz_median']:.1f} Hz)  "
              f"null {verdict['beat_band_power_null']:.3f} "
              f"(peak {verdict['beat_peak_null_hz']:.1f} Hz)")
    if wvcl_acf_gap:
        for k, d in wvcl_acf_gap.items():
            print(f"wVCL ACF {k:>7}: GT {d['gt']:+.3f}  null {d['null']:+.3f}")
    for thr in THRESHOLDS:
        kk = f"{thr:g}um_s"
        g, n = gt["threshold_crossings"][kk], nl["threshold_crossings"][kk]
        print(f"crossings @ {thr:g} um/s: GT rate {g['rate_per_s_mean']:.3f}/s "
              f"(median iv {g['interval_s_quantiles']['50'] if g['interval_s_quantiles'] else None} s, "
              f"CV {g['interval_cv']}) | null rate {n['rate_per_s_mean']:.3f}/s "
              f"(median iv {n['interval_s_quantiles']['50'] if n['interval_s_quantiles'] else None} s, "
              f"CV {n['interval_cv']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
