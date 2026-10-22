#!/usr/bin/env python3
"""pbr_correlation.py - how related the PBR maps of a set of materials are.

Scans a directory of extracted AmbientCG material folders (NAME_1K-PNG_*.png) and
measures, per material, how much structure the maps share. Two levels of answer:

  * pairwise : a TEXTURE-LEVEL score per unordered pair of maps -- the mean (and
               max) of the absolute Pearson correlation and of the normalised
               mutual information over all channel pairs between the two textures;
  * triples  : a score per unordered TRIPLE of maps -- the mean of the three
               pairwise |r| (the "clique correlation") and the EFFECTIVE RANK of
               the triple's channel correlation matrix (eigenvalue entropy). A low
               effective rank means the three textures live in a low-dimensional
               subspace, which is exactly what NNTC's few shared latent channels
               exploit. Triples whose mean pairwise |r| is below --min-r are
               PRUNED (obviously decorrelated; no effective rank is computed).

Colour handling, so the numbers mean what a PBR pipeline means:

  * albedo (_Color)          : sRGB -> linear before anything is measured;
  * normal (_NormalGL)       : decoded to [-1, 1] (linear, no sRGB);
  * roughness / metalness /  : linear, single-channel, [0, 1];
    ambient occlusion /
    displacement (_Displacement = height)

The map is identified by its AmbientCG suffix. _NormalDX is skipped (same map,
Y-flipped; keeping both would double-weight the normal). The material preview
NAME.png and the .blend/.usdc/.mtlx/.tres sidecars are ignored.

Caveat: Pearson and MI are POINTWISE statistics. The normal map encodes the
GRADIENT of the height (normal ~ d(height)), so normal<->height reads near zero
here even though the two are tightly, if non-locally, coupled. Treat triples
containing `normal` as a lower bound, and validate them with the encoder.

Outputs (into --out):

  * materials/NAME_corr.csv, materials/NAME_mi.csv   pairwise matrices, per material;
  * aggregate_corr.csv, aggregate_mi.csv             dataset mean per pair;
  * aggregate_counts.csv                            how many materials had each pair;
  * triples.csv                                     ranked 3-texture combinations;
  * report.json                                     the same numbers as JSON.

A pair / triple is averaged only over the materials that carry ALL its maps.

Usage:
  python pbr_correlation.py PATH/TO/pbr-materials -o correlation_out [--min-r 0.25]
"""

import argparse
import itertools
import json
import os

import numpy as np
from PIL import Image

# canonical order of the texture-level rows/columns
MAPS = [
    ("Color",             "albedo",    "rgb"),
    ("NormalGL",          "normal",    "rgb"),
    ("Roughness",         "roughness", "gray"),
    ("Metalness",         "metalness", "gray"),
    ("AmbientOcclusion",  "ao",        "gray"),
    ("Displacement",      "height",    "gray"),
]
LABELS = [label for _, label, _ in MAPS]
KIND = {label: kind for _, label, kind in MAPS}
N_MAPS = len(MAPS)
MI_BINS = 64


def srgb_to_linear(u8):
    """8-bit sRGB -> linear float in [0, 1] (the IEC piecewise curve)."""
    f = u8.astype(np.float32) / 255.0
    lo = f <= 0.04045
    out = np.empty_like(f)
    out[lo] = f[lo] / 12.92
    out[~lo] = ((f[~lo] + 0.055) / 1.055) ** 2.4
    return out


def load_map(path, kind):
    """Load one map as a float HxWxC array in its working colour space.

    kind is "rgb" or "gray". Albedo is linearised; the normal is decoded to
    [-1, 1]; everything else is a linear [0, 1] plane.
    """
    im = Image.open(path)
    im.load()
    if kind == "rgb":
        a = np.asarray(im.convert("RGB"))
        if "Color" in os.path.basename(path):
            return srgb_to_linear(a)
        return a.astype(np.float32) / 255.0 * 2.0 - 1.0
    # grayscale. 16-bit PNGs (AmbientCG Displacement) must be read from the raw
    # uint16 array: Pillow's convert("L") mangles I;16 (see tools/crop_set.py).
    if im.mode in ("I;16", "I;16L", "I;16B"):
        a = np.asarray(im, dtype=np.float32) / 65535.0
    else:
        a = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return a[:, :, None]


def find_maps(folder):
    """Return {label: path} for every recognised map in `folder`, in MAPS order."""
    found = {}
    for name in os.listdir(folder):
        if not name.lower().endswith(".png"):
            continue
        for suffix, label, _ in MAPS:
            if name.endswith(f"_{suffix}.png"):
                found[label] = os.path.join(folder, name)
                break
    return found


def normalised_mi(x, y, bins=MI_BINS):
    """Normalised mutual information in [0, 1] from a 2-D histogram."""
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    cxy = np.histogram2d(x, y, bins=bins)[0]
    pxy = cxy / cxy.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))
    if hx + hy == 0.0:
        return 0.0
    return float(2.0 * (hx + hy - hxy) / (hx + hy))


def pearson_matrix(chans):
    """KxK Pearson correlation matrix of the channel vectors; diagonal 1."""
    r = np.corrcoef(np.stack(chans, axis=0))
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(r, 1.0)
    return r


def mi_matrix(chans, bins=MI_BINS):
    """KxK normalised-MI matrix of the channel vectors; diagonal 1."""
    k = len(chans)
    m = np.zeros((k, k))
    for i in range(k):
        m[i, i] = 1.0
        for j in range(i + 1, k):
            v = normalised_mi(chans[i], chans[j], bins)
            m[i, j] = m[j, i] = v
    return m


def block_mean_abs(mat, chan_label, labels):
    """Collapse a channel-level matrix to texture-level: mean |mat| per map block."""
    n = len(labels)
    out = np.full((n, n), np.nan)
    idx = {l: [k for k, cl in enumerate(chan_label) if cl == l] for l in labels}
    for a in range(n):
        for b in range(n):
            out[a, b] = float(np.mean(np.abs(mat[np.ix_(idx[labels[a]], idx[labels[b]])])))
    return out


def effective_rank(corr):
    """Effective rank (eigenvalue entropy) of a correlation matrix, in [1, K]."""
    ev = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    s = ev.sum()
    if s <= 0.0:
        return 1.0
    p = ev[ev > 0] / s
    return float(np.exp(-np.sum(p * np.log(p))))


def table_text(matrix, labels, fmt="%6.3f"):
    """A fixed-width text table (no pandas); NaN prints as a dot."""
    head = "          " + " ".join(f"{l:>9}" for l in labels)
    rows = [head]
    for i, l in enumerate(labels):
        cells = []
        for j in range(len(labels)):
            v = matrix[i, j]
            cells.append("        ." if (v is None or np.isnan(v)) else fmt % v)
        rows.append(f"{l:>10}" + "".join(cells))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Texture-level Pearson + MI correlation of AmbientCG PBR maps, "
                    "plus ranked 3-texture combinations.")
    ap.add_argument("pbr_dir", nargs="?", default=".",
                    help="directory of extracted material folders (default .)")
    ap.add_argument("-o", "--out", default="correlation_out",
                    help="output directory (default correlation_out)")
    ap.add_argument("--min-r", type=float, default=0.25,
                    help="prune a triple below this mean pairwise |r| (default 0.25)")
    args = ap.parse_args()

    root = args.pbr_dir
    out_abs = os.path.abspath(args.out)
    folders = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.abspath(os.path.join(root, d)) != out_abs
    )
    if not folders:
        ap.error(f"no material folders under {root}")

    mats_dir = os.path.join(args.out, "materials")
    os.makedirs(mats_dir, exist_ok=True)

    corr_sum = np.zeros((N_MAPS, N_MAPS))
    corr_cnt = np.zeros((N_MAPS, N_MAPS), dtype=np.int64)
    mi_sum = np.zeros((N_MAPS, N_MAPS))
    # triple -> [sum_r, sum_erank, n_scored, n_pruned]
    triple_acc = {}
    report = {}

    for folder in folders:
        paths = find_maps(os.path.join(root, folder))
        labels = [l for l in LABELS if l in paths]
        if not labels:
            print(f"skip {folder}: no recognised maps")
            continue

        loaded = {}
        size = None
        for l in labels:
            a = load_map(paths[l], KIND[l])
            hw = a.shape[:2]
            if size is None:
                size = hw
            elif size != hw:
                ap.error(f"{folder}: {l} is {hw[1]}x{hw[0]}, expected {size[1]}x{size[0]}")
            loaded[l] = a

        # flatten every channel, in map order (labels is canonical, so is chan_label)
        chans, chan_label = [], []
        for l in labels:
            for c in range(loaded[l].shape[2]):
                chans.append(loaded[l][:, :, c].ravel().astype(np.float64))
                chan_label.append(l)

        r_chan = pearson_matrix(chans)
        m_chan = mi_matrix(chans)
        n = len(labels)
        corr_loc = block_mean_abs(r_chan, chan_label, labels)
        mi_loc = block_mean_abs(m_chan, chan_label, labels)

        # pairwise: accumulate into the full N_MAPS x N_MAPS aggregates
        idx = [LABELS.index(l) for l in labels]
        corr_full = np.full((N_MAPS, N_MAPS), np.nan)
        mi_full = np.full((N_MAPS, N_MAPS), np.nan)
        for p in range(n):
            for q in range(n):
                corr_full[idx[p], idx[q]] = corr_loc[p, q]
                mi_full[idx[p], idx[q]] = mi_loc[p, q]
        for p in range(n):
            for q in range(p, n):
                i, j = idx[p], idx[q]
                corr_sum[i, j] += corr_loc[p, q]
                corr_sum[j, i] += corr_loc[p, q]
                mi_sum[i, j] += mi_loc[p, q]
                mi_sum[j, i] += mi_loc[p, q]
                corr_cnt[i, j] += 1
                corr_cnt[j, i] += 1

        # triples: enumerate, prune the obviously-low ones
        chan_idx = {l: [k for k, cl in enumerate(chan_label) if cl == l] for l in labels}
        for a, b, c in itertools.combinations(range(n), 3):
            r_mean = (corr_loc[a, b] + corr_loc[b, c] + corr_loc[a, c]) / 3.0
            key = tuple(sorted((labels[a], labels[b], labels[c]), key=LABELS.index))
            acc = triple_acc.setdefault(key, [0.0, 0.0, 0, 0])
            if r_mean < args.min_r:
                acc[3] += 1
                continue
            sel = sorted(chan_idx[labels[a]] + chan_idx[labels[b]] + chan_idx[labels[c]])
            acc[0] += r_mean
            acc[1] += effective_rank(r_chan[np.ix_(sel, sel)])
            acc[2] += 1

        print(f"=== {folder}  ({', '.join(labels)}) ===")
        print(table_text(corr_full, LABELS))
        print()

        stem = folder.replace(os.sep, "_")
        np.savetxt(os.path.join(mats_dir, f"{stem}_corr.csv"), corr_full,
                   delimiter=",", header=",".join(LABELS), comments="", fmt="%.4f")
        np.savetxt(os.path.join(mats_dir, f"{stem}_mi.csv"), mi_full,
                   delimiter=",", header=",".join(LABELS), comments="", fmt="%.4f")
        report[folder] = {"maps": labels, "corr": corr_full.tolist(),
                          "mi": mi_full.tolist()}

    # pairwise aggregate
    with np.errstate(invalid="ignore", divide="ignore"):
        agg_corr = corr_sum / corr_cnt
        agg_mi = mi_sum / corr_cnt
    np.nan_to_num(agg_corr, copy=False, nan=0.0)
    np.nan_to_num(agg_mi, copy=False, nan=0.0)
    for name, mat in (("aggregate_corr.csv", agg_corr), ("aggregate_mi.csv", agg_mi)):
        np.savetxt(os.path.join(args.out, name), mat, delimiter=",",
                   header=",".join(LABELS), comments="", fmt="%.4f")
    np.savetxt(os.path.join(args.out, "aggregate_counts.csv"), corr_cnt,
               delimiter=",", header=",".join(LABELS), comments="", fmt="%d")

    # triples: rank the scored ones, list the fully-pruned ones
    rows, pruned_all = [], []
    for key, (sr, se, nsc, npr) in triple_acc.items():
        if nsc:
            rows.append((key, nsc, sr / nsc, se / nsc, npr))
        else:
            pruned_all.append((key, npr))
    rows.sort(key=lambda t: -t[2])

    print("=== 3-texture combos, ranked by mean pairwise |r| "
          f"(pruned below --min-r {args.min_r}) ===")
    print(f"{'triple':>34} {'n':>3} {'mean|r|':>8} {'erank':>6}")
    for key, nsc, r, er, _ in rows:
        print(f"{' <-> '.join(key):>34} {nsc:3d} {r:8.3f} {er:6.2f}")

    if pruned_all:
        print(f"\n=== pruned (mean pairwise |r| < {args.min_r} everywhere) ===")
        for key, npr in sorted(pruned_all, key=lambda t: LABELS.index(t[0][0])):
            print(f"  {' <-> '.join(key):>34}  (seen {npr} materials, never scored)")

    with open(os.path.join(args.out, "triples.csv"), "w", newline="\n") as f:
        f.write("a,b,c,n,mean_r,mean_erank,n_pruned\n")
        for key, nsc, r, er, npr in rows:
            f.write(f"{key[0]},{key[1]},{key[2]},{nsc},{r:.4f},{er:.4f},{npr}\n")

    report["triples"] = [{"maps": list(k), "n": nsc, "mean_r": r,
                          "mean_erank": er, "n_pruned": npr}
                         for k, nsc, r, er, npr in rows]
    report["pruned_triples"] = [{"maps": list(k), "n_seen": npr}
                                for k, npr in pruned_all]
    report["aggregate_corr"] = agg_corr.tolist()
    report["aggregate_mi"] = agg_mi.tolist()
    report["counts"] = corr_cnt.tolist()
    with open(os.path.join(args.out, "report.json"), "w", newline="\n") as f:
        json.dump(report, f, indent=2)

    print(f"\nwrote {args.out}/")


if __name__ == "__main__":
    main()
