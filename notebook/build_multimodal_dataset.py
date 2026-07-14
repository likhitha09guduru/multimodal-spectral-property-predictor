"""
Downloads the REAL, published IR-NMR multimodal spectroscopy dataset and
builds a fused tabular feature file for the ML pipeline.

Dataset (real, not synthetic):
  "IR-NMR Multimodal Computational Spectra Dataset for 177K Patent-Extracted
  Organic Molecules" - Zipoli, Alberts, Laino (IBM Research Europe - Zurich)
  Zenodo record: https://zenodo.org/records/16417648
  License: Community Data License Agreement - Permissive 2.0 (CDLA-Permissive-2.0)
  Paper: https://chemrxiv.org/engage/chemrxiv/article-details/684f1f86c1cb1ecda0230ceb

  Spectra were computed with a hybrid MD + DFT + ML pipeline (not hand-made
  synthetic numbers): NMR shifts come from CPMD molecular-dynamics-averaged
  shielding calculations, IR spectra come from the Fourier transform of
  dipole-moment autocorrelation functions sampled along MD trajectories.

Contents downloaded:
  - NMR_data.parquet            (~23 MB,  1,255 molecules, 1H + 13C NMR shifts)
  - IR_data_chunkNNN_of_009.parquet (~900 MB each, 20,000 molecules/chunk,
    177,461 molecules total across 9 chunks)

Because the full IR data is ~8.1 GB, by default this script only downloads
IR chunk 1 (~900 MB) and keeps whichever molecules in that chunk also have
NMR data (partial overlap - enough for a working demo). Pass --chunks with
more chunk numbers (1-9) for broader molecule coverage; note this pipeline
does not need to be re-run once notebook/data/multimodal_spectra_dataset.csv
exists.

This script needs internet access and the extra packages listed in
requirements.txt (pyarrow, rdkit, requests) - it is meant to be run on your
own machine, not inside a network-restricted sandbox.

Usage:
    python notebook/build_multimodal_dataset.py --chunks 1
    python notebook/build_multimodal_dataset.py --chunks 1 2 3   # broader coverage
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests

ZENODO_BASE = "https://zenodo.org/records/16417648/files"
NMR_URL = f"{ZENODO_BASE}/NMR_data.parquet?download=1"
IR_CHUNK_URL_TEMPLATE = f"{ZENODO_BASE}/IR_data_chunk{{:03d}}_of_009.parquet?download=1"

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "multimodal_spectra_dataset.csv")

# IR functional-group band windows (cm^-1) used to turn a full IR spectrum
# into fixed-size, chemically interpretable features (standard practice in
# IR spectral analysis - these are textbook functional-group regions).
IR_BANDS = {
    "ir_band_ohnh_stretch_3200_3550": (3200.0, 3550.0),   # O-H / N-H stretch
    "ir_band_ch_stretch_2850_3000": (2850.0, 3000.0),     # C-H stretch
    "ir_band_carbonyl_1650_1750": (1650.0, 1750.0),       # C=O stretch
    "ir_band_aromatic_1450_1600": (1450.0, 1600.0),       # aromatic C=C
    "ir_band_fingerprint_500_1500": (500.0, 1500.0),      # fingerprint region
}


def download_file(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        print(f"Already downloaded: {dest_path}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} -> {dest_path}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:8.1f} / {total / 1e6:8.1f} MB", end="")
        print()


def summarize_shift_array(values) -> dict:
    """Turns a variable-length list of NMR chemical shifts into fixed-size
    summary features (mean/std/min/max/count) - a standard way to featurize
    a spectrum for tabular ML models."""
    arr = np.asarray(values, dtype=float) if values is not None else np.array([])
    arr = arr[~np.isnan(arr)] if arr.size else arr
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


def extract_nmr_row_features(row) -> dict:
    """Pulls the averaged 1H and 13C shift arrays out of a NMR_data.parquet
    row. The real schema nests shift arrays either directly under
    'averaged_frames' or under the first entry of 'frames'; this handles
    both, which is necessary because Parquet/JSON nesting depth in the
    published file isn't guaranteed to be flat."""
    h_shifts, c_shifts = None, None

    averaged = row.get("averaged_frames")
    if isinstance(averaged, dict):
        h_shifts = averaged.get("h_nmr_peaks_ave")
        c_shifts = averaged.get("c_nmr_peaks_ave")

    if h_shifts is None or c_shifts is None:
        frames = row.get("frames")
        if isinstance(frames, dict):
            h_shifts = h_shifts if h_shifts is not None else frames.get("h_nmr_peaks_ave")
            c_shifts = c_shifts if c_shifts is not None else frames.get("c_nmr_peaks_ave")
        elif isinstance(frames, (list, tuple)) and len(frames) > 0 and isinstance(frames[0], dict):
            h_shifts = h_shifts if h_shifts is not None else frames[0].get("h_nmr_peaks_ave")
            c_shifts = c_shifts if c_shifts is not None else frames[0].get("c_nmr_peaks_ave")

    h_summary = summarize_shift_array(h_shifts)
    c_summary = summarize_shift_array(c_shifts)

    return {
        "h_nmr_shift_mean": h_summary["mean"],
        "h_nmr_shift_std": h_summary["std"],
        "h_nmr_shift_max": h_summary["max"],
        "h_nmr_peak_count": h_summary["count"],
        "c_nmr_shift_mean": c_summary["mean"],
        "c_nmr_shift_std": c_summary["std"],
        "c_nmr_shift_max": c_summary["max"],
        "c_nmr_peak_count": c_summary["count"],
    }


def extract_ir_row_features(row) -> dict:
    """Integrates the real IR spectrum intensity within standard
    functional-group wavenumber windows to produce fixed-size band
    features from the variable-length spectrum."""
    freq = np.asarray(row["Frequency(cm^-1)"], dtype=float)
    spec = np.asarray(row["ir_spectra"], dtype=float)

    integrate = getattr(np, "trapezoid", None) or np.trapz

    features = {}
    for name, (lo, hi) in IR_BANDS.items():
        mask = (freq >= lo) & (freq <= hi)
        features[name] = float(integrate(spec[mask], freq[mask])) if mask.sum() > 1 else 0.0
    return features


def composition_flags_from_smiles(smiles: str) -> dict:
    """Derives real categorical descriptors from the molecule's actual
    structure (SMILES), used as the third ('descriptor') modality."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("invalid SMILES")
        symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        mw = None
        from rdkit.Chem import Descriptors
        mw = Descriptors.MolWt(mol)
        return {
            "contains_nitrogen": "yes" if "N" in symbols else "no",
            "contains_oxygen": "yes" if "O" in symbols else "no",
            "contains_halogen": "yes" if symbols & {"F", "Cl", "Br", "I"} else "no",
            "contains_sulfur": "yes" if "S" in symbols else "no",
            "molecular_weight": mw,
        }
    except Exception:
        return {
            "contains_nitrogen": "no",
            "contains_oxygen": "no",
            "contains_halogen": "no",
            "contains_sulfur": "no",
            "molecular_weight": None,
        }


def build_dataset(chunks):
    os.makedirs(RAW_DIR, exist_ok=True)

    nmr_path = os.path.join(RAW_DIR, "NMR_data.parquet")
    download_file(NMR_URL, nmr_path)
    nmr_df = pd.read_parquet(nmr_path)
    print(f"Loaded {len(nmr_df)} molecules with real NMR data")

    nmr_records = []
    for _, row in nmr_df.iterrows():
        rec = {"id": row["id"], "smiles": row["smiles"]}
        rec.update(extract_nmr_row_features(row))
        nmr_records.append(rec)
    nmr_features_df = pd.DataFrame(nmr_records)

    ir_records = []
    for chunk_num in chunks:
        chunk_path = os.path.join(RAW_DIR, f"IR_data_chunk{chunk_num:03d}_of_009.parquet")
        download_file(IR_CHUNK_URL_TEMPLATE.format(chunk_num), chunk_path)
        ir_df = pd.read_parquet(chunk_path)
        print(f"Loaded {len(ir_df)} molecules from IR chunk {chunk_num}")

        matched = ir_df[ir_df["id"].isin(set(nmr_features_df["id"]))]
        print(f"  {len(matched)} of these also have real NMR data (fusable pairs)")
        for _, row in matched.iterrows():
            rec = {"id": row["id"]}
            rec.update(extract_ir_row_features(row))
            ir_records.append(rec)

    if not ir_records:
        print(
            "No overlapping molecules found between the downloaded IR chunk(s) "
            "and the NMR set. Try passing more --chunks (1-9) for broader coverage."
        )
        sys.exit(1)

    ir_features_df = pd.DataFrame(ir_records)

    fused = nmr_features_df.merge(ir_features_df, on="id", how="inner")
    print(f"Fused IR + NMR multimodal dataset: {len(fused)} molecules")

    descriptor_rows = fused["smiles"].apply(composition_flags_from_smiles).apply(pd.Series)
    fused = pd.concat([fused, descriptor_rows], axis=1)

    fused = fused.dropna(subset=["molecular_weight"])
    fused = fused.drop(columns=["id", "smiles"])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fused.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(fused)} rows to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks", type=int, nargs="+", default=[1],
        help="Which IR chunk numbers (1-9) to download and fuse with the NMR set. "
             "Default: [1] (~900 MB). Add more for broader molecule coverage.",
    )
    args = parser.parse_args()
    build_dataset(args.chunks)
