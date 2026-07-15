"""
Builds notebook/data/multimodal_spectra_dataset.csv, the fused dataset that
src/components/data_ingestion.py reads.

Source: the REAL, published "IR-NMR Multimodal Computational Spectra Dataset
for 177K Patent-Extracted Organic Molecules" (Zipoli, Alberts, Laino; IBM
Research; Zenodo record 16417648; CDLA-Permissive-2.0).
https://doi.org/10.5281/zenodo.16417648

The Zenodo record ships two kinds of files:
  - NMR_data.parquet                      -> 1,255 molecules (id, smiles, ...)
  - IR_data_chunk001..009_of_009.parquet   -> 177,461 molecules (id, smiles,
                                               Frequency(cm^-1), ir_spectra)

Only molecules present in BOTH files (i.e. the 1,255 NMR molecules) are kept,
since the GNN+CNN model needs a SMILES string (for the graph) and an IR
spectrum (for the CNN) for every row. This script:
  1. Downloads NMR_data.parquet and scans the 9 IR chunk files, keeping only
     rows whose `id` also appears in the NMR data.
  2. Bins/interpolates each molecule's raw IR spectrum onto a fixed-length
     grid (IR_BINS points) so every row has a same-shape CNN input.
  3. Computes molecular_weight for each molecule from its SMILES with RDKit
     -- this is the regression target.
  4. Writes id, smiles, ir_spectrum_binned (JSON list), molecular_weight to
     notebook/data/multimodal_spectra_dataset.csv.

This is a large, slow download (~8GB across all 9 IR chunks, since a
molecule's `id` can land in any chunk). Use --demo for a small synthetic
stand-in dataset (same schema, RDKit-computed target, made-up spectra) so you
can exercise the rest of the pipeline without waiting on the full download.

Usage:
    python notebook/build_multimodal_dataset.py            # real data (slow)
    python notebook/build_multimodal_dataset.py --demo      # fast synthetic
    python notebook/build_multimodal_dataset.py --demo --n 500
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

from src.exception import CustomException
from src.logger import logging
from src.config import IR_BINS  # fixed-length CNN input grid, shared with app/pipeline

ZENODO_RECORD = "https://zenodo.org/records/16417648/files"
NMR_FILE = "NMR_data.parquet"
IR_CHUNK_TEMPLATE = "IR_data_chunk{:03d}_of_009.parquet"
N_IR_CHUNKS = 9

OUT_PATH = os.path.join("notebook", "data", "multimodal_spectra_dataset.csv")


def _download_parquet(filename: str) -> pd.DataFrame:
    url = f"{ZENODO_RECORD}/{filename}?download=1"
    logging.info(f"Downloading {url}")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))


def _bin_spectrum(freq: np.ndarray, intensity: np.ndarray, n_bins: int = IR_BINS) -> np.ndarray:
    """Resample a raw IR spectrum onto a fixed-length grid via interpolation."""
    freq = np.asarray(freq, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    order = np.argsort(freq)
    freq, intensity = freq[order], intensity[order]
    grid = np.linspace(freq.min(), freq.max(), n_bins)
    binned = np.interp(grid, freq, intensity)
    return binned


def _molecular_weight(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)


def build_real_dataset() -> pd.DataFrame:
    try:
        nmr_df = _download_parquet(NMR_FILE)
        wanted_ids = set(nmr_df["id"].unique())
        logging.info(f"NMR dataset has {len(wanted_ids)} molecules to match against IR chunks")

        rows = []
        for chunk_idx in range(1, N_IR_CHUNKS + 1):
            chunk_name = IR_CHUNK_TEMPLATE.format(chunk_idx)
            ir_df = _download_parquet(chunk_name)
            matched = ir_df[ir_df["id"].isin(wanted_ids)]
            logging.info(f"{chunk_name}: matched {len(matched)} molecules")

            for _, row in matched.iterrows():
                mw = _molecular_weight(row["smiles"])
                if mw is None:
                    continue
                binned = _bin_spectrum(row["Frequency(cm^-1)"], row["ir_spectra"])
                rows.append({
                    "id": row["id"],
                    "smiles": row["smiles"],
                    "ir_spectrum_binned": json.dumps(binned.tolist()),
                    "molecular_weight": mw,
                })

        df = pd.DataFrame(rows).drop_duplicates(subset="id").reset_index(drop=True)
        logging.info(f"Fused real IR+NMR dataset built with {len(df)} molecules")
        return df
    except Exception as e:
        raise CustomException(e, sys)


def build_demo_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Small, fast, synthetic stand-in with the SAME schema as the real dataset,
    for exercising the pipeline end-to-end without an 8GB download. The
    SMILES are real (drawn from a fixed list of common small organic
    molecules) and molecular_weight is computed for real via RDKit; only the
    IR spectra are randomly generated, so they are NOT physically meaningful.
    """
    from rdkit import Chem

    rng = np.random.default_rng(seed)
    base_smiles = [
        "CCO", "CC(=O)O", "c1ccccc1", "CCN", "CC(C)O", "CCOCC", "CC(=O)OC",
        "c1ccc(O)cc1", "CCCCO", "CC(N)C(=O)O", "c1ccncc1", "CC(C)=O",
        "CCOC(=O)C", "CCCl", "c1ccc(N)cc1", "CCCCCC", "OCC(O)CO",
        "CC(C)(C)O", "c1ccc(Cl)cc1", "CCCCN",
    ]
    smiles_pool = [s for s in base_smiles if Chem.MolFromSmiles(s) is not None]

    rows = []
    for i in range(n):
        smi = smiles_pool[i % len(smiles_pool)]
        mw = _molecular_weight(smi)
        if mw is None:
            continue
        freq = np.linspace(400, 4000, 512)
        # a few random Lorentzian-ish peaks -- placeholder only, not real physics
        intensity = np.zeros_like(freq)
        for _ in range(rng.integers(3, 7)):
            center = rng.uniform(500, 3800)
            width = rng.uniform(20, 80)
            height = rng.uniform(0.2, 1.0)
            intensity += height / (1 + ((freq - center) / width) ** 2)
        intensity += rng.normal(0, 0.02, size=freq.shape)
        binned = _bin_spectrum(freq, intensity)

        rows.append({
            "id": f"demo_{i}",
            "smiles": smi,
            "ir_spectrum_binned": json.dumps(binned.tolist()),
            "molecular_weight": mw,
        })

    df = pd.DataFrame(rows)
    logging.info(f"Synthetic demo dataset built with {len(df)} rows (SYNTHETIC SPECTRA)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                         help="Build a small synthetic dataset instead of downloading the real 8GB dataset")
    parser.add_argument("--n", type=int, default=500, help="Number of rows for --demo mode")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    if args.demo:
        dataset = build_demo_dataset(n=args.n)
    else:
        dataset = build_real_dataset()

    dataset.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(dataset)} rows to {OUT_PATH}")
