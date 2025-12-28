# src/store/label_store.py
import os
from datetime import datetime

import pandas as pd

LABELS_DIR = os.environ.get("LABELS_DIR", "data/labels")
os.makedirs(LABELS_DIR, exist_ok=True)


def append_labels_to_store(records: list[dict], labels_dir: str = LABELS_DIR) -> str:
    """Append accepted label records to a timestamped CSV file in labels_dir."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = os.path.join(labels_dir, f"labels_{ts}.parquet")

    # Convert records to DataFrame
    df = pd.DataFrame(records)

    # Append to Parquet (if the file already exists, append)
    if os.path.exists(fname):
        df.to_parquet(fname, mode="append", index=False)
    else:
        df.to_parquet(fname, mode="w", index=False)

    # Optionally: version with DVC or timestamp (simple versioning in this case)
    versioned_fname = os.path.join(labels_dir, f"labels_{ts}_v1.parquet")
    df.to_parquet(versioned_fname, index=False)

    # Create metadata file (JSON) for tracking
    meta = {"created_at": ts, "n_records": len(records), "filename": fname}
    with open(fname + ".meta.json", "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file)

    return fname
