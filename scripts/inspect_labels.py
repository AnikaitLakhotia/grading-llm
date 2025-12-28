# scripts/inspect_labels.py
import json
import os

import pandas as pd

LABELS_DIR = os.environ.get("LABELS_DIR", "data/labels")


def list_labels():
    """List all labels files in the LABELS_DIR."""
    files = [f for f in os.listdir(LABELS_DIR) if f.endswith(".parquet")]
    return sorted(files, reverse=True)


def inspect_label_file(label_file: str):
    """Inspect the contents of a specific label file."""
    label_path = os.path.join(LABELS_DIR, label_file)

    if os.path.exists(label_path):
        # Read the Parquet file
        df = pd.read_parquet(label_path)
        print(f"Showing first 10 rows of {label_file}:")
        print(df.head(10))

        # Optionally: Read associated metadata
        meta_file = label_path + ".meta.json"
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                metadata = json.load(f)
            print("\nMetadata:")
            print(json.dumps(metadata, indent=2))
    else:
        print(f"File {label_file} does not exist.")


def main():
    # List files
    files = list_labels()
    print("Label files found:", files)

    # Inspect the latest file
    if files:
        inspect_label_file(files[0])


if __name__ == "__main__":
    main()
