# ui/streamlit_app.py
"""
Streamlit Teacher Review UI for grading-llm.

Features:
- Upload sanitized CSV or use sample data.
- Batch-call grading API (/grade) or use local MockLLMClient when API not provided.
- Show model suggestions (score, confidence, feedback).
- Inline edit, accept per-row, add notes, set final_score.
- Export graded CSV and append accepted rows to data/labels/labels_{timestamp}.csv (versioned).
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Try to import the prompt-runner and mock client if available; otherwise fall back to simple mock
try:
    from src.baseline.llm_client import MockLLMClient  # type: ignore
    from src.baseline.prompt_runner import PromptRunner  # type: ignore

    HAS_PROMPT_RUNNER = True
except Exception:
    HAS_PROMPT_RUNNER = False

import httpx

# Constants
REQUIRED_COLUMNS = ["essay_id", "score", "full_text"]
DEFAULT_API_URL = os.environ.get("GRADING_API_URL", "http://localhost:8000/grade")
DEFAULT_USER_ID = os.environ.get("USER_ID", "grader_local")
LABELS_DIR = os.environ.get("LABELS_DIR", "data/labels")
MAX_UPLOAD_MB = 50


def ensure_labels_dir():
    os.makedirs(LABELS_DIR, exist_ok=True)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read uploaded file (Streamlit UploadedFile or path string) into DataFrame."""
    if isinstance(uploaded_file, str):
        return pd.read_csv(uploaded_file)
    else:
        # uploaded_file is a BytesIO-like
        return pd.read_csv(uploaded_file)


def validate_input_df(df: pd.DataFrame) -> List[str]:
    errors = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    return errors


def call_grade_api_batch(
    api_url: str, rows: List[Dict], timeout: float = 20.0
) -> List[Dict]:
    """Call grading API for a list of rows; returns list of model outputs."""
    outputs = []
    client = httpx.Client(timeout=timeout)
    for r in rows:
        payload = {
            "essay_id": r.get("essay_id"),
            "full_text": r.get("full_text"),
            "assignment": r.get("assignment"),
            "prompt_name": r.get("prompt_name"),
            "grade_level": r.get("grade_level"),
        }
        try:
            resp = client.post(api_url, json=payload)
            if resp.status_code == 200:
                outputs.append(resp.json())
            else:
                outputs.append(
                    {
                        "score": None,
                        "feedback": f"API error {resp.status_code}",
                        "evidence": "",
                        "confidence": 0.0,
                    }
                )
        except Exception as exc:
            outputs.append(
                {
                    "score": None,
                    "feedback": f"Request failed: {exc}",
                    "evidence": "",
                    "confidence": 0.0,
                }
            )
    client.close()
    return outputs


def run_local_mock_runner(essays: List[str]) -> List[Dict]:
    """Use MockLLMClient + PromptRunner locally to produce deterministic outputs."""
    if not HAS_PROMPT_RUNNER:
        # fallback trivial deterministic rule: map length -> score
        outs = []
        for t in essays:
            l = len(str(t))
            score = max(1, min(6, (l // 100) + 1))
            outs.append(
                {
                    "score": int(score),
                    "feedback": "Local fallback heuristic",
                    "evidence": "",
                    "confidence": 0.5,
                }
            )
        return outs
    client = MockLLMClient()
    runner = PromptRunner(llm_client=client)
    outs = []
    for t in essays:
        out = runner.run_single(t)
        outs.append(out)
    return outs


def append_labels_to_store(records: List[Dict], labels_dir: str = LABELS_DIR) -> str:
    """Append accepted label records to a timestamped CSV file in labels_dir."""
    ensure_labels_dir()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = os.path.join(labels_dir, f"labels_{ts}.csv")
    df = pd.DataFrame(records)
    df.to_csv(fname, index=False)
    # also write a small JSON metadata file
    meta = {"created_at": ts, "n_records": len(records)}
    with open(fname + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return fname


def build_initial_state(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure necessary columns exist and create UI-state columns."""
    df2 = df.copy()
    # normalize column names to simple ones
    if "essay_id" not in df2.columns and "id" in df2.columns:
        df2 = df2.rename(columns={"id": "essay_id"})
    # create UI columns
    if "model_score" not in df2.columns:
        df2["model_score"] = None
    if "model_confidence" not in df2.columns:
        df2["model_confidence"] = None
    if "model_feedback" not in df2.columns:
        df2["model_feedback"] = ""
    if "final_score" not in df2.columns:
        df2["final_score"] = (
            df2["score"].astype("Int64").where(pd.notnull(df2["score"]), None)
        )
    if "accepted" not in df2.columns:
        df2["accepted"] = False
    if "grader_notes" not in df2.columns:
        df2["grader_notes"] = ""
    return df2


def main():
    st.set_page_config(page_title="Grading-LLM - Teacher Review", layout="wide")
    st.title("Grading-LLM - Teacher Review")

    # Sidebar controls
    st.sidebar.header("Settings")
    api_url = st.sidebar.text_input(
        "Grading API URL (leave blank to use local Mock)",
        value=os.environ.get("GRADING_API_URL", ""),
    )
    user_id = st.sidebar.text_input(
        "Your grader id", value=os.environ.get("USER_ID", DEFAULT_USER_ID)
    )
    max_rows = st.sidebar.number_input(
        "Max rows to display", min_value=10, max_value=1000, value=200, step=10
    )

    st.sidebar.markdown("**Actions**")
    uploaded_file = st.sidebar.file_uploader("Upload essays CSV", type=["csv"])
    use_sample = st.sidebar.button("Use sample data (safe)")

    # show existing label-store files (quick view)
    if st.sidebar.checkbox("Show recent label files"):
        ensure_labels_dir()
        files = sorted(
            [f for f in os.listdir(LABELS_DIR) if f.endswith(".csv")], reverse=True
        )[:10]
        st.sidebar.write(files)

    # load dataframe
    df = None
    if uploaded_file is not None:
        # check size
        uploaded_file.seek(0, io.SEEK_END)
        size_mb = uploaded_file.tell() / (1024 * 1024)
        uploaded_file.seek(0)
        if size_mb > MAX_UPLOAD_MB:
            st.sidebar.error(
                f"Uploaded file too large ({size_mb:.1f} MB). Max {MAX_UPLOAD_MB} MB."
            )
            return
        try:
            df = read_uploaded_csv(uploaded_file)
        except Exception as exc:
            st.sidebar.error(f"Failed to read CSV: {exc}")
            return
    elif use_sample:
        sample_path = "data/sample_sanitized.csv"
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
        else:
            st.sidebar.error(
                "Sample data not found. Run synthetic data generator first."
            )
            return
    else:
        st.info(
            "Upload a sanitized CSV (columns: essay_id, score, full_text, ...), or click 'Use sample data'."
        )
        return

    # Validate df
    errors = validate_input_df(df)
    if errors:
        st.error("Input CSV validation failed:")
        for e in errors:
            st.write(" - " + e)
        return

    df_state = build_initial_state(df)
    st.success(f"Loaded {len(df_state)} essays.")

    # Show quick stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Essays", len(df_state))
    col2.metric(
        "Unique prompts",
        df_state["prompt_name"].nunique() if "prompt_name" in df_state.columns else 0,
    )
    col3.metric(
        "Grade levels",
        df_state["grade_level"].nunique() if "grade_level" in df_state.columns else 0,
    )

    # Select subset to display
    st.subheader("Grading batch")
    display_df = df_state.copy().reset_index(drop=True)
    if len(display_df) > max_rows:
        display_df = display_df.sample(n=max_rows, random_state=42).reset_index(
            drop=True
        )
        st.caption(
            f"Showing a sampled subset of {max_rows} rows (dataset has {len(df_state)} rows)."
        )

    # Run model suggestions
    if st.button("Run model suggestions"):
        with st.spinner("Fetching suggestions..."):
            essays = display_df["full_text"].astype(str).tolist()
            # choose API vs local mock
            if api_url:
                outputs = call_grade_api_batch(
                    api_url, display_df.to_dict(orient="records")
                )
            else:
                outputs = run_local_mock_runner(essays)
            # attach to df_state (by essay_id)
            for idx, row in display_df.iterrows():
                out = outputs[idx]
                eid = row["essay_id"]
                mask = df_state["essay_id"] == eid
                if out is None:
                    continue
                df_state.loc[mask, "model_score"] = (
                    int(out.get("score")) if out.get("score") is not None else None
                )
                df_state.loc[mask, "model_confidence"] = float(
                    out.get("confidence", 0.0)
                )
                df_state.loc[mask, "model_feedback"] = str(out.get("feedback", ""))
        st.success("Model suggestions attached.")

    # Render editable table row-by-row
    st.write(
        "Edit suggestions, accept/override, and add notes. Click 'Save accepted' to persist accepted rows."
    )
    edited_rows: List[Dict] = []
    to_save_records: List[Dict] = []

    # Use an expander per essay for clarity
    for i, row in display_df.iterrows():
        eid = row["essay_id"]
        with st.expander(f"Essay {eid} - current score: {row.get('score', '')}"):
            st.write(
                row.get("full_text", "")[:500]
                + ("..." if len(str(row.get("full_text", ""))) > 500 else "")
            )
            col_left, col_right = st.columns([2, 1])
            with col_left:
                model_score = (
                    int(row["model_score"])
                    if pd.notna(row.get("model_score"))
                    else None
                )
                model_conf = (
                    float(row.get("model_confidence"))
                    if pd.notna(row.get("model_confidence"))
                    else 0.0
                )
                st.markdown(
                    f"**Model suggestion:** {model_score} (confidence {model_conf:.2f})"
                )
                st.markdown(f"**Model feedback:** {row.get('model_feedback', '')}")
                final_score = st.selectbox(
                    f"Final score for {eid}",
                    options=[1, 2, 3, 4, 5, 6],
                    index=(model_score - 1) if model_score else 0,
                    key=f"final_{i}",
                )
                notes = st.text_area(
                    f"Notes (grader)",
                    value=str(row.get("grader_notes", "")),
                    key=f"notes_{i}",
                )
            with col_right:
                accept = st.checkbox(
                    "Accept (save)",
                    value=bool(row.get("accepted", False)),
                    key=f"accept_{i}",
                )
                # quick action buttons
                if st.button("Copy model -> final", key=f"copy_{i}"):
                    if model_score:
                        final_score = model_score
                # Save per-row to memory list - deferred actual file write until Save accepted
            # update df_state in-memory for this row
            mask = df_state["essay_id"] == eid
            df_state.loc[mask, "final_score"] = int(final_score)
            df_state.loc[mask, "accepted"] = bool(accept)
            df_state.loc[mask, "grader_notes"] = str(notes)
            edited_rows.append(
                {
                    "essay_id": eid,
                    "model_score": int(model_score) if model_score else None,
                    "model_confidence": float(model_conf),
                    "final_score": int(final_score),
                    "accepted": bool(accept),
                    "grader_notes": str(notes),
                }
            )
            if accept:
                to_save_records.append(
                    {
                        "essay_id": eid,
                        "assignment": row.get("assignment"),
                        "prompt_name": row.get("prompt_name"),
                        "grade_level": row.get("grade_level"),
                        "original_score": row.get("score"),
                        "model_score": int(model_score) if model_score else None,
                        "model_confidence": float(model_conf),
                        "final_score": int(final_score),
                        "user_id": user_id,
                        "notes": str(notes),
                        "timestamp_utc": datetime.utcnow().isoformat(),
                    }
                )

    # Save accepted records button
    if st.button("Save accepted rows to label store"):
        if not to_save_records:
            st.warning(
                "No rows marked as accepted. Mark 'Accept (save)' to persist rows."
            )
        else:
            fname = append_labels_to_store(to_save_records)
            st.success(f"Saved {len(to_save_records)} accepted records to {fname}")
            st.balloons()

    # Export graded CSV for all displayed rows
    if st.button("Export displayed graded CSV"):
        out_df = df_state.loc[display_df.index, :].copy()
        # ensure final_score exists
        if "final_score" not in out_df.columns:
            st.error("No final_score column present.")
        else:
            to_download = out_df.drop(
                columns=["full_text", "model_feedback"], errors="ignore"
            )
            csv_bytes = to_download.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download graded CSV",
                data=csv_bytes,
                file_name="graded_export.csv",
                mime="text/csv",
            )

    # Show summary of edits (counts)
    st.subheader("Batch summary")
    accepted_count = int(df_state["accepted"].sum())
    st.metric("Accepted rows", accepted_count)
    st.write(
        df_state[["essay_id", "score", "model_score", "final_score", "accepted"]].head(
            20
        )
    )

    st.write(
        "End of UI. Use 'Save accepted rows' to persist grader-accepted final scores to `data/labels/`."
    )


if __name__ == "__main__":
    main()
