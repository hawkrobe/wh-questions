#!/usr/bin/env python3
"""De-identify participant data prior to public release.

What this does:
  1. Walks every CSV in `analysis/` (top-level) and `data/experiment{1,2}/`.
  2. Builds a global mapping prolific_pid -> subj_NNN (3-digit zero-padded,
     sorted by PID for determinism).
  3. Replaces the prolific_pid column with the anonymous ID.
  4. Drops the study_id, session_id, and stimulus columns where present.
  5. Renames raw per-participant CSV files (which currently use random
     session tokens as filenames) to subj_NNN.csv. If a participant
     appears in both experiments, they get the same subj ID in both dirs.
  6. Writes the PID -> subj_NNN mapping to .pid_mapping.json at the repo
     root. **This file is sensitive** -- keep it out of the public repo
     (it is added to .gitignore by this script).

Run from repo root:
    python deidentify.py --dry-run    # preview only
    python deidentify.py              # apply
"""
import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
ANALYSIS_CSVS = [
    REPO / "analysis" / "exp1_coded.csv",
    REPO / "analysis" / "exp1_toCode.csv",
    REPO / "analysis" / "exp2_processed.csv",
]
RAW_DIRS = [REPO / "data" / "experiment1", REPO / "data" / "experiment2"]
DROP_COLS_RAW = {"study_id", "session_id", "stimulus"}
DROP_COLS_ANALYSIS = {"study_id", "session_id"}
MAPPING_FILE = REPO / ".pid_mapping.json"
GITIGNORE = REPO / ".gitignore"


def collect_pids():
    """Return sorted list of unique prolific_pids across all files."""
    pids = set()
    for f in ANALYSIS_CSVS:
        if not f.exists():
            continue
        with f.open() as fp:
            for row in csv.DictReader(fp):
                if row.get("prolific_pid"):
                    pids.add(row["prolific_pid"])
    for d in RAW_DIRS:
        if not d.exists():
            continue
        for path in d.iterdir():
            if path.suffix != ".csv":
                continue
            with path.open() as fp:
                for row in csv.DictReader(fp):
                    if row.get("prolific_pid"):
                        pids.add(row["prolific_pid"])
                    break  # one row is enough
    return sorted(pids)


def build_mapping(pids):
    width = max(3, len(str(len(pids))))
    return {pid: f"subj_{i + 1:0{width}d}" for i, pid in enumerate(pids)}


def rewrite_csv(src, dst, mapping, drop_cols):
    """Rewrite CSV: replace prolific_pid via mapping, drop drop_cols.

    Returns (rows_written, kept_cols) tuple.
    """
    with src.open() as fp:
        reader = csv.DictReader(fp)
        original_cols = reader.fieldnames or []
        kept_cols = [c for c in original_cols if c not in drop_cols]
        rows = list(reader)
    for row in rows:
        if "prolific_pid" in row and row["prolific_pid"] in mapping:
            row["prolific_pid"] = mapping[row["prolific_pid"]]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=kept_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in kept_cols})
    return len(rows), kept_cols


def first_pid_in(path):
    with path.open() as fp:
        for row in csv.DictReader(fp):
            return row.get("prolific_pid")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="preview without writing files")
    args = ap.parse_args()

    pids = collect_pids()
    print(f"Found {len(pids)} unique prolific_pids")
    mapping = build_mapping(pids)

    print("\nAnalysis CSVs:")
    for f in ANALYSIS_CSVS:
        if not f.exists():
            print(f"  SKIP (missing): {f.relative_to(REPO)}")
            continue
        if args.dry_run:
            with f.open() as fp:
                cols = csv.DictReader(fp).fieldnames or []
            kept = [c for c in cols if c not in DROP_COLS_ANALYSIS]
            print(f"  {f.relative_to(REPO)}: would drop {sorted(set(cols) & DROP_COLS_ANALYSIS) or 'nothing'}; "
                  f"keep {len(kept)}/{len(cols)} cols")
        else:
            n, kept = rewrite_csv(f, f, mapping, DROP_COLS_ANALYSIS)
            print(f"  {f.relative_to(REPO)}: rewrote {n} rows, kept {len(kept)} cols")

    print("\nRaw per-participant CSVs:")
    for d in RAW_DIRS:
        if not d.exists():
            print(f"  SKIP (missing): {d.relative_to(REPO)}")
            continue
        files = sorted(p for p in d.iterdir() if p.suffix == ".csv")
        if args.dry_run:
            sample = files[0] if files else None
            if sample:
                with sample.open() as fp:
                    cols = csv.DictReader(fp).fieldnames or []
                kept = [c for c in cols if c not in DROP_COLS_RAW]
                print(f"  {d.relative_to(REPO)}: {len(files)} files; would drop "
                      f"{sorted(set(cols) & DROP_COLS_RAW)}; keep {len(kept)}/{len(cols)} cols; "
                      f"rename to subj_NNN.csv")
        else:
            renamed = 0
            for src in files:
                pid = first_pid_in(src)
                if not pid or pid not in mapping:
                    print(f"    WARN: no PID in {src.name}, skipping")
                    continue
                subj = mapping[pid]
                dst = d / f"{subj}.csv"
                tmp = d / f".{subj}.tmp.csv"
                rewrite_csv(src, tmp, mapping, DROP_COLS_RAW)
                if dst.exists():
                    print(f"    NOTE: {dst.name} already exists "
                          f"(participant has multiple files in this dir); "
                          f"appending suffix")
                    suffix = 2
                    while (d / f"{subj}_{suffix}.csv").exists():
                        suffix += 1
                    dst = d / f"{subj}_{suffix}.csv"
                tmp.rename(dst)
                src.unlink()
                renamed += 1
            print(f"  {d.relative_to(REPO)}: rewrote+renamed {renamed} files")

    if not args.dry_run:
        with MAPPING_FILE.open("w") as fp:
            json.dump(mapping, fp, indent=2)
        print(f"\nWrote PID mapping to {MAPPING_FILE.relative_to(REPO)}")
        print("  (this file is SENSITIVE -- keep it out of the public repo)")
        # Ensure mapping is gitignored
        if GITIGNORE.exists():
            existing = GITIGNORE.read_text()
            if ".pid_mapping.json" not in existing:
                with GITIGNORE.open("a") as fp:
                    if not existing.endswith("\n"):
                        fp.write("\n")
                    fp.write(".pid_mapping.json\n")
                print(f"  Added .pid_mapping.json to .gitignore")
    else:
        print("\n(dry run -- no files modified)")


if __name__ == "__main__":
    main()
