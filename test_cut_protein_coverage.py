#!/usr/bin/env python3
"""Test that cut_protein's cover_range intervals cover each residue exactly once.

Examples
--------
python test_cut_protein_coverage.py --source batch_predict.py
python test_cut_protein_coverage.py --source batch_predict.py --csv data/curated/NR_Sequences_New.csv

The --chunk-size argument is the number of amino acids per model window
(excluding START and END tokens). For a cut_protein(sequence, seq_len)
function, seq_len is automatically set to chunk_size + 2.

"""

import argparse
import ast
import csv
import inspect
import random
from pathlib import Path

import numpy as np


def load_cut_protein(source_path):
    """Load the actual function definition without executing the source script."""
    path = Path(source_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "cut_protein"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one top-level cut_protein function in {path}; found {len(matches)}")
    node = matches[0]
    if isinstance(node, ast.AsyncFunctionDef):
        raise ValueError("cut_protein must be a synchronous function")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    actual = namespace["cut_protein"]
    params = inspect.signature(actual).parameters
    if "seq_len" in params:
        # batch_predict.py expects the *tokenized* model length, i.e. a window
        # of chunk_size amino acids plus the START and END special tokens.
        def call(sequence, *, chunk_size):
            return actual(sequence, seq_len=chunk_size + 2)
    elif "chunk_size" in params:
        # Support variants that directly accept a number of amino acids.
        def call(sequence, *, chunk_size):
            return actual(sequence, chunk_size=chunk_size)
    else:
        raise ValueError(
            f"{path}: unsupported cut_protein signature {inspect.signature(actual)}; "
            "expected a seq_len or chunk_size parameter"
        )
    return call


def check_sequence(cut_protein, sequence, chunk_size, name):
    """Return a list of problems, or an empty list when all checks pass."""
    problems = []
    length = len(sequence)
    half = chunk_size // 2
    chunks = cut_protein(sequence, chunk_size=chunk_size)
    coverage = np.zeros(length, dtype=np.int32)

    if not chunks:
        return [f"{name}: no chunks returned"]

    for expected_id, record in enumerate(chunks):
        chunk_id = record["chunk_id"]
        chunk_seq = record["seq"]
        local_start, local_end = record["cover_range"]
        offset = 0 if chunk_id is None else chunk_id * half

        if length <= chunk_size:
            if chunk_id is not None:
                problems.append(f"{name}: short sequence has chunk_id={chunk_id}, expected None")
        elif chunk_id != expected_id:
            problems.append(f"{name}: chunk {expected_id} has chunk_id={chunk_id}")

        if offset < 0 or offset + len(chunk_seq) > length:
            problems.append(f"{name}: chunk {expected_id} extends outside original sequence")
            continue
        if chunk_seq != sequence[offset:offset + len(chunk_seq)]:
            problems.append(f"{name}: chunk {expected_id} contains incorrect sequence data")
        if len(chunk_seq) > chunk_size:
            problems.append(f"{name}: chunk {expected_id} is longer than {chunk_size}")

        if not (0 <= local_start <= local_end <= len(chunk_seq)):
            problems.append(
                f"{name}: chunk {expected_id} invalid cover_range "
                f"{(local_start, local_end)} for length {len(chunk_seq)}"
            )
            continue

        global_start = offset + local_start
        global_end = offset + local_end
        coverage[global_start:global_end] += 1

    # Tests effective supervised/evaluation ranges, NOT overlapping context.
    missing = np.flatnonzero(coverage == 0)
    repeated = np.flatnonzero(coverage > 1)
    if missing.size:
        problems.append(
            f"{name} (length {length}): {missing.size} UNCOVERED positions; "
            f"first 1-based positions: {(missing[:10] + 1).tolist()}"
        )
    if repeated.size:
        problems.append(
            f"{name} (length {length}): {repeated.size} MULTIPLY COVERED positions; "
            f"first 1-based positions: {(repeated[:10] + 1).tolist()}"
        )
    return problems


def test_csv(cut_protein, csv_path, chunk_size, failures, max_examples):
    count = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if "Sequence" not in (reader.fieldnames or []):
            raise ValueError(f"CSV {csv_path} has no 'Sequence' column")
        for row_num, row in enumerate(reader, start=2):
            uid = row.get("UniprotID") or f"row {row_num}"
            seq = row["Sequence"]
            if not seq:
                failures.append(f"CSV row {row_num}, {uid}: missing/empty Sequence")
                count += 1
                continue
            issues = check_sequence(cut_protein, seq, chunk_size, f"CSV row {row_num}, {uid}")
            failures.extend(issues[:max_examples])
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Python file containing your cut_protein function")
    parser.add_argument("--csv", default="data/curated/NR_Sequences_New.csv", help="Input CSV with Sequence column (default: data/curated/NR_Sequences_New.csv)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Amino acids per window, excluding START/END (default: 512; seq_len=514)")
    parser.add_argument("--synthetic-max", type=int, default=4096, help="Test every sequence length from 0 through this value")
    parser.add_argument("--random-tests", type=int, default=300)
    parser.add_argument("--random-max-length", type=int, default=50000)
    args = parser.parse_args()

    if args.chunk_size <= 0 or args.chunk_size % 4:
        parser.error("--chunk-size must be a positive multiple of 4")
    if args.synthetic_max < 0 or args.random_tests < 0 or args.random_max_length < 0:
        parser.error("test lengths/counts must be nonnegative")

    cut_protein = load_cut_protein(args.source)
    failures = []
    rng = random.Random(0)

    # Exhaustive length checks are especially useful at chunk boundaries.
    n_synthetic = 0
    for length in range(args.synthetic_max + 1):
        failures.extend(check_sequence(cut_protein, "A" * length, args.chunk_size, f"synthetic length {length}"))
        n_synthetic += 1

    for k in range(args.random_tests):
        length = rng.randint(0, args.random_max_length)
        failures.extend(check_sequence(cut_protein, "A" * length, args.chunk_size, f"random test {k + 1}"))
        n_synthetic += 1

    csv_path = Path(args.csv)
    if csv_path.is_file():
        n_real = test_csv(cut_protein, csv_path, args.chunk_size, failures, max_examples=10)
        csv_message = f"{n_real} real sequences from {csv_path}"
    else:
        n_real = 0
        csv_message = f"SKIPPED real sequences (CSV not found: {csv_path})"

    print(f"Synthetic sequences tested: {n_synthetic}")
    print(f"Real sequences tested:      {csv_message}")
    if failures:
        print(f"\nFAIL: {len(failures)} detected issue(s). First 20:")
        for issue in failures[:20]:
            print("  -", issue)
        raise SystemExit(1)
    print("\nPASS: Every residue is covered EXACTLY ONCE by cover_range.")
    print("      Chunk sequence slices and local-to-global offsets are also valid.")


if __name__ == "__main__":
    main()
