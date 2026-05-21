#!/usr/bin/env python3
"""Build clean daily return data from Kenneth French library ZIP files.

The default dataset is the CRSP-backed 49 industry portfolio daily returns.
Outputs are decimal returns, not percentage returns.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import urllib.request
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np


FRENCH49_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "49_Industry_Portfolios_daily_CSV.zip"
)

SECTION_MARKERS = {
    "value": "Average Value Weighted Returns -- Daily",
    "equal": "Average Equal Weighted Returns -- Daily",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download/parse Kenneth French daily industry returns."
    )
    parser.add_argument("--url", default=FRENCH49_DAILY_URL)
    parser.add_argument("--input-zip", type=Path, default=None)
    parser.add_argument(
        "--raw-zip",
        type=Path,
        default=repo_root / "data/raw/french/49_Industry_Portfolios_daily_CSV.zip",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "data/processed/french49_daily_returns.csv",
    )
    parser.add_argument(
        "--csdi-dir",
        type=Path,
        default=repo_root / "CSDI/data/french49_daily",
    )
    parser.add_argument("--weighting", choices=sorted(SECTION_MARKERS), default="value")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--keep-other", action="store_true")
    parser.add_argument("--max-missing-frac", type=float, default=0.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-csdi", action="store_true")
    return parser.parse_args()


def parse_iso_date(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def ensure_zip(args: argparse.Namespace) -> Path:
    args.raw_zip.parent.mkdir(parents=True, exist_ok=True)
    if args.input_zip is not None:
        shutil.copyfile(args.input_zip, args.raw_zip)
        return args.raw_zip
    if args.raw_zip.exists() and not args.force_download:
        return args.raw_zip

    request = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        args.raw_zip.write_bytes(response.read())
    return args.raw_zip


def read_csv_from_zip(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {csv_names}")
        return archive.read(csv_names[0]).decode("utf-8-sig")


def parse_french_section(text: str, weighting: str) -> tuple[list[str], list[date], np.ndarray]:
    lines = text.splitlines()
    marker = SECTION_MARKERS[weighting]
    try:
        marker_index = next(i for i, line in enumerate(lines) if marker in line)
    except StopIteration as exc:
        raise ValueError(f"Could not find section marker: {marker}") from exc

    header_index = None
    for i in range(marker_index + 1, len(lines)):
        if lines[i].strip():
            header_index = i
            break
    if header_index is None:
        raise ValueError(f"Could not find header after section marker: {marker}")

    header = next(csv.reader([lines[header_index]]))
    columns = [name.strip() for name in header[1:]]
    dates: list[date] = []
    rows: list[list[float]] = []

    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        first_field = stripped.split(",", 1)[0].strip()
        if len(first_field) != 8 or not first_field.isdigit():
            break

        fields = next(csv.reader([line]))
        dates.append(datetime.strptime(fields[0].strip(), "%Y%m%d").date())
        values = []
        for raw in fields[1:]:
            value = float(raw.strip())
            if value in {-99.99, -999.0}:
                values.append(np.nan)
            else:
                values.append(value / 100.0)
        rows.append(values)

    if not rows:
        raise ValueError(f"No data rows found for section: {marker}")
    return columns, dates, np.asarray(rows, dtype=np.float64)


def filter_rows(
    dates: list[date],
    values: np.ndarray,
    start: date | None,
    end: date | None,
) -> tuple[list[date], np.ndarray]:
    keep = np.ones(len(dates), dtype=bool)
    if start is not None:
        keep &= np.asarray([d >= start for d in dates])
    if end is not None:
        keep &= np.asarray([d <= end for d in dates])
    return [d for d, keep_row in zip(dates, keep) if keep_row], values[keep]


def filter_columns(
    columns: list[str],
    values: np.ndarray,
    keep_other: bool,
    max_missing_frac: float,
) -> tuple[list[str], np.ndarray, list[str]]:
    missing_frac = np.isnan(values).mean(axis=0)
    keep = missing_frac <= max_missing_frac
    if not keep_other:
        keep &= np.asarray([col != "Other" for col in columns])

    dropped = [col for col, keep_col in zip(columns, keep) if not keep_col]
    kept_columns = [col for col, keep_col in zip(columns, keep) if keep_col]
    kept_values = values[:, keep]
    if kept_values.shape[1] == 0:
        raise ValueError("Column filters removed every return series.")
    return kept_columns, kept_values, dropped


def write_returns_csv(path: Path, columns: list[str], dates: list[date], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", *columns])
        for row_date, row_values in zip(dates, values):
            formatted = ["" if np.isnan(x) else f"{x:.10g}" for x in row_values]
            writer.writerow([row_date.isoformat(), *formatted])


def masked_mean_std(values: np.ndarray, train_rows: int) -> tuple[np.ndarray, np.ndarray]:
    train_values = values[:train_rows]
    mean = np.nanmean(train_values, axis=0)
    std = np.nanstd(train_values, axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def write_csdi_files(
    output_dir: Path,
    columns: list[str],
    dates: list[date],
    values: np.ndarray,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_length = 252
    test_length = 252
    history_length = 231
    pred_length = 21
    train_rows = max(1, len(dates) - valid_length - test_length)

    main_data = np.nan_to_num(values, nan=0.0).astype(np.float32)
    mask_data = np.isfinite(values).astype(np.float32)
    mean, std = masked_mean_std(values, train_rows)

    with (output_dir / "data.pkl").open("wb") as f:
        pickle.dump([main_data, mask_data], f)
    with (output_dir / "meanstd.pkl").open("wb") as f:
        pickle.dump([mean, std], f)

    csdi_metadata = {
        **metadata,
        "features": columns,
        "csdi": {
            "history_length": history_length,
            "pred_length": pred_length,
            "valid_length": valid_length,
            "test_length": test_length,
            "train_rows_for_normalizer": train_rows,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(csdi_metadata, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    start = parse_iso_date(args.start)
    end = parse_iso_date(args.end)

    zip_path = ensure_zip(args)
    text = read_csv_from_zip(zip_path)
    columns, dates, values = parse_french_section(text, args.weighting)
    dates, values = filter_rows(dates, values, start, end)
    columns, values, dropped = filter_columns(
        columns,
        values,
        keep_other=args.keep_other,
        max_missing_frac=args.max_missing_frac,
    )

    metadata = {
        "dataset": "french49_daily",
        "source_url": args.url,
        "source_zip": str(zip_path),
        "weighting": args.weighting,
        "return_scale": "decimal",
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "rows": len(dates),
        "columns": len(columns),
        "dropped_columns": dropped,
        "missing_values": int(np.isnan(values).sum()),
    }

    write_returns_csv(args.output_csv, columns, dates, values)
    args.output_csv.with_suffix(".metadata.json").write_text(
        json.dumps({**metadata, "features": columns}, indent=2) + "\n"
    )
    if not args.no_csdi:
        write_csdi_files(args.csdi_dir, columns, dates, values, metadata)

    print(
        "Built French daily returns: "
        f"{len(dates)} rows x {len(columns)} columns, "
        f"{dates[0]} to {dates[-1]}"
    )
    print(f"CSV: {args.output_csv}")
    if not args.no_csdi:
        print(f"CSDI: {args.csdi_dir}")
    if dropped:
        print(f"Dropped columns: {', '.join(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
