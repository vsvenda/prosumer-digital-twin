import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\vanja.svenda\Downloads\filtered_pv_power_measurements_sc.csv"
)

OUTPUT_FILE = Path(
    r"C:\Users\vanja.svenda\Downloads\utrecht_selected_4_households.csv"
)

SELECTED_IDS = [
    "ID003",
    "ID022",
    "ID061",
    "ID089",
]

CHUNK_SIZE = 200_000


# ============================================================
# Columns to keep
# ============================================================

usecols = [
    "DateTime",
    *SELECTED_IDS,
]


# ============================================================
# Process file chunk-by-chunk
# ============================================================

first_output_chunk = True
total_rows = 0


for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ),
    start=1,
):

    print(f"Processing chunk {chunk_number}...")

    # Parse timestamp.
    # Utrecht dataset timestamps are treated as UTC.
    chunk["DateTime"] = pd.to_datetime(
        chunk["DateTime"],
        utc=True,
        errors="raise",
    )

    # Make sure PV columns are numeric.
    # Existing missing/invalid measurements remain NaN.
    for col in SELECTED_IDS:
        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce",
        )

    # Save incrementally.
    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if first_output_chunk else "a",
        header=first_output_chunk,
        index=False,
    )

    total_rows += len(chunk)
    first_output_chunk = False


print()
print("Finished.")
print(f"Rows written: {total_rows:,}")
print(f"Saved to: {OUTPUT_FILE}")