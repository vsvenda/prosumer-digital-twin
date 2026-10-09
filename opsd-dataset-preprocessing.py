import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\vanja.svenda\Downloads\household_data_1min_singleindex.csv"
)

OUTPUT_FILE = Path(
    r"C:\Users\vanja.svenda\Downloads\resident_pv_data_1min.csv"
)

RESIDENTS = {
    "residential1": "DE_KN_residential1_pv",
    "residential3": "DE_KN_residential3_pv",
    "residential4": "DE_KN_residential4_pv",
    "residential6": "DE_KN_residential6_pv",
}

CHUNK_SIZE = 200_000


# ============================================================
# Columns to read
# ============================================================

usecols = [
    "utc_timestamp",
    "interpolated",
    *RESIDENTS.values(),
]


# ============================================================
# Process file chunk-by-chunk
# ============================================================

first_output_chunk = True

# We need the previous row because power at t uses E(t) - E(t-1)
previous_energy = {
    resident: None
    for resident in RESIDENTS
}

previous_interpolated = {
    resident: False
    for resident in RESIDENTS
}

previous_timestamp = None


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

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    chunk["utc_timestamp"] = pd.to_datetime(
        chunk["utc_timestamp"],
        utc=True,
        errors="raise",
    )

    # --------------------------------------------------------
    # Determine which original energy readings were interpolated
    # --------------------------------------------------------

    interpolation_text = (
        chunk["interpolated"]
        .fillna("")
        .astype(str)
    )

    current_interpolated = {}

    for resident, source_column in RESIDENTS.items():

        current_interpolated[resident] = (
            interpolation_text.str.contains(
                source_column,
                regex=False,
            )
        )

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    out = pd.DataFrame()

    out["utc_timestamp"] = chunk["utc_timestamp"]

    affected_by_interpolation = pd.DataFrame(
        False,
        index=chunk.index,
        columns=RESIDENTS.keys(),
    )

    # --------------------------------------------------------
    # Convert cumulative energy [kWh] -> average power [W]
    # --------------------------------------------------------

    for resident, source_column in RESIDENTS.items():

        energy = pd.to_numeric(
            chunk[source_column],
            errors="coerce",
        )

        # Previous energy value, including continuity between chunks
        prev_energy = energy.shift(1)

        if previous_energy[resident] is not None:
            prev_energy.iloc[0] = previous_energy[resident]

        # Previous timestamp
        prev_timestamp_series = chunk["utc_timestamp"].shift(1)

        if previous_timestamp is not None:
            prev_timestamp_series.iloc[0] = previous_timestamp

        # Time difference in hours
        delta_hours = (
            chunk["utc_timestamp"]
            - prev_timestamp_series
        ).dt.total_seconds() / 3600.0

        # Energy difference in kWh
        delta_energy = energy - prev_energy

        # Average power:
        #
        # P[kW] = delta_E[kWh] / delta_t[h]
        # P[W]  = P[kW] * 1000
        #
        power_W = (
            delta_energy
            / delta_hours
            * 1000.0
        )

        # Only discard mathematically invalid time intervals
        power_W = power_W.where(delta_hours > 0)

        out[f"{resident}_pv_power_W"] = power_W

        # ----------------------------------------------------
        # Mark derived power interval as interpolated if
        # either endpoint was interpolated
        # ----------------------------------------------------

        prev_interp = current_interpolated[resident].shift(
            1,
            fill_value=False,
        )

        if len(prev_interp) > 0:
            prev_interp.iloc[0] = previous_interpolated[resident]

        affected_by_interpolation[resident] = (
            current_interpolated[resident]
            | prev_interp
        )

        # Save final row for next chunk
        if len(energy) > 0:
            previous_energy[resident] = energy.iloc[-1]
            previous_interpolated[resident] = bool(
                current_interpolated[resident].iloc[-1]
            )

    if len(chunk) > 0:
        previous_timestamp = chunk["utc_timestamp"].iloc[-1]

    # --------------------------------------------------------
    # Create one compact interpolation column
    # --------------------------------------------------------

    def interpolation_label(row):
        affected = [
            resident
            for resident, flag in row.items()
            if flag
        ]

        return ";".join(affected)

    out["interpolated"] = affected_by_interpolation.apply(
        interpolation_label,
        axis=1,
    )

    # --------------------------------------------------------
    # Save incrementally
    # --------------------------------------------------------

    out.to_csv(
        OUTPUT_FILE,
        mode="w" if first_output_chunk else "a",
        header=first_output_chunk,
        index=False,
    )

    first_output_chunk = False


print()
print("Finished.")
print(f"Saved to: {OUTPUT_FILE}")