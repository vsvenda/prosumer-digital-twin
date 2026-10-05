import pandas as pd
import cdsapi

dataset = "cams-solar-radiation-timeseries"

request = {
    "sky_type": "observed_cloud",
    "location": {
        "longitude": 19.1122,
        "latitude": 45.7742,
    },
    "altitude": ["-999"],
    "date": ["2023-04-07/2023-04-08"],
    "time_step": "1minute",
    "time_reference": "universal_time",
    "data_format": "csv",
}

client = cdsapi.Client()

client.retrieve(
    dataset,
    request,
).download(
    "cams_test_sombor.csv",
)


CAMS_COLUMNS = [
    "observation_period",
    "toa",
    "clear_sky_ghi",
    "clear_sky_bhi",
    "clear_sky_dhi",
    "clear_sky_bni",
    "ghi",
    "bhi",
    "dhi",
    "bni",
    "reliability",
]


df = pd.read_csv(
    "cams_test_sombor.csv",
    sep=";",
    comment="#",
    header=None,
    names=CAMS_COLUMNS,
)


df[["timestamp_start", "timestamp_end"]] = (
    df["observation_period"]
    .str.split("/", expand=True)
)

df["timestamp"] = pd.to_datetime(
    df["timestamp_start"],
    utc=True,
)

df = df[
    [
        "timestamp",
        "ghi",
        "dhi",
        "bni",
        "reliability",
    ]
].copy()

df = df.rename(
    columns={
        "ghi": "ghi_wh_m2",
        "dhi": "dhi_wh_m2",
        "bni": "bni_wh_m2",
    }
)


# 1-minute clean dataset
df.to_csv(
    "cams_sombor_1min.csv",
    index=False,
)


# 10-minute dataset
cams_10min = (
    df
    .set_index("timestamp")
    .resample("10min")
    .agg(
        {
            "ghi_wh_m2": "sum",
            "dhi_wh_m2": "sum",
            "bni_wh_m2": "sum",
            "reliability": "mean",
        }
    )
    .reset_index()
)

cams_10min.to_csv(
    "cams_sombor_10min.csv",
    index=False,
)


print("1-minute data:")
print(df.head())

print("\n10-minute data:")
print(cams_10min.head())