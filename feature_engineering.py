import pandas as pd

# Loading ouravailable Data

df = pd.read_csv("FINAL_DATA_SET.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Columns Found:", df.columns.tolist())


# Basic Cleaning

df = df.dropna(subset=["impressions"])
df[["likes", "comments", "shares"]] = df[["likes", "comments", "shares"]].fillna(0)

# Handle date column
if "date" in df.columns:
    date_col = "date"
elif "post_date" in df.columns:
    date_col = "post_date"
else:
    raise ValueError("No date column found")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])


# Engagement Rate (ONLY for label)


df["engagement_rate"] = (
    (df["likes"] + df["comments"] + df["shares"]) / df["impressions"]
) * 100


# Binary Target (SAFE)

threshold = df["engagement_rate"].median()

df["performance_binary"] = (df["engagement_rate"] >= threshold).astype(int)

# Time Features

df["day_code"] = df[date_col].dt.dayofweek
df["hour"] = df[date_col].dt.hour

def time_slot(h):
    if h < 12:
        return 0  # Morning
    elif h < 17:
        return 1  # Afternoon
    elif h < 21:
        return 2  # Evening
    else:
        return 3  # Night

df["time_code"] = df["hour"].apply(time_slot)


# Hashtag Features

df["hashtag_count"] = (
    df["hashtag"]
    .fillna("")
    .astype(str)
    .apply(lambda x: len(x.split()))
)

# Encode Categorical

df["post_type_code"] = df["post_type"].astype("category").cat.codes
df["platform_code"] = df["platform"].astype("category").cat.codes

# Dropinf the Leakage Columns

DROP_COLS = [
    "likes",
    "comments",
    "shares",
    "engagement_rate",
    date_col
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])


# Saveing ML-Ready Data


df.to_csv("ml_ready_posts.csv", index=False)

print(" Feature engineering completed")
print(" Saved: ml_ready_posts.csv")
