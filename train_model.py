"""
train_model.py — Data Cleaning + Model Training Script
========================================================
Loads data.xlsx, cleans data, engineers features,
trains crowd-level classifier & temperature regressor,
builds weather lookup table, and saves everything to models/
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

# ══════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════

DATA_FILE = "data.xlsx"
MODEL_DIR = "models"

# Crowd level thresholds based on Crowd_Count (in Thousands)
CROWD_THRESHOLDS = {
    "Low":      (0, 9999),
    "Moderate": (10000, 21999),
    "High":     (22000, 34999),
    "Extreme":  (35000, float("inf")),
}

# Feature columns used for training
FEATURE_COLS = [
    "place_enc", "month", "day_of_month", "Week_of_Year",
    "is_weekend", "is_holiday", "is_festival", "event_enc"
]


def load_data():
    """Load the Excel dataset into a DataFrame."""
    print("=" * 60)
    print("STEP 1: Loading dataset...")
    print("=" * 60)
    df = pd.read_excel(DATA_FILE)
    print(f"  ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    """Apply all cleaning steps: parse dates, drop columns, handle NaN, fix types."""
    print("\n" + "=" * 60)
    print("STEP 2: Cleaning data...")
    print("=" * 60)

    # 1. Parse Date column as datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    print("  ✓ Parsed 'Date' column as datetime")

    # 2. Drop the Time column — no predictive value
    df = df.drop(columns=["Time"])
    print("  ✓ Dropped 'Time' column")

    # 3. Handle missing values
    df["Special_Features"] = df["Special_Features"].fillna("None")
    print(f"  ✓ Filled {683} NaN values in 'Special_Features' with 'None'")

    # Drop rows with NaN in critical columns
    critical_cols = ["Place", "Date", "Crowd_Count (in Thousands)", "Weather", "Event", "Holiday"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    dropped = before - len(df)
    print(f"  ✓ Dropped {dropped} rows with NaN in critical columns")

    # 4. Remove duplicate rows by Date + Place
    before = len(df)
    df = df.drop_duplicates(subset=["Date", "Place"])
    dropped = before - len(df)
    print(f"  ✓ Removed {dropped} duplicate rows (by Date + Place)")

    # 5. Fix data types
    df["Crowd_Count (in Thousands)"] = df["Crowd_Count (in Thousands)"].astype(int)
    df["Temperature (°C)"] = df["Temperature (°C)"].astype(int)
    df["Holiday"] = df["Holiday"].str.strip()
    print("  ✓ Fixed data types (Crowd_Count→int, Temperature→int, Holiday→stripped)")

    return df


def engineer_features(df):
    """Create new feature columns from existing data."""
    print("\n" + "=" * 60)
    print("STEP 3: Engineering features...")
    print("=" * 60)

    # 6. Create feature columns
    df["month"] = df["Date"].dt.month
    df["day_of_month"] = df["Date"].dt.day
    df["is_weekend"] = df["Day_of_Week"].apply(
        lambda x: 1 if x in ["Saturday", "Sunday"] else 0
    )
    df["is_holiday"] = df["Holiday"].apply(lambda x: 1 if x == "Yes" else 0)
    df["is_festival"] = df["Event"].apply(
        lambda x: 1 if x in ["Festival", "National Holiday", "Cultural Event"] else 0
    )
    print("  ✓ Created: month, day_of_month, is_weekend, is_holiday, is_festival")

    # 7. Create target variable crowd_level
    def assign_crowd_level(count):
        """Assign crowd level category based on count thresholds."""
        if count <= 9999:
            return "Low"
        elif count <= 21999:
            return "Moderate"
        elif count <= 34999:
            return "High"
        else:
            return "Extreme"

    df["crowd_level"] = df["Crowd_Count (in Thousands)"].apply(assign_crowd_level)
    print("  ✓ Created target variable 'crowd_level'")

    return df


def encode_features(df):
    """Encode categorical columns using LabelEncoder and save encoders."""
    print("\n" + "=" * 60)
    print("STEP 4: Encoding categorical features...")
    print("=" * 60)

    encoders = {}

    # 8. Encode categorical columns
    for col in ["Place", "Weather", "Event", "Day_of_Week", "crowd_level"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  ✓ Encoded '{col}' → {len(le.classes_)} classes: {list(le.classes_)}")

    # Rename for clarity
    df.rename(columns={
        "Place_enc": "place_enc",
        "Event_enc": "event_enc",
        "Day_of_Week_enc": "day_of_week_enc",
        "Weather_enc": "weather_enc",
        "crowd_level_enc": "crowd_level_enc"
    }, inplace=True)

    return df, encoders


def print_diagnostics(df):
    """Print cleaning diagnostics for verification."""
    print("\n" + "=" * 60)
    print("DIAGNOSTICS — verify cleaning results")
    print("=" * 60)
    print(f"\n  DataFrame shape: {df.shape}")
    print(f"\n  Null values per column:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"    {col}: {count}")
    if null_counts.sum() == 0:
        print("    ✓ No null values found!")

    print(f"\n  Crowd level distribution:")
    for level, count in df["crowd_level"].value_counts().items():
        print(f"    {level}: {count}")

    print(f"\n  Cleaned sample (first 5 rows):")
    print(df[["Date", "Place", "Crowd_Count (in Thousands)", "crowd_level", "month", "is_weekend", "is_holiday", "is_festival"]].head(5).to_string(index=False))


def build_weather_lookup(df):
    """Build a weather lookup dict: (Place, month) → most common weather."""
    print("\n" + "=" * 60)
    print("STEP 5: Building weather lookup table...")
    print("=" * 60)

    weather_lookup = {}
    for place in df["Place"].unique():
        for month in range(1, 13):
            subset = df[(df["Place"] == place) & (df["month"] == month)]
            if len(subset) > 0:
                most_common = subset["Weather"].mode()[0]
            else:
                most_common = "Clear"  # default fallback
            weather_lookup[(place, month)] = most_common

    print(f"  ✓ Built lookup for {len(weather_lookup)} (Place, Month) combinations")

    # Print a sample
    print("  Sample entries:")
    sample_keys = list(weather_lookup.keys())[:4]
    for key in sample_keys:
        print(f"    {key} → {weather_lookup[key]}")

    return weather_lookup


def train_crowd_model(df):
    """Train RandomForestClassifier for crowd level prediction."""
    print("\n" + "=" * 60)
    print("STEP 6: Training Crowd Level Classifier...")
    print("=" * 60)

    X = df[FEATURE_COLS]
    y = df["crowd_level_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  ★ Crowd Model Accuracy: {accuracy * 100:.1f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importances
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    print("  Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {feat:18s} {imp:.4f} {bar}")

    return model, importances


def train_temp_model(df):
    """Train RandomForestRegressor for temperature prediction."""
    print("\n" + "=" * 60)
    print("STEP 7: Training Temperature Predictor...")
    print("=" * 60)

    X = df[FEATURE_COLS]
    y = df["Temperature (°C)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    model = RandomForestRegressor(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n  ★ Temperature MAE: {mae:.1f} °C")

    return model


def save_models(crowd_model, temp_model, encoders, weather_lookup, importances):
    """Save all models, encoders, and lookup tables to the models/ directory."""
    print("\n" + "=" * 60)
    print("STEP 8: Saving models...")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(crowd_model, os.path.join(MODEL_DIR, "crowd_model.pkl"))
    print(f"  ✓ Saved crowd_model.pkl")

    joblib.dump(temp_model, os.path.join(MODEL_DIR, "temp_model.pkl"))
    print(f"  ✓ Saved temp_model.pkl")

    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    print(f"  ✓ Saved encoders.pkl")

    joblib.dump(weather_lookup, os.path.join(MODEL_DIR, "weather_lookup.pkl"))
    print(f"  ✓ Saved weather_lookup.pkl")

    joblib.dump(importances, os.path.join(MODEL_DIR, "feature_importance.pkl"))
    print(f"  ✓ Saved feature_importance.pkl")


def main():
    """Main pipeline: load → clean → feature eng → encode → train → save."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " Should I Visit? — Model Training Pipeline ".center(58) + "║")
    print("╚" + "═" * 58 + "╝\n")

    # Load
    df = load_data()

    # Clean
    df = clean_data(df)

    # Feature engineering
    df = engineer_features(df)

    # Encode
    df, encoders = encode_features(df)

    # Diagnostics
    print_diagnostics(df)

    # Weather lookup (no ML — just mode per Place+Month)
    weather_lookup = build_weather_lookup(df)

    # Train crowd classifier
    crowd_model, importances = train_crowd_model(df)

    # Train temperature regressor
    temp_model = train_temp_model(df)

    # Save everything
    save_models(crowd_model, temp_model, encoders, weather_lookup, importances)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " ✅ All models saved successfully! ".center(58) + "║")
    print("║" + " Run 'python app.py' to start the web app ".center(58) + "║")
    print("╚" + "═" * 58 + "╝\n")


if __name__ == "__main__":
    main()
