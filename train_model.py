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
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

# Force UTF-8 output on Windows to avoid UnicodeEncodeError in print statements
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

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
    """Train multiple classifiers for crowd level prediction."""
    print("\n" + "=" * 60)
    print("STEP 6: Training Crowd Level Classifiers...")
    print("=" * 60)

    X = df[FEATURE_COLS]
    y = df["crowd_level_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    # 1. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"  ✓ Random Forest Classifier Accuracy: {rf_acc * 100:.1f}%")

    # 2. Decision Tree
    dt_model = DecisionTreeClassifier(
        max_depth=10, random_state=42
    )
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    print(f"  ✓ Decision Tree Classifier Accuracy: {dt_acc * 100:.1f}%")

    # 3. Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=50, max_depth=5, random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    print(f"  ✓ Gradient Boosting Classifier Accuracy: {gb_acc * 100:.1f}%")

    # Feature importances for RF (primary model)
    importances = dict(zip(FEATURE_COLS, rf_model.feature_importances_))
    print("\n  RF Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {feat:18s} {imp:.4f} {bar}")

    metrics = {
        "Random Forest": {"accuracy": round(rf_acc, 4)},
        "Decision Tree": {"accuracy": round(dt_acc, 4)},
        "Gradient Boosting": {"accuracy": round(gb_acc, 4)}
    }

    return rf_model, dt_model, gb_model, importances, metrics


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


def train_crowd_count_model(df):
    """Train multiple regressors for raw crowd count prediction."""
    print("\n" + "=" * 60)
    print("STEP 7B: Training Crowd Count Regressors...")
    print("=" * 60)

    X = df[FEATURE_COLS]
    y = df["Crowd_Count (in Thousands)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    # 1. Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=150, max_depth=12, random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"  ✓ Random Forest Regressor - MAE: {rf_mae:.0f} visitors | R2: {rf_r2:.3f}")

    # 2. Decision Tree
    dt_model = DecisionTreeRegressor(
        max_depth=12, random_state=42
    )
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_mae = mean_absolute_error(y_test, dt_pred)
    dt_r2 = r2_score(y_test, dt_pred)
    print(f"  ✓ Decision Tree Regressor - MAE: {dt_mae:.0f} visitors | R2: {dt_r2:.3f}")

    # 3. Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=50, max_depth=5, random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    print(f"  ✓ Gradient Boosting Regressor - MAE: {gb_mae:.0f} visitors | R2: {gb_r2:.3f}")

    # Show sample RF predictions vs actual
    print("\n  Sample Predictions vs Actual (Random Forest):")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = int(round(rf_pred[i]))
        print(f"    Actual: {actual:,} | Predicted: {predicted:,}")

    metrics = {
        "Random Forest": {"mae": round(float(rf_mae), 2), "r2": round(float(rf_r2), 4)},
        "Decision Tree": {"mae": round(float(dt_mae), 2), "r2": round(float(dt_r2), 4)},
        "Gradient Boosting": {"mae": round(float(gb_mae), 2), "r2": round(float(gb_r2), 4)}
    }

    return rf_model, dt_model, gb_model, metrics


def save_models(
    rf_crowd, dt_crowd, gb_crowd,
    temp_model,
    rf_cc, dt_cc, gb_cc,
    encoders, weather_lookup, importances,
    metrics
):
    """Save all models, encoders, lookup tables, and model metrics."""
    print("\n" + "=" * 60)
    print("STEP 8: Saving models...")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Dump classifier models
    joblib.dump(rf_crowd, os.path.join(MODEL_DIR, "crowd_model.pkl"))
    joblib.dump(dt_crowd, os.path.join(MODEL_DIR, "dt_crowd_model.pkl"))
    joblib.dump(gb_crowd, os.path.join(MODEL_DIR, "gb_crowd_model.pkl"))
    print("  ✓ Saved crowd level classification models (RF, DT, GB)")

    # Dump temperature model
    joblib.dump(temp_model, os.path.join(MODEL_DIR, "temp_model.pkl"))
    print("  ✓ Saved temp_model.pkl")

    # Dump regressor models (crowd count)
    joblib.dump(rf_cc, os.path.join(MODEL_DIR, "crowd_count_model.pkl"))
    joblib.dump(dt_cc, os.path.join(MODEL_DIR, "dt_crowd_count_model.pkl"))
    joblib.dump(gb_cc, os.path.join(MODEL_DIR, "gb_crowd_count_model.pkl"))
    print("  ✓ Saved crowd count regression models (RF, DT, GB)")

    # Save remaining utilities
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(weather_lookup, os.path.join(MODEL_DIR, "weather_lookup.pkl"))
    joblib.dump(importances, os.path.join(MODEL_DIR, "feature_importance.pkl"))
    print("  ✓ Saved encoders, weather lookup, and feature importance")

    # Save metrics JSON
    import json
    metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("  ✓ Saved model_metrics.json")


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

    # Train crowd classifiers
    rf_crowd, dt_crowd, gb_crowd, importances, class_metrics = train_crowd_model(df)

    # Train temperature regressor
    temp_model = train_temp_model(df)

    # Train crowd count regressors
    rf_cc, dt_cc, gb_cc, reg_metrics = train_crowd_count_model(df)

    # Combine metrics
    metrics = {
        "classification": class_metrics,
        "regression": reg_metrics
    }

    # Save everything
    save_models(
        rf_crowd, dt_crowd, gb_crowd,
        temp_model,
        rf_cc, dt_cc, gb_cc,
        encoders, weather_lookup, importances,
        metrics
    )

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " ✅ All models saved successfully! ".center(58) + "║")
    print("║" + " Run 'python app.py' to start the web app ".center(58) + "║")
    print("╚" + "═" * 58 + "╝\n")


if __name__ == "__main__":
    main()
