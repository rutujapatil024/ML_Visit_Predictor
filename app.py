"""
app.py — Flask Backend for "Should I Visit?" Crowd Predictor
=============================================================
Loads trained ML models, serves the frontend, and handles
prediction requests via a REST API endpoint.

Features:
  - Crowd level classification (Low/Moderate/High/Extreme)
  - Crowd count regression (actual visitor numbers)
  - Temperature prediction
  - Weather API integration (Open-Meteo live forecast + ML fallback)
  - Monthly crowd trend chart data
  - Landmark comparison
  - Better date suggestions
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory


# ══════════════════════════════════════════════════════
# CONSTANTS — Indian Public Holidays & Festival Calendar
# ══════════════════════════════════════════════════════

INDIAN_HOLIDAYS = {
    # ── 2025 National & Public Holidays ──────────────────
    "2025-01-01": {"name": "New Year's Day",             "type": "Public Holiday"},
    "2025-01-06": {"name": "Guru Gobind Singh Jayanti",  "type": "Religious Holiday"},
    "2025-01-14": {"name": "Makar Sankranti / Pongal",   "type": "Festival"},
    "2025-01-23": {"name": "Netaji Subhas Chandra Bose Jayanti", "type": "National Day"},
    "2025-01-26": {"name": "Republic Day",               "type": "National Holiday"},
    "2025-02-02": {"name": "Basant Panchami",            "type": "Festival"},
    "2025-02-19": {"name": "Chhatrapati Shivaji Maharaj Jayanti", "type": "Regional Holiday"},
    "2025-02-26": {"name": "Maha Shivaratri",            "type": "Festival"},
    "2025-03-13": {"name": "Holika Dahan",               "type": "Festival"},
    "2025-03-14": {"name": "Holi",                       "type": "Festival"},
    "2025-03-30": {"name": "Ram Navami",                 "type": "Festival"},
    "2025-03-31": {"name": "Eid ul-Fitr",                "type": "Festival"},
    "2025-04-06": {"name": "Mahavir Jayanti",            "type": "Religious Holiday"},
    "2025-04-10": {"name": "Maundy Thursday",            "type": "Religious Holiday"},
    "2025-04-13": {"name": "Baisakhi / Vishu",           "type": "Festival"},
    "2025-04-14": {"name": "Ambedkar Jayanti / Tamil New Year", "type": "Regional Holiday"},
    "2025-04-18": {"name": "Good Friday",                "type": "Public Holiday"},
    "2025-04-20": {"name": "Easter Sunday",              "type": "Religious Holiday"},
    "2025-05-12": {"name": "Buddha Purnima",             "type": "Religious Holiday"},
    "2025-06-07": {"name": "Eid ul-Adha (Bakrid)",       "type": "Festival"},
    "2025-06-27": {"name": "Rath Yatra",                 "type": "Festival"},
    "2025-07-06": {"name": "Muharram",                   "type": "Religious Holiday"},
    "2025-08-09": {"name": "Raksha Bandhan",             "type": "Festival"},
    "2025-08-15": {"name": "Independence Day",           "type": "National Holiday"},
    "2025-08-16": {"name": "Janmashtami",                "type": "Festival"},
    "2025-08-27": {"name": "Ganesh Chaturthi",           "type": "Festival"},
    "2025-09-05": {"name": "Milad-un-Nabi (Eid-e-Milad)", "type": "Religious Holiday"},
    "2025-09-22": {"name": "Navratri Begins",            "type": "Festival"},
    "2025-10-01": {"name": "Navratri Ends / Durga Ashtami", "type": "Festival"},
    "2025-10-02": {"name": "Gandhi Jayanti / Dussehra",  "type": "National Holiday"},
    "2025-10-20": {"name": "Diwali / Deepavali",         "type": "Festival"},
    "2025-10-21": {"name": "Govardhan Puja",             "type": "Festival"},
    "2025-10-22": {"name": "Bhai Dooj",                  "type": "Festival"},
    "2025-11-05": {"name": "Guru Nanak Jayanti",         "type": "Religious Holiday"},
    "2025-11-15": {"name": "Jharkhand Foundation Day",   "type": "Regional Holiday"},
    "2025-12-25": {"name": "Christmas Day",              "type": "Public Holiday"},

    # ── 2026 National & Public Holidays ──────────────────
    "2026-01-01": {"name": "New Year's Day",             "type": "Public Holiday"},
    "2026-01-14": {"name": "Makar Sankranti / Pongal",   "type": "Festival"},
    "2026-01-26": {"name": "Republic Day",               "type": "National Holiday"},
    "2026-02-15": {"name": "Maha Shivaratri",            "type": "Festival"},
    "2026-03-03": {"name": "Holi",                       "type": "Festival"},
    "2026-03-20": {"name": "Ram Navami",                 "type": "Festival"},
    "2026-03-21": {"name": "Eid ul-Fitr",                "type": "Festival"},
    "2026-03-26": {"name": "Mahavir Jayanti",            "type": "Religious Holiday"},
    "2026-04-02": {"name": "Good Friday",                "type": "Public Holiday"},
    "2026-04-14": {"name": "Ambedkar Jayanti",           "type": "Regional Holiday"},
    "2026-05-01": {"name": "Maharashtra / Gujarat Day",  "type": "Regional Holiday"},
    "2026-05-31": {"name": "Buddha Purnima",             "type": "Religious Holiday"},
    "2026-06-27": {"name": "Eid ul-Adha (Bakrid)",       "type": "Festival"},
    "2026-07-29": {"name": "Raksha Bandhan",             "type": "Festival"},
    "2026-08-15": {"name": "Independence Day",           "type": "National Holiday"},
    "2026-08-05": {"name": "Janmashtami",                "type": "Festival"},
    "2026-08-19": {"name": "Ganesh Chaturthi",           "type": "Festival"},
    "2026-10-02": {"name": "Gandhi Jayanti",             "type": "National Holiday"},
    "2026-10-11": {"name": "Dussehra",                   "type": "Festival"},
    "2026-11-08": {"name": "Diwali",                     "type": "Festival"},
    "2026-11-24": {"name": "Guru Nanak Jayanti",         "type": "Religious Holiday"},
    "2026-12-25": {"name": "Christmas Day",              "type": "Public Holiday"},
}

# ── Festival Season Windows ───────────────────────────
FESTIVAL_SEASONS = [
    (1,  12, 1,  16, "Makar Sankranti / Pongal Season"),
    (1,  24, 1,  27, "Republic Day Long Weekend"),
    (3,  10, 3,  18, "Holi Season"),
    (3,  28, 4,   5, "Ram Navami / Navratri Season"),
    (4,  10, 4,  15, "Baisakhi / Easter Season"),
    (8,  10, 8,  20, "Independence Day / Janmashtami Season"),
    (8,  25, 9,   5, "Ganesh Chaturthi Season"),
    (9,  20, 10,  5, "Navratri / Durga Puja Season"),
    (10,  1, 10, 25, "Dussehra / Diwali Season"),
    (12, 20, 12, 31, "Christmas / New Year Season"),
]

# ── Place-specific Peak Seasons ───────────────────────
PLACE_PEAK_MONTHS = {
    "Taj Mahal, Agra":               [10, 11, 12,  1,  2,  3],
    "Varanasi Ghats, Varanasi":      [10, 11, 12,  1,  2,  3],
    "Red Fort, Delhi":               [ 8, 10, 11, 12,  1,  2],
    "India Gate, Delhi":             [ 1,  8, 10, 12],
    "Qutub Minar, Delhi":           [10, 11, 12,  1,  2,  3],
    "Gateway of India, Mumbai":      [12,  1,  2,  8],
    "Mecca Masjid, Hyderabad":       [ 3,  4,  6],
    "Sanchi Stupa, Madhya Pradesh":  [10, 11, 12,  1,  2],
}

# ── Crowd Level Score Impact ──────────────────────────
CROWD_SCORE_BASE = {
    "Low":      90,
    "Moderate": 65,
    "High":     38,
    "Extreme":  15,
}

WEATHER_SCORE_PENALTY = {
    "Rainy":  -12,
    "Cloudy":  -3,
    "Clear":    0,
    "Sunny":   -5,
}

# ── Crowd count estimate ranges ──────────────────────
CROWD_COUNT_RANGES = {
    "Low":      "0 – 9,999",
    "Moderate": "10,000 – 21,999",
    "High":     "22,000 – 34,999",
    "Extreme":  "35,000+",
}

# ── List of 8 landmarks ──────────────────────────────
PLACES = [
    "Sanchi Stupa, Madhya Pradesh",
    "Qutub Minar, Delhi",
    "Varanasi Ghats, Varanasi",
    "Red Fort, Delhi",
    "Mecca Masjid, Hyderabad",
    "Taj Mahal, Agra",
    "Gateway of India, Mumbai",
    "India Gate, Delhi",
]

# ── Place Coordinates for Weather API ─────────────────
PLACE_COORDINATES = {
    "Taj Mahal, Agra":               {"lat": 27.1751, "lon": 78.0421},
    "Red Fort, Delhi":               {"lat": 28.6562, "lon": 77.2410},
    "India Gate, Delhi":             {"lat": 28.6129, "lon": 77.2295},
    "Qutub Minar, Delhi":           {"lat": 28.5245, "lon": 77.1855},
    "Varanasi Ghats, Varanasi":      {"lat": 25.3176, "lon": 83.0107},
    "Gateway of India, Mumbai":      {"lat": 18.9220, "lon": 72.8347},
    "Mecca Masjid, Hyderabad":       {"lat": 17.3604, "lon": 78.4736},
    "Sanchi Stupa, Madhya Pradesh":  {"lat": 23.4793, "lon": 77.7398},
}

# ── WMO Weather Code Mapping ─────────────────────────
WMO_WEATHER_MAP = {
    0: "Clear", 1: "Clear", 2: "Cloudy", 3: "Cloudy",
    45: "Cloudy", 48: "Cloudy",
    51: "Rainy", 53: "Rainy", 55: "Rainy",
    56: "Rainy", 57: "Rainy",
    61: "Rainy", 63: "Rainy", 65: "Rainy",
    66: "Rainy", 67: "Rainy",
    71: "Cloudy", 73: "Cloudy", 75: "Cloudy",
    77: "Cloudy",
    80: "Rainy", 81: "Rainy", 82: "Rainy",
    85: "Cloudy", 86: "Cloudy",
    95: "Rainy", 96: "Rainy", 99: "Rainy",
}


# ══════════════════════════════════════════════════════
# FLASK APP SETUP
# ══════════════════════════════════════════════════════

app = Flask(__name__)

# ── Load Models ───────────────────────────────────────


def load_models():
    """Load all trained ML models, encoders, and metrics from disk."""
    model_dir = "models"
    required_files = [
        "crowd_model.pkl", "temp_model.pkl", "crowd_count_model.pkl",
        "dt_crowd_model.pkl", "dt_crowd_count_model.pkl",
        "gb_crowd_model.pkl", "gb_crowd_count_model.pkl",
        "encoders.pkl", "weather_lookup.pkl", "model_metrics.json"
    ]

    # Check models directory exists
    if not os.path.exists(model_dir):
        print("\n" + "=" * 60)
        print("  ❌ ERROR: Models not found!")
        print("  Please run: python train_model.py")
        print("=" * 60 + "\n")
        sys.exit(1)

    # Check all required files exist
    for f in required_files:
        if not os.path.exists(os.path.join(model_dir, f)):
            print(f"\n  ❌ ERROR: {f} not found in models/")
            print("  Please run: python train_model.py")
            sys.exit(1)

    # Load model binaries
    rf_crowd = joblib.load(os.path.join(model_dir, "crowd_model.pkl"))
    dt_crowd = joblib.load(os.path.join(model_dir, "dt_crowd_model.pkl"))
    gb_crowd = joblib.load(os.path.join(model_dir, "gb_crowd_model.pkl"))

    temp_model = joblib.load(os.path.join(model_dir, "temp_model.pkl"))

    rf_cc = joblib.load(os.path.join(model_dir, "crowd_count_model.pkl"))
    dt_cc = joblib.load(os.path.join(model_dir, "dt_crowd_count_model.pkl"))
    gb_cc = joblib.load(os.path.join(model_dir, "gb_crowd_count_model.pkl"))

    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    weather_lookup = joblib.load(os.path.join(model_dir, "weather_lookup.pkl"))

    # Load metrics JSON
    with open(os.path.join(model_dir, "model_metrics.json"), "r", encoding="utf-8") as f:
        model_metrics = json.load(f)

    print("  ✓ All models and metrics loaded successfully!")
    
    return {
        "classifiers": {"Random Forest": rf_crowd, "Decision Tree": dt_crowd, "Gradient Boosting": gb_crowd},
        "temp_model": temp_model,
        "regressors": {"Random Forest": rf_cc, "Decision Tree": dt_cc, "Gradient Boosting": gb_cc},
        "encoders": encoders,
        "weather_lookup": weather_lookup,
        "metrics": model_metrics
    }


# Load models at startup
loaded = load_models()
crowd_model = loaded["classifiers"]["Random Forest"]
temp_model = loaded["temp_model"]
crowd_count_model = loaded["regressors"]["Random Forest"]
encoders = loaded["encoders"]
weather_lookup = loaded["weather_lookup"]
model_metrics = loaded["metrics"]
classifiers = loaded["classifiers"]
regressors = loaded["regressors"]

# Landmark-specific hourly density profiles (weights for each hour from 6 AM to 9 PM)
HOURLY_PROFILES = {
    "Taj Mahal, Agra": [
        12, 14, 10, 8, 7, 6, 5, 4, 4, 6, 10, 14, 10, 0, 0, 0 # Closed after sunset (6 PM onwards set to 0)
    ],
    "Varanasi Ghats, Varanasi": [
        18, 14, 8, 5, 4, 3, 3, 2, 2, 3, 6, 12, 16, 12, 6, 2  # Peaks at 6 AM and 6 PM for Aarti
    ],
    "India Gate, Delhi": [
        1, 2, 3, 3, 4, 3, 2, 2, 4, 8, 14, 20, 22, 18, 12, 6   # Massively peaks in the evening (5 PM - 9 PM)
    ],
    "Gateway of India, Mumbai": [
        3, 4, 5, 6, 6, 5, 4, 4, 6, 8, 12, 16, 18, 14, 8, 4    # Sunset/evening peak
    ],
    "Red Fort, Delhi": [
        0, 0, 0, 8, 12, 14, 15, 12, 10, 10, 12, 12, 5, 0, 0, 0 # Open 9 AM - 6 PM, peaks mid-day
    ],
    "Qutub Minar, Delhi": [
        0, 2, 4, 8, 10, 12, 10, 8, 8, 10, 12, 14, 8, 4, 0, 0  # Open 7 AM - 9 PM, peaks mid-day and sunset
    ],
    "Mecca Masjid, Hyderabad": [
        0, 0, 4, 6, 8, 14, 16, 10, 8, 12, 14, 8, 4, 0, 0, 0   # Peaks around Dhuhr (12 PM - 2 PM) and Asr (4 PM - 5 PM)
    ],
    "Sanchi Stupa, Madhya Pradesh": [
        2, 4, 6, 10, 14, 16, 15, 12, 10, 8, 6, 4, 2, 0, 0, 0  # Open 6:30 AM - 6:30 PM, peak mid-day
    ]
}


def generate_hourly_data(place, daily_count, daily_temp):
    """Generate hourly crowd counts and temperatures for the landmark."""
    hours = ["6:00 AM", "7:00 AM", "8:00 AM", "9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM",
             "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM", "5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM"]
    
    # Temperature hourly offsets (relative to daily average)
    temp_offsets = [-4, -3, -1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, -1, -2]
    
    # Get profile or fallback to a standard bell curve
    profile = HOURLY_PROFILES.get(place, [4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 4, 4, 4])
    
    # Normalize profile weights
    total_weight = sum(profile)
    normalized_weights = [w / total_weight for w in profile]
    
    hourly_data = []
    for i, hour in enumerate(hours):
        h_weight = normalized_weights[i]
        
        # Calculate crowd count at this hour
        h_count = int(round(daily_count * h_weight))
        
        # Temperature
        h_temp = int(round(daily_temp + temp_offsets[i]))
        
        hourly_data.append({
            "hour": hour,
            "crowd_count": max(0, h_count),
            "temperature": h_temp
        })
        
    return hourly_data


def predict_ensemble(place, date_str):
    """Generate predictions from all three ensemble models (RF, DT, GB)."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

    month = date_obj.month
    day_of_month = date_obj.day
    week_of_year = date_obj.isocalendar()[1]
    day_of_week = get_day_of_week_string(date_obj)
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    holiday_info_data = is_indian_holiday(date_str)
    is_holiday = 1 if holiday_info_data else 0
    event_type = get_event_type(date_str, month, day_of_month)
    is_festival = 1 if event_type in ["Festival", "National Holiday", "Cultural Event"] else 0

    place_enc = encoders["Place"].transform([place])[0]
    event_enc = encoders["Event"].transform([event_type])[0]

    features = np.array([[
        place_enc, month, day_of_month, week_of_year,
        is_weekend, is_holiday, is_festival, event_enc
    ]])

    predictions = {}
    for name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
        # Predict crowd level
        clf = classifiers[name]
        cl_enc = clf.predict(features)[0]
        cl = encoders["crowd_level"].inverse_transform([cl_enc])[0]

        # Predict raw crowd count
        reg = regressors[name]
        cc = int(round(reg.predict(features)[0]))
        cc = max(0, cc)

        predictions[name] = {
            "crowd_level": cl,
            "crowd_count": cc
        }

    return predictions

# ══════════════════════════════════════════════════════
# WEATHER API (Open-Meteo)
# ══════════════════════════════════════════════════════


def fetch_live_weather(place, date_str):
    """Fetch weather from Open-Meteo API for dates within forecast range.

    Returns dict with temperature, weather, weather_source or None if unavailable.
    """
    coords = PLACE_COORDINATES.get(place)
    if not coords:
        return None

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    days_ahead = (target_date - today).days

    # Open-Meteo provides 16-day forecasts
    if days_ahead < 0 or days_ahead > 15:
        return None

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            return None

        data = resp.json()
        daily = data.get("daily", {})

        if not daily.get("temperature_2m_max"):
            return None

        temp_max = daily["temperature_2m_max"][0]
        temp_min = daily["temperature_2m_min"][0]
        weather_code = daily["weather_code"][0]

        avg_temp = int(round((temp_max + temp_min) / 2))
        weather = WMO_WEATHER_MAP.get(weather_code, "Clear")

        return {
            "temperature": avg_temp,
            "weather": weather,
            "weather_source": "Live Forecast",
            "temp_max": int(round(temp_max)),
            "temp_min": int(round(temp_min)),
        }
    except (requests.RequestException, KeyError, TypeError, ValueError) as e:
        print(f"  ⚠ Weather API error: {e}")
        return None


# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════


def is_indian_holiday(date_str):
    """Check if a date string is a known Indian holiday. Returns holiday info dict or None."""
    return INDIAN_HOLIDAYS.get(date_str, None)


def is_in_festival_season(month, day):
    """Check if a date falls within any festival season window."""
    for sm, sd, em, ed, name in FESTIVAL_SEASONS:
        if sm == em:
            if month == sm and sd <= day <= ed:
                return name
        elif month == sm and day >= sd:
            return name
        elif month == em and day <= ed:
            return name
    return None


def get_event_type(date_str, month, day):
    """Determine the event type for a given date based on holiday calendar."""
    holiday = is_indian_holiday(date_str)
    if holiday:
        htype = holiday["type"]
        if htype in ["Festival"]:
            return "Festival"
        elif htype in ["National Holiday", "National Day"]:
            return "National Holiday"
        elif htype in ["Religious Holiday", "Regional Holiday"]:
            return "Cultural Event"
        else:
            return "Cultural Event"

    season = is_in_festival_season(month, day)
    if season:
        return "Cultural Event"

    return "Regular Day"


def get_day_of_week_string(date_obj):
    """Return the day of week as a string (Monday, Tuesday, etc.)."""
    return date_obj.strftime("%A")

def generate_tips(temperature, weather, crowd_level):
    """Generate contextual travel tips based on temperature, weather, and crowd levels."""
    tips = []
    # Temperature tips
    if temperature >= 38:
        tips.append("⚠️ Extreme Heat: High temperature of " + str(temperature) + "°C expected. Limit outdoor exposure between 12 PM - 4 PM. Wear a hat, sunglasses, and carry hydration.")
    elif temperature >= 33:
        tips.append("☀️ Warm Day: Wear lightweight clothing, sunglasses, and sunscreen. Keep a water bottle handy.")
    elif temperature <= 16:
        tips.append("❄️ Cool Day: Temperatures are cool (around " + str(temperature) + "°C). Layered clothing or a light jacket is recommended for early morning and evening.")

    # Weather tips
    if weather == "Rainy":
        tips.append("🌧️ Rain Forecast: Bring an umbrella or raincoat. Outdoor stone steps and paths may be slippery.")
    elif weather == "Sunny":
        tips.append("🕶️ UV Protection: Sunny sky expected. Sunscreen and a hat are highly recommended.")
    elif weather == "Cloudy":
        tips.append("📸 Photography Tip: Overcast skies offer beautifully soft, even lighting—perfect for clear photos without harsh shadows.")

    # Crowd tips
    if crowd_level in ["High", "Extreme"]:
        tips.append("🎟️ Ticket Booking: Book entry tickets online in advance to skip the long physical ticketing lines.")
        tips.append("⏰ Early Arrival: Arrive early in the morning (before 8 AM) to beat the crowds and queues.")
    elif crowd_level == "Low":
        tips.append("✨ Peaceful Visit: Excellent day to explore at a relaxed pace with minimal wait times.")

    return tips


def predict_for_date(place, date_str):
    """Run full prediction pipeline for a given place and date string."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    # Extract date features
    month = date_obj.month
    day_of_month = date_obj.day
    week_of_year = date_obj.isocalendar()[1]
    day_of_week = get_day_of_week_string(date_obj)

    # Determine is_weekend
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0

    # Determine is_holiday
    holiday_info_data = is_indian_holiday(date_str)
    is_holiday = 1 if holiday_info_data else 0

    # Determine event type and is_festival
    event_type = get_event_type(date_str, month, day_of_month)
    is_festival = 1 if event_type in ["Festival", "National Holiday", "Cultural Event"] else 0

    # Holiday info string for display
    if holiday_info_data:
        holiday_info = f"{holiday_info_data['name']} – {holiday_info_data['type']}"
    else:
        season = is_in_festival_season(month, day_of_month)
        if season:
            holiday_info = f"{season}"
        else:
            holiday_info = "Regular Day"

    # Encode place
    try:
        place_enc = encoders["Place"].transform([place])[0]
    except ValueError:
        return {"error": f"Unknown place: {place}"}

    # Encode event
    try:
        event_enc = encoders["Event"].transform([event_type])[0]
    except ValueError:
        event_enc = encoders["Event"].transform(["Regular Day"])[0]

    # Build feature array in same order as training
    features = np.array([[
        place_enc, month, day_of_month, week_of_year,
        is_weekend, is_holiday, is_festival, event_enc
    ]])

    # Predict crowd level
    crowd_level_enc = crowd_model.predict(features)[0]
    crowd_level = encoders["crowd_level"].inverse_transform([crowd_level_enc])[0]

    # Predict raw crowd count
    crowd_count_predicted = int(round(crowd_count_model.predict(features)[0]))
    crowd_count_predicted = max(0, crowd_count_predicted)  # Clamp to 0

    # Predict temperature (ML fallback)
    ml_temperature = int(round(temp_model.predict(features)[0]))

    # Get weather — try live API first, then fallback to lookup
    live_weather = fetch_live_weather(place, date_str)
    if live_weather:
        temperature = live_weather["temperature"]
        weather = live_weather["weather"]
        weather_source = live_weather["weather_source"]
        temp_max = live_weather.get("temp_max")
        temp_min = live_weather.get("temp_min")
    else:
        temperature = ml_temperature
        weather = weather_lookup.get((place, month), "Clear")
        weather_source = "ML Prediction"
        temp_max = None
        temp_min = None

    # Compute visit score
    visit_score = CROWD_SCORE_BASE.get(crowd_level, 50)

    # Apply weather penalty
    visit_score += WEATHER_SCORE_PENALTY.get(weather, 0)

    # Apply temperature penalty
    if temperature > 38:
        visit_score -= 12
    elif temperature > 36:
        visit_score -= 8

    # Add small random variation based on date hash for natural feel
    date_hash = hash(date_str) % 11 - 5  # -5 to +5
    visit_score += date_hash

    # Clamp score to 0-100
    visit_score = max(0, min(100, visit_score))

    # Determine recommendation
    if visit_score >= 70:
        recommendation = "YES"
    elif visit_score >= 40:
        recommendation = "MAYBE"
    else:
        recommendation = "NO"

    # Crowd count estimate (category range + predicted number)
    crowd_count_est = CROWD_COUNT_RANGES.get(crowd_level, "Unknown")

    # Generate travel tips
    tips = generate_tips(temperature, weather, crowd_level)

    result = {
        "crowd_level": crowd_level,
        "crowd_count_est": crowd_count_est,
        "crowd_count_predicted": crowd_count_predicted,
        "temperature": temperature,
        "weather": weather,
        "weather_source": weather_source,
        "visit_score": visit_score,
        "recommendation": recommendation,
        "holiday_info": holiday_info,
        "event_type": event_type,
        "day_of_week": day_of_week,
        "date": date_str,
        "tips": tips,
    }

    if temp_max is not None:
        result["temp_max"] = temp_max
        result["temp_min"] = temp_min

    return result


def find_better_dates(place, start_date_str):
    """Scan next 60 days and return top 3 dates with best visit scores."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    all_dates = []

    for i in range(1, 61):
        check_date = start_date + timedelta(days=i)
        check_str = check_date.strftime("%Y-%m-%d")
        result = predict_for_date(place, check_str)

        if "error" in result:
            continue

        all_dates.append({
            "date": check_str,
            "day_of_week": result["day_of_week"],
            "crowd_level": result["crowd_level"],
            "crowd_count_predicted": result["crowd_count_predicted"],
            "temperature": result["temperature"],
            "weather": result["weather"],
            "visit_score": result["visit_score"],
            "holiday_info": result["holiday_info"],
        })

    # Sort by visit score descending and return top 3
    all_dates.sort(key=lambda x: -x["visit_score"])
    return all_dates[:3]


# ══════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════


@app.route("/")
def index():
    """Render the main frontend page with place list."""
    return render_template("index.html", places=PLACES)


@app.route("/static/index.css")
def serve_css():
    """Serve the CSS file from the templates directory."""
    return send_from_directory("templates", "index.css", mimetype="text/css")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests. Receives JSON with place and date."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received. Please send JSON."}), 400

        place = data.get("place", "").strip()
        date_str = data.get("date", "").strip()

        # Validate inputs
        if not place:
            return jsonify({"error": "Please select a destination."}), 400
        if not date_str:
            return jsonify({"error": "Please select a date."}), 400
        if place not in PLACES:
            return jsonify({"error": f"Unknown destination: {place}"}), 400

        # Validate date format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        # Run prediction
        result = predict_for_date(place, date_str)

        if "error" in result:
            return jsonify(result), 400

        # Always find better dates for the selected place
        result["better_dates"] = find_better_dates(place, date_str)

        # Generate ensemble predictions
        ensemble = predict_ensemble(place, date_str)

        # Generate hourly distribution
        hourly = generate_hourly_data(place, result["crowd_count_predicted"], result["temperature"])
        
        # Get best hour to visit
        operating_hours = [h for h in hourly if h["crowd_count"] > 0]
        if not operating_hours:
            operating_hours = hourly
        best_hour_data = min(operating_hours, key=lambda x: (x["crowd_count"], x["temperature"]))
        
        # Inject into result
        result["ensemble_predictions"] = ensemble
        result["model_metrics"] = model_metrics
        result["hourly_distribution"] = hourly
        result["best_time_to_visit"] = best_hour_data["hour"]

        return jsonify(result)

    except Exception as e:
        print(f"  ❌ Prediction error: {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


@app.route("/compare", methods=["POST"])
def compare():
    """Handle comparison requests. Receives JSON with list of places and date."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received. Please send JSON."}), 400

        places = data.get("places", [])
        date_str = data.get("date", "").strip()

        # Validate inputs
        if not places or not isinstance(places, list):
            return jsonify({"error": "Please select landmarks to compare."}), 400
        if len(places) < 2 or len(places) > 3:
            return jsonify({"error": "Please select 2 or 3 landmarks."}), 400
        if not date_str:
            return jsonify({"error": "Please select a date."}), 400

        # Validate date format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        results = []
        for place in places:
            place = place.strip()
            if place not in PLACES:
                return jsonify({"error": f"Unknown destination: {place}"}), 400

            res = predict_for_date(place, date_str)
            if "error" in res:
                return jsonify(res), 400
            res["place"] = place
            results.append(res)

        # Sort results by visit_score descending
        results.sort(key=lambda x: -x["visit_score"])
        for idx, res in enumerate(results):
            res["is_best"] = (idx == 0)

        return jsonify(results)

    except Exception as e:
        print(f"  ❌ Comparison error: {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500



@app.route("/chart-data", methods=["POST"])
def chart_data():
    """Return monthly crowd trend data for Chart.js visualization."""
    try:
        data = request.get_json()
        place = data.get("place", "").strip()
        date_str = data.get("date", "").strip()

        if place not in PLACES:
            return jsonify({"error": f"Unknown place: {place}"}), 400

        # Parse the target date to get month/year
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        year = target_date.year
        month = target_date.month

        # Generate predictions for each day of the month
        import calendar
        days_in_month = calendar.monthrange(year, month)[1]

        dates = []
        crowd_counts = []
        crowd_levels = []
        temperatures = []
        visit_scores = []

        for day in range(1, days_in_month + 1):
            day_str = f"{year}-{month:02d}-{day:02d}"
            result = predict_for_date(place, day_str)

            if "error" in result:
                continue

            dates.append(day_str)
            crowd_counts.append(result["crowd_count_predicted"])
            crowd_levels.append(result["crowd_level"])
            temperatures.append(result["temperature"])
            visit_scores.append(result["visit_score"])

        return jsonify({
            "dates": dates,
            "crowd_counts": crowd_counts,
            "crowd_levels": crowd_levels,
            "temperatures": temperatures,
            "visit_scores": visit_scores,
            "selected_date": date_str,
            "month_name": target_date.strftime("%B %Y"),
        })

    except Exception as e:
        print(f"  ❌ Chart data error: {e}")
        return jsonify({"error": "Something went wrong."}), 500


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " Should I Visit? — Web App Running ".center(58) + "║")
    print("║" + " Open: http://localhost:5000 ".center(58) + "║")
    print("╚" + "═" * 58 + "╝\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
