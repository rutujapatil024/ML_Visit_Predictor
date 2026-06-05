"""
app.py — Flask Backend for "Should I Visit?" Crowd Predictor
=============================================================
Loads trained ML models, serves the frontend, and handles
prediction requests via a REST API endpoint.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
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

# ══════════════════════════════════════════════════════
# FLASK APP SETUP
# ══════════════════════════════════════════════════════

app = Flask(__name__)

# ── Load Models ───────────────────────────────────────


def load_models():
    """Load all trained models, encoders, and lookup tables from models/ directory."""
    model_dir = "models"
    required_files = [
        "crowd_model.pkl", "temp_model.pkl",
        "encoders.pkl", "weather_lookup.pkl"
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

    crowd_model = joblib.load(os.path.join(model_dir, "crowd_model.pkl"))
    temp_model = joblib.load(os.path.join(model_dir, "temp_model.pkl"))
    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    weather_lookup = joblib.load(os.path.join(model_dir, "weather_lookup.pkl"))

    print("  ✓ All models loaded successfully!")
    return crowd_model, temp_model, encoders, weather_lookup


# Load models at startup
crowd_model, temp_model, encoders, weather_lookup = load_models()


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

    # Predict temperature
    temperature = int(round(temp_model.predict(features)[0]))

    # Get weather from lookup
    weather = weather_lookup.get((place, month), "Clear")

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

    # Crowd count estimate
    crowd_count_est = CROWD_COUNT_RANGES.get(crowd_level, "Unknown")

    # Generate travel tips
    tips = generate_tips(temperature, weather, crowd_level)

    return {
        "crowd_level": crowd_level,
        "crowd_count_est": crowd_count_est,
        "temperature": temperature,
        "weather": weather,
        "visit_score": visit_score,
        "recommendation": recommendation,
        "holiday_info": holiday_info,
        "event_type": event_type,
        "day_of_week": day_of_week,
        "date": date_str,
        "tips": tips,
    }


def find_better_dates(place, start_date_str):
    """Scan next 60 days and return top 3 dates with Low or Moderate crowd."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    better = []

    for i in range(1, 61):
        check_date = start_date + timedelta(days=i)
        check_str = check_date.strftime("%Y-%m-%d")
        result = predict_for_date(place, check_str)

        if "error" in result:
            continue

        if result["crowd_level"] in ["Low", "Moderate"]:
            better.append({
                "date": check_str,
                "day_of_week": result["day_of_week"],
                "crowd_level": result["crowd_level"],
                "temperature": result["temperature"],
                "weather": result["weather"],
                "visit_score": result["visit_score"],
                "holiday_info": result["holiday_info"],
            })

        if len(better) >= 3:
            break

    # Sort by visit score descending and return top 3
    better.sort(key=lambda x: -x["visit_score"])
    return better[:3]


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

        # Find better dates if recommendation is NO or MAYBE
        if result["recommendation"] in ["NO", "MAYBE"]:
            result["better_dates"] = find_better_dates(place, date_str)
        else:
            result["better_dates"] = []

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


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " Should I Visit? — Web App Running ".center(58) + "║")
    print("║" + " Open: http://localhost:5000 ".center(58) + "║")
    print("╚" + "═" * 58 + "╝\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
