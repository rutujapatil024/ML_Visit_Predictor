"""
app.py — Flask Backend for "Should I Visit?" Crowd Predictor
=============================================================
Loads trained ML models, serves the frontend, handles
user authentication (Register / Login), real-time confirmed
visit tracking, and prediction blending.

Features:
  - User Signup / Login Authentication (Password Hashing via Werkzeug)
  - Real-Time User Visit Confirmation & Cancellation
  - SQLite Database (`visitors.db`) for User Data & Confirmed Visits
  - Blended Real-Time Prediction Engine (ML + Live Users)
  - Dynamic Live Calendar Integration (holidays library + Nager.Date API)
  - Weather Forecast Integration (Open-Meteo live API + ML fallback)
  - Multi-Model Ensemble predictions (Random Forest, Decision Tree, Gradient Boosting)
  - Interactive Chart.js monthly trends & hourly density curves
"""

import os
import sys
import json
import uuid
import sqlite3
import numpy as np
import pandas as pd
import joblib
import requests
try:
    import holidays  # type: ignore # pyright: ignore[reportMissingImports]
    HAS_HOLIDAYS_LIB = True
except ImportError:
    holidays = None
    HAS_HOLIDAYS_LIB = False
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

# Force UTF-8 output on Windows to avoid UnicodeEncodeError with emoji/unicode
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ══════════════════════════════════════════════════════
# FEATURE COLUMNS FOR ML INFERENCE
# ══════════════════════════════════════════════════════
FEATURE_COLS = [
    "place_enc", "month", "day_of_month", "Week_of_Year",
    "is_weekend", "is_holiday", "is_festival", "event_enc"
]


# ══════════════════════════════════════════════════════
# DYNAMIC HOLIDAY CALENDAR — Live Fetching
# ══════════════════════════════════════════════════════

# In-memory cache: { year: { "YYYY-MM-DD": { "name": ..., "type": ... } } }
_holiday_cache = {}

# Nager.Date API cache: { year: { "YYYY-MM-DD": { "name": ..., "type": ... } } }
_nager_cache = {}


def _get_holidays_lib_data(year):
    """Fetch Indian holidays for a given year using the 'holidays' Python library."""
    if year in _holiday_cache:
        return _holiday_cache[year]

    if not HAS_HOLIDAYS_LIB or holidays is None:
        _holiday_cache[year] = {}
        return {}

    try:
        india_holidays = holidays.India(years=year)
        result = {}
        for date_obj, name in sorted(india_holidays.items()):
            date_str = date_obj.strftime("%Y-%m-%d")
            htype = _classify_holiday_type(name)
            result[date_str] = {"name": name, "type": htype}
        
        _holiday_cache[year] = result
        print(f"  ✓ Loaded {len(result)} holidays for {year} from holidays library")
        return result
    except Exception as e:
        print(f"  ⚠ holidays library error for {year}: {e}")
        _holiday_cache[year] = {}
        return {}


def _fetch_nager_holidays(year):
    """Fetch Indian public holidays from Nager.Date API as supplementary source."""
    if year in _nager_cache:
        return _nager_cache[year]

    try:
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/IN"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            _nager_cache[year] = {}
            return {}

        data = resp.json()
        result = {}
        for entry in data:
            date_str = entry.get("date", "")
            name = entry.get("localName", entry.get("name", "Holiday"))
            htype = _classify_holiday_type(name)
            result[date_str] = {"name": name, "type": htype}

        _nager_cache[year] = result
        print(f"  ✓ Fetched {len(result)} holidays for {year} from Nager.Date API")
        return result
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"  ⚠ Nager.Date API error for {year}: {e}")
        _nager_cache[year] = {}
        return {}


def _classify_holiday_type(name):
    """Classify a holiday name into a type category for the prediction engine."""
    name_lower = name.lower()

    national_keywords = ["republic day", "independence day", "gandhi jayanti",
                         "gandhi birthday", "mahatma gandhi"]
    if any(kw in name_lower for kw in national_keywords):
        return "National Holiday"

    festival_keywords = ["diwali", "deepavali", "holi", "dussehra", "navratri",
                        "durga puja", "ganesh chaturthi", "raksha bandhan",
                        "janmashtami", "pongal", "onam", "baisakhi",
                        "makar sankranti", "chhath", "lohri", "bihu",
                        "eid", "ramadan", "eid ul-fitr", "eid ul-adha",
                        "bakrid", "christmas", "easter", "govardhan",
                        "bhai dooj", "karva chauth", "ram navami",
                        "rath yatra", "guru nanak"]
    if any(kw in name_lower for kw in festival_keywords):
        return "Festival"

    religious_keywords = ["maha shivaratri", "shivaratri", "buddha purnima",
                         "mahavir jayanti", "good friday", "milad",
                         "muharram", "guru gobind", "basant panchami",
                         "holika"]
    if any(kw in name_lower for kw in religious_keywords):
        return "Religious Holiday"

    regional_keywords = ["jayanti", "foundation day", "ambedkar",
                        "shivaji", "subhas", "netaji", "vishu",
                        "tamil new year", "ugadi", "gudi padwa"]
    if any(kw in name_lower for kw in regional_keywords):
        return "Regional Holiday"

    return "Public Holiday"


def get_holidays_for_year(year):
    """Get merged holiday data for a year from all sources."""
    lib_data = _get_holidays_lib_data(year)
    nager_data = _fetch_nager_holidays(year)
    merged = dict(lib_data)
    for date_str, info in nager_data.items():
        if date_str not in merged:
            merged[date_str] = info
    return merged


# ── Festival Season Windows (approximate busy periods) ───
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

# ── Real Operating Hours for Each Landmark ────────────
# Format: (open_hour_24h, close_hour_24h, closed_day or None)
# closed_day: day name string (e.g. "Friday") or None if open daily
PLACE_OPERATING_HOURS = {
    "Taj Mahal, Agra":               {"open": 6,  "close": 18, "closed_day": "Friday"},
    "Red Fort, Delhi":               {"open": 9,  "close": 16, "closed_day": "Monday"},
    "India Gate, Delhi":             {"open": 0,  "close": 24, "closed_day": None},      # 24 hours
    "Qutub Minar, Delhi":           {"open": 7,  "close": 17, "closed_day": None},
    "Varanasi Ghats, Varanasi":      {"open": 0,  "close": 24, "closed_day": None},      # 24 hours
    "Gateway of India, Mumbai":      {"open": 0,  "close": 24, "closed_day": None},      # 24 hours
    "Mecca Masjid, Hyderabad":       {"open": 4,  "close": 21, "closed_day": None},
    "Sanchi Stupa, Madhya Pradesh":  {"open": 8,  "close": 18, "closed_day": None},
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

# ── Confirmed Visitor Scale Factor ────────────────────
CONFIRMED_VISITOR_SCALE = 50

# ── ML weight for blending predictions ────────────────
ML_WEIGHT = 0.6  # 60% ML prediction, 40% scaled confirmed visitors


# ══════════════════════════════════════════════════════
# SQLITE — Database Setup (Users & Confirmed Visits)
# ══════════════════════════════════════════════════════

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visitors.db")


def init_db():
    """Initialize the SQLite database with users and confirmed_visits tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # Confirmed Visits Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS confirmed_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            place TEXT NOT NULL,
            date TEXT NOT NULL,
            session_id TEXT NOT NULL,
            user_id INTEGER,
            username TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(place, date, session_id)
        )
    """)

    # Ensure missing columns exist
    c.execute("PRAGMA table_info(confirmed_visits)")
    cols = [row[1] for row in c.fetchall()]
    if "user_id" not in cols:
        c.execute("ALTER TABLE confirmed_visits ADD COLUMN user_id INTEGER")
    if "username" not in cols:
        c.execute("ALTER TABLE confirmed_visits ADD COLUMN username TEXT")

    conn.commit()
    conn.close()
    print("  ✓ SQLite database initialized (users + visitors.db)")


# ── USER AUTHENTICATION DATABASE FUNCTIONS ───────────

def register_user(username, email, password):
    """Register a new user in the SQLite database."""
    username = username.strip()
    email = email.strip().lower()

    if len(username) < 3:
        return {"error": "Username must be at least 3 characters long."}
    if "@" not in email or "." not in email:
        return {"error": "Please enter a valid email address."}
    if len(password) < 6:
        return {"error": "Password must be at least 6 characters long."}

    pwd_hash = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, pwd_hash)
        )
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return {
            "success": True,
            "user": {"id": user_id, "username": username, "email": email},
            "message": f"Welcome, {username}! Account created successfully."
        }
    except sqlite3.IntegrityError as e:
        conn.close()
        err_str = str(e).lower()
        if "username" in err_str:
            return {"error": "Username is already taken. Please choose another."}
        elif "email" in err_str:
            return {"error": "Email is already registered. Please log in."}
        else:
            return {"error": "User with this username or email already exists."}


def authenticate_user(login_id, password):
    """Authenticate user with username/email and password."""
    login_id = login_id.strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, username, email, password_hash FROM users WHERE username = ? OR email = ?",
        (login_id, login_id.lower())
    )
    row = c.fetchone()
    conn.close()

    if not row:
        return {"error": "Invalid username/email or password."}

    uid, uname, uemail, phash = row
    if not check_password_hash(phash, password):
        return {"error": "Invalid username/email or password."}

    return {
        "success": True,
        "user": {"id": uid, "username": uname, "email": uemail},
        "message": f"Welcome back, {uname}!"
    }


def get_user_by_id(user_id):
    """Retrieve user details by ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1], "email": row[2]}
    return None


# ── CONFIRMED VISITS DATABASE FUNCTIONS ──────────────

def get_confirmed_info(place, date_str):
    """Get count and usernames of confirmed visitors for a place and date."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Total count
    c.execute(
        "SELECT COUNT(*) FROM confirmed_visits WHERE place = ? AND date = ?",
        (place, date_str)
    )
    total_count = c.fetchone()[0]

    # Named users who confirmed
    c.execute(
        "SELECT DISTINCT username FROM confirmed_visits WHERE place = ? AND date = ? AND username IS NOT NULL AND username != ''",
        (place, date_str)
    )
    user_rows = c.fetchall()
    conn.close()

    confirmed_users = [r[0] for r in user_rows]
    return {
        "count": total_count,
        "users": confirmed_users
    }


def add_confirmed_visit(place, date_str, session_id, user_id=None, username=None):
    """Add or update a confirmed visit for a user/session. Returns True if new, False if updated/duplicate."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check existing visit for user_id or session_id
    if user_id:
        c.execute(
            "SELECT id FROM confirmed_visits WHERE place = ? AND date = ? AND (user_id = ? OR session_id = ?)",
            (place, date_str, user_id, session_id)
        )
        row = c.fetchone()
        if row:
            c.execute(
                "UPDATE confirmed_visits SET user_id = ?, username = ? WHERE id = ?",
                (user_id, username, row[0])
            )
            conn.commit()
            conn.close()
            return False

    try:
        c.execute(
            "INSERT INTO confirmed_visits (place, date, session_id, user_id, username, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (place, date_str, session_id, user_id, username, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def remove_confirmed_visit(place, date_str, session_id, user_id=None):
    """Remove a confirmed visit for a user or session."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if user_id:
        c.execute(
            "DELETE FROM confirmed_visits WHERE place = ? AND date = ? AND (user_id = ? OR session_id = ?)",
            (place, date_str, user_id, session_id)
        )
    else:
        c.execute(
            "DELETE FROM confirmed_visits WHERE place = ? AND date = ? AND session_id = ?",
            (place, date_str, session_id)
        )
    removed = c.rowcount > 0
    conn.commit()
    conn.close()
    return removed


def is_visit_confirmed(place, date_str, session_id, user_id=None):
    """Check if a specific session or user has confirmed a visit."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if user_id:
        c.execute(
            "SELECT COUNT(*) FROM confirmed_visits WHERE place = ? AND date = ? AND (user_id = ? OR session_id = ?)",
            (place, date_str, user_id, session_id)
        )
    else:
        c.execute(
            "SELECT COUNT(*) FROM confirmed_visits WHERE place = ? AND date = ? AND session_id = ?",
            (place, date_str, session_id)
        )
    confirmed = c.fetchone()[0] > 0
    conn.close()
    return confirmed


def get_all_confirmed_for_date(date_str):
    """Get confirmed visitor counts and usernames for all places on a date."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT place, COUNT(*) as cnt FROM confirmed_visits WHERE date = ? GROUP BY place",
        (date_str,)
    )
    result = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    return result


# ══════════════════════════════════════════════════════
# FLASK APP SETUP
# ══════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = "visit_predictor_antigravity_secret_key"

# Initialize SQLite tables
init_db()


def load_models():
    """Load all trained ML models, encoders, and metrics from disk."""
    model_dir = "models"
    required_files = [
        "crowd_model.pkl", "temp_model.pkl", "crowd_count_model.pkl",
        "dt_crowd_model.pkl", "dt_crowd_count_model.pkl",
        "gb_crowd_model.pkl", "gb_crowd_count_model.pkl",
        "encoders.pkl", "weather_lookup.pkl", "model_metrics.json"
    ]

    if not os.path.exists(model_dir):
        print("\n" + "=" * 60)
        print("  ❌ ERROR: Models not found!")
        print("  Please run: python train_model.py")
        print("=" * 60 + "\n")
        sys.exit(1)

    for f in required_files:
        if not os.path.exists(os.path.join(model_dir, f)):
            print(f"\n  ❌ ERROR: {f} not found in models/")
            print("  Please run: python train_model.py")
            sys.exit(1)

    rf_crowd = joblib.load(os.path.join(model_dir, "crowd_model.pkl"))
    dt_crowd = joblib.load(os.path.join(model_dir, "dt_crowd_model.pkl"))
    gb_crowd = joblib.load(os.path.join(model_dir, "gb_crowd_model.pkl"))

    temp_model = joblib.load(os.path.join(model_dir, "temp_model.pkl"))

    rf_cc = joblib.load(os.path.join(model_dir, "crowd_count_model.pkl"))
    dt_cc = joblib.load(os.path.join(model_dir, "dt_crowd_count_model.pkl"))
    gb_cc = joblib.load(os.path.join(model_dir, "gb_crowd_count_model.pkl"))

    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    weather_lookup = joblib.load(os.path.join(model_dir, "weather_lookup.pkl"))

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
crowd_model = loaded["classifiers"]["Gradient Boosting"]     # Best classifier: 68.5% accuracy
temp_model = loaded["temp_model"]
crowd_count_model = loaded["regressors"]["Random Forest"]     # Best regressor: R²=0.895, MAE=2771
encoders = loaded["encoders"]
weather_lookup = loaded["weather_lookup"]
model_metrics = loaded["metrics"]
classifiers = loaded["classifiers"]
regressors = loaded["regressors"]

HOURLY_PROFILES = {
    "Taj Mahal, Agra":               [12, 14, 10, 8, 7, 6, 5, 4, 4, 6, 10, 14, 10, 0, 0, 0],
    "Varanasi Ghats, Varanasi":      [18, 14, 8, 5, 4, 3, 3, 2, 2, 3, 6, 12, 16, 12, 6, 2],
    "India Gate, Delhi":             [1, 2, 3, 3, 4, 3, 2, 2, 4, 8, 14, 20, 22, 18, 12, 6],
    "Gateway of India, Mumbai":      [3, 4, 5, 6, 6, 5, 4, 4, 6, 8, 12, 16, 18, 14, 8, 4],
    "Red Fort, Delhi":               [0, 0, 0, 8, 12, 14, 15, 12, 10, 10, 12, 12, 5, 0, 0, 0],
    "Qutub Minar, Delhi":           [0, 2, 4, 8, 10, 12, 10, 8, 8, 10, 12, 14, 8, 4, 0, 0],
    "Mecca Masjid, Hyderabad":       [0, 0, 4, 6, 8, 14, 16, 10, 8, 12, 14, 8, 4, 0, 0, 0],
    "Sanchi Stupa, Madhya Pradesh":  [2, 4, 6, 10, 14, 16, 15, 12, 10, 8, 6, 4, 2, 0, 0, 0]
}


def generate_hourly_data(place, daily_count, daily_temp):
    """Generate hourly crowd counts and temperatures for the landmark."""
    hours = ["6:00 AM", "7:00 AM", "8:00 AM", "9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM",
             "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM", "5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM"]
    temp_offsets = [-4, -3, -1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, -1, -2]
    profile = HOURLY_PROFILES.get(place, [4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 4, 4, 4])
    total_weight = sum(profile)
    normalized_weights = [w / total_weight for w in profile]
    
    hourly_data = []
    for i, hour in enumerate(hours):
        h_weight = normalized_weights[i]
        h_count = int(round(daily_count * h_weight))
        h_temp = int(round(daily_temp + temp_offsets[i]))
        hourly_data.append({
            "hour": hour,
            "crowd_count": max(0, h_count),
            "temperature": h_temp
        })
    return hourly_data


def predict_ensemble(place, date_str):
    """Generate predictions from all three ensemble models using pandas DataFrame for feature names."""
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

    # Create DataFrame with exact column names to avoid sklearn UserWarnings
    features_df = pd.DataFrame([[
        place_enc, month, day_of_month, week_of_year,
        is_weekend, is_holiday, is_festival, event_enc
    ]], columns=FEATURE_COLS)

    predictions = {}
    for name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
        clf = classifiers[name]
        cl_enc = clf.predict(features_df)[0]
        cl = encoders["crowd_level"].inverse_transform([cl_enc])[0]

        reg = regressors[name]
        cc = int(round(reg.predict(features_df)[0]))
        cc = max(0, cc)

        predictions[name] = {
            "crowd_level": cl,
            "crowd_count": cc
        }

    return predictions


def fetch_live_weather(place, date_str):
    """Fetch weather from Open-Meteo API for dates within forecast range."""
    coords = PLACE_COORDINATES.get(place)
    if not coords:
        return None

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    days_ahead = (target_date - today).days

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
    except Exception as e:
        print(f"  ⚠ Weather API error: {e}")
        return None


def is_indian_holiday(date_str):
    """Check if a date string is a known Indian holiday using live calendar data."""
    try:
        year = int(date_str[:4])
    except (ValueError, IndexError):
        return None
    year_holidays = get_holidays_for_year(year)
    return year_holidays.get(date_str, None)


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
    """Determine event type for a date using live holiday calendar."""
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
    return date_obj.strftime("%A")


def generate_tips(temperature, weather, crowd_level):
    tips = []
    if temperature >= 38:
        tips.append("⚠️ Extreme Heat: High temperature of " + str(temperature) + "°C expected. Limit outdoor exposure between 12 PM - 4 PM. Wear a hat, sunglasses, and carry hydration.")
    elif temperature >= 33:
        tips.append("☀️ Warm Day: Wear lightweight clothing, sunglasses, and sunscreen. Keep a water bottle handy.")
    elif temperature <= 16:
        tips.append("❄️ Cool Day: Temperatures around " + str(temperature) + "°C. Layered clothing or a light jacket is recommended.")

    if weather == "Rainy":
        tips.append("🌧️ Rain Forecast: Bring an umbrella or raincoat. Paths and outdoor stone steps may be slippery.")
    elif weather == "Sunny":
        tips.append("🕶️ UV Protection: Sunny skies expected. Sunscreen and sunglasses recommended.")
    elif weather == "Cloudy":
        tips.append("📸 Photography Tip: Overcast skies offer soft, even lighting—perfect for clear monument photos.")

    if crowd_level in ["High", "Extreme"]:
        tips.append("🎟️ Ticket Booking: Book entry tickets online in advance to skip physical queue lines.")
        tips.append("⏰ Early Arrival: Arrive early in the morning (before 8 AM) to beat peak crowd entry queues.")
    elif crowd_level == "Low":
        tips.append("✨ Peaceful Visit: Excellent day to explore at a relaxed pace with minimal wait times.")

    return tips


def blend_prediction(ml_count, confirmed_count):
    """Blend ML predicted crowd count with confirmed visitor data."""
    if confirmed_count == 0:
        return ml_count
    estimated_real = confirmed_count * CONFIRMED_VISITOR_SCALE
    blended = int(round(ML_WEIGHT * ml_count + (1 - ML_WEIGHT) * estimated_real))
    return max(blended, estimated_real)


def predict_for_date(place, date_str):
    """Run full prediction pipeline for a given place and date string."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    month = date_obj.month
    day_of_month = date_obj.day
    week_of_year = date_obj.isocalendar()[1]
    day_of_week = get_day_of_week_string(date_obj)
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    holiday_info_data = is_indian_holiday(date_str)
    is_holiday = 1 if holiday_info_data else 0
    event_type = get_event_type(date_str, month, day_of_month)
    is_festival = 1 if event_type in ["Festival", "National Holiday", "Cultural Event"] else 0

    if holiday_info_data:
        holiday_info = f"{holiday_info_data['name']} – {holiday_info_data['type']}"
    else:
        season = is_in_festival_season(month, day_of_month)
        holiday_info = f"{season}" if season else "Regular Day"

    try:
        place_enc = encoders["Place"].transform([place])[0]
    except ValueError:
        return {"error": f"Unknown place: {place}"}

    try:
        event_enc = encoders["Event"].transform([event_type])[0]
    except ValueError:
        event_enc = encoders["Event"].transform(["Regular Day"])[0]

    # Create DataFrame with column names to fix sklearn warnings
    features_df = pd.DataFrame([[
        place_enc, month, day_of_month, week_of_year,
        is_weekend, is_holiday, is_festival, event_enc
    ]], columns=FEATURE_COLS)

    crowd_level_enc = crowd_model.predict(features_df)[0]
    crowd_level = encoders["crowd_level"].inverse_transform([crowd_level_enc])[0]

    crowd_count_ml = int(round(crowd_count_model.predict(features_df)[0]))
    crowd_count_ml = max(0, crowd_count_ml)

    # Confirmed visitors info
    conf_info = get_confirmed_info(place, date_str)
    confirmed_count = conf_info["count"]
    confirmed_users = conf_info["users"]
    
    crowd_count_predicted = blend_prediction(crowd_count_ml, confirmed_count)

    ml_temperature = int(round(temp_model.predict(features_df)[0]))

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

    visit_score = CROWD_SCORE_BASE.get(crowd_level, 50)
    visit_score += WEATHER_SCORE_PENALTY.get(weather, 0)
    if temperature > 38:
        visit_score -= 12
    elif temperature > 36:
        visit_score -= 8

    date_hash = hash(date_str) % 11 - 5
    visit_score = max(0, min(100, visit_score + date_hash))

    if visit_score >= 70:
        recommendation = "YES"
    elif visit_score >= 40:
        recommendation = "MAYBE"
    else:
        recommendation = "NO"

    crowd_count_est = CROWD_COUNT_RANGES.get(crowd_level, "Unknown")
    tips = generate_tips(temperature, weather, crowd_level)

    result = {
        "crowd_level": crowd_level,
        "crowd_count_est": crowd_count_est,
        "crowd_count_predicted": crowd_count_predicted,
        "crowd_count_ml": crowd_count_ml,
        "confirmed_visitors": confirmed_count,
        "confirmed_users": confirmed_users,
        "prediction_source": "ML + Real-Time Live Users" if confirmed_count > 0 else "ML Prediction Only",
        "temperature": temperature,
        "weather": weather,
        "weather_source": weather_source,
        "visit_score": visit_score,
        "recommendation": recommendation,
        "holiday_info": holiday_info,
        "holiday_source": "Live Calendar",
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
    all_dates.sort(key=lambda x: -x["visit_score"])
    return all_dates[:3]


# ══════════════════════════════════════════════════════
# ROUTES & AUTHENTICATION ENDPOINTS
# ══════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", places=PLACES)


@app.route("/static/index.css")
def serve_css():
    return send_from_directory("templates", "index.css", mimetype="text/css")


# ── AUTHENTICATION ROUTES ─────────────────────────────

@app.route("/register", methods=["POST"])
def auth_register():
    """Register a new user."""
    try:
        data = request.get_json() or {}
        username = data.get("username", "").strip()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not username or not email or not password:
            return jsonify({"error": "Please provide username, email, and password."}), 400

        res = register_user(username, email, password)
        if "error" in res:
            return jsonify(res), 400
        return jsonify(res)
    except Exception as e:
        print(f"  ❌ Register error: {e}")
        return jsonify({"error": "Registration failed. Please try again."}), 500


@app.route("/login", methods=["POST"])
def auth_login():
    """Log in an existing user."""
    try:
        data = request.get_json() or {}
        login_id = data.get("login_id", "").strip()
        password = data.get("password", "").strip()

        if not login_id or not password:
            return jsonify({"error": "Please enter your username/email and password."}), 400

        res = authenticate_user(login_id, password)
        if "error" in res:
            return jsonify(res), 400
        return jsonify(res)
    except Exception as e:
        print(f"  ❌ Login error: {e}")
        return jsonify({"error": "Login failed. Please try again."}), 500


@app.route("/current-user", methods=["POST"])
def current_user_route():
    """Get profile info for a user ID."""
    try:
        data = request.get_json() or {}
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"user": None})
        user = get_user_by_id(user_id)
        return jsonify({"user": user})
    except Exception as e:
        return jsonify({"user": None})


# ── PREDICTION & VISITOR ROUTES ───────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received."}), 400

        place = data.get("place", "").strip()
        date_str = data.get("date", "").strip()
        session_id = data.get("session_id", "")
        user_id = data.get("user_id")

        if not place or not date_str:
            return jsonify({"error": "Please select a destination and date."}), 400
        if place not in PLACES:
            return jsonify({"error": f"Unknown destination: {place}"}), 400

        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        result = predict_for_date(place, date_str)
        if "error" in result:
            return jsonify(result), 400

        result["better_dates"] = find_better_dates(place, date_str)
        ensemble = predict_ensemble(place, date_str)
        hourly = generate_hourly_data(place, result["crowd_count_predicted"], result["temperature"])
        
        # Filter hourly data to only include actual operating hours
        op_hours = PLACE_OPERATING_HOURS.get(place, {"open": 6, "close": 21, "closed_day": None})
        open_h, close_h = op_hours["open"], op_hours["close"]

        # Parse hour strings like "7:00 AM" to 24h integers for filtering
        def hour_str_to_24(h_str):
            """Convert '7:00 AM' or '2:00 PM' to 24-hour int."""
            parts = h_str.replace(":", " ").split()
            hr = int(parts[0])
            ampm = parts[2]
            if ampm == "PM" and hr != 12:
                hr += 12
            elif ampm == "AM" and hr == 12:
                hr = 0
            return hr

        # Filter to hours within operating window
        if open_h == 0 and close_h == 24:
            # 24-hour place — use all hours but prefer reasonable tourist hours
            open_hours = [h for h in hourly if 6 <= hour_str_to_24(h["hour"]) <= 20]
        else:
            open_hours = [h for h in hourly if open_h <= hour_str_to_24(h["hour"]) < close_h]

        if not open_hours:
            open_hours = hourly  # fallback

        best_hour_data = min(open_hours, key=lambda x: (x["crowd_count"], x["temperature"]))

        # Add closure warning if the place is closed on this day
        closed_day = op_hours.get("closed_day")
        if closed_day:
            try:
                visit_day = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
                if visit_day == closed_day:
                    result["closure_warning"] = f"⚠️ {place.split(',')[0]} is closed on {closed_day}s!"
            except ValueError:
                pass

        # Check confirmation status for this user or session
        result["user_confirmed"] = is_visit_confirmed(place, date_str, session_id, user_id=user_id)

        result["ensemble_predictions"] = ensemble
        result["model_metrics"] = model_metrics
        result["hourly_distribution"] = hourly
        result["best_time_to_visit"] = best_hour_data["hour"]
        result["operating_hours"] = f"{op_hours['open']}:00 AM – {op_hours['close'] if op_hours['close'] <= 12 else op_hours['close']-12}:00 {'AM' if op_hours['close'] <= 12 else 'PM'}" if not (open_h == 0 and close_h == 24) else "Open 24 Hours"

        return jsonify(result)

    except Exception as e:
        print(f"  ❌ Prediction error: {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


@app.route("/confirm-visit", methods=["POST"])
def confirm_visit():
    """Handle visit confirmation. Requires user authentication."""
    try:
        data = request.get_json() or {}
        place = data.get("place", "").strip()
        date_str = data.get("date", "").strip()
        session_id = data.get("session_id", "").strip()
        user_id = data.get("user_id")
        username = data.get("username", "").strip()

        if not place or not date_str or not session_id:
            return jsonify({"error": "Missing required details (place, date, session)."}), 400
        if place not in PLACES:
            return jsonify({"error": f"Unknown destination: {place}"}), 400

        # Require login for real-time user confirmation if user_id is missing
        if not user_id or not username:
            return jsonify({
                "login_required": True,
                "error": "Please sign in or create an account to confirm your visit and help others get real-time crowd predictions!"
            }), 401

        is_new = add_confirmed_visit(place, date_str, session_id, user_id=user_id, username=username)
        conf_info = get_confirmed_info(place, date_str)

        return jsonify({
            "success": True,
            "is_new": is_new,
            "confirmed_count": conf_info["count"],
            "confirmed_users": conf_info["users"],
            "message": f"Awesome {username}! Your visit is confirmed for {date_str}."
        })

    except Exception as e:
        print(f"  ❌ Confirm visit error: {e}")
        return jsonify({"error": "Something went wrong."}), 500


@app.route("/cancel-visit", methods=["POST"])
def cancel_visit():
    """Handle visit cancellation."""
    try:
        data = request.get_json() or {}
        place = data.get("place", "").strip()
        date_str = data.get("date", "").strip()
        session_id = data.get("session_id", "").strip()
        user_id = data.get("user_id")

        if not place or not date_str:
            return jsonify({"error": "Missing place or date."}), 400

        removed = remove_confirmed_visit(place, date_str, session_id, user_id=user_id)
        conf_info = get_confirmed_info(place, date_str)

        return jsonify({
            "success": True,
            "removed": removed,
            "confirmed_count": conf_info["count"],
            "confirmed_users": conf_info["users"],
            "message": "Visit cancelled."
        })

    except Exception as e:
        print(f"  ❌ Cancel visit error: {e}")
        return jsonify({"error": "Something went wrong."}), 500


@app.route("/live-stats", methods=["POST"])
def live_stats():
    """Return live confirmed visitor stats."""
    try:
        data = request.get_json() or {}
        date_str = data.get("date", "").strip()
        place = data.get("place", "").strip()

        if not date_str:
            return jsonify({"error": "Missing date."}), 400

        if place:
            conf_info = get_confirmed_info(place, date_str)
            return jsonify({
                "date": date_str,
                "place": place,
                "confirmed_count": conf_info["count"],
                "confirmed_users": conf_info["users"],
                "estimated_real": conf_info["count"] * CONFIRMED_VISITOR_SCALE
            })
        else:
            all_stats = get_all_confirmed_for_date(date_str)
            return jsonify({
                "date": date_str,
                "places": {p: {"confirmed": c, "estimated": c * CONFIRMED_VISITOR_SCALE} for p, c in all_stats.items()}
            })

    except Exception as e:
        print(f"  ❌ Live stats error: {e}")
        return jsonify({"error": "Something went wrong."}), 500


@app.route("/compare", methods=["POST"])
def compare():
    """Handle landmark comparisons."""
    try:
        data = request.get_json() or {}
        places = data.get("places", [])
        date_str = data.get("date", "").strip()

        if not places or not isinstance(places, list) or len(places) < 2 or len(places) > 3:
            return jsonify({"error": "Please select 2 or 3 landmarks."}), 400
        if not date_str:
            return jsonify({"error": "Please select a date."}), 400

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

        results.sort(key=lambda x: -x["visit_score"])
        for idx, res in enumerate(results):
            res["is_best"] = (idx == 0)

        return jsonify(results)

    except Exception as e:
        print(f"  ❌ Comparison error: {e}")
        return jsonify({"error": "Something went wrong."}), 500


@app.route("/chart-data", methods=["POST"])
def chart_data():
    """Return monthly crowd trend data."""
    try:
        data = request.get_json() or {}
        place = data.get("place", "").strip()
        date_str = data.get("date", "").strip()

        if place not in PLACES:
            return jsonify({"error": f"Unknown place: {place}"}), 400

        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        year = target_date.year
        month = target_date.month

        import calendar
        days_in_month = calendar.monthrange(year, month)[1]

        dates, crowd_counts, crowd_levels, temperatures, visit_scores = [], [], [], [], []

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


@app.route("/holidays", methods=["POST"])
def get_holidays():
    """Return holiday data for a given year."""
    try:
        data = request.get_json() or {}
        year = data.get("year", datetime.now().year)
        year_holidays = get_holidays_for_year(int(year))
        return jsonify({
            "year": year,
            "holidays": year_holidays,
            "count": len(year_holidays),
            "source": "Live Calendar (holidays library + Nager.Date API)"
        })
    except Exception as e:
        print(f"  ❌ Holidays error: {e}")
        return jsonify({"error": "Something went wrong."}), 500


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    current_year = datetime.now().year
    get_holidays_for_year(current_year)
    get_holidays_for_year(current_year + 1)
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " Should I Visit? — Web App Running ".center(58) + "║")
    print("║" + " Open: http://localhost:5000 ".center(58) + "║")
    print("║" + " 👤 User Accounts · 👥 Live Confirmed Visitors ".center(58) + "║")
    print("╚" + "═" * 58 + "╝\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
