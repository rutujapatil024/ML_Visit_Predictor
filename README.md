# "Should I Visit?" — Indian Landmark Crowd Predictor 

**"Should I Visit?"** is a machine learning-powered web application that helps tourists decide the best time to visit famous Indian landmarks. The system predicts crowd levels, exact visitor counts, temperatures, and weather conditions for any queried landmark and date, providing an actionable recommendation (**YES**, **MAYBE**, or **NO**).

---

## Key Features

- **Tuned Machine Learning Engine (Gradient Boosting Primary)**: Powered by a high-accuracy **Gradient Boosting Classifier (68.5% test accuracy)** for crowd level classification and a **Random Forest Regressor (R² = 0.895, MAE = 2,771)** for exact visitor volume prediction.
- **Multi-Model Ensemble Engine**: Simultaneously runs Gradient Boosting, Random Forest, and Decision Tree side-by-side, displaying cross-model confidence metrics, accuracy, and predictions in parallel.
- **Real Landmark Operating Hours & Closure Verification**: Cross-references real-world opening/closing hours and closed days for each monument (e.g. Taj Mahal closed on Fridays, Red Fort closed on Mondays). Guarantees that the **Optimal Visit Time** is only suggested during hours when the landmark is actually open.
- **Dynamic Live Calendar (No Hardcoded Dates)**: Automatically calculates Indian public holidays and major festivals for any year (2026, 2027, 2028, etc.) using the Python `holidays` library + Nager.Date API fallback.
- **User Authentication & Real-Time Confirmed Visitors**: Secure user registration and login with PBKDF2/SHA-256 password hashing. Confirmed visits are stored in SQLite (`visitors.db`) and blended dynamically with ML baseline predictions (60% ML / 40% Live User Intent).
- **Live Weather Integration**: Queries the Open-Meteo API for live 16-day temperature and WMO weather code forecasts (with statistical ML fallback for dates beyond 16 days).
- **Interactive Crowd Trend Charts**: Displays monthly visitor trends and hourly density curves (using Chart.js) to pinpoint the single optimal hour to visit within open operating times.
- **Landmark Comparison Drawer**: Enables side-by-side comparison of 2 or 3 destinations for any date to find the best option.
- **Top 3 Better Dates**: Scans the next 60 days to automatically suggest alternative dates with the best visit scores.

---

## Project Architecture

```
Visit_Predictor/
├── app.py                      # Flask backend API, auth routes & prediction pipeline
├── train_model.py              # ML model training script (creates models/)
├── learn_visit_predictor.ipynb # Interactive Jupyter Notebook for EDA & model exploration
├── data.xlsx                   # 2-year daily historical dataset (730 observations)
├── requirements.txt            # Python dependencies (Flask, scikit-learn, holidays, etc.)
├── explanation.txt             # Comprehensive project documentation
├── visitors.db                 # SQLite database for user accounts & confirmed visits
├── templates/
│   ├── index.html              # Main single-page web app dashboard & modal auth
│   └── index.css               # Modern dark-mode glassmorphism stylesheet
└── models/                     # Trained ML models (GB, RF, DT), encoders & lookups
```

---

## Operating Hours & Landmark Details

| Landmark | Hours | Closed Day |
|---|---|---|
| **Taj Mahal, Agra** | 6:00 AM – 6:00 PM | Fridays |
| **Red Fort, Delhi** | 9:00 AM – 4:00 PM | Mondays |
| **India Gate, Delhi** | Open 24 Hours | Open Daily |
| **Qutub Minar, Delhi** | 7:00 AM – 5:00 PM | Open Daily |
| **Varanasi Ghats, Varanasi** | Open 24 Hours | Open Daily |
| **Gateway of India, Mumbai** | Open 24 Hours | Open Daily |
| **Mecca Masjid, Hyderabad** | 4:00 AM – 9:00 PM | Open Daily |
| **Sanchi Stupa, Madhya Pradesh** | 8:00 AM – 6:00 PM | Open Daily |

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train / Refresh Models** *(Optional)*:
   ```bash
   python train_model.py
   ```

3. **Start the Flask Server**:
   ```bash
   python app.py
   ```

4. **Open in Browser**:
   Navigate to `http://localhost:5000`

