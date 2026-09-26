# "Should I Visit?" — Indian Landmark Crowd Predictor 

**"Should I Visit?"** is a machine learning-powered web application that helps tourists decide the best time to visit famous Indian landmarks. The system predicts crowd levels, exact visitor counts, temperatures, and weather conditions for any queried landmark and date, providing an actionable recommendation (**YES**, **MAYBE**, or **NO**).

---

## Key Features

- **Dynamic Live Calendar (No Hardcoded Dates)**: Automatically calculates Indian public holidays and major festivals for any year (2026, 2027, 2028, etc.) using the Python `holidays` library + Nager.Date API fallback.
- **Real-Time Confirmed Visitor System**: Allows users to confirm their planned visits via an interactive card. The backend SQLite database (`visitors.db`) tracks confirmed visitors and dynamically blends real-time user intent with ML baseline predictions.
- **Multi-Model Ensemble Engine**: Computes predictions using Random Forest, Decision Tree, and Gradient Boosting models side-by-side, displaying confidence metrics and predictions in parallel.
- **Live Weather Integration**: Queries the Open-Meteo API for live 16-day temperature and WMO weather code forecasts (with statistical ML fallback for dates beyond 16 days).
- **Interactive Crowd Trend Charts**: Displays monthly visitor trends and hourly density curves (using Chart.js) to pinpoint the single optimal hour to visit.
- **Landmark Comparison Drawer**: Enables side-by-side comparison of 2 or 3 destinations for any date to find the best option.
- **Top 3 Better Dates**: Scans the next 60 days to automatically suggest alternative dates with the best visit scores.

---

## Project Architecture

```
Visit_Predictor/
├── app.py                      # Flask backend API & prediction pipeline
├── train_model.py              # ML model training script (creates models/)
├── data.xlsx                   # 2-year daily historical dataset (730 observations)
├── requirements.txt            # Python dependencies (Flask, scikit-learn, holidays, etc.)
├── explanation.txt             # Comprehensive project documentation
├── visitors.db                 # SQLite database for confirmed user visits
├── templates/
│   ├── index.html              # Main single-page web app dashboard
│   └── index.css               # Modern dark-mode glassmorphism stylesheet
└── models/                     # Trained ML models, encoders & weather lookups
```

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
