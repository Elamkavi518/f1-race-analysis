# 🏎️ F1 Race Strategy, Analysis & Prediction

A Python project that analyses real Formula 1 race data using the official FastF1 API — covering lap times, tyre strategy, telemetry, and a machine learning model to predict race finishing positions.

---

## 📊 What It Does

| # | Analysis | Description |
|---|----------|-------------|
| 1 | **Lap Time Comparison** | Race pace of selected drivers across all laps |
| 2 | **Tyre Strategy Map** | Pit stop timing and compound choices for all drivers |
| 3 | **Speed & Telemetry** | Speed, throttle, and brake traces on the fastest lap |
| 4 | **Pace Consistency** | Box plot showing average pace and consistency |
| 5 | **Position Changes** | How driver positions changed lap by lap |
| 6 | **ML Prediction Model** | Gradient Boosting model to predict finishing position |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **FastF1** — Official F1 timing & telemetry API
- **Pandas** — Data processing
- **Matplotlib** — Visualisations
- **Scikit-learn** — Machine learning (Gradient Boosting Regressor)

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install fastf1 pandas matplotlib scikit-learn

# 2. Run the script
python f1_analysis.py
```

---

## ✏️ Customise It

At the top of `f1_analysis.py`, change these variables:

```python
YEAR    = 2024
RACE    = "Bahrain"          # Monaco | Silverstone | Monza | Abu Dhabi
DRIVERS = ["VER", "LEC", "HAM"]
```

For the custom prediction:

```python
GRID_POS  = 3      # Starting position
AVG_LAP   = 95.8   # Expected lap time (seconds)
PIT_COUNT = 2      # Number of pit stops
```

---

## 📁 Output Files

Running the script saves 6 chart images in the same folder:

```
1_lap_times.png
2_tyre_strategy.png
3_telemetry.png
4_pace_consistency.png
5_position_changes.png
6_feature_importance.png
```

---

## 💡 Key Concepts Covered

- **Undercut** — Pitting before a rival to gain track position on fresh tyres
- **Overcut** — Staying out longer while rival loses pace in traffic
- **Tyre degradation** — Rising lap times as rubber wears
- **DRS** — Speed advantage on straights visible in speed traces
- **ML prediction** — Using grid position, avg lap time & pit stops to predict finish

---

## 📌 Data Source

Data is sourced from the [FastF1](https://docs.fastf1.dev/) Python library, which pulls official F1 timing and telemetry data. FastF1 is unofficial and not associated with Formula 1 companies.

---

## 🧑‍💻 Author

**Your Name**  
Built as a one-day F1 data science project.  
Connect on [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)
