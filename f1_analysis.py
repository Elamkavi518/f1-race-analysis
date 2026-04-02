import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import fastf1
import fastf1.plotting

warnings.filterwarnings("ignore")
fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

# ============================================================
# CHANGE THESE to analyse any race / drivers you want!
# ============================================================
YEAR    = 2024
RACE    = "Bahrain"          # Monaco | Silverstone | Monza | Abu Dhabi
DRIVERS = ["VER", "LEC", "HAM"]

BG     = "#0d0d0d"
ACCENT = "#e8002d"
GRID_A = 0.15

# ============================================================
# HELPERS
# ============================================================
def to_sec(series):
    """Convert LapTime (timedelta or float) to float seconds."""
    try:
        out = pd.to_timedelta(series, errors="coerce").dt.total_seconds()
        if out.dropna().max() > 10:
            return out
    except Exception:
        pass
    return pd.to_numeric(series, errors="coerce")

def dark(ax, title="", xlabel="", ylabel=""):
    """Apply dark theme to a matplotlib axis."""
    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    if title:  ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=GRID_A, linestyle="--")

# ============================================================
# 0 - LOAD SESSION
# ============================================================
print(f"\n{'='*55}")
print(f"  F1 Analysis  |  {YEAR} {RACE} Grand Prix")
print(f"{'='*55}")
print("Loading session ... (first load ~60 s)\n")

session = fastf1.get_session(YEAR, RACE, "R")
session.load()

event = session.event
print(f"  Event  : {event['EventName']}")
print(f"  Date   : {event['EventDate'].strftime('%d %B %Y')}")
print(f"  Track  : {event['Location']}, {event['Country']}\n")

# ============================================================
# 1 - RACE RESULTS TABLE
# ============================================================
print("RACE RESULTS\n" + "-"*50)
res = session.results[["Abbreviation","FullName","TeamName","Position","Points","Status"]].copy()
res["Position"] = pd.to_numeric(res["Position"], errors="coerce").fillna(99).astype(int)
res = res.sort_values("Position").reset_index(drop=True)
for _, r in res.iterrows():
    print(f"  P{int(r['Position']):02d}  {r['Abbreviation']}  "
          f"{r['FullName']:<22}  {r['TeamName']:<28}  "
          f"{int(r['Points']):>2} pts  {r['Status']}")

# ============================================================
# 2 - LAP TIME COMPARISON
# ============================================================
print("\nBuilding Lap Time chart ...")

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor(BG)

for drv in DRIVERS:
    try:
        laps  = session.laps.pick_driver(drv).pick_quicklaps().copy()
        times = to_sec(laps["LapTime"])
        mask  = (times > 60) & (times < 200)
        color = fastf1.plotting.driver_color(drv)
        ax.plot(
            laps["LapNumber"][mask],
            times[mask],
            label=drv, color=color,
            linewidth=2, marker="o", markersize=3, zorder=3
        )
    except Exception as e:
        print(f"  WARNING {drv}: {e}")

dark(ax,
     title=f"Lap Time Comparison - {YEAR} {RACE} GP",
     xlabel="Lap Number",
     ylabel="Lap Time (seconds)")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f s"))
ax.legend(facecolor="#1a1a1a", labelcolor="white",
          fontsize=11, framealpha=0.9, edgecolor="#444")
plt.tight_layout()
plt.savefig("1_lap_times.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("  Saved: 1_lap_times.png")

# ============================================================
# 3 - TYRE STRATEGY MAP
# ============================================================
print("\nBuilding Tyre Strategy chart ...")

TYRE_COLOR = {
    "SOFT":         "#e8002d",
    "MEDIUM":       "#ffd700",
    "HARD":         "#ebebeb",
    "INTERMEDIATE": "#39b54a",
    "WET":          "#0067ff",
}

top15    = session.results.sort_values("Position")["Abbreviation"].tolist()[:15]
all_laps = session.laps

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for i, drv in enumerate(top15):
    try:
        dlaps = all_laps.pick_driver(drv)[["LapNumber","Compound"]].dropna()
        for _, lap in dlaps.iterrows():
            c = TYRE_COLOR.get(str(lap["Compound"]).upper(), "#888")
            ax.barh(i, 1, left=lap["LapNumber"] - 1,
                    color=c, alpha=0.9, height=0.65, edgecolor="none")
    except Exception:
        pass

ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15, color="white", fontsize=10)
ax.set_xlabel("Lap Number", color="white", fontsize=11)
ax.set_title(f"Tyre Strategy Map - {YEAR} {RACE} GP",
             color="white", fontsize=14, fontweight="bold", pad=12)
ax.tick_params(colors="white")
ax.grid(True, axis="x", alpha=GRID_A, linestyle="--")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
patches = [mpatches.Patch(color=v, label=k) for k, v in TYRE_COLOR.items()]
ax.legend(handles=patches, loc="lower right",
          facecolor="#1a1a1a", labelcolor="white",
          fontsize=10, framealpha=0.9, edgecolor="#444")
plt.tight_layout()
plt.savefig("2_tyre_strategy.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("  Saved: 2_tyre_strategy.png")

# ============================================================
# 4 - SPEED + THROTTLE + BRAKE TELEMETRY
# ============================================================
print("\nBuilding Telemetry chart ...")

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(BG)
gs     = GridSpec(3, 1, figure=fig, hspace=0.06)
ax_spd = fig.add_subplot(gs[0])
ax_thr = fig.add_subplot(gs[1], sharex=ax_spd)
ax_brk = fig.add_subplot(gs[2], sharex=ax_spd)

for drv in DRIVERS:
    try:
        fl  = session.laps.pick_driver(drv).pick_fastest()
        tel = fl.get_telemetry()
        col = fastf1.plotting.driver_color(drv)
        dist     = pd.to_numeric(tel["Distance"],  errors="coerce")
        speed    = pd.to_numeric(tel["Speed"],     errors="coerce")
        throttle = pd.to_numeric(tel["Throttle"],  errors="coerce")
        brake    = pd.to_numeric(tel["Brake"],     errors="coerce")
        ax_spd.plot(dist, speed,    color=col, linewidth=1.6, label=drv)
        ax_thr.plot(dist, throttle, color=col, linewidth=1.4, alpha=0.85)
        ax_brk.plot(dist, brake,    color=col, linewidth=1.4, alpha=0.85)
    except Exception as e:
        print(f"  WARNING {drv}: {e}")

dark(ax_spd, title=f"Fastest Lap Telemetry - {YEAR} {RACE} GP", ylabel="Speed (km/h)")
dark(ax_thr, ylabel="Throttle (%)")
dark(ax_brk, ylabel="Brake", xlabel="Distance (m)")
ax_spd.legend(facecolor="#1a1a1a", labelcolor="white",
              fontsize=11, framealpha=0.9, edgecolor="#444")
plt.setp(ax_spd.get_xticklabels(), visible=False)
plt.setp(ax_thr.get_xticklabels(), visible=False)
plt.savefig("3_telemetry.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("  Saved: 3_telemetry.png")

# ============================================================
# 5 - PIT STOP ANALYSIS (text)
# ============================================================
print("\nPIT STOP ANALYSIS\n" + "-"*50)
for drv in DRIVERS:
    try:
        dl   = session.laps.pick_driver(drv)
        pits = dl[dl["PitOutTime"].notna()]["LapNumber"].tolist()
        comp = dl.groupby("Stint")["Compound"].first().tolist()
        sts  = dl.groupby("Stint")["LapNumber"].count().tolist()
        print(f"\n  {drv}")
        print(f"    Pit laps  : {pits}")
        print(f"    Compounds : {' -> '.join(str(c) for c in comp)}")
        print(f"    Stints    : {sts} laps  |  Strategy: {len(pits)}-STOP")
    except Exception:
        print(f"  WARNING: No pit data for {drv}")

# ============================================================
# 6 - PACE CONSISTENCY BOX PLOT
# ============================================================
print("\nBuilding Pace Consistency chart ...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

pace_data, labels, colors = [], [], []
for drv in DRIVERS:
    try:
        laps  = session.laps.pick_driver(drv).pick_quicklaps()
        times = to_sec(laps["LapTime"]).dropna()
        times = times[(times > 60) & (times < 200)]
        if len(times) > 5:
            pace_data.append(times.values)
            labels.append(drv)
            colors.append(fastf1.plotting.driver_color(drv))
    except Exception:
        pass

if pace_data:
    bp = ax.boxplot(
        pace_data, labels=labels, patch_artist=True, widths=0.5,
        medianprops={"color": "white", "linewidth": 2.5},
        whiskerprops={"color": "#aaa"},
        capprops={"color": "#aaa"},
        flierprops={"marker": "o", "markersize": 4,
                    "markerfacecolor": "#aaa", "alpha": 0.5}
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)

dark(ax,
     title=f"Pace Consistency - {YEAR} {RACE} GP",
     xlabel="Driver",
     ylabel="Lap Time (seconds)")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f s"))
plt.tight_layout()
plt.savefig("4_pace_consistency.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("  Saved: 4_pace_consistency.png")

# ============================================================
# 7 - POSITION CHANGES CHART
# ============================================================
print("\nBuilding Position Changes chart ...")

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for drv in DRIVERS:
    try:
        dl  = session.laps.pick_driver(drv)[["LapNumber","Position"]].dropna()
        col = fastf1.plotting.driver_color(drv)
        ax.plot(dl["LapNumber"], dl["Position"],
                label=drv, color=col, linewidth=2.5,
                marker="o", markersize=2.5, zorder=3)
    except Exception as e:
        print(f"  WARNING {drv}: {e}")

ax.invert_yaxis()
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
dark(ax,
     title=f"Race Position Changes - {YEAR} {RACE} GP",
     xlabel="Lap Number",
     ylabel="Position")
ax.legend(facecolor="#1a1a1a", labelcolor="white",
          fontsize=11, framealpha=0.9, edgecolor="#444")
plt.tight_layout()
plt.savefig("5_position_changes.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("  Saved: 5_position_changes.png")

# ============================================================
# 8 - ML PREDICTION MODEL
# ============================================================
print("\nTraining ML Race Prediction Model ...")

TRAIN_RACES = [
    (2024, "Bahrain"), (2024, "Saudi Arabia"), (2024, "Australia"),
    (2024, "Japan"),   (2024, "China"),         (2024, "Miami"),
    (2024, "Monaco"),  (2024, "Canada"),
]

rows = []
for yr, rc in TRAIN_RACES:
    try:
        s = fastf1.get_session(yr, rc, "R")
        s.load(telemetry=False, weather=False, messages=False)
        for _, row in s.results.iterrows():
            try:
                drv  = row["Abbreviation"]
                dl   = s.laps.pick_driver(drv)
                avg  = to_sec(dl["LapTime"]).dropna()
                avg  = avg[(avg > 60) & (avg < 200)].mean()
                pits = int(dl["PitOutTime"].notna().sum())
                grid = int(pd.to_numeric(row.get("GridPosition", 10), errors="coerce") or 10)
                fin  = int(pd.to_numeric(row.get("Position", 20),     errors="coerce") or 20)
                if not np.isnan(avg):
                    rows.append({"avg_lap": avg, "grid": grid,
                                 "pits": pits,   "finish": fin})
            except Exception:
                pass
        print(f"  Loaded: {yr} {rc}")
    except Exception as e:
        print(f"  Skipped {rc}: {e}")

df = pd.DataFrame(rows)
print(f"\n  Training samples: {len(df)}")

model = None
if len(df) >= 20:
    X = df[["avg_lap","grid","pits"]]
    y = df["finish"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3,
        learning_rate=0.05, random_state=42
    )
    model.fit(Xtr, ytr)
    mae = mean_absolute_error(yte, model.predict(Xte))
    print(f"  Model MAE: +/-{mae:.2f} positions")

    # Feature importance chart
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    feats = ["Avg Lap Time", "Grid Position", "Pit Stops"]
    imp   = model.feature_importances_
    bars  = ax.barh(feats, imp * 100,
                    color=[ACCENT, "#ffd700", "#39b54a"],
                    edgecolor="none", height=0.5)
    for bar, val in zip(bars, imp):
        ax.text(bar.get_width() + 0.4,
                bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center",
                color="white", fontsize=10, fontweight="bold")
    dark(ax,
         title="What Predicts Race Finishing Position?",
         xlabel="Feature Importance (%)")
    ax.set_xlim(0, max(imp * 100) + 15)
    plt.tight_layout()
    plt.savefig("6_feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.show()
    print("  Saved: 6_feature_importance.png")

# ============================================================
# 9 - CUSTOM PREDICTION
# ============================================================
print("\nCUSTOM RACE PREDICTION")
print("-" * 40)

# CHANGE THESE values to predict any scenario!
GRID_POS  = 3      # Starting grid position (1-20)
AVG_LAP   = 95.8   # Expected average lap time in seconds
PIT_COUNT = 2      # Number of pit stops planned

if model:
    inp  = pd.DataFrame([{"avg_lap": AVG_LAP, "grid": GRID_POS, "pits": PIT_COUNT}])
    pred = int(np.clip(round(model.predict(inp)[0]), 1, 20))
    print(f"  Grid position : P{GRID_POS}")
    print(f"  Avg lap time  : {AVG_LAP}s")
    print(f"  Pit stops     : {PIT_COUNT}")
    print("-" * 40)
    print(f"  Predicted finish -> P{pred}")
    if pred <= 3:    print("  PODIUM FINISH!")
    elif pred <= 10: print("  Points finish!")
    else:            print("  Outside points - rethink strategy!")
else:
    print("  Model not ready - check step 8 output above")

# ============================================================
# FINAL SUMMARY
# ============================================================
charts = [f for f in [
    "1_lap_times.png", "2_tyre_strategy.png", "3_telemetry.png",
    "4_pace_consistency.png", "5_position_changes.png", "6_feature_importance.png"
] if os.path.exists(f)]

print(f"\n{'='*55}")
print("  ALL DONE!")
print(f"{'='*55}")
print(f"\n  {len(charts)} charts saved:")
for c in charts:
    print(f"    {c}")
print("""
  What you built:
    - Real F1 lap time race pace comparison
    - Full tyre strategy map (all drivers)
    - Speed + Throttle + Brake telemetry
    - Pace consistency box plot
    - Race position changes chart
    - ML race outcome prediction model (GBR)
    - Custom scenario predictor

  GitHub README tags:
    Python | FastF1 | Pandas | Matplotlib | Scikit-learn
""")
print("="*55)