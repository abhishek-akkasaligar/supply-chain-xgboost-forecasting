"""
Generates a synthetic demand forecasting dataset for the public portfolio repo.
Structure mirrors the real production pipeline (TE Connectivity Korea->EU lanes)
but every material number, customer, and quantity is fabricated. Safe to publish.

Run: python scripts/generate_sample_data.py
Output: data/sample/weekly_demand.csv, data/sample/material_master.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_MATERIALS = 60          # smaller than the real 531, enough to demo segmentation
N_WEEKS = 104              # 2 years of weekly history
START_DATE = "2024-01-07"  # a Sunday, week-start convention

dates = pd.date_range(START_DATE, periods=N_WEEKS, freq="W")

# ---------------------------------------------------------------------------
# 1. Material master — assign each fake material a demand "personality"
# ---------------------------------------------------------------------------
segments = (
    ["high_runner"] * 8
    + ["regular_mover"] * 10
    + ["lumpy"] * 12
    + ["sporadic"] * 15
    + ["mto"] * 15
)
np.random.shuffle(segments)

materials = [f"SAMPLE-MAT-{i:04d}" for i in range(1, N_MATERIALS + 1)]

material_master = pd.DataFrame({
    "material": materials,
    "segment": segments,
    "planned_lead_time_days": np.random.choice([45, 60, 79, 90, 120], N_MATERIALS),
    "abc_indicator": np.random.choice(["A", "B", "C"], N_MATERIALS, p=[0.15, 0.35, 0.5]),
    "transport_mode": np.random.choice(["OCEAN", "AIR"], N_MATERIALS, p=[0.7, 0.3]),
})

# ---------------------------------------------------------------------------
# 2. Weekly demand history — shape depends on segment
# ---------------------------------------------------------------------------
rows = []
for mat, seg in zip(materials, segments):
    base = np.random.randint(200, 2000)

    if seg == "high_runner":
        weeks = np.arange(N_WEEKS)
        # Smooth upward/downward trend over 2 years
        trend = base * np.random.uniform(-0.002, 0.003) * weeks
        # Strong annual seasonality (52-week cycle) -- the signal a model
        # with week_sin/week_cos + 52-week lag can learn, but a naive
        # "repeat last week" forecast cannot.
        annual = base * 0.4 * np.sin(2 * np.pi * weeks / 52 + np.random.uniform(0, 2 * np.pi))
        # Small amount of genuine random noise
        noise = np.random.normal(0, base * 0.04, N_WEEKS)
        qty = base + trend + annual + noise
        qty = np.clip(qty, base * 0.2, None)

    elif seg == "regular_mover":
        noise = np.random.normal(0, 0.2, N_WEEKS)
        qty = base * (1 + noise)
        qty = np.clip(qty, 0, None)

    elif seg == "lumpy":
        active = np.random.binomial(1, 0.5, N_WEEKS)
        qty = active * base * np.random.uniform(0.5, 2.5, N_WEEKS)

    elif seg == "sporadic":
        active = np.random.binomial(1, 0.25, N_WEEKS)
        qty = active * base * np.random.uniform(0.3, 3.0, N_WEEKS)

    else:  # mto - mostly zero, occasional order-driven spike
        active = np.random.binomial(1, 0.08, N_WEEKS)
        qty = active * base * np.random.uniform(1.0, 4.0, N_WEEKS)

    for d, q in zip(dates, qty):
        rows.append((mat, d, max(0, round(q))))

weekly_demand = pd.DataFrame(rows, columns=["material", "week_start", "shipped_qty"])

# ---------------------------------------------------------------------------
# 3. Forecast + stock snapshot (as-of last week, for the stock-risk check)
# ---------------------------------------------------------------------------
latest = weekly_demand[weekly_demand["week_start"] == dates[-1]].copy()
avg_recent = (
    weekly_demand[weekly_demand["week_start"] >= dates[-13]]
    .groupby("material")["shipped_qty"].mean()
    .rename("avg_weekly_demand_13wk")
)

stock_snapshot = material_master[["material"]].merge(avg_recent, on="material", how="left")
stock_snapshot["avg_weekly_demand_13wk"] = stock_snapshot["avg_weekly_demand_13wk"].fillna(0)
stock_snapshot["stock_on_hand"] = (
    stock_snapshot["avg_weekly_demand_13wk"] * np.random.uniform(0.5, 8, N_MATERIALS)
).round()
stock_snapshot["weeks_of_cover"] = np.where(
    stock_snapshot["avg_weekly_demand_13wk"] > 0,
    (stock_snapshot["stock_on_hand"] / stock_snapshot["avg_weekly_demand_13wk"]).round(1),
    np.nan,
)

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
material_master.to_csv("data/sample/material_master.csv", index=False)
weekly_demand.to_csv("data/sample/weekly_demand.csv", index=False)
stock_snapshot.to_csv("data/sample/stock_snapshot.csv", index=False)

print(f"Generated {len(materials)} materials, {len(weekly_demand)} weekly demand rows")
print("Saved to data/sample/")
