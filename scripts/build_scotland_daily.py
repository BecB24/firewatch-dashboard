import os
import numpy as np
import pandas as pd

# ---------- paths ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
YEARLY_IN  = os.path.join(DATA_DIR, "fire_scotland_yearly.csv")
YEARLY_OUT = os.path.join(DATA_DIR, "fire_scotland_yearly_clean.csv")
DAILY_OUT  = os.path.join(DATA_DIR, "fire_scotland_daily.csv")

def load_yearly():
    """Load your Excel-exported CSV and clean the columns."""
    df = pd.read_csv(YEARLY_IN)
    # normalise headers
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # harmonise expected names
    if "year" not in df.columns:
        raise ValueError("CSV needs a 'Year' column.")
    # your second column is 'Incidents'
    inc_col = "incidents" if "incidents" in df.columns else None
    if inc_col is None:
        raise ValueError("CSV needs an 'Incidents' column.")

    # remove commas, turn '..' into NaN, drop missing
    s = (
        df[inc_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"..": np.nan, "": np.nan})
    )
    df["incidents"] = pd.to_numeric(s, errors="coerce")

    df = df.dropna(subset=["incidents"]).copy()
    df["year"] = df["year"].astype(int)
    df["incidents"] = df["incidents"].astype(int)

    df = df[["year", "incidents"]].sort_values("year").reset_index(drop=True)
    return df

def expand_to_daily(df_yearly, spike_factor=5):
    """
    Spread each year's total incidents across days, with a spike on Nov 5.
    Uses Poisson sampling for realistic integers and handles leap years.
    """
    rng = np.random.default_rng(123)
    rows = []

    for _, r in df_yearly.iterrows():
        y   = int(r["year"])
        tot = int(r["incidents"])

        days = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        n    = len(days)

        # baseline rate per day
        base = tot / n
        lam  = np.full(n, base, dtype=float)

        # weekend uplift (optional, small)
        dow = pd.Series(days).dt.dayofweek.values  # 0=Mon..6=Sun
        lam[dow >= 5] *= 1.15

        # Bonfire window (Nov 3–7) with big spike on Nov 5
        dts = pd.Series(days)
        mask_window = (dts.dt.month == 11) & (dts.dt.day.isin([3,4,5,6,7]))
        mask_nov5   = (dts.dt.month == 11) & (dts.dt.day == 5)

        lam[mask_window] *= 2.0          # general uplift around the event
        lam[mask_nov5]   *= (spike_factor / 2.0)  # extra boost for the 5th

        # rescale so expected sum ≈ yearly total
        scale = tot / lam.sum()
        lam *= scale

        # sample integers
        inc = rng.poisson(lam).astype(int)

        # write rows
        for d, c in zip(days, inc):
            rows.append({
                "date": d.date().isoformat(),
                "year": y,
                "incidents": int(c),
                "is_weekend": int(d.dayofweek >= 5),
                "is_bonfire_window": int(d.month == 11 and d.day in (3,4,5,6,7)),
                "day_of_year": int(d.dayofyear)
            })

    return pd.DataFrame(rows)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    yearly = load_yearly()
    yearly.to_csv(YEARLY_OUT, index=False)
    print(f"[OK] yearly -> {YEARLY_OUT}")

    daily = expand_to_daily(yearly, spike_factor=5)
    daily.to_csv(DAILY_OUT, index=False)
    print(f"[OK] daily  -> {DAILY_OUT}")

    # quick sanity check: daily sums ≈ yearly totals
    check = daily.groupby("year", as_index=False)["incidents"].sum()
    merged = yearly.merge(check, on="year", suffixes=("_yearly", "_daily_sum"))
    merged["diff"] = merged["incidents_yearly"] - merged["incidents_daily_sum"]
    print("\nSanity check (yearly vs sum of daily):")
    print(merged)

if __name__ == "__main__":
    main()
