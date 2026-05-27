from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

app = FastAPI(title="Weather Prophet API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CITIES = {
    "delhi":     {"name": "New Delhi",  "lat": 28.6139, "lon": 77.2090},
    "mumbai":    {"name": "Mumbai",     "lat": 19.0760, "lon": 72.8777},
    "bangalore": {"name": "Bangalore",  "lat": 12.9716, "lon": 77.5946},
    "kolkata":   {"name": "Kolkata",    "lat": 22.5726, "lon": 88.3639},
    "london":    {"name": "London",     "lat": 51.5074, "lon": -0.1278},
    "nyc":       {"name": "New York",   "lat": 40.7128, "lon": -74.0060},
    "tokyo":     {"name": "Tokyo",      "lat": 35.6762, "lon": 139.6503},
    "sydney":    {"name": "Sydney",     "lat": -33.8688, "lon": 151.2093},
}

VARIABLE_MAP = {
    "temp":     {"param": "temperature_2m_mean",        "unit": "°C",   "label": "Temperature"},
    "rain":     {"param": "precipitation_sum",           "unit": "mm",   "label": "Rainfall"},
    "humidity": {"param": "relative_humidity_2m_mean",   "unit": "%",    "label": "Humidity"},
    "wind":     {"param": "wind_speed_10m_mean",         "unit": "km/h", "label": "Wind Speed"},
}

# Simple in-memory cache so same city+variable isn't re-fetched every request
_cache = {}


def fetch_open_meteo(lat: float, lon: float, variable: str, years: int = 3) -> pd.DataFrame:
    """
    Fetch real daily weather data from Open-Meteo Historical Weather API.
    Free, no API key required. Data goes back to 1940.
    """
    cache_key = f"real_{lat}_{lon}_{variable}_{years}"
    if cache_key in _cache:
        return _cache[cache_key]

    end   = datetime.now().date() - timedelta(days=5)   # API has ~5 day lag
    start = end - timedelta(days=365 * years)
    param = VARIABLE_MAP[variable]["param"]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "daily":      param,
        "timezone":   "auto",
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, f"Open-Meteo API error: {e}")

    data = r.json()

    if "daily" not in data or param not in data["daily"]:
        raise HTTPException(502, f"Open-Meteo returned unexpected response: {data}")

    df = pd.DataFrame({
        "ds": pd.to_datetime(data["daily"]["time"]),
        "y":  data["daily"][param],
    })
    df = df.dropna()

    if df.empty:
        raise HTTPException(502, "Open-Meteo returned no data for this location/variable.")

    _cache[cache_key] = df
    return df


def run_prophet(df: pd.DataFrame, horizon: int, ci: float) -> dict:
    model = Prophet(
        interval_width=ci / 100,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        n_changepoints=25,
    )
    model.fit(df)

    future   = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)

    hist_len = len(df)
    hist_fc  = forecast.iloc[:hist_len]
    future_fc = forecast.iloc[hist_len:]

    trend_vals  = forecast["trend"].values
    slope_30    = float(trend_vals[-1] - trend_vals[max(0, len(trend_vals) - 30)])

    # Yearly seasonality — sample one full year
    yearly_df   = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=365, freq="D")})
    yearly_pred = model.predict(yearly_df)
    monthly_labels  = yearly_pred["ds"].dt.strftime("%b").tolist()[::30][:12]
    monthly_seasonal = yearly_pred["yearly"].iloc[::30][:12].round(2).tolist()

    # Weekly seasonality
    weekly_df   = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=7, freq="D")})
    weekly_pred = model.predict(weekly_df)
    weekly_labels = weekly_pred["ds"].dt.strftime("%a").tolist()
    weekly_effect = weekly_pred["weekly"].round(2).tolist()

    # Sample every 3rd day to keep response size manageable
    step        = 3
    hist_s      = hist_fc.iloc[::step]
    df_s        = df.iloc[::step]

    return {
        "historical": {
            "dates":  df_s["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "values": df_s["y"].round(2).tolist(),
            "trend":  hist_s["trend"].round(2).tolist(),
        },
        "forecast": {
            "dates":  future_fc["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "values": future_fc["yhat"].round(2).tolist(),
            "lower":  future_fc["yhat_lower"].round(2).tolist(),
            "upper":  future_fc["yhat_upper"].round(2).tolist(),
            "trend":  future_fc["trend"].round(2).tolist(),
        },
        "seasonality": {
            "monthly_labels":  monthly_labels,
            "monthly_values":  monthly_seasonal,
            "weekly_labels":   weekly_labels,
            "weekly_values":   weekly_effect,
        },
        "stats": {
            "mean_historical": round(float(df["y"].mean()), 2),
            "std_historical":  round(float(df["y"].std()),  2),
            "mean_forecast":   round(float(future_fc["yhat"].mean()), 2),
            "trend_slope_30d": round(slope_30, 2),
        },
    }


@app.get("/cities")
def get_cities():
    return [{"id": k, "name": v["name"], "lat": v["lat"], "lon": v["lon"]}
            for k, v in CITIES.items()]


@app.get("/variables")
def get_variables():
    return [{"id": k, "label": v["label"], "unit": v["unit"]}
            for k, v in VARIABLE_MAP.items()]


@app.get("/forecast")
def forecast(
    city:     str   = Query("delhi"),
    variable: str   = Query("temp"),
    horizon:  int   = Query(30, ge=7, le=365),
    ci:       float = Query(80, ge=50, le=99),
    years:    int   = Query(3, ge=1, le=5),
):
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")
    if variable not in VARIABLE_MAP:
        raise HTTPException(400, f"Unknown variable: {variable}")

    cfg = CITIES[city]
    df  = fetch_open_meteo(cfg["lat"], cfg["lon"], variable, years)
    result = run_prophet(df, horizon, ci)

    result["meta"] = {
        "city":        cfg["name"],
        "variable":    VARIABLE_MAP[variable]["label"],
        "unit":        VARIABLE_MAP[variable]["unit"],
        "horizon":     horizon,
        "ci":          ci,
        "data_points": len(df),
        "data_from":   df["ds"].min().strftime("%Y-%m-%d"),
        "data_to":     df["ds"].max().strftime("%Y-%m-%d"),
        "data_source": "Open-Meteo Historical Weather API (real observed data)",
    }
    return result


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}
