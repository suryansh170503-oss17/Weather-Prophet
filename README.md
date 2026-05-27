# 🌦 Prophet Weather Forecaster

A full-stack weather forecasting web app powered by **Facebook Prophet** — a real ML model developed by Meta's Core Data Science team. It fetches genuine historical weather data from the **Open-Meteo API** and predicts future trends with uncertainty bands.

---

## 🚀 Live Demo

> Start the backend locally and open `frontend/index.html` in your browser.

---

## 📸 Features

- 🌍 **8 cities** — New Delhi, Mumbai, Bangalore, Kolkata, London, New York, Tokyo, Sydney
- 🌡️ **4 weather variables** — Temperature, Rainfall, Humidity, Wind Speed
- 📈 **Real historical data** from Open-Meteo (free, no API key needed, data back to 1940)
- 🤖 **Facebook Prophet ML model** — piecewise trend + Fourier seasonality + MCMC uncertainty
- 📊 **4 decomposition views** — Full forecast, Trend, Seasonality, Residuals
- 📅 **Forecast up to 365 days** ahead with 80/90/95% confidence intervals
- 📆 **7-day outlook table** with weather condition badges
- ⚡ **In-memory caching** — second request for same city is instant

---

## 🗂️ Project Structure

```
Weather-Prophet/
├── backend/
│   ├── main.py             # FastAPI server + Prophet model + Open-Meteo integration
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Complete single-file frontend (no build step)
└── README.md
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip
- A modern web browser

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/weather-prophet.git
cd weather-prophet
```

### 2. Install Python dependencies

```bash
cd backend
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> ⚠️ **Fedora/Linux users:** If you hit a metadata build error with pandas, run:
> ```bash
> pip install pandas --only-binary=:all:
> pip install -r requirements.txt
> ```

### 3. Start the backend server

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

You should see:
```
INFO: Uvicorn running on http://127.0.0.1:8000
```

### 4. Open the frontend

Open `frontend/index.html` directly in your browser — no build step needed.

Or serve it with Python:
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

---

## 🌐 Data Source

All weather data is fetched in real time from the **[Open-Meteo Historical Weather API](https://open-meteo.com/)**:

| Property | Detail |
|----------|--------|
| Provider | Open-Meteo (open source) |
| API key  | Not required — completely free |
| Coverage | Global, data from 1940 onwards |
| Update frequency | Daily (~5 day lag) |
| Resolution | Daily aggregates |

### Variables and their Open-Meteo parameters

| Variable | Open-Meteo parameter |
|----------|----------------------|
| Temperature | `temperature_2m_mean` |
| Rainfall | `precipitation_sum` |
| Humidity | `relative_humidity_2m_mean` |
| Wind Speed | `wind_speed_10m_mean` |

---

## 🤖 How the Prophet Model Works

**Facebook Prophet** decomposes a time series into three additive components:

```
y(t) = trend(t) + seasonality(t) + noise(t)
```

| Component | How Prophet models it |
|-----------|-----------------------|
| **Trend** | Piecewise linear regression with automatic changepoint detection |
| **Yearly seasonality** | Fourier series (captures annual weather cycles) |
| **Weekly seasonality** | Day-of-week effect on weather patterns |
| **Uncertainty** | MCMC sampling to produce confidence intervals |

### Model configuration used

```python
Prophet(
    interval_width        = ci / 100,   # 0.80 / 0.90 / 0.95
    yearly_seasonality    = True,        # Fourier annual pattern
    weekly_seasonality    = True,        # Day-of-week effect
    daily_seasonality     = False,       # Not needed for daily data
    changepoint_prior_scale = 0.05,      # Trend flexibility
    seasonality_prior_scale = 10,        # Seasonality strength
    n_changepoints          = 25,        # Max trend change points
)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| GET | `/cities` | List all available cities |
| GET | `/variables` | List all weather variables |
| GET | `/forecast` | Run Prophet forecast |

### Forecast query parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `city` | string | `delhi` | see cities | City ID |
| `variable` | string | `temp` | temp, rain, humidity, wind | Weather variable |
| `horizon` | int | `30` | 7–365 | Days to forecast |
| `ci` | float | `80` | 50–99 | Confidence interval % |
| `years` | int | `3` | 1–5 | Years of training data |

### Example request

```bash
curl "http://127.0.0.1:8000/forecast?city=mumbai&variable=rain&horizon=60&ci=90&years=3"
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | [Facebook Prophet 1.1+](https://facebook.github.io/prophet/) |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Data | [Open-Meteo API](https://open-meteo.com/) |
| Data processing | Pandas, NumPy |
| Frontend | Vanilla HTML/CSS/JS |
| Charts | [Chart.js 4.4](https://www.chartjs.org/) |

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Acknowledgements

- [Facebook Prophet](https://facebook.github.io/prophet/) — by Meta Core Data Science
- [Open-Meteo](https://open-meteo.com/) — free open-source weather API
- [FastAPI](https://fastapi.tiangolo.com/) — modern Python web framework
