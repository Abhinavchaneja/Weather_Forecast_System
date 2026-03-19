# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fpdf import FPDF
import io
import base64
import os
import requests

# ------------- Page config -------------
st.set_page_config(page_title="Pro Weather Dashboard (A+C)", page_icon="🌦️", layout="wide")

# ------------- Helper functions -------------
def get_current_weather(api_key, city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        
        if data.get("cod") != 200:
            return None

        weather_info = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind": data["wind"]["speed"],
            "condition": data["weather"][0]["description"]
        }
        return weather_info
    st.subheader("Real-Time Weather")

api_key = st.text_input("Enter your OpenWeather API Key")
city_name = st.text_input("Enter City Name")

if api_key and city_name:
    current = get_current_weather(api_key, city_name)
    if current:
        st.success(f"Current Weather in {current['city']}, {current['country']}")

        st.write(f"**Temperature:** {current['temp']} °C")
        st.write(f"**Humidity:** {current['humidity']} %")
        st.write(f"**Pressure:** {current['pressure']} hPa")
        st.write(f"**Wind Speed:** {current['wind']} m/s")
        st.write(f"**Condition:** {current['condition'].title()}")

        st.info("You can use this current weather as the starting point for forecasting.")
    else:
        st.error("Could not fetch weather. Check city name or API key.")

def find_column(df_cols, patterns):
    for pat in patterns:
        regex = re.compile(pat, flags=re.I)
        for c in df_cols:
            if regex.search(c):
                return c
    return None

def detect_columns(df):
    cols = df.columns.tolist()
    temp_patterns = [r'\btemp(?:erature)?\b', r'\bmean\s*temp\b', r'\btemperature\s*\(c\)', r'\bapparent temperature\b', r'\btemp_c\b']
    humid_patterns = [r'\bhumid', r'\brelative\s*humidity\b']
    wind_patterns = [r'wind[_\s\-]*speed', r'wind\s*speed', r'windspeed']
    pressure_patterns = [r'press', r'millibar', r'\bmb\b', r'meanpressure', r'pressure']
    date_patterns = [r'date', r'time', r'formatted date', r'datetime']

    return {
        "temp": find_column(cols, temp_patterns),
        "humidity": find_column(cols, humid_patterns),
        "wind": find_column(cols, wind_patterns),
        "pressure": find_column(cols, pressure_patterns),
        "date": find_column(cols, date_patterns)
    }

def ensure_temp_prev(df, temp_col):
    df = df.copy()
    if 'temp_prev' not in df.columns:
        df['temp_prev'] = df[temp_col].shift(1)
    return df

def load_csv_any(buffer):
    df = pd.read_csv(buffer)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_resource
def load_model_cached(path="weather_model.pkl"):
    return joblib.load(path)

def train_and_save_model(df, features, target, model_path="weather_model.pkl"):
    X = df[features].dropna()
    y = df.loc[X.index, target]
    if len(X) < 10:
        raise ValueError("Not enough rows to train. Need at least 10 clean rows.")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    joblib.dump(model, model_path)
    return model, {"mae": mae, "rmse": rmse, "x_test": x_test, "y_test": y_test, "preds": preds}

def predict_next(model, Humidity, WindSpeed, Pressure, temp_prev):
    X = np.array([[Humidity, WindSpeed, Pressure, temp_prev]])
    return float(model.predict(X)[0])

def iterative_forecast(model, start_dict, days, future_inputs=None):
    temp_prev = float(start_dict.get('Temp', start_dict.get('temp_prev', np.nan)))
    hum_base = float(start_dict.get('Humidity', np.nan))
    wind_base = float(start_dict.get('WindSpeed', np.nan))
    press_base = float(start_dict.get('Pressure', np.nan))
    if future_inputs is None:
        future_inputs = {}
    hums = future_inputs.get('Humidity', [hum_base])
    winds = future_inputs.get('WindSpeed', [wind_base])
    press = future_inputs.get('Pressure', [press_base])
    def fill_to_days(lst, base):
        if len(lst) >= days:
            return [float(x) for x in lst[:days]]
        else:
            if len(lst) == 0:
                return [float(base)] * days
            else:
                last = float(lst[-1])
                return [float(x) for x in lst] + [last] * (days - len(lst))
    hums = fill_to_days(hums, hum_base)
    winds = fill_to_days(winds, wind_base)
    press = fill_to_days(press, press_base)
    records = []
    for i in range(days):
        pred = predict_next(model, hums[i], winds[i], press[i], temp_prev)
        records.append({
            "Day": i+1,
            "PredTemp": round(pred, 3),
            "Humidity_used": hums[i],
            "WindSpeed_used": winds[i],
            "Pressure_used": press[i],
            "temp_prev_used": round(temp_prev, 3)
        })
        temp_prev = pred
    return pd.DataFrame(records)

def fetch_openweather_current(api_key, city, units="metric"):
    base = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": units}
    r = requests.get(base, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    # extract relevant fields
    return {
        "temp": data["main"]["temp"],  # degrees C if units=metric
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data.get("wind", {}).get("speed", 0),
        "desc": data["weather"][0]["description"]
    }

def fig_to_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def create_pdf_report(metrics, forecast_df, figure_bytes, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Weather Forecast Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"MAE: {metrics.get('mae', 'N/A'):.3f}    RMSE: {metrics.get('rmse', 'N/A'):.3f}", ln=True)
    pdf.ln(6)
    # Insert figure
    img_path = "tmp_plot.png"
    with open(img_path, "wb") as f:
        f.write(figure_bytes.getvalue())
    pdf.image(img_path, x=15, w=180)
    pdf.ln(6)
    pdf.cell(0, 8, "Forecast (first 10 rows):", ln=True)
    pdf.ln(2)
    # table (simple)
    pdf.set_font("Courier", size=9)
    rows = forecast_df.head(10).to_string(index=False).split("\n")
    for row in rows:
        pdf.cell(0, 6, row, ln=True)
    pdf.output(filename)
    # cleanup
    if os.path.exists(img_path):
        os.remove(img_path)
    return filename

# ------------- UI Layout -------------
st.title("🌦️ Pro Weather Dashboard — ML + Real-time API + Upload")
st.markdown("Upload any weather dataset, train a RandomForest model, or use existing model. Choose forecast horizon (1–30 days).")

# Left column: controls
with st.sidebar:
    st.header("Dataset & Model")
    uploaded = st.file_uploader("Upload weather CSV (optional)", type=["csv"])
    default_file = Path("weatherHistory.csv")
    if uploaded is None and default_file.exists():
        if st.checkbox(f"Use default dataset: {default_file.name}", value=True):
            df = load_csv_any(str(default_file))
        else:
            df = None
    elif uploaded is not None:
        df = load_csv_any(uploaded)
    else:
        df = None

    st.write("---")
    st.subheader("OpenWeatherMap API (optional)")
    use_api = st.checkbox("Fetch current weather from OpenWeatherMap", value=False)
    api_key = st.text_input("OpenWeatherMap API key (get at openweathermap.org)", type="password")
    city = st.text_input("City name for current weather (e.g., London,IN or New York,US)", value="Delhi,IN")
    st.write("---")
    st.subheader("Model file")
    model_path_input = st.text_input("Model filename to save/load", value="weather_model.pkl")
    load_saved = st.button("Load saved model")

# Try loading saved model early if requested
model = None
if 'model_obj' not in st.session_state:
    st.session_state.model_obj = None

if 'train_metrics' not in st.session_state:
    st.session_state.train_metrics = {}

if load_saved:
    try:
        st.session_state.model_obj = load_model_cached(model_path_input)
        st.success("Loaded saved model")
    except Exception as e:
        st.error(f"Could not load model: {e}")

# Middle column: dataset preview and mapping
col1, col2 = st.columns([1.2, 2.5])
with col1:
    st.subheader("Step 1 — Data")
    if df is None:
        st.info("Upload a CSV dataset or place weatherHistory.csv in same folder.")
    else:
        st.write(f"Dataset loaded — {df.shape[0]} rows, {df.shape[1]} columns")
        with st.expander("Preview dataset (first 5 rows)"):
            st.dataframe(df.head(5))

        detected = detect_columns(df)
        st.markdown("**Detected columns (auto)** — override if incorrect.")
        col_temp = st.text_input("Temperature column", value=detected["temp"] or "")
        col_hum = st.text_input("Humidity column", value=detected["humidity"] or "")
        col_wind = st.text_input("Wind Speed column", value=detected["wind"] or "")
        col_press = st.text_input("Pressure column", value=detected["pressure"] or "")
        col_date = st.text_input("Date column (optional)", value=detected["date"] or "")

        st.markdown("If your dataset doesn't include pressure/wind/humidity, use numeric fallbacks below.")
        fallback_temp = st.number_input("Fallback previous-day temp (°C)", value=25.0)
        fallback_hum = st.number_input("Fallback humidity (%)", value=70.0)
        fallback_wind = st.number_input("Fallback wind speed", value=1.5)
        fallback_press = st.number_input("Fallback pressure (hPa)", value=1010.0)

# Right column: training, forecast controls
with col2:
    st.subheader("Step 2 — Train / Forecast")
    st.markdown("Model: **RandomForestRegressor** (n=200).")

    train_now = st.button("Train model on this dataset")
    if train_now:
        if df is None:
            st.error("No dataset to train on. Upload CSV first.")
        else:
            # prepare df
            working = df.copy()
            if col_temp and col_temp in working.columns:
                working["Temp"] = pd.to_numeric(working[col_temp], errors='coerce')
            else:
                working["Temp"] = float(fallback_temp)

            if col_hum and col_hum in working.columns:
                working["Humidity"] = pd.to_numeric(working[col_hum], errors='coerce')
            else:
                working["Humidity"] = float(fallback_hum)

            if col_wind and col_wind in working.columns:
                working["WindSpeed"] = pd.to_numeric(working[col_wind], errors='coerce')
            else:
                working["WindSpeed"] = float(fallback_wind)

            if col_press and col_press in working.columns:
                working["Pressure"] = pd.to_numeric(working[col_press], errors='coerce')
            else:
                working["Pressure"] = float(fallback_press)

            working = ensure_temp_prev(working, "Temp")
            # drop rows with na in core features
            working_clean = working.dropna(subset=["Temp", "temp_prev", "Humidity", "WindSpeed", "Pressure"]).reset_index(drop=True)

            try:
                model_obj, metrics = train_and_save_model(working_clean, ["Humidity", "WindSpeed", "Pressure", "temp_prev"], "Temp", model_path=model_path_input)
                st.session_state.model_obj = model_obj
                st.session_state.train_metrics = metrics
                st.success(f"Model trained and saved to {model_path_input}")
                st.write(f"MAE: {metrics['mae']:.3f}   RMSE: {metrics['rmse']:.3f}")
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.markdown("---")
    st.markdown("### Forecast settings")
    days = st.slider("How many days to forecast?", min_value=1, max_value=30, value=7)
    start_choice = st.radio("Start forecast from:", options=["Last row of dataset", "Current real-time (OpenWeather)", "Custom values"], index=0)
    if start_choice == "Custom values":
        c_temp_prev = st.number_input("Custom previous day temp (temp_prev)", value=25.0)
        c_hum = st.number_input("Custom Humidity", value=70.0)
        c_wind = st.number_input("Custom Wind Speed", value=1.5)
        c_press = st.number_input("Custom Pressure", value=1010.0)

    st.markdown("Optional: Provide future Humidity/Wind/Pressure as comma-separated (app will repeat last if shorter).")
    fut_hum = st.text_input("Future Humidity list (e.g., 70,68,67)", value="")
    fut_wind = st.text_input("Future WindSpeed list (e.g., 1.2,1.5)", value="")
    fut_press = st.text_input("Future Pressure list (e.g., 1012,1010)", value="")

    run_forecast = st.button("Run Forecast")

# ------------- Forecast execution -------------
if run_forecast:
    model_obj = st.session_state.model_obj
    if model_obj is None:
        # try load from disk
        try:
            model_obj = load_model_cached(model_path_input)
            st.session_state.model_obj = model_obj
            st.success("Loaded model from disk.")
        except Exception as e:
            st.error("No trained model available. Train a model or load saved model first.")
            st.stop()

    future_inputs = {}
    def parse_list_text(txt):
        if not txt or str(txt).strip() == "":
            return []
        try:
            return [float(x.strip()) for x in str(txt).split(",") if x.strip() != ""]
        except:
            return []

    fh = parse_list_text(fut_hum)
    fw = parse_list_text(fut_wind)
    fp = parse_list_text(fut_press)
    if fh:
        future_inputs["Humidity"] = fh
    if fw:
        future_inputs["WindSpeed"] = fw
    if fp:
        future_inputs["Pressure"] = fp

    # determine starting baseline
    if start_choice == "Last row of dataset":
        if df is None:
            st.error("No dataset loaded to take last row from.")
            st.stop()
        working = df.copy()
        if col_temp and col_temp in working.columns:
            working["Temp"] = pd.to_numeric(working[col_temp], errors='coerce')
        else:
            working["Temp"] = float(fallback_temp)
        if col_hum and col_hum in working.columns:
            working["Humidity"] = pd.to_numeric(working[col_hum], errors='coerce')
        else:
            working["Humidity"] = float(fallback_hum)
        if col_wind and col_wind in working.columns:
            working["WindSpeed"] = pd.to_numeric(working[col_wind], errors='coerce')
        else:
            working["WindSpeed"] = float(fallback_wind)
        if col_press and col_press in working.columns:
            working["Pressure"] = pd.to_numeric(working[col_press], errors='coerce')
        else:
            working["Pressure"] = float(fallback_press)
        working = ensure_temp_prev(working, "Temp")
        last_row = working.dropna(subset=["Temp", "temp_prev", "Humidity", "WindSpeed", "Pressure"]).iloc[-1]
        start = {
            "Temp": float(last_row["Temp"]),
            "temp_prev": float(last_row["temp_prev"]),
            "Humidity": float(last_row["Humidity"]),
            "WindSpeed": float(last_row["WindSpeed"]),
            "Pressure": float(last_row["Pressure"])
        }
    elif start_choice == "Current real-time (OpenWeather)":
        if not api_key:
            st.error("Please supply OpenWeatherMap API key in the sidebar.")
            st.stop()
        try:
            current = fetch_openweather_current(api_key, city, units="metric")
            st.success(f"Fetched current weather: {current['desc']}, {current['temp']} °C")
            start = {
                "Temp": current["temp"],
                "temp_prev": current["temp"],  # treat current as prev day baseline
                "Humidity": current["humidity"],
                "WindSpeed": current["wind_speed"],
                "Pressure": current["pressure"]
            }
        except Exception as e:
            st.error(f"Could not fetch OpenWeather data: {e}")
            st.stop()
    else:  # custom
        start = {
            "Temp": float(c_temp_prev),
            "temp_prev": float(c_temp_prev),
            "Humidity": float(c_hum),
            "WindSpeed": float(c_wind),
            "Pressure": float(c_press)
        }

    # run forecast
    df_fore = iterative_forecast(model_obj, start, days, future_inputs)
    st.success("Forecast generated ✅")

    # Dashboard: top metrics & small table
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Next day (pred)", f"{df_fore.iloc[0]['PredTemp']:.2f} °C")
    colB.metric("Max (forecast)", f"{df_fore['PredTemp'].max():.2f} °C")
    colC.metric("Min (forecast)", f"{df_fore['PredTemp'].min():.2f} °C")
    colD.metric("Avg (forecast)", f"{df_fore['PredTemp'].mean():.2f} °C")

    st.markdown("### Forecast Table")
    st.dataframe(df_fore)

    # Plot forecast
    fig1, ax1 = plt.subplots(figsize=(8,3))
    sns.lineplot(x="Day", y="PredTemp", data=df_fore, marker="o", ax=ax1)
    ax1.set_title(f"{days}-Day Temperature Forecast")
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid(True)
    st.pyplot(fig1)

    # Feature importance & model evaluation (if trained metrics exist)
    if 'train_metrics' in st.session_state and st.session_state.train_metrics:
        metrics = st.session_state.train_metrics
        st.markdown("### Model evaluation (on test set)")
        st.write(f"MAE: {metrics['mae']:.3f}   RMSE: {metrics['rmse']:.3f}")
        # plot actual vs predicted
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.plot(metrics['y_test'].values, label="Actual")
        ax2.plot(metrics['preds'], label="Predicted")
        ax2.legend()
        ax2.set_title("Actual vs Predicted (test set)")
        st.pyplot(fig2)

    # Feature importance from loaded model
    try:
        fi = st.session_state.model_obj.feature_importances_
        features = ["Humidity","WindSpeed","Pressure","temp_prev"]
        fi_df = pd.DataFrame({"feature": features, "importance": fi}).sort_values("importance", ascending=False)
        st.markdown("### Feature importance")
        fig3, ax3 = plt.subplots(figsize=(5,3))
        sns.barplot(x="importance", y="feature", data=fi_df, ax=ax3)
        ax3.set_xlim(0, fi.max()*1.2)
        st.pyplot(fig3)
    except Exception:
        pass

    # Correlation heatmap (if df present)
    if df is not None:
        try:
            corr_cols = []
            working_small = working.dropna(subset=["Temp", "temp_prev", "Humidity", "WindSpeed", "Pressure"])
            corr_cols = ["Temp","temp_prev","Humidity","WindSpeed","Pressure"]
            fig4, ax4 = plt.subplots(figsize=(5,4))
            sns.heatmap(working_small[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
            ax4.set_title("Correlation matrix")
            st.pyplot(fig4)
        except Exception:
            pass

    # Download CSV
    csv_bytes = df_fore.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download forecast CSV", csv_bytes, file_name=f"forecast_{days}d.csv", mime="text/csv")

    # PDF report (simple)
    if st.button("📄 Generate PDF report"):
        fig_buf = io.BytesIO()
        fig1.savefig(fig_buf, format="png", bbox_inches="tight")
        fig_buf.seek(0)
        metrics_for_pdf = st.session_state.train_metrics if 'train_metrics' in st.session_state else {}
        pdf_file = create_pdf_report(metrics_for_pdf, df_fore, fig_buf, filename="weather_report.pdf")
        with open(pdf_file, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_file}">⬇️ Download PDF report</a>'
        st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.info("Notes: For best accuracy retrain the model on the location-specific dataset. The OpenWeather fetch uses current weather as the starting baseline if chosen.")
