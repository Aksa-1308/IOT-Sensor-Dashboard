import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="IoT Sensor Analysis & Prediction", page_icon="🌡️", layout="wide")
st.title("🌡️ IoT Sensor Data Analysis & Prediction")
st.caption("Hourly sensor analytics, next-hour temperature forecasting, and overheating alerts")

@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=60*24, freq="h")
    nodes = [
        ("N01", "Server Room", 24.0, 3.0, 28.0),
        ("N02", "Classroom", 23.0, 4.5, 30.0),
        ("N03", "Warehouse", 25.0, 5.0, 32.0),
        ("N04", "Greenhouse", 26.0, 7.0, 35.0),
        ("N05", "Server Room", 24.5, 2.5, 28.0),
        ("N06", "Classroom", 22.5, 4.0, 30.0),
        ("N07", "Warehouse", 25.5, 4.5, 32.0),
        ("N08", "Greenhouse", 25.0, 6.5, 35.0),
        ("N09", "Classroom", 23.5, 4.0, 30.0),
        ("N10", "Warehouse", 24.5, 5.0, 32.0),
    ]
    out = []
    for node, loc, base, amp, limit in nodes:
        for t in dates:
            hour = t.hour
            daily = amp * np.sin(2*np.pi*(hour-7)/24)
            weekly = 0.7 if t.dayofweek < 5 and loc == "Classroom" else 0
            temp = base + daily + weekly + rng.normal(0, 0.65)
            humidity = np.clip(58 - 0.65*(temp-base) + rng.normal(0, 4), 15, 95)
            co2 = np.clip(450 + (350 if loc == "Classroom" and t.dayofweek < 5 and 8 <= hour <= 16 else 80) + rng.normal(0, 45), 350, 1800)
            light = np.clip(20 + 600*max(0, np.sin(np.pi*(hour-6)/12)) + rng.normal(0, 35), 0, 1000)
            battery = np.clip(3.3 - (t-dates[0]).total_seconds()/3600*0.00008 + rng.normal(0, 0.015), 2.6, 3.4)
            out.append((t, node, loc, temp, humidity, co2, light, battery, limit))
    df = pd.DataFrame(out, columns=["timestamp","node_id","location_type","temperature_c","humidity_pct","co2_ppm","light_lux","battery_v","safe_limit_c"])
    # Introduce a small amount of realistic missingness
    for col in ["humidity_pct","co2_ppm","light_lux"]:
        ix = rng.choice(df.index, size=20, replace=False)
        df.loc[ix, col] = np.nan
    return df

uploaded = st.sidebar.file_uploader("Upload sensor CSV (optional)", type=["csv"])
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        required = {"timestamp","node_id","location_type","temperature_c","humidity_pct","co2_ppm","light_lux","battery_v"}
        if not required.issubset(raw.columns):
            st.error("CSV needs columns: " + ", ".join(sorted(required)))
            st.stop()
        df = raw.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "safe_limit_c" not in df:
            df["safe_limit_c"] = df["location_type"].map({"Server Room":28, "Classroom":30, "Warehouse":32, "Greenhouse":35}).fillna(30)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    df = make_demo_data()
    st.sidebar.info("Using the included synthetic demonstration dataset.")

df = df.dropna(subset=["timestamp","node_id","temperature_c"]).drop_duplicates().sort_values(["node_id","timestamp"])
df = df.reset_index(drop=True)
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["temperature_lag1"] = df.groupby("node_id")["temperature_c"].shift(1)
df["temperature_lag24"] = df.groupby("node_id")["temperature_c"].shift(24)
df["target_next_temp"] = df.groupby("node_id")["temperature_c"].shift(-1)
model_df = df.dropna(subset=["temperature_lag1","target_next_temp"]).copy()
model_df["temperature_lag24"] = model_df["temperature_lag24"].fillna(model_df["temperature_lag1"])

with st.sidebar:
    st.header("Filters")
    locations = ["All"] + sorted(df["location_type"].dropna().astype(str).unique().tolist())
    selected_loc = st.selectbox("Location", locations)
    nodes = ["All"] + sorted(df["node_id"].dropna().astype(str).unique().tolist())
    selected_node = st.selectbox("Sensor node", nodes)
view = df.copy()
if selected_loc != "All":
    view = view[view["location_type"] == selected_loc]
if selected_node != "All":
    view = view[view["node_id"] == selected_node]

k1,k2,k3,k4 = st.columns(4)
k1.metric("Sensor readings", f"{len(view):,}")
k2.metric("Sensor nodes", f"{view['node_id'].nunique():,}")
k3.metric("Mean temperature", f"{view['temperature_c'].mean():.1f} °C")
k4.metric("Missing sensor values", f"{int(view[['humidity_pct','co2_ppm','light_lux','battery_v']].isna().sum().sum()):,}")

tab1,tab2,tab3,tab4,tab5 = st.tabs(["Overview","Explore Data","Train & Evaluate","Predict Next Hour","Dataset"])
with tab1:
    st.subheader("Temperature over time")
    fig, ax = plt.subplots(figsize=(11,4))
    for node, g in view.groupby("node_id"):
        ax.plot(g["timestamp"], g["temperature_c"], linewidth=1, label=node)
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Time")
    ax.grid(alpha=.25)
    if view["node_id"].nunique() <= 10: ax.legend(ncol=5, fontsize=8)
    st.pyplot(fig)
    st.subheader("Average readings by location")
    summary = view.groupby("location_type")[["temperature_c","humidity_pct","co2_ppm","light_lux","battery_v"]].mean().round(2)
    st.dataframe(summary, use_container_width=True)
    hot = view[view["temperature_c"] > view["safe_limit_c"]]
    st.metric("Current dataset readings above safe limit", len(hot))
    if not hot.empty:
        st.warning("Some readings exceed their configured temperature limit. Review the table in Dataset.")
with tab2:
    left,right = st.columns(2)
    with left:
        metric = st.selectbox("Sensor metric", ["temperature_c","humidity_pct","co2_ppm","light_lux","battery_v"])
        fig, ax = plt.subplots(figsize=(7,4))
        view.boxplot(column=metric, by="location_type", ax=ax, rot=20)
        ax.set_title(f"{metric} by location"); fig.suptitle("")
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(view["humidity_pct"], view["temperature_c"], alpha=.25)
        ax.set_xlabel("Humidity (%)"); ax.set_ylabel("Temperature (°C)")
        ax.set_title("Humidity vs temperature"); ax.grid(alpha=.2)
        st.pyplot(fig)
    st.subheader("Hourly temperature pattern")
    hourly = view.groupby("hour")["temperature_c"].mean()
    st.line_chart(hourly)
with tab3:
    st.write("Models predict the next hour's temperature from time, location, and previous readings. The latest 20% of timestamps are held out for testing; rows are not randomly shuffled.")
    feature_cols = ["hour","day_of_week","is_weekend","temperature_lag1","temperature_lag24","humidity_pct","co2_ppm","light_lux","battery_v","location_type"]
    model_df = model_df.sort_values("timestamp")
    cut = model_df["timestamp"].quantile(.8)
    train = model_df[model_df["timestamp"] < cut]
    test = model_df[model_df["timestamp"] >= cut]
    if len(train) < 30 or len(test) < 10:
        st.warning("Not enough valid rows for a time-based train/test evaluation.")
    else:
        numeric = [c for c in feature_cols if c != "location_type"]
        prep = ColumnTransformer([
            ("num", Pipeline([("imputer",SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), ["location_type"])
        ])
        algos = {
            "Previous-hour baseline": None,
            "Linear Regression": Pipeline([("prep",prep),("model",LinearRegression())]),
            "Random Forest": Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=150,max_depth=14,min_samples_leaf=3,random_state=42,n_jobs=-1))])
        }
        rows = []
        for name, model in algos.items():
            if model is None:
                pred = test["temperature_lag1"].to_numpy()
            else:
                model.fit(train[feature_cols],train["target_next_temp"])
                pred = model.predict(test[feature_cols])
            y = test["target_next_temp"].to_numpy()
            rows.append({"Model":name,"MAE (°C)":mean_absolute_error(y,pred),"RMSE (°C)":mean_squared_error(y,pred)**0.5,"R²":r2_score(y,pred)})
        scores = pd.DataFrame(rows).round(4)
        st.dataframe(scores,use_container_width=True,hide_index=True)
        st.bar_chart(scores.set_index("Model")[["MAE (°C)","RMSE (°C)"]])
        st.caption("These metrics are computed from the data currently loaded. Synthetic demonstration data is not real sensor telemetry.")
with tab4:
    st.subheader("Next-hour temperature estimate")
    available = sorted(df["node_id"].astype(str).unique())
    chosen = st.selectbox("Choose node to forecast", available, key="forecast_node")
    g = df[df["node_id"].astype(str)==chosen].sort_values("timestamp")
    last = g.iloc[-1]
    next_time = last["timestamp"] + pd.Timedelta(hours=1)
    lag24 = g.iloc[-24]["temperature_c"] if len(g)>=24 else last["temperature_c"]
    next_hour = next_time.hour
    loc = str(last["location_type"])
    row = pd.DataFrame([{"hour":next_hour,"day_of_week":next_time.dayofweek,"is_weekend":int(next_time.dayofweek>=5),"temperature_lag1":last["temperature_c"],"temperature_lag24":lag24,"humidity_pct":last["humidity_pct"],"co2_ppm":last["co2_ppm"],"light_lux":last["light_lux"],"battery_v":last["battery_v"],"location_type":loc}])
    feat = row.drop(columns=["location_type"])
    feat["location_type"] = loc
    # Fit a model on all available historical rows for a demonstration forecast.
    fcols = ["hour","day_of_week","is_weekend","temperature_lag1","temperature_lag24","humidity_pct","co2_ppm","light_lux","battery_v","location_type"]
    train_all = model_df.dropna(subset=["target_next_temp"])
    numeric = [c for c in fcols if c!="location_type"]
    prep = ColumnTransformer([("num",SimpleImputer(strategy="median"),numeric),("cat",OneHotEncoder(handle_unknown="ignore"),["location_type"])])
    pipe = Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=150,max_depth=14,min_samples_leaf=3,random_state=42,n_jobs=-1))])
    pipe.fit(train_all[fcols],train_all["target_next_temp"])
    forecast = float(pipe.predict(row[fcols])[0])
    limit = float(last["safe_limit_c"])
    a,b,c = st.columns(3)
    a.metric("Predicted temperature",f"{forecast:.2f} °C")
    b.metric("Forecast time",next_time.strftime("%Y-%m-%d %H:%M"))
    c.metric("Configured safe limit",f"{limit:.1f} °C")
    if forecast > limit:
        st.error("⚠️ Forecast exceeds the configured limit. Investigate the sensor/location.")
    else:
        st.success("Forecast is below the configured limit.")
    st.caption("This forecast uses a model trained on the currently loaded historical data; it is a demonstration, not a live device reading.")
with tab5:
    st.subheader("Sensor data")
    st.dataframe(view, use_container_width=True, height=450)
    st.download_button("Download filtered CSV",view.to_csv(index=False).encode("utf-8"),"iot_sensor_data_filtered.csv","text/csv")
    st.download_button("Download full dataset CSV",df.to_csv(index=False).encode("utf-8"),"iot_sensor_data.csv","text/csv")
    st.caption("If you uploaded a CSV, the downloadable full dataset is the uploaded data after basic cleaning and feature creation.")
