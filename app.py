import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

st.title("🚗 Traffic Accident Severity Prediction System")

st.markdown("Predict accident severity using environmental and road conditions")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Enter Conditions")

# Numerical
temp = st.sidebar.slider("Temperature (F)", -20, 120, 70)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
pressure = st.sidebar.slider("Pressure (in)", 28.0, 32.0, 30.0)
visibility = st.sidebar.slider("Visibility (mi)", 0.0, 10.0, 5.0)
wind_speed = st.sidebar.slider("Wind Speed (mph)", 0.0, 50.0, 10.0)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

# Categorical
weather = st.sidebar.selectbox("Weather Condition", [
    "Clear", "Rain", "Snow", "Fog", "Cloudy"
])

wind_dir = st.sidebar.selectbox("Wind Direction", [
    "N", "S", "E", "W", "NE", "NW", "SE", "SW"
])

# Boolean
amenity = st.sidebar.checkbox("Amenity Nearby")
bump = st.sidebar.checkbox("Bump")
crossing = st.sidebar.checkbox("Crossing")
give_way = st.sidebar.checkbox("Give Way")
junction = st.sidebar.checkbox("Junction")
stop = st.sidebar.checkbox("Stop Sign")
traffic_signal = st.sidebar.checkbox("Traffic Signal")

# Day/Night
sun = st.sidebar.selectbox("Day or Night", ["Day", "Night"])

# Convert Day/Night
sun_val = 1 if sun == "Day" else 0

# -----------------------------
# CREATE INPUT DATA
# -----------------------------
input_data = pd.DataFrame([{
    "Temperature(F)": temp,
    "Humidity(%)": humidity,
    "Pressure(in)": pressure,
    "Visibility(mi)": visibility,
    "Wind_Speed(mph)": wind_speed,
    "Weather_Condition": weather,
    "Wind_Direction": wind_dir,
    "Amenity": int(amenity),
    "Bump": int(bump),
    "Crossing": int(crossing),
    "Give_Way": int(give_way),
    "Junction": int(junction),
    "Stop": int(stop),
    "Traffic_Signal": int(traffic_signal),
    "Sunrise_Sunset": sun_val,
    "hour": hour
}])

# -----------------------------
# PREDICTION
# -----------------------------
if st.sidebar.button("Predict Severity"):
    result = model.predict(input_data)[0]

    st.subheader("🚨 Prediction Result")

    if result == 1:
        st.success("Low Severity")
    elif result == 2:
        st.info("Moderate Severity")
    elif result == 3:
        st.warning("High Severity")
    else:
        st.error("Very High Severity")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
---
### 📊 About Project
- Built using 200k+ accident records  
- Used ML pipeline with preprocessing  
- Integrated environmental & road features  
- Developed interactive Streamlit dashboard  
""")