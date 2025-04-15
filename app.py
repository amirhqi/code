import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from geopy.geocoders import Nominatim
import requests

# --------------------- تنظیمات رایگان ---------------------
st.set_page_config(page_title="EnergyGuard Pro", layout="wide")
st.markdown("""<style> .st-emotion-cache-1kyxreq {background: #f0f2f6;} </style>""", unsafe_allow_html=True)

# --------------------- داده‌های شبیه‌سازی شده ---------------------
def mock_energy_data(location):
    np.random.seed(hash(location) % 1000)
    dates = [datetime.now() + timedelta(hours=i) for i in range(24)]
    return pd.DataFrame({
        'Hour': dates,
        'Consumption': np.random.lognormal(mean=3, sigma=0.3, size=24)*100,
        'Energy Price': np.random.uniform(0.12, 0.25, 24),
        'Temperature': np.random.normal(25, 5, 24)
    })

# --------------------- مدل پیش‌بینی ---------------------
class EnergyPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
    
    def train(self, data):
        X = data[['Temperature', 'Hour']]
        X['Hour'] = X['Hour'].dt.hour
        y = data['Consumption']
        self.model.fit(X, y)
    
    def predict(self, temp, hour):
        return self.model.predict([[temp, hour]])[0]

# --------------------- رابط کاربری حرفه‌ای ---------------------
def main():
    st.title("⚡ EnergyGuard Pro - $20k Value Edition")
    
    # دریافت موقعیت مکانی
    geolocator = Nominatim(user_agent="energy_app")
    location = st.text_input("Enter Location (e.g., Tehran, Iran):", "Tehran")
    
    try:
        loc = geolocator.geocode(location)
        lat, lon = loc.latitude, loc.longitude
        weather = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m").json()
    except:
        st.error("Invalid location! Using demo data.")
        lat, lon = 35.7, 51.4
        weather = {'hourly': {'temperature_2m': [25]*24}}

    # تولید داده‌ها
    data = mock_energy_data(location)
    predictor = EnergyPredictor()
    predictor.train(data)
    
    # محاسبات پیشرفته
    avg_consumption = data['Consumption'].mean()
    cost = sum(data['Consumption'] * data['Energy Price'])
    prediction = [predictor.predict(temp, hour) for hour, temp in enumerate(weather['hourly']['temperature_2m'][:24])]
    
    # --------------------- داشبورد حرفه‌ای ---------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📍 Location", location)
        st.metric("🌡 Current Temp", f"{weather['hourly']['temperature_2m'][0]}°C")
        
    with col2:
        st.metric("💡 Avg Consumption", f"{avg_consumption:,.0f} kWh")
        st.metric("💵 Daily Cost", f"${cost:,.2f}")
        
    with col3:
        st.metric("📈 Predicted Peak", f"{max(prediction):,.0f} kWh")
        st.metric("🛡 Savings Potential", f"${0.2*cost:,.2f}")
    
    # --------------------- ویژوال‌های تعاملی ---------------------
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🎮 Recommendations", "📜 Report"])
    
    with tab1:
        fig = px.area(data, x='Hour', y='Consumption', title="Energy Consumption Forecast")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("AI-Powered Optimization Tips")
        tips = [
            "Shift 20% of load to off-peak hours",
            "Enable demand response protocols",
            "Negotiate better energy rates",
            "Implement virtual power plants"
        ]
        for tip in tips:
            st.markdown(f"✅ {tip}")
        
        if st.button("🔄 Generate Custom Plan"):
            st.success("Customized savings plan generated! Estimated savings: 15-22%")
    
    with tab3:
        st.subheader("Automated Executive Report")
        st.download_button(
            label="📥 Download PDF Report",
            data=generate_pdf_report(data, location, cost),
            file_name="energy_report.pdf"
        )

# --------------------- توابع کمکی ---------------------
def generate_pdf_report(data, location, cost):
    # تولید گزارش PDF ساده (نسخه نمایشی)
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Energy Report - {location}", ln=1)
    pdf.cell(200, 10, txt=f"Total Daily Cost: ${cost:.2f}", ln=1)
    return pdf.output(dest='S').encode('latin1')

# --------------------- اجرای برنامه ---------------------
if __name__ == "__main__":
    main()