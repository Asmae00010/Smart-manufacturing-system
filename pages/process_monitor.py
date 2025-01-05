import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.express as px
import os
import json

# Function to save data
def save_data(data, parameter):
    if not os.path.exists('data'):
        os.makedirs('data')
    filename = f'data/steel_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    data.to_csv(filename, index=False)
    metadata = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'parameter': parameter,
        'data_points': len(data),
        'file_path': filename
    }
    with open('data/forecast_metadata.json', 'a') as f:
        json.dump(metadata, f)
        f.write('\n')
    return filename

def generate_data(start_date, periods=100):
    dates = pd.date_range(start=start_date, periods=periods, freq='H')
    
    # Original parameters
    temp = np.random.normal(1450, 30, periods)
    temp = np.clip(temp, 1370, 1540)
    elongation = np.random.normal(20, 2, periods)
    elongation = np.clip(elongation, 15, 25)
    energy = np.random.normal(500, 50, periods)
    energy = np.clip(energy, 400, 600)
    
    # New parameters
    cooling_rate = np.random.normal(50, 5, periods)  # °C/min
    cooling_rate = np.clip(cooling_rate, 40, 60)
    
    contact_pressure = np.random.normal(200, 20, periods)  # MPa
    contact_pressure = np.clip(contact_pressure, 150, 250)
    
    vibration = np.random.normal(2.5, 0.5, periods)  # mm/s
    vibration = np.clip(vibration, 1.5, 3.5)
    
    df = pd.DataFrame({
        'datetime': dates,
        'Temperature': temp,
        'Elongation': elongation,
        'Energy_Consumption': energy,
        'Cooling_Rate': cooling_rate,
        'Contact_Pressure': contact_pressure,
        'Vibration': vibration
    })
    
    return df

def create_forecast(data, parameter, periods=24):
    df_prophet = pd.DataFrame({
        'ds': data['datetime'],
        'y': data[parameter]
    })
    
    model = Prophet(
        interval_width=0.95,
        daily_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        changepoint_range=0.9
    )
    
    model.add_seasonality(
        name='hourly',
        period=24,
        fourier_order=5
    )
    
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)
    
    return forecast, model

def show_page():
    st.title("Steel Manufacturing Monitoring System")
    st.markdown("""
    This dashboard monitors six critical parameters in steel manufacturing:
    - *Temperature*: Melting temperature (1370-1540°C)
    - *Elongation*: Material elongation percentage (15-25%)
    - *Energy Consumption*: Energy used per ton (400-600 kWh/ton)
    - *Cooling Rate*: Cooling speed (40-60°C/min)
    - *Contact Pressure*: Applied pressure (150-250 MPa)
    - *Vibration*: Equipment vibration (1.5-3.5 mm/s)
    """)

    # Generate sample data
    if 'monitor_data' not in st.session_state:
        start_date = datetime.now() - timedelta(days=4)
        st.session_state.monitor_data = generate_data(start_date)

    # Parameter ranges and units
    parameter_info = {
        "Temperature": {"range": (1370, 1540), "unit": "°C"},
        "Elongation": {"range": (15, 25), "unit": "%"},
        "Energy_Consumption": {"range": (400, 600), "unit": "kWh/ton"},
        "Cooling_Rate": {"range": (40, 60), "unit": "°C/min"},
        "Contact_Pressure": {"range": (150, 250), "unit": "MPa"},
        "Vibration": {"range": (1.5, 3.5), "unit": "mm/s"}
    }

    # Sidebar controls
    st.sidebar.header("Controls")
    parameter = st.sidebar.selectbox(
        "Select Parameter",
        list(parameter_info.keys()),
        format_func=lambda x: x.replace('_', ' ')
    )

    # Add forecast explanation in sidebar
    st.sidebar.markdown("""
    ### How the Forecast Works

    The forecasting system uses Facebook's Prophet algorithm, which is particularly good at:

    1. *Time Series Decomposition*:
       - Trend: Overall direction
       - Seasonality: Daily, weekly patterns
       - Holiday effects (if applicable)

    2. *Key Components*:
       - Changepoint detection for trend changes
       - Multiple seasonality patterns
       - Robust to missing data
       - Handles outliers well
    """)

    # Create main dashboard layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"{parameter.replace('_', ' ')} Monitoring")
        
        # Historical data plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.monitor_data['datetime'],
            y=st.session_state.monitor_data[parameter],
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add range lines
        param_range = parameter_info[parameter]["range"]
        fig.add_hline(y=param_range[0], line_dash="dash", line_color="red", annotation_text="Min Limit")
        fig.add_hline(y=param_range[1], line_dash="dash", line_color="red", annotation_text="Max Limit")
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=f"{parameter.replace('_', ' ')} ({parameter_info[parameter]['unit']})",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast section
        st.subheader("Forecast Analysis")
        forecast, model = create_forecast(st.session_state.monitor_data, parameter)
        
        # Save the data and forecast
        saved_file = save_data(st.session_state.monitor_data, parameter)
        st.info(f"Data saved to: {saved_file}")
        
        # Forecast plot
        fig_forecast = go.Figure()
        
        fig_forecast.add_trace(go.Scatter(
            x=st.session_state.monitor_data['datetime'],
            y=st.session_state.monitor_data[parameter],
            name='Historical',
            line=dict(color='blue')
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'][-24:],
            y=forecast['yhat'][-24:],
            name='Forecast',
            line=dict(color='green', dash='dash')
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'][-24:],
            y=forecast['yhat_upper'][-24:],
            fill=None,
            mode='lines',
            line_color='rgba(0,255,0,0)',
            showlegend=False
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'][-24:],
            y=forecast['yhat_lower'][-24:],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,255,0,0)',
            name='95% Confidence'
        ))
        
        fig_forecast.update_layout(
            xaxis_title="Time",
            yaxis_title=f"{parameter.replace('_', ' ')} ({parameter_info[parameter]['unit']})",
            height=400
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    with col2:
        st.subheader("Statistics")
        
        # Current value card
        current_value = st.session_state.monitor_data[parameter].iloc[-1]
        st.metric(
            label="Current Value",
            value=f"{current_value:.2f} {parameter_info[parameter]['unit']}"
        )
        
        # Statistics
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
            'Value': [
                f"{st.session_state.monitor_data[parameter].mean():.2f}",
                f"{st.session_state.monitor_data[parameter].std():.2f}",
                f"{st.session_state.monitor_data[parameter].min():.2f}",
                f"{st.session_state.monitor_data[parameter].max():.2f}"
            ]
        })
        st.table(stats_df)
        
        # Forecast statistics
        st.subheader("Forecast Statistics")
        forecast_stats = pd.DataFrame({
            'Metric': ['Mean Forecast', 'Upper Bound', 'Lower Bound'],
            'Value': [
                f"{forecast['yhat'].mean():.2f}",
                f"{forecast['yhat_upper'].mean():.2f}",
                f"{forecast['yhat_lower'].mean():.2f}"
            ]
        })
        st.table(forecast_stats)
        
        # Distribution plot
        fig_dist = px.histogram(
            st.session_state.monitor_data,
            x=parameter,
            title=f"{parameter.replace('_', ' ')} Distribution"
        )
        fig_dist.update_layout(height=300)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Download buttons for data
    st.sidebar.markdown("### Download Data")
    if st.sidebar.button("Download CSV"):
        csv = st.session_state.monitor_data.to_csv(index=False)
        st.sidebar.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f'steel_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    st.set_page_config(page_title="Steel Manufacturing Monitor", layout="wide")
    show_page()