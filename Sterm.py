import streamlit as st
import pandas as pd
import plotly.express as px

# Sample data
data = {
    "Date": pd.date_range(start="2023-01-01", periods=6, freq="M"),
    "Sales": [100, 120, 90, 150, 180, 160]
}
df = pd.DataFrame(data)

st.title("Line Chart: Streamlit vs Plotly")

# --- Streamlit's built-in line_chart ---
st.subheader("Using st.line_chart (Streamlit built-in)")
st.line_chart(df.set_index("Date")["Sales"])

# --- Plotly chart ---
st.subheader("Using st.plotly_chart (Plotly)")
fig = px.line(df, x="Date", y="Sales", title="Sales over Time (Plotly)")
st.plotly_chart(fig, use_container_width=True)