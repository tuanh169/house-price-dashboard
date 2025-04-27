import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="Denmark House Price Dashboard", layout="wide")

# Load data
df = pd.read_csv('C:/Users/Vuong/Downloads/train_merged.csv')

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Sidebar - Filters
st.sidebar.header("Filters")
selected_city = st.sidebar.selectbox("Select City/District", sorted(df['city'].unique()))
price_range = st.sidebar.slider("Select Price Range", 
                                 int(df['price'].min()), 
                                 int(df['price'].max()), 
                                 (int(df['price'].min()), int(df['price'].max())))

property_type = st.sidebar.multiselect("Property Type", df['property_type'].unique(), default=df['property_type'].unique())

# Filter data based on sidebar selections
filtered_df = df[(df['city'] == selected_city) &
                 (df['price'].between(price_range[0], price_range[1])) &
                 (df['property_type'].isin(property_type))]

# Main title
st.title("🏠 Denmark Real Estate Dashboard")

# Show average price
st.header(f"Average House Price in {selected_city}")
avg_price = filtered_df['price'].mean()
st.metric(label="Average Price", value=f"${avg_price:,.0f}")

# Bar chart - Price by District
st.subheader("House Prices by District")
fig_bar = px.bar(filtered_df.groupby('district')['price'].mean().reset_index(),
                 x='district', y='price', color='district', title='Average Price by District')
st.plotly_chart(fig_bar, use_container_width=True)

# Heatmap (if available coordinates)
if 'latitude' in df.columns and 'longitude' in df.columns:
    st.subheader("Heatmap of House Prices")
    fig_map = px.density_mapbox(filtered_df, lat='latitude', lon='longitude', z='price', radius=10,
                                center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
                                zoom=9, mapbox_style="stamen-terrain")
    st.plotly_chart(fig_map, use_container_width=True)

# Line chart - Price trend over time
st.subheader("Price Trend Over Time")
if 'date_sold' in df.columns:
    df['date_sold'] = pd.to_datetime(df['date_sold'])
    fig_line = px.line(filtered_df.groupby('date_sold')['price'].mean().reset_index(),
                       x='date_sold', y='price', title='Average Price Over Time')
    st.plotly_chart(fig_line, use_container_width=True)

# Prediction Section
st.header("🔢 Predict House Price")
st.markdown("Input property details below to predict price:")

col1, col2 = st.columns(2)
with col1:
    size = st.number_input('Size (sqm)', min_value=10, max_value=500, value=100)
    rooms = st.number_input('Number of Rooms', min_value=1, max_value=10, value=3)
with col2:
    year_built = st.number_input('Year Built', min_value=1800, max_value=2025, value=2000)
    lot_size = st.number_input('Lot Size (sqm)', min_value=0, max_value=5000, value=500)

# Button to predict
if st.button("Predict Price"):
    input_data = np.array([[size, rooms, year_built, lot_size]])
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ${prediction[0]:,.0f}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit 🚀")
