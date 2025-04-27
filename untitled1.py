

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load City Mapping ---
city_mapping_df = pd.read_csv('city_mapping.csv')  # file chứa city_code, city_name
city_code_to_name = dict(zip(city_mapping_df['city_code'], city_mapping_df['city_name']))
city_name_to_code = dict(zip(city_mapping_df['city_name'], city_mapping_df['city_code']))

# --- Load Model & Data ---
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_model.pkl")
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("train_merged.csv")
    return df

model = load_model()
df = load_data()

# --- Sidebar: User Input ---  
st.sidebar.title("🔍 Nhập thông tin dự đoán")  
area = st.sidebar.number_input("Diện tích (m²)", min_value=10, max_value=1000, value=120)  
rooms = st.sidebar.slider("Số phòng", 1, 10, 3)  
zipcode = st.sidebar.selectbox("Mã vùng (zipcode)", df['zipcode'].unique())  
house_type = st.sidebar.selectbox("Loại nhà", df['house_type'].unique())  
sales_type = st.sidebar.selectbox("Loại bán", df['sales_type'].unique())  
year_build = st.sidebar.number_input("Năm xây dựng", min_value=1900, max_value=2025, value=2000)  
sqm_price = st.sidebar.number_input("Giá/m²", min_value=0, value=0)  

# city phần sửa:
city_name = st.sidebar.selectbox("Thành phố", list(city_name_to_code.keys()))
city = city_name_to_code[city_name]  # lấy city_code để predict

region = st.sidebar.selectbox("Vùng", df['region'].unique())  
nom_interest_rate = st.sidebar.number_input("Lãi suất cho vay (%)", value=0.0)  
dk_ann_infl_rate = st.sidebar.number_input("Lạm phát dự kiến (%)", value=0.0)  
yield_on_mortgage_credit_bonds = st.sidebar.number_input("Lợi suất trái phiếu tín dụng thế chấp (%)", value=0.0)  

# --- Dự đoán ---
input_data = pd.DataFrame({
    'house_type': [house_type],
    'sales_type': [sales_type],
    'year_build': [year_build],
    'no_rooms': [rooms],
    'sqm': [area],
    'sqm_price': [sqm_price],
    'zip_code': [zipcode],
    'city': [city],
    'area': [area],
    'region': [region],
    'nom_interest_rate%': [nom_interest_rate],
    'dk_ann_infl_rate%': [dk_ann_infl_rate],
    'yield_on_mortgage_credit_bonds%': [yield_on_mortgage_credit_bonds],
})

predicted_price = model.predict(input_data)[0]

# --- Hiển thị kết quả ---
st.title("🏠 Dự đoán giá nhà ở Đan Mạch")
st.subheader("Giá nhà dự đoán:")
st.success(f"💰 {predicted_price:,.0f} DKK")

# --- Biểu đồ EDA ---
st.subheader("Phân tích dữ liệu (EDA)")

with st.expander("Heatmap Tương Quan"):
    corr = df.corr(numeric_only=True)
    if 'purchaseprice' in corr.columns:
        top_corr = corr['purchaseprice'].abs().sort_values(ascending=False)[1:11].index
        selected_corr = corr.loc[top_corr, top_corr]
    else:
        selected_corr = corr
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(selected_corr, annot=True, cmap="coolwarm", linewidths=0.5, square=True, fmt=".2f")
    ax.set_title("Heatmap các biến tương quan")
    st.pyplot(fig)

with st.expander("Biểu đồ phân bố diện tích"):
    fig2, ax2 = plt.subplots()
    sns.histplot(df['area'], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

with st.expander("ROI theo khu vực"):
    roi_data = df.groupby('zipcode')['purchaseprice'].mean()
    st.line_chart(roi_data)

# --- User Guide ---
st.sidebar.markdown("---")
st.sidebar.markdown(" **Hướng dẫn:** Nhập thông tin ở trên để dự đoán giá nhà. Xem biểu đồ và phân tích bên dưới.")
