# Loading in all the required libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import joblib
from utils import create_dataset

# loading in the data
data = pd.read_csv('clean_data-2.csv')

# limiting the year column to 2020
data_1 = data[data['year'] <= 2020]
# another dataframe for the next 4 years
data_2 = data[data['year'] > 2020]

# setting the title for the page
st.title('Price Forecast Analysis')


# loading pipelines and model
cat_encoder = joblib.load('cat_encoder.joblib')
feature_scaler = joblib.load('feature_scaler.joblib')
price_scaler = joblib.load('price_scaler.joblib')
model = keras.models.load_model('forecast_model.h5')

# List of unique values for stateDescription and sectorName
unique_states = sorted(data_1['stateDescription'].unique())
unique_sectors = sorted(data_1['sectorName'].unique())



# Sidebar widgets for user selection
state = st.sidebar.selectbox('State Description', unique_states)
sector = st.sidebar.selectbox('Sector Name', unique_sectors)

st.subheader("Data Preview")

@st.cache_data
# Data filtering based on user selection
def filter_data(data_1, state, sector):
    return data_1[(data_1['stateDescription'] == state) & (data_1['sectorName'] == sector)].reset_index(drop=True)

# Filtered data based on user selection
filtered_data = filter_data(data_1, state, sector)


# Displaying the filtered data in a container
with st.container():
    st.write(f"Data for {state} and {sector}")
    st.dataframe(filtered_data, use_container_width=True)

st.subheader("Data Analysis")
# A button to show some analysis 
if st.button('Show Analysis', key='analysis'):
    with st.container():
        st.write('Analysis of the data')
        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        sns.lineplot(x='year', y='price', data=filtered_data, ax=ax[0])
        ax[0].set_title('Price vs Year')
        sns.lineplot(x='year', y='sales', data=filtered_data, ax=ax[1])
        ax[1].set_title('Sales vs Year')
        st.pyplot(fig)

        # showing correalation matrix
        st.write('Correlation Matrix')
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(filtered_data[['price', 'revenue', 'sales', 'customers']].corr(), annot=True)
        st.pyplot(fig)

        # showing linecharts
        st.write('Line Charts')
        st.line_chart(filtered_data[['price', 'revenue', 'sales', 'customers']])    

st.subheader("Forecast Analysis")
# A button to show the forecast
if st.button('Show Forecast', key='forecast'):
    with st.container():
        st.write('Forecast for the data till 2024')
        # Forecasting the price
        X = data_2[['year', 'month', 'stateDescription', 'sectorName', 'customers', 'revenue', 'sales']]
        # encoding the categorical features [stateDescription, sectorName]
        cat_features = ['stateDescription', 'sectorName']
        X[cat_features] = cat_encoder.transform(X[cat_features])
        # scaling the features
        X_scaled = feature_scaler.transform(X.values)
        # the price column
        y = data_2['price']
        # reshaping the data for the model
        X_scaled, y = create_dataset(X_scaled, y)
        # making the forecast
        y_pred = model.predict(X_scaled)
        # inverse scaling the forecasted price
        y_pred = price_scaler.inverse_transform(y_pred)
        # reducing the dshape of filtered data to match the forecasted price
        result_df = data_2.iloc[4:].reset_index(drop=True)
        result_df['forecasted_price'] = y_pred
        result_df['price'] = y
        # filtering the data for the selected state and sector
        result_df = result_df[(result_df['stateDescription'] == state) & (result_df['sectorName'] == sector)].reset_index(drop=True)
        # plotting the actual and forecasted price
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.lineplot(x='year', y='price', data=result_df, ax=ax, label='Actual Price')
        sns.lineplot(x='year', y='forecasted_price', data=result_df, ax=ax, label='Forecasted Price')
        ax.set_title('Actual vs Forecasted Price')
        st.pyplot(fig)