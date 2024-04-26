import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import pickle

st.write("""
# California House Price Prediction App

This app predicts the **california House Price**!
""")
st.write('---')

# Loads the california california house california
california=pd.read_csv('housing_california/housing.csv')


def preprocessing_california(california):
    """
    preprocess sequence for the california
    """
    # preprocess the california
    # get median income in readable format for end-user
    california['median_income'] = california['median_income'] * 10000

    # drop all rows missing values
    california.dropna(inplace=True)

    # need to convert near ocean column into numerical for analysis
    df_dummies = pd.get_dummies(california['ocean_proximity'])

    # Ensure column names are string and replace unsupported characters
    df_dummies.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in df_dummies.columns]

    # categorical california being changed into numerical values whether true or false
    california = pd.concat([california, df_dummies], axis=1)

    # ocean proximity column replaced by dummy columns and not needed anymore
    california.drop(['ocean_proximity'], axis=1, inplace=True)

    california.columns = california.columns.str.lower()
    california.columns = california.columns.str.strip()
    california.columns = california.columns.str.replace(' ', '_')

    return california

california =preprocessing_california(california)

X = california.drop(['median_house_value'], axis=1)
Y = california['median_house_value']

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    longitude = st.sidebar.slider('longitude', X.longitude.min(), X.longitude.max(), X.longitude.mean())
    latitude = st.sidebar.slider('latitude', X.latitude.min(), X.latitude.max(), X.latitude.mean())
    housing_media_age = st.sidebar.slider('housing_media_age', X.housing_media_age.min(), X.housing_media_age.max(), X.housing_media_age.mean())
    total_rooms = st.sidebar.slider('total_rooms', X.total_rooms.min(), X.total_rooms.max(), X.total_rooms.mean())
    total_bedrooms = st.sidebar.slider('total_bedrooms', X.total_bedrooms.min(), X.total_bedrooms.max(), X.total_bedrooms.mean())
    population = st.sidebar.slider('population', X.population.min(), X.population.max(), X.population.mean())
    households = st.sidebar.slider('households', X.households.min(), X.households.max(), X.households.mean())
    median_income = st.sidebar.slider('median_income', X.median_income.min(), X.median_income.max(), X.median_income.mean())
    median_house_value = st.sidebar.slider('median_house_value', X.median_house_value.min(), X.median_house_value.max(), X.median_house_value.mean())
    less_than_one_hr_from_ocean  = st.sidebar.slider('1h_ocean', X.1h_ocean.min(), X.1h_ocean.max(), X.1h_ocean.mean())
    inland = st.sidebar.slider('inland', X.inland.min(), X.inland.max(), X.inland.mean())
    island = st.sidebar.slider('island', X.island.min(), X.island.max(), X.island.mean())
    near_bay = st.sidebar.slider('near_bay', X.near_bay.min(), X.near_bay.max(), X.near_bay.mean())
    near_ocean = st.sidebar.slider('near_ocean', X.near_ocean.min(), X.near_ocean.max(), X.near_ocean.mean())
    bedroom_ratio = st.sidebar.slider('bedroom_ratio', X.bedroom_ratio.min(), X.bedroom_ratio.max(), X.bedroom_ratio.mean())
    household_rooms = st.sidebar.slider('household_rooms', X.household_rooms.min(), X.household_rooms.max(), X.household_rooms.mean())

    data = {'longitude':longitude,
            'latitude' : latitude,
            'housing_media_age' : housing_media_age,
            'total_rooms' : total_rooms,
            'total_bedrooms' : total_bedrooms,
            'population' : population ,
            'households' : households ,
            'median_income' : median_income,
            'median_house_value' : median_house_value,
            'less_than_one_hr_from_ocean' : less_than_one_hr_from_ocean,
            'inland' : inland ,
            'island' : island,
            'near_bay' : near_bay ,
            'near_ocean' : near_ocean,
            'bedroom_ratio' : bedroom_ratio,
            'household_rooms' : household_rooms}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# load pre-trained model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')