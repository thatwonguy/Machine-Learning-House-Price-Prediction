import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import pickle
import folium

# remove warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# California House Price Prediction App

This app predicts **california House Prices** based on your chosen inputs using Machine Learning.
""")
st.write('---')

# Loads the california california house california
california=pd.read_csv('housing_data/housing.csv')


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
    # df_dummies = pd.get_dummies(california['ocean_proximity'])

    # Ensure column names are string and replace unsupported characters
    california.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in california.columns]

    # categorical california being changed into numerical values whether true or false
    # california = pd.concat([california, df_dummies], axis=1)

    # ocean proximity column replaced by dummy columns and not needed anymore
    california.drop(['ocean_proximity'], axis=1, inplace=True)

    california.columns = california.columns.str.lower()
    california.columns = california.columns.str.strip()
    california.columns = california.columns.str.replace(' ', '_')

    california['bedroom_ratio'] = california['total_bedrooms'] / california['total_rooms']
    california['household_rooms'] = california['total_rooms'] / california['households']

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
    housing_median_age= st.sidebar.slider('housing_median_age', X.housing_median_age.min(), X.housing_median_age.max(), X.housing_median_age.mean())
    total_rooms = st.sidebar.slider('total_rooms', X.total_rooms.min(), X.total_rooms.max(), X.total_rooms.mean())
    total_bedrooms = st.sidebar.slider('total_bedrooms', X.total_bedrooms.min(), X.total_bedrooms.max(), X.total_bedrooms.mean())
    population = st.sidebar.slider('population', X.population.min(), X.population.max(), X.population.mean())
    households = st.sidebar.slider('households', X.households.min(), X.households.max(), X.households.mean())
    median_income = st.sidebar.slider('median_income', X.median_income.min(), X.median_income.max(), X.median_income.mean())
    bedroom_ratio = st.sidebar.slider('bedroom_ratio', X.bedroom_ratio.min(), X.bedroom_ratio.max(), X.bedroom_ratio.mean())
    household_rooms = st.sidebar.slider('household_rooms', X.household_rooms.min(), X.household_rooms.max(), X.household_rooms.mean())

    data = {'longitude':longitude,
            'latitude' : latitude,
            'housing_median_age' : housing_median_age,
            'total_rooms' : total_rooms,
            'total_bedrooms' : total_bedrooms,
            'population' : population ,
            'households' : households ,
            'median_income' : median_income,
            'bedroom_ratio' : bedroom_ratio,
            'household_rooms' : household_rooms}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


def show_map(latitude, longitude):
    # Create a map centered around California
    california_map = folium.Map(location=[latitude, longitude], zoom_start=6)

    # Add a marker for the specified latitude and longitude
    folium.Marker([latitude, longitude], popup='Location').add_to(california_map)

    # Render the map
    st.write(california_map)

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Print Map location
st.header('Map showing prediction location proximity')
show_map(df['latitude'], df['longitude'])


# load pre-trained model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
# Apply Model to Make Prediction
prediction = model.predict(df)[0]
formatted_prediction = "${:.2f}".format(prediction)


st.header('Machine Learning Predicted House Value')
st.write(formatted_prediction)
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