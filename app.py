import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import pickle
import folium

# Remove warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# California House Price Prediction App

This app predicts **California House Prices** based on your chosen inputs using Machine Learning.
""")
st.write('---')

# Loads the California housing data
california = pd.read_csv('housing_data/housing.csv')

def preprocessing_california(california):
    """
    Preprocess sequence for the California housing data.
    """
    # Preprocess the data
    # Get median income in a readable format for end-user
    california['median_income'] = california['median_income'] * 10000

    # Drop all rows with missing values
    california.dropna(inplace=True)

    # Ensure column names are string and replace unsupported characters
    california.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in california.columns]

    # Ocean proximity column replaced by dummy columns and not needed anymore
    california.drop(['ocean_proximity'], axis=1, inplace=True)

    california.columns = california.columns.str.lower()
    california.columns = california.columns.str.strip()
    california.columns = california.columns.str.replace(' ', '_')

    # Additional features
    california['bedroom_ratio'] = california['total_bedrooms'] / california['total_rooms']
    california['household_rooms'] = california['total_rooms'] / california['households']

    return california

california = preprocessing_california(california)

X = california.drop(['median_house_value'], axis=1)
Y = california['median_house_value']

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    longitude = st.sidebar.slider('Longitude', X.longitude.min(), X.longitude.max(), X.longitude.mean())
    latitude = st.sidebar.slider('Latitude', X.latitude.min(), X.latitude.max(), X.latitude.mean())
    housing_median_age = st.sidebar.slider('Housing Median Age', X.housing_median_age.min(), X.housing_median_age.max(), X.housing_median_age.mean())
    total_rooms = st.sidebar.slider('Total Rooms', X.total_rooms.min(), X.total_rooms.max(), X.total_rooms.mean())
    total_bedrooms = st.sidebar.slider('Total Bedrooms', X.total_bedrooms.min(), X.total_bedrooms.max(), X.total_bedrooms.mean())
    population = st.sidebar.slider('Population', X.population.min(), X.population.max(), X.population.mean())
    households = st.sidebar.slider('Households', X.households.min(), X.households.max(), X.households.mean())
    median_income = st.sidebar.slider('Median Income', X.median_income.min(), X.median_income.max(), X.median_income.mean())
    bedroom_ratio = st.sidebar.slider('Bedroom Ratio', X.bedroom_ratio.min(), X.bedroom_ratio.max(), X.bedroom_ratio.mean())
    household_rooms = st.sidebar.slider('Household Rooms', X.household_rooms.min(), X.household_rooms.max(), X.household_rooms.mean())

    data = {'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'bedroom_ratio': bedroom_ratio,
            'household_rooms': household_rooms}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


def show_map(latitude, longitude):
    # Create a map centered around California
    california_map = folium.Map(location=[latitude, longitude], zoom_start=6)

    # Add a marker for the specified latitude and longitude
    folium.Marker([latitude, longitude], popup='Location').add_to(california_map)

    # Render the map to HTML
    map_html = california_map._repr_html_()

    # Display the HTML map in Streamlit
    st.components.v1.html(map_html, width=700, height=500)

# Main Panel

# Print specified input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Print Map location
st.header('Map showing prediction location proximity')
map = show_map(df['latitude'], df['longitude'])


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
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
st.image('shap_summary_plot.png', use_column_width=True)
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot_bar.png')
st.image('shap_summary_plot_bar.png', use_column_width=True)
