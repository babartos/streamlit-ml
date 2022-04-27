import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from IPython.display import IFrame
import pydeck as pdk
import requests
import urllib.parse
## HELPER FUNCTIONS:
# initialize session state:
def intilize_session():
    if 'price' not in st.session_state:
        st.session_state.price = '$0'
    if 'form_header' not in st.session_state:
        st.session_state.form_header = 'Estimate your California house price:'

## longitude and latitude plotting over California:
def scatter_plot(housing):
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(x=housing['longitude'], y=housing['latitude'], alpha=0.4,s=housing['population']/100, label='population', c=housing['housing_inflination'], cmap=plt.get_cmap('jet'))
    plt.colorbar()
    st.pyplot(fig)

## display pydeck map. Provides most styling
def display_map():
    # housing data point map:
    df.dropna(inplace=True)
    view = pdk.ViewState(latitude=36.7783, longitude=-119.4179, pitch=35, zoom=5)
    # layer
    column_layer = pdk.Layer('ColumnLayer',
        data=df,
        get_position=['longitude', 'latitude'],
        get_elevation='housing_inflination',
        elevation_scale=.10,
        radius=800,
        get_fill_color=[255, 165, 0, 80],
        pickable=True,
        auto_highlight=True)
    # render map
    st.pydeck_chart(pdk.Deck(layers=column_layer, initial_view_state=view))

def display_mapbox_map():
    df_map = pd.DataFrame(data={'lat': df['latitude'], 'lon': df['longitude']})
    print(df_map)
    st.map(df_map)

# dataframe cleaning
def clean_dataframe(housing):
    # # account for inflination rate:
    # get rid of null values:
    housing.dropna(inplace=True)
    # get amt bedrooms using amt rooms data, assume half the rooms are bedrooms
    housing['impute_bedrooms'] = housing['total_rooms'] / housing['households'] / 2
    # use label encoder to encode categorical data:
    encoder = OrdinalEncoder()
    housing_cat_encoded = encoder.fit_transform(housing[['ocean_proximity']])
    housing['ocean_prox_encoded'] = housing_cat_encoded
    enc_cats = encoder.categories_
    print(enc_cats[0])
    ocean_prox_map = {'<1H OCEAN': 0, 'INLAND':1,  'ISLAND': 2,  'NEAR BAY': 3, 'NEAR OCEAN': 4}
    # drop columns not in use
    housing.drop(columns=['population', 'ocean_proximity', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'households', 'median_income'], axis=1, inplace=True)
    #return the value
    housing_target = np.array(housing['housing_inflination'])
    housing.drop(columns=['median_house_value', 'housing_inflination'], axis=1, inplace=True)
    housing_train = housing.to_numpy()
    print(housing)
    return housing_train, housing_target, ocean_prox_map

# get the linear regression model
def get_model_linearRegression(housing_train, housing_target):
    lin_reg = LinearRegression()
    lin_reg.fit(housing_train, housing_target)
    return lin_reg

def submit_address():
    # https://stackoverflow.com/questions/25888396/how-to-get-latitude-longitude-with-python
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
    response = requests.get(url).json()
    # error handling
    if not response:
        st.session_state.form_header = '** ERROR COULD NOT FIND ADDRESS'
        return
    else:
        st.session_state.form_header = 'Estimate your house price:'
    # use model to predict
    lat = response[0]['lat']
    lon = response[0]['lon']
    x = np.array([[float(lon), float(lat), float(user_bedrooms), float(option_int)]])
    y = abs(housing_model.predict(x)[0])
    # format the price output
    y = '$' + '{:,}'.format(y)  # add commas and $
    y = y.rsplit('.')[0]
    # update the session state
    st.session_state.price = y

## SEE TEXTBOOK Chapter 2 'End to End Machine Learning model'
# hands on machine learning with scikit-learn and tensorflow
# DRIVER CODE:
intilize_session()
st.title('California House Prices')
st.text('By Matthew Brent Bartos, Jarod William Cox, and Mitchell Gibson Morrison')
# get dataframe:
df = pd.read_csv('./housing.csv')
df['housing_inflination'] = df['median_house_value'] * 3  # 2.5x inlfination rate from online calculation
# display map:
display_map()
# clean dataframe:
housing_train, housing_target, ocean_prox_map = clean_dataframe(df)
# get model:
housing_model = get_model_linearRegression(housing_train, housing_target) # y = housing_model.predict(np.array([[-122.23,37.88,8]]))
## form submission
# form header:
st.title(str(st.session_state.price))
st.subheader(st.session_state.form_header)
# address form:
address = st.text_input('Enter address', placeholder=None)
user_bedrooms = st.text_input('Enter amount of bedrooms in household', placeholder=None)
option = st.selectbox(
     'How close are you to the ocean?',
     ('<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'))
option_int = ocean_prox_map[option]
st.button(label='Submit', key='submit2', on_click=submit_address, args=None, kwargs=None)
#st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.write('A Linear Regression Model was utilized to predict housing price using longitude, latitude, bedrooms, and ocean proximity')