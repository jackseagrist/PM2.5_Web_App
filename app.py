import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import plotly_express as px
import folium
from folium.plugins import HeatMap
import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
from pandas import read_csv
import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import ast
import csv
import requests
import tensorflow as tf

# Load Model
@st.cache(allow_output_mutation=True)
def load_my_model():
    model = tf.keras.models.load_model('all_v4_2')
    #model._make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    #session = K.get_session()
    return model#, session

# Date selection
today_date = datetime.date.today()
day_limit = today_date #+ datetime.timedelta(days=-1)
date = st.date_input('Date', max_value=day_limit)


# Button to run prediction
if st.button('Run Prediction'):
    model = load_my_model()
    # ----- Part 1 CIMIS ----- #
    # CIMIS WEB API https://et.water.ca.gov/Rest/Index
    cimis_api = 'http://et.water.ca.gov/api/station/'  # API for individual station data
    api_key = '273cd116-fc49-462c-9104-ea513ccd1192'

    cimis_api_1 = 'http://et.water.ca.gov/api/data?appKey='
    cimis_api_2 = '&targets='
    cimis_api_3 = '&startDate='
    cimis_api_4 = '&endDate='
    cimis_api_5 = '&dataItems='

    # previous day
    prev_day = date + datetime.timedelta(days=0)
    prev_day = prev_day.strftime("%Y-%m-%d")

    # 3 days before
    prev_day3 = date + datetime.timedelta(days=-3)
    prev_day3 = prev_day3.strftime("%Y-%m-%d")

    targets = '157,254'  # ,253,104,111,211,171,191,178,213,157,187,109,144,158,83,170,247,47,139,121,212'  From file 47.0,77.0,83.0,103.0,104.0,109.0,111.0,121.0,139.0,144.0,157.0,158.0,171.0,178.0,187.0,191.0,211.0,212.0,213.0,247.0,253.0,254.0,
    startDate = prev_day3
    endDate = prev_day
    dataItems = 'day-dew-pnt,day-air-tmp-avg,day-vap-pres-avg,day-wind-run,day-wind-spd-avg,day-precip'
    measure = '&unitOfMeasure=M'

    # Get data from CIMIS api
    full_request = cimis_api_1 + api_key + cimis_api_2 + targets + cimis_api_3 + startDate + cimis_api_4 + endDate + cimis_api_5 + dataItems + measure
    cimis_data = requests.get(full_request)
    cimis_data_json = cimis_data.json()

    # Import latlon
    latlon = pd.read_csv('latlon_dict.csv')
    latlon = latlon[['Number', 'name', 'latitude', 'longitude']]
    latlon = latlon.dropna()
    latlon = latlon.reset_index(drop=True)

    # create a function that loops through and pulls latest hour of the day record for each station get data into a format for making prediction
    data_length = len(cimis_data_json['Data']['Providers'][0]['Records'])
    data = cimis_data_json['Data']['Providers'][0]['Records']
    cimis_data_df = pd.DataFrame(
        columns=['date', 'pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'rain', 'lat', 'lon', 'station'])

    for i in range(data_length):
        # Get list of data dictionaries
        record = data[i]

        # Get data points from list of data dictionaries
        date = record['Date']
        dew = record['DayDewPnt']['Value']
        temp = record['DayAirTmpAvg']['Value']
        press = record['DayVapPresAvg']['Value']
        wnd_dir = record['DayWindRun']['Value']
        wnd_spd = record['DayWindSpdAvg']['Value']
        rain = record['DayPrecip']['Value']
        station = float(record['Station'])

        cimis_data_df = cimis_data_df.append(
            {'date': date, 'dew': dew, 'temp': temp, 'press': press, 'wnd_dir': wnd_dir, 'wnd_spd': wnd_spd,
             'rain': rain,
             'station': station}, ignore_index=True)

    # add in the Lat Lon from the other dictionary

    for i in range(len(cimis_data_df)):
        number = cimis_data_df.iloc[i]['station']

        latlonrecord = latlon[latlon['Number'] == number]

        lat = latlonrecord['latitude']
        lon = latlonrecord['longitude']

        cimis_data_df.at[i, 'lat'] = lat
        cimis_data_df.at[i, 'lon'] = lon

    st.write(cimis_data_df)
    # ----- Part 2 EPA -----#
    # Collect Air Pollution Data to predict
    # https://aqs.epa.gov/aqsweb/documents/data_api.html#signup
    # https://docs.airnowapi.org/
    startDate = prev_day3 + str('T00')
    endDate = prev_day + str('T23')
    epa_api_1 = 'https://www.airnowapi.org/aq/data/?startDate='
    epa_api_2 = '&endDate='
    epa_api_3 = '&parameters=PM25&BBOX=-123.006465,37.429665,-121.869380,38.512226&dataType=B&format=application/json&verbose=1&nowcastonly=0&includerawconcentrations=0&API_KEY=BE986A1F-8C49-47A5-B60E-B42EC0D080A5'

    epa_full_request = epa_api_1 + startDate + epa_api_2 + endDate + epa_api_3
    epa_data = requests.get(epa_full_request)
    epa_data_json = epa_data.json()

    # Turn Json into DataFrame
    epa_data_length = len(epa_data_json)
    epa_data_df = pd.DataFrame(columns=['date', 'pollution', 'SiteName', 'lat', 'lon'])

    st.write("check")
    for i in range(epa_data_length):
        # Get individual data dictionay
        record = epa_data_json[i]

        # Get all data points
        date = record['UTC']
        pollution = record['Value']
        SiteName = record['SiteName']
        lat = record['Latitude']
        lon = record['Longitude']

        epa_data_df = epa_data_df.append(
            {'date': date, 'pollution': pollution, 'SiteName': SiteName, 'lat': lat, 'lon': lon}, ignore_index=True)

    # Dictionary mapping cimis stations to epa stations ****** IMPORTANT**************
    cimis_epa_dict = {
        157.0: 'San Rafael',
        254.0: 'Oakland West',
    }

    # Filter based on stations in dictionary
    epa_data_df = epa_data_df[epa_data_df.SiteName.isin(cimis_epa_dict.values())]

    # Loop through both stations and get np arrays of averages
    days = pd.date_range(start=prev_day3, end=prev_day, freq='D')
    epa_data_df_avg = pd.DataFrame(columns=['date', 'pollution', 'SiteName', 'lat', 'lon'])

    st.write("check")

    for i in range(len(cimis_epa_dict.values())):
        # Get Site
        SiteName = list(cimis_epa_dict.values())[i]

        # Filter based on station
        dataset = epa_data_df[epa_data_df['SiteName'] == SiteName]
        dataset['date'] = pd.to_datetime(dataset['date'])

        # Get average
        average = np.array(dataset.groupby(pd.Grouper(key='date', freq='D')).mean())

        # Create Dataframe
        station_df = pd.DataFrame(days, columns=['date'])

        # Arrays to fill DF
        pollution = []
        for j in range(len(days)):
            poll_value = average[j][0]
            pollution.append(poll_value)
        sites = np.full((len(days), 1), SiteName)
        lat = np.full((len(days), 1), average[0][1])
        lon = np.full((len(days), 1), average[0][2])

        station_df['pollution'] = pd.DataFrame(pollution)
        station_df['SiteName'] = pd.DataFrame(sites)
        station_df['lat'] = pd.DataFrame(lat)
        station_df['lon'] = pd.DataFrame(lon)

        # Append to epa_data_df_avg
        epa_data_df_avg = epa_data_df_avg.append(station_df, ignore_index=True)

    st.write("check")
    # Create mapping to populate all data into Cimis df

    for i in range(len(cimis_data_df)):
        site = cimis_data_df.iloc[i]['station']
        date = cimis_data_df.iloc[i]['date']
        epa_name = cimis_epa_dict[site]

        # Filter EPA data based on name
        filtered_epa = epa_data_df_avg[epa_data_df_avg['SiteName'] == epa_name]
        value = filtered_epa[filtered_epa['date'] == date]['pollution']

        cimis_data_df.at[i, 'pollution'] = value

    st.write("check")
    st.write(cimis_data_df)

    # Do code to prep data
    cimis_data_df.set_index('date', inplace=True)
    cimis_data_df.index.name = 'date'
    cimis_data_df.fillna(0, inplace=True)

    # Convert number columns to float
    cimis_data_df['pollution'] = pd.to_numeric(cimis_data_df['pollution'])
    cimis_data_df['dew'] = pd.to_numeric(cimis_data_df['dew'])
    cimis_data_df['temp'] = pd.to_numeric(cimis_data_df['temp'])
    cimis_data_df['press'] = pd.to_numeric(cimis_data_df['press'])
    cimis_data_df['wnd_dir'] = pd.to_numeric(cimis_data_df['wnd_dir'])
    cimis_data_df['wnd_spd'] = pd.to_numeric(cimis_data_df['wnd_spd'])
    cimis_data_df['rain'] = pd.to_numeric(cimis_data_df['rain'])
    cimis_data_df['lat'] = pd.to_numeric(cimis_data_df['lat'])
    cimis_data_df['lon'] = pd.to_numeric(cimis_data_df['lon'])

    # Lets normalize all features, and remove the weather variables for the day to be predicted.
    def s_to_super(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # Only Selecting One Station right now
    #TODO don't drop station here, need to update and later for prediction
    # station_one = list(cimis_epa_dict.keys())[0]
    # cimis_data_df = cimis_data_df[cimis_data_df['station'] == station_one]
    # Drop Station Column
    cimis_data_df = cimis_data_df.drop(axis=1, labels=['station'])

    values = cimis_data_df.values
    encoder = preprocessing.LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = s_to_super(scaled, 1, 1)

    reframed.drop(reframed.columns[[10, 11, 12, 13, 14, 15, 16, 17]], axis=1, inplace=True)

    values = reframed.values
    X = values[:, :-1]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # ----- Predict ----- #
    pred = model.predict(X)
    X = X.reshape((X.shape[0], X.shape[2]))
    inv_yhat = concatenate((pred, X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    inv_y = scaler.inverse_transform(X)
    inv_y = inv_y[:, 0]


    # ----- Add Values to Map ----- #
    cimis_data_df = cimis_data_df.reset_index(inplace=False)

    # Create Final dataframe for mapping
    cimis_data_df['predictions'] = np.nan

    for i in range(len(cimis_data_df) - 1):
        value = float(inv_yhat[i])
        cimis_data_df.at[i, 'predictions'] = value

    cimis_data_df = cimis_data_df.dropna()

    # Only show day 2 days ago
    map_date = date
    #map_date = map_date - datetime.timedelta(days=2)
    #map_date = date.strftime("%Y-%m-%d")

    cimis_data_df_map = cimis_data_df[cimis_data_df['date'] == map_date]

    base_map = folium.Map([36.7783, -119.4179], zoom_start=6, tiles='cartodbpositron')

    for i in range(len(cimis_data_df_map)):
        row = cimis_data_df_map.iloc[i]
        string = 'PM2.5 in micrograms/m^3' + '\n' + 'Predicted:' + str(
            np.round(row['predictions'], decimals=1)) + '\n' + 'True:' + str(np.round(row['pollution'], decimals=1))
        color_value = row['predictions']
        if color_value < 50:
            color = 'green'
        elif 50 < color_value < 100:
            color = 'yellow'
        elif 100 < color_value < 150:
            color = 'orange'
        elif 150 < color_value:
            color = 'red'
        folium.CircleMarker(location=[row['lat'], row['lon']], radius=row['predictions'], popup=string, fill=True,
                            color=color).add_to(base_map)
    folium_static(base_map)
    st.write(cimis_data_df_map.head())
    st.write(cimis_data_df.head())

def main():
    st.header("PM 2.5 Forecasting")
    st.subheader("A demo on how to use Streamlit")
    st.image("images/Cal_tahoe.jpg", width=600)
    #base_map = folium.Map([36.7783, -119.4179], zoom_start=6, tiles='cartodbpositron')
    #folium_static(base_map)
if __name__ == "__main__":
    main()