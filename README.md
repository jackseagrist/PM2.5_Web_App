# PM2.5_Web_App
Live web application for predicting PM 2.5 concentration using streamlit based on CS230 project. Project is hosted using Heroku.

For details on the dataset and model training, check out the [Project Repository](https://github.com/jackseagrist/Forecasting_PM25_LSTM).

## Outline of project contents:

1. app.py - Main application file. Web app built using [streamlit](https://www.streamlit.io/)

2. all_v4_2 - Trained model from CS230 project.

3. latlon_dict.csv - Mapping of CIMIS stations and coordinate locations.

4. requirements.txt - Python package requirements. Main packages include requests, numpy, pandas, tensorflow, and sklearn.

## Data

[EPA airnow api information](https://docs.airnowapi.org/) - collect PM 2.5 concentration in micrograms / meter^3

[CIMIS web api link](https://et.water.ca.gov/Rest/Index) - collect the dew point temperature, temperature, pressure, wind run, wind speed, and precipitation
