import requests
import urllib.request
import time
# from bs4 import BeautifulSoup
import json
import pandas as pd


def fetch_sentiment(limit):
    """This function pulls sentiment data from the crypto fear and greed index. That data is updated daily.
    This is the link to the website: https://alternative.me/crypto/fear-and-greed-index/#fng-history
    The 'limit' argument is the number of data points to fetch (one for each day).
    The given value is on a scale of 0 - 100, with 0 being extreme fear and 100 being extreme greed."""

    url = "https://api.alternative.me/fng/?limit="+ str(limit) +"&date_format=us"

    data = requests.get(url).json()["data"] #returns a list of dictionaries

    sentiment_df = pd.DataFrame(data)

    #Drop unnecessaary columns
    sentiment_df.drop(columns = ["time_until_update", "value_classification"], inplace = True)
    #Rename the columns
    sentiment_df.rename(columns={'timestamp': 'Date', 'value': 'Value', 'value_classification': 'Value Classification'}, inplace = True)
    #Format the dates
    sentiment_df['Date'] = pd.to_datetime(sentiment_df["Date"], format = "%m-%d-%Y")
    #Convert value to int, and center the sentiment value at 0
    sentiment_df["Value"] = sentiment_df['Value'].apply(int)
    sentiment_df['Value'] = sentiment_df['Value'] - 50

    # print(sentiment_df)
    return sentiment_df

df = fetch_sentiment(10)
print(df)
