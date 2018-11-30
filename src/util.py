"""
    Functions for loading and processing data
"""
from __future__ import division
import os
import logging
from datetime import timedelta
import pandas as pd
import numpy as np

def load_data(date, data_dir):
    """ Compile all availiable data for a particular date """

    next_date = date + timedelta(days=1)
    files = [f for f in os.listdir(data_dir) if '.csv' in f]
    logging.info("Loading data from:\n  %s", "\n  ".join(files))

    data = list()
    for file in files:
        data_name = file[:file.find('USD')]
        tmp_df = (
            pd.read_csv(os.path.join(data_dir, file),
                        usecols=['Timestamp', 'Weighted_Price'])
            .assign(Timestamp=lambda df: pd.to_datetime(df['Timestamp'], unit='s'))
            .loc[lambda df: (df['Timestamp'] >= date) & (df['Timestamp'] < next_date)]
            .set_index('Timestamp')
            .sort_index()
            .fillna(method='ffill')
            .fillna(0)
            .astype(float)
            .rename(columns={'Weighted_Price': 'price_%s' % data_name})
        )
        data.append(tmp_df)

    logging.info("Loaded %d data sets", len(data))
    return pd.concat(data, axis=1, join='outer')

def split_data(df, test_periods):
    """ Create the input features and labels, split into train/test """
    train_features = df.iloc[:-test_periods, :]
    train_targets = df.shift(-1).iloc[:-test_periods, :]
    test_targets = df.shift(-1).iloc[-test_periods:-1, :]
    return train_features, train_targets, test_targets

def smape(forecast, actual, normalizer=None):
    """ Symmetric Mean Absolute Percent Error """
    forecast = np.reshape(forecast, (-1,))
    actual = np.reshape(actual, (-1,))

    if normalizer:
        forecast = normalizer.inverse_transform(forecast)
        actual = normalizer.inverse_transform(actual)

    N = len(forecast)
    return 200 / N * np.sum(np.abs(forecast - actual) / (np.abs(forecast) + np.abs(actual)))


class Normalizer(object):
    """ Simple object for doing a mean-centering stddev scaling """

    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, df):
        """ Fit based on all values in df """
        if isinstance(df, (pd.DataFrame, pd.Series)):
            df = df.values
        self.mean = np.mean(df)
        self.std = np.std(df)
        return self
    
    def transform(self, df):
        """ Transform df according to the mean found in .fit() """ 
        if not self.mean:
            raise IOError("You must call .fit() before .transform()")

        return (df - self.mean) / self.std

    def inverse_transform(self, df):
        """ Undo .transform """
        if not self.mean:
            raise IOError("You must call .fit() before .inverse_transform()")

        return df * self.std + self.mean

    def fit_transform(self, df):
        """ Call .fit() -> .transform() """
        self.fit(df)
        return self.transform(df)
