# pylint: disable=import-error
# pylint:  disable=missing-module-docstring
# TODO: structure config

from process_weather import OrderWeatherPrediction, FreshWeatherPrediction, \
    AggregatedWeatherPrediction


# Params for WeatherPrediction classes. Required to use one for one class realisation
wide_weather_params = {'date_col': 'date',
                       'cols_to_transform': ['u', 'v', 'ws', 'wd'],
                       'reset_index': True}

fresh_weather_params = {'sort': ['date', 'hors'],
                        'grouby_col': 'date',
                        'col_idx_min': 'hors',
                        'col_na_check': 'ws',
                        'check': False  # Works for wp1 only
                        }

agg_weather_params = {'date_col': 'date',
                      'grouped_length': 4*(36+48),
                      'sort': ['date', 'hors'],
                      'extend_df': [{'date': '2012-06-23 13:00:00',
                                     'hors': 36},
                                    {'date': '2012-06-24 01:00:00',
                                     'hors': 24},
                                    {'date': '2012-06-24 13:00:00',
                                     'hors': 12},],
                      'fit_to_test_params': {'mul': 4,
                                             'group_len': 36+48,
                                             'group_start_idx': 36+12,
                                             'group_start_hors': [1, 13, 25],
                                             'group_end_hors': [12, 24, 36]},
                      'agg_params': {'wd_sin_cos_transform': True,
                                     'wd_col': 'wd',
                                     'wd_drop': True,
                                     'exclude_cols': ['hors'],
                                     'agg_fun': 'mean'},
                      'check': False  # Works for wp1 only
                      }

# Could be used to create 4 datasets with 4 descending_orders => increase date amount
# and increase prediction stability.
order_weather_params = {'sort': ['date', 'hors'], 
                        'date_col': 'date', 
                        'grouby_col': 'date', 
                        'col_idx': 'hors', 
                        'col_na_check': 'ws', 
                        'descending_order': 3,}  # Optional: [0, 1, 2, 3] or -1 - random

# MAIN dict params for weather
# Dict for different aggregation type for train and test OR with the same type
agg_params_train_test = {'train': {'class': OrderWeatherPrediction.get_df,
                                   'params': order_weather_params},
                         'nans': {'class': FreshWeatherPrediction.get_df, 
                                  'params': fresh_weather_params}}
agg_params_all = {'all': {'class': AggregatedWeatherPrediction.get_df,
                          'params': agg_weather_params}}
