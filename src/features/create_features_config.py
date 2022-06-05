# pylint: disable=missing-module-docstring
# TODO: make config simpler
# TODO: add feature order inside feature dict
from collections import OrderedDict

import numpy as np

from math_functions import (Exp, Log, # pylint: disable=unused-import,import-error
                            Power, Sigmoid, Trig)
from process_features import (AggregateFeature, ComplexFeature, DateFeature, # pylint: disable=import-error
                               DerivativeFeature, DirectionFeature,
                               DropFeatures, LagFeature, MathFeature, 
                               PairedFeature)

feature_dict = {
    'general_params': {
        'sep': '_',
        'link': '-',
        # could be used for case when 
        # for ex. derivatives feature is going to be aggregated
        'additional_drops': 0
    },
    'attach': {  # Attach previously predicted wp or not
        'attach': False,
        'path': './data/' + 'wp_feature_df_20210929_173650.csv',
        'feature_name': 'wp',
        # replace or not specific date in train datset which going to be taken from 
        # 'wp_path' (predicted vals), if None -> replace all dates in train dataset 
        # from 'wp_path'
        'predicted_dates': None 
    },
    'date': {'col': 'date',
             'feature': ['hour', 'month'],
    },
    'math': {'feature':
             {'wd': {'sin': Trig(add=0, fun=np.sin),
                     'cos': Trig(add=0, fun=np.cos)},
              'ws': {'log': Log(add=1),
                     'frac': Power(p=-1, add=1),
                     'exp': np.exp},
             }
    },
    'side': {'col': 'wd',
             'feature': {'is_numeric': True, 'directions': 12, 'name': 'side'}
    },
    'complex_exp': {'feature':
                       [{'cols': ['ws', 'wd_sin'], 'fun': 'cp_exp'},
                        {'cols': ['ws', 'wd_cos'], 'fun': 'cp_exp'},
                        ]
    },
    'paired': {'feature':
               [
                {'cols': ['ws', 'wd_sin'], 'fun': np.prod},
                {'cols': ['ws', 'wd_cos'], 'fun': np.prod}
               ]
    },
    'derivatives': {'feature': 
                    # order in tuple is important: (power, shift/step)
                    # # for (1, 3) name: wd_sin_d1s3 for ex
                       {'ws': {'power-shift': [(2, 1), (2, 3)]},
                        'wd_sin': {'power-shift': [(2, 1), (2, 3)]},
                        'wd_cos': {'power-shift': [(2, 1), (2, 3)]}
                        }
    },
    'aggregate': {'feature':
                  [
                   {'col': 'ws',
                    'fun': np.median,
                    'name': 'med', 
                    'shift': [24, 24*7]},
                   {'col': 'ws-wd_cos',
                    'fun': np.median,
                    'name': 'med',
                    'shift': [24, 24*7]},
                   {'col': 'ws-wd_cos',
                    'fun': np.median,
                    'name': 'med',
                    'shift': [24, 24*7]},
                   {'col': 'ws',
                    'fun': np.std,
                    'name': 'std',
                    'shift': [24, 24*7]},
                   {'col': 'ws-wd_sin',
                    'fun': np.std,
                    'name': 'std',
                    'shift': [24, 24*7]},
                   {'col': 'ws-wd_cos',
                    'fun': np.std,
                    'name': 'std',
                    'shift': [24, 24*7]},
                  ]
    },
    'lags': {'feature': {'ws': [3, 6, 9, 12, 24],
                         'ws-wd_sin': [3, 6, 9, 12, 24],
                         'ws-wd_cos': [3, 6, 9, 12, 24]}
    },
    # mention features which you want to drop
    'drop': {'feature': ['u', 'v', 'wd', 'hors']
    },
}


def return_feature_order() -> OrderedDict:
    """Set order for feature and return OrderedDict with ordered classes

    Returns:
        OrderedDict: ordered classes for creating features
    """
    # Order is important - define in which order classes are going to be used
    feature_order = OrderedDict()
    feature_order['date'] = DateFeature
    feature_order['math'] = MathFeature
    feature_order['side'] = DirectionFeature
    feature_order['complex_exp'] = ComplexFeature
    feature_order['paired'] = PairedFeature
    feature_order['derivatives'] = DerivativeFeature
    feature_order['lags'] = LagFeature
    feature_order['aggregate'] = AggregateFeature
    feature_order['drop'] = DropFeatures
    return feature_order
