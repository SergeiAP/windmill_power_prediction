# pylint: disable=missing-module-docstring
from abc import ABC, abstractmethod
from typing import Union

import ewtpy
import numpy as np
import pandas as pd
from scipy.special import comb as binom


# ======================== Feature Interface =================================
class Feature(ABC):

    @classmethod
    @abstractmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict) -> tuple[pd.DataFrame, int]:
        """
        Create new features inside df using params in dict. del_rows - to 
        check/increase rows in df head required to del due to feature 
        """
        
    @staticmethod
    def update_del_rows(del_rows: int, val: int) -> int: 
        return val if del_rows < val else del_rows

# ========================New feature classes=================================

class RollingFeature(Feature):
    """
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'cols': {'ws': np.mean, 'u': np.std,},
                       'shift': 8,
                       'accumulate': 24}
    """
    
    @classmethod
    def create(cls, df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        separator = general_params['sep']
        params = specific_params
        if params is not None and len(params['cols']) > 0:
            # Copy
            df = df.copy()

            del_rows = cls.update_del_rows(del_rows=del_rows,
            val=params['shift']+params['accumulate']) 
            for col, fun in params['cols'].items():
                df[col+separator+f'rol{params["shift"]}a{params["accumulate"]}'] = (
                    df[col].shift(params['shift'])
                    .rolling(params['accumulate']).apply(fun, raw=True))
        return df, del_rows


class DecompositionFeature(Feature):
    """
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'cols': {'ws': 5, 'wd': 6,}}
    """

    @staticmethod
    def decompose(col: pd.Series, k: int, window: int) -> np.ndarray:
        # n_row = col.shape[0]
        col_arr = np.array(col)
        n_row = col_arr.shape[0]
        dec = np.empty((n_row, k))
        ewt1,  mfb ,boundaries = ewtpy.EWT1D(col_arr[:window], N = k)
        dec[:window, :] = ewt1
        for i in range(window+1, n_row):
            ewt1,  mfb ,boundaries = ewtpy.EWT1D(col_arr[i-window:i], N = k)
            dec[i] = ewt1[-1, :]
        return dec
        
    @classmethod
    def create(cls, df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        separator = general_params['sep']
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            # Copy
            df = df.copy()
            for col, par in params['feature'].items():
                k = par[0]
                window = par[1]
                # decomposition
                dec = cls.decompose(df[col], k, window)
                # to dataframe
                for i in range(dec.shape[1]):
                    df[col+separator+'decomp'+separator+str(i)] = dec[:, i]
        return df, del_rows

class DateFeature(Feature):
    """
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'date': {'col': 'date', 
                                'feature': ['hour', 'day', 'week', 'month']}
    """

    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        df = df.copy()
        params = specific_params
        df.loc[:,params['col']] = pd.to_datetime(df[params['col']])
        
        if params is not None and len(params['feature']) > 0:
            # Copy
            df = df.copy()

            for elem in params['feature']:
                if elem == 'hour':
                    df['hour'] = df[params['col']].dt.hour
                elif elem == 'day':
                    df['day'] = df[params['col']].dt.day
                elif elem == 'week':
                    df['week'] = df[params['col']].dt.week
                elif elem == 'month':
                    df['month'] = df[params['col']].dt.month            
            
        return df, del_rows


class DirectionFeature(Feature):
    """
    Class to create direction feature denoted side of the world.
    Possible input:
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'col': 'wd',
                       'feature': {'is_numeric': False, 
                                   'directions': 12,
                                   'name': 'side'}}
    """

    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            # Copy
            df = df.copy()

            compare_list = cls.get_geographical_side(params['feature']['directions'])
            directions_list = cls.get_directions_list(params['feature']['is_numeric'],
                                                      params['feature']['directions'])
            df[params['feature']['name']] = (
                df[params['col']].apply(
                    cls.comparision_function,
                    directions=directions_list,
                    compare_list=compare_list))
        return df, del_rows

    @staticmethod
    def get_geographical_side(directions: int = 6):
        interval = 360 / directions
        half = float(interval / 2)
        vals_spread = [((360 - half, 360), (0, half))]
        side_options = [(half + interval*i, half + interval*(i + 1)) 
                        for i in range(directions-1)]
        vals_spread = vals_spread + side_options # type: ignore
        return vals_spread

    @staticmethod
    def get_directions_list(is_numeric: bool, length: int):
        if is_numeric:
            return [idx for idx in range(length)]
        alpha = 'd'
        return [alpha+str(idx) for idx in range(length)]

    @staticmethod
    def comparision_function(row: pd.Series, 
                             directions: list[Union[str, int]],
                             compare_list: list[tuple]):
        for idx in range(1, len(compare_list)):
            if compare_list[idx][0] < row <= compare_list[idx][1]:
                return directions[idx]
        return directions[0]


class MathFeature(Feature):
    """
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'feature': {
        'ws': {'pr3': Power(3), 'sqrt': np.sqrt},
        'wd_sin': {'40': Trig(add=40, fun=np.sin, arg_fun=np.arcsin)}
        }
    """

    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        separator = general_params['sep']
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            df = df.copy()
            for col in params['feature']:
                for fun in params['feature'][col]:
                    df[col+separator+fun] = params['feature'][col][fun](df[col])
        return df, del_rows


class PairedFeature(Feature):
    """
    np.prod, np.sum
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'feature':
                       [{'cols': ['ws', 'wd_sin'], 'fun': np.prod},
                        {'cols': ['wd_cos', 'wd_sin'], 'fun': np.prod},
                        ]
                       }
    """

    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        link = general_params['link']
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            # Copy
            df = df.copy()
            for elem in params['feature']:
                df_features = elem['fun']([df[col] for col in elem['cols']], axis=0)
                df[link.join(elem['cols'])] = df_features
        return df, del_rows


class AggregateFeature(Feature):
    """
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'feature':
                       [{'col': 'ws',
                       'fun': np.mean, 'name': 'mean', 'shift': [3, 8, 12]},
                        {'col': 'wd_sin',
                        'fun': np.median,
                        'name': 'med',
                        'shift': [3, 8, 12]},
                        ]
                       }
    """
    
    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        separator = general_params['sep']
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            # Copy
            df = df.copy()
            for elem in params['feature']:
                for shift in elem['shift']:
                    del_rows = cls.update_del_rows(del_rows=del_rows, val=shift) 
                    df[elem['col']+separator+elem['name']+str(shift)] = (
                        df[elem['col']]
                        .rolling(shift)
                        .apply(elem['fun'], raw=True)
                        )
        return df, del_rows


class LagFeature(Feature):
    """
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'feature':
                       {'ws': [1, 2, 3, 4],
                        'wd_sin': [1, 2, 3, 4],
                        }
    }
    """

    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        separator = general_params['sep']
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            # Copy
            df = df.copy()
            for key in params['feature']:
                for lag in params['feature'][key]:
                    del_rows = cls.update_del_rows(del_rows=del_rows, val=lag) 
                    df[key+separator+'l'+str(lag)] = df[key].shift(lag)
        return df, del_rows


class LagDiffFeature(Feature):
    """
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'feature':
                       {'ws': [1, 2, 3, 4],
                        'wd_sin': [1, 2, 3, 4],
                        }
    }
    """

    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        separator = general_params['sep']
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            # Copy
            df = df.copy()
            for key in params['feature']:
                for lag in params['feature'][key]:
                    del_rows = cls.update_del_rows(del_rows=del_rows, val=lag) 
                    df[key+separator+'ld'+str(lag)] = df[key] - df[key].shift(lag)
        return df, del_rows


class SmoothFeature(Feature):
    """
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'feature': ['ws', 'wd_sin'],
                       'inplace': False,
                       'alpha': 0.2
    },
    """

    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        separator = general_params['sep']
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            # Copy
            df = df.copy()
            cols = params['feature']
            alpha = params['alpha']
            if params['inplace']:
                df[cols] = df[cols].ewm(alpha=alpha).mean()
            else:
                df[[col + "_smth" for col in cols]] = df[cols].ewm(alpha=alpha).mean()
        return df, del_rows


class DerivativeFeature(Feature):
    """
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'feature':
                       {# order in tuple is important: (power, shift/step)
                           'ws': {'power-shift': [(3, 1), (3, 2)]},
                           'wd_sin': {'power-shift': [(3, 1), (3, 3)]},
                        }
    }
    """

    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        
        def calculate_differences(df_col: pd.Series,
                                  power: int,
                                  shift: int) -> pd.Series:
            """from https://en.wikipedia.org/wiki/Finite_difference"""
            der_feature = sum([(-1) ** (i) 
                               * binom(power, i, exact=True) 
                               * df_col.shift(i*shift) 
                               for i in range(power+1)])
            return der_feature # type: ignore

        separator = general_params['sep']
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            # Copy
            df = df.copy()
            for key in params['feature']:
                for power, shift in params['feature'][key]['power-shift']:
                    del_rows = cls.update_del_rows(del_rows=del_rows,
                                                   val=power*shift) 
                    df[key+separator+'d'+str(power)+'s'+str(shift)] = (
                        calculate_differences(df[key], power, shift) / (shift**power)
                        )
        return df, del_rows


class DropFeatures(Feature):
    """
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'feature': 
                        ['u', 'v', 'wd', 
                        'ws_d1s1', 'wd_d1s1','ws_d2s1', 
                        'wd_d2s1', 'ws_d3s1', 'wd_d3s1',]
    }
    """
    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        separator = general_params['sep']
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            df = df.copy()
            df.drop(columns=params['feature'], inplace=True)
        return df, del_rows


class ComplexFeature(Feature):
    """
    for np.prod, np.sum"
    Possible input
    general_params = {'sep': '_', 'link': '-',},
    specific_params = {'feature':
                       [{'cols': ['ws', 'wd_sin'], 'fun': 'cp_exp'},
                        {'cols': ['ws', 'wd_cos'], 'fun': 'cp_exp'},
                        ]
                       }
    """

    @classmethod
    def create(cls,
               df: pd.DataFrame,
               del_rows: int,
               general_params: dict,
               specific_params: dict
               ) -> tuple[pd.DataFrame, int]:
        """Create new features inside df using params in dict"""
        fun_dict = {'cp_exp': cls._cp_exp_formula}
        link = general_params['link']
        separator = general_params['sep']
        params = specific_params
        if params is not None and len(params['feature']) > 0:
            # Copy
            df = df.copy()
            for elem in params['feature']:
                feature =  fun_dict[elem['fun']](*[df[col] for col in elem['cols']])
                df[separator.join(elem['cols'])+link+elem['fun']] = feature
        return df, del_rows

    @staticmethod
    def _cp_exp_formula(df_ws: pd.Series,
                        df_angle: pd.Series
                        ) -> pd.Series:
        return (df_ws-df_angle**2)*np.exp(-1/(1/df_ws+1/(1+df_angle**3)))


class GenerateFrequencies(Feature):
    """Create feature with frequencies

    Args:
        Feature (ABC): interface
    """
