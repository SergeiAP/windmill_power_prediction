# pylint: disable=missing-module-docstring
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd


# ===================== Math operations to create feature =====================
class MathOperation(ABC):
    """
    To build custom function acceptable for Feature engineering part of pipeline
    """

    @abstractmethod
    def __init__(self) -> None:
        """Init"""

    @abstractmethod
    def __call__(self):
        """Callable class"""


class Power(MathOperation):
    """Power in (pd.Series + adds) ** power format

    Args:
        MathOperation (ABC): interface
    """
    
    def __init__(self, # pylint
                 p: Union[int, float],
                 add: Optional[int] = 0
                 ) -> None:
        """Save attributes to power feature when called

        Args:
            p (Union[int, float]): power for (series + add)
            add (Optional[int], optional): adds inside of power function. 
            Defaults to 0.
        """
        super().__init__()
        self.p = p # pylint: disable=invalid-name
        self.add = add

    def __call__(self, df: pd.Series) -> pd.Series:
        """Make power feature in pd.Series

        Args:
            df (pd.Series): feature to make powered

        Returns:
            pd.Series: powered feature
        """
        df = (df + self.add) ** self.p # type: ignore 
        return df


class Sigmoid(MathOperation):
    """Add feature as sigmoid function 1 / (1 + exp(-(pd.Series + adds)))

    Args:
        MathOperation (ABC): interface
    """
    
    def __init__(self, add: Optional[int] = 0) -> None:
        """Save attributes to sigmoid feature when called

        Args:
            add (Optional[int], optional): adds inside exp. Defaults to 0.
        """
        super().__init__()
        self.add = add

    def __call__(self, df: pd.Series) -> pd.Series:
        """Make feature sigmoided

        Args:
            df (pd.Series): feature to make sigmoid

        Returns:
            pd.Series: sigmoided feature
        """
        df = 1 / (1 + np.exp(-(df + self.add))) # type: ignore
        return df


class Exp(MathOperation):
    """Make pd.Series exponented by np.exp(self.fun(pd.Series))

    Args:
        MathOperation (ABC): interface
    """
    
    def __init__(self, fun: Callable) -> None:
        """Save attributes to make exponented feature when called

        Args:
            fun (Callable): function inside np.exp
        """
        super().__init__()
        self.fun = fun

    def __call__(self, df: pd.Series) -> pd.Series:
        """Make feature exponended

        Args:
            df (pd.Series): feature to make exponenta

        Returns:
            pd.Series: exponented feature
        """
        return np.exp(self.fun(df))


class Log(MathOperation):
    """Make pd.Series logarithmed by np.log(pd.Series + self.add)

    Args:
        MathOperation (ABC): interface
    """
    
    def __init__(self, add: int):
        """Save attributes to logarithm feature when called

        Args:
            add (int): adds inside log
        """
        super().__init__()
        self.add = add

    def __call__(self, df: pd.Series) -> pd.Series:
        """Make feature logarithmed

        Args:
            df (pd.Series): feature to make logarithmed

        Returns:
            pd.Series: logarithmed feature
        """
        df = np.log(df + self.add) # type: ignore 
        return df
    

class Trig(MathOperation):
    """
    Apply to pd.Series trig feature like 
    np.trig(self.arg_fun(pd.Sereis) + adss in deg)

    Args:
        MathOperation (ABC): interface
    """

    def __init__(self,
                 add: Union[int, float],
                 fun: Callable,
                 arg_fun: Optional[Callable] = None
                 ) -> None:
        """Save attributes to make trigonometrical feature when called

        Args:
            add (Union[int, float]): add some degrees inside of trig function
            fun (Callable): trig function like np.sin
            arg_fun (Optional[Callable], optional): transformation function for 
            pd.Series. Defaults to None.
        """
        super().__init__()
        self.add = add
        self.fun = fun
        if arg_fun:
            self.arg_fun = arg_fun
        else:
            self.arg_fun = np.deg2rad

    def __call__(self, df: pd.Series) -> pd.Series:
        """Make feature trigonometrical

        Args:
            df (pd.Series): feature to make trigonometrical

        Returns:
            pd.Series: trigonometrical feature
        """
        return self.fun(self.arg_fun(df) + np.deg2rad(self.add))
