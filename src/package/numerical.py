import numpy as np
import pandas as pd


def autocorr_single_tp(a: np.array, t: int) -> float:
    """Do autocorrelation for a single time point.

    Parameters
    ----------
    a : np.array
        The array to correlate (complex or real number)
    t : int
        The distance (in the index)

    Returns
    -------
    float
        The autocorrelation as a real number.
    """
    return np.real(np.sum(a[0] * np.conj(a[t])))


def autocorr(df: pd.DataFrame) -> pd.DataFrame:
    """Do autocorrelation for all possible time steps over all columns.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame to correlate

    Returns
    -------
    pd.DataFrame
        The resulting dataframe with timestep as index and one column named autocorr
    """
    df_result = pd.DataFrame()
    df_result['autocorr'] = [autocorr_single_tp(df.values, i) for i in range(df.shape[0])]
    df_result.index.name = 'timestep'
    return df_result


def fourier_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Fourier transform a dataframe column-wise. The shape of the dataframe is
    not changed, only the column names are appended with _ft.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to transform

    Returns
    -------
    pd.DataFrame
        The dataframe with the fourier transform of all columns (they are named {}_ft)
    """
    df_result = df.apply(np.fft.fft)
    df_result.index.name = 'frequency'
    df_result.columns = [f'{c}_ft' for c in df_result.columns]
    return df_result
