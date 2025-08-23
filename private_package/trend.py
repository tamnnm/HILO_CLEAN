
import numpy as np
from scipy import stats as sps
import pymannkendall as mk
<<<<<<< HEAD
=======
import xarray as xr
>>>>>>> c80f4457 (First commit)
from typing import Union, Optional, List, Tuple, Dict, Literal, Callable
EPSILON = 1e-10

# ------------------------------------------------------------------ #
#                   Check the dimension of the data                  #
# ------------------------------------------------------------------ #


<<<<<<< HEAD
def check_dim(array):
    """
    Check if the input is a pandas dataframe or series
    """
=======
def check_dim(array, var_name: Optional[str] = None):
    """
    Check if the input is a pandas dataframe or series
    """
    if isinstance(array, xr.Dataset):
        if var_name is not None:
            array = array[var_name].values
        else:
            raise ValueError("The input data is a dataset, please specify the variable name")
>>>>>>> c80f4457 (First commit)
    if not isinstance(array, np.ndarray):
        array = array.values
    return np.array([element for sublist in array for element in sublist]) if len(array.shape) > 1 else np.array(array)

def non_nan(actual, predicted):
    """ Remove NaN values from both array """
    # print(actual.shape,predicted.shape)
    valid_ind = ~np.isnan(actual) & ~np.isnan(predicted)
    return actual[valid_ind], predicted[valid_ind]

# ------------------------------------------------------------------ #
#                   Evaluation on a single dataset                   #
# ------------------------------------------------------------------ #

# ------------------------ Declare function ------------------------ #
def kurtosis(actual):
    """Kurtosis"""
    return sps.kurtosis(check_dim(actual))

def skewness(actual):
    """Skewness"""
    return sps.skew(check_dim(actual))

def std(actual):
    """Standard deviation"""
    return np.std(check_dim(actual))

def mann_kendall(actual, alpha=0.05, method: Optional[Literal["Seasonal", "PW", "no-trend PW", "Seasonal PW", "Sens slope"]]="no-trend PW"):
    """ Mann Kandall trend test + Sen's slope """
    methods={
        None: mk.original_test,
        "Seasonal": mk.seasonal_test,
        "PW": mk.pre_whitening_modification_test,
        "no-trend PW": mk.trend_free_pre_whitening_modification_test,
    }
    result = methods[method](check_dim(actual), alpha=alpha)
    result_dict = {
        "trend": result.trend,
        "h": result.h,
        "p": result.p,
        "z": result.z,
        "Tau": result.Tau,
        "s": result.s,
        "var_s": result.var_s,
        "slope": result.slope,
        "intercept": result.intercept
    }
    return result_dict
def sen_slope(actual):
    """ Sen's slope """
    return mk.sens_slope(check_dim(actual))

# ------------------------------------------------------------------ #
#                        Evaluate two datasets                       #
# ------------------------------------------------------------------ #

# ------------------------ Declare function ------------------------ #
def _error(actual, predicted):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual, predicted):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    actual=check_dim(actual)
    predicted=check_dim(predicted)
<<<<<<< HEAD
    return _error(actual, predicted) / abs(actual + EPSILON)
=======
    return _error(actual, predicted) / abs(actual + EPSILON) * 100
>>>>>>> c80f4457 (First commit)


def _naive_forecasting(actual, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(actual, predicted, benchmark = None):
    """ Relative Error """
    actual=check_dim(actual)
    predicted=check_dim(predicted)
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) /\
               (_error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON)

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


def _bounded_relative_error(actual, predicted, benchmark = None):
    actual=check_dim(actual)
    predicted=check_dim(predicted)
    """ Bounded Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        abs_err = np.abs(_error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(_error(actual[seasonality:], _naive_forecasting(actual, seasonality)))
    else:
        abs_err = np.abs(_error(actual, predicted))
        abs_err_bench = np.abs(_error(actual, benchmark))

    return abs_err / (abs_err + abs_err_bench + EPSILON)

def _geometric_mean(a, axis=0, dtype=None):
    actual=check_dim(actual)
    predicted=check_dim(predicted)
    """ Geometric mean """
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))

def _mse(actual, predicted):
    """ Mean Squared Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean(np.square(_error(actual, predicted)))

def _rmse(actual, predicted):
    """ Root Mean Squared Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.sqrt(_mse(actual, predicted))

def _me(actual, predicted):
    """ Mean Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean(_error(actual, predicted))

def _mae(actual, predicted):
    """ Mean Absolute Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean(np.abs(_error(actual, predicted)))

def _mad(actual, predicted):
    """ Mean Absolute Deviation """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    error = _error(actual, predicted)
    return np.mean(np.abs(error - np.mean(error)))

def _gmae(actual, predicted):
    """ Geometric Mean Absolute Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return _geometric_mean(np.abs(_error(actual, predicted)))

def _mdae(actual, predicted):
    """ Median Absolute Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.median(np.abs(_error(actual, predicted)))

def _mpe(actual, predicted):
    """ Mean Percentage Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean(_percentage_error(actual, predicted))

def _mape(actual, predicted):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean(np.abs(_percentage_error(actual, predicted)))

def _nmse(actual, predicted):
    """ Normalized Mean Squared Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return _mse(actual, predicted) / (np.mean(np.abs(actual)**2))

def _nme(actual, predicted):
    """ Normalized Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return _me(actual, predicted) / np.mean(np.abs(actual))

def _nmae(actual, predicted):
    """ Normalized Absolute Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return _mae(actual, predicted) / np.mean(np.abs(actual))

def _nrmse(actual, predicted):
    """ Normalized Root Mean Squared Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return _rmse(actual, predicted) / np.mean(np.abs(actual))

def _mdape(actual, predicted):
    """
    Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.median(np.abs(_percentage_error(actual, predicted)))


def _smape(actual, predicted):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def _smdape(actual, predicted):
    """
    Symmetric Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.median(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def _maape(actual, predicted):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def _mase(actual, predicted, seasonality: int = 1):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return _mae(actual, predicted) / _mae(actual[seasonality:], _naive_forecasting(actual, seasonality))


def _std_ae(actual, predicted):
    """ Normalized Absolute Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    __mae = _mae(actual, predicted)
    return np.sqrt(np.sum(np.square(_error(actual, predicted) - __mae))/(len(actual) - 1))


def _std_ape(actual, predicted):
    """ Normalized Absolute Percentage Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    __mape = _mape(actual, predicted)
    return np.sqrt(np.sum(np.square(_percentage_error(actual, predicted) - __mape))/(len(actual) - 1))


def _rmspe(actual, predicted):
    """
    Root Mean Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def _rmdspe(actual, predicted):
    """
    Root Median Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def _rmsse(actual, predicted, seasonality: int = 1):
    """ Root Mean Squared Scaled Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    q = np.abs(_error(actual, predicted)) / _mae(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(np.mean(np.square(q)))


def _inrse(actual, predicted):
    """ Integral Normalized Root Squared Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.sqrt(np.sum(np.square(_error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual))))


def _rrse(actual, predicted):
    """ Root Relative Squared Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))


def _mre(actual, predicted, benchmark = None):
    """ Mean Relative Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean(_relative_error(actual, predicted, benchmark))


def _rae(actual, predicted):
    """ Relative Absolute Error (aka Approximation Error) """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)


def _mrae(actual, predicted, benchmark = None):
    """ Mean Relative Absolute Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def _mdrae(actual, predicted, benchmark = None):
    """ Median Relative Absolute Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def _gmrae(actual, predicted, benchmark = None):
    """ Geometric Mean Relative Absolute Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))


def _mbrae(actual, predicted, benchmark = None):
    """ Mean Bounded Relative Absolute Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def _umbrae(actual, predicted, benchmark = None):
    """ Unscaled Mean Bounded Relative Absolute Error """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    __mbrae = _mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae)


def _mda(actual, predicted):
    """ Mean Directional Accuracy """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1])).astype(int))

def _linregress(x, y):
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return np.polyfit(x, y, 1)


# ---------------------- Correlation function ---------------------- #
<<<<<<< HEAD
def pearson_correlation(actual, predicted,method=None, R_only=True):
    """ Pearson correlation coefficient """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    if method is None:
    # Assume data is linear
        result= sps.pearsonr(actual, predicted)
    elif method == "Permutation":
        rng=np.random.default_rng()
        method = sps.PermutationMethod(n_resamples=np.inf, random_state=rng)
        result= sps.pearsonr(actual, predicted,method=method)
    elif method == "Monte Carlos":
        rng=np.random.default_rng()
        method = sps.MonteCarloMethod(n_resamples=np.inf, random_state=rng)
        result= sps.pearsonr(actual, predicted,method=method)
    if R_only:
        return result[0]
    else: return result
=======
def pearson_correlation(actual, predicted, method=None, R_only=False):
    """ Pearson correlation coefficient """
    actual, predicted = non_nan(check_dim(actual), check_dim(predicted))
    if method is None:
        # Assume data is linear
        result = sps.pearsonr(actual, predicted)
    elif method == "Permutation":
        rng = np.random.default_rng()
        method = sps.PermutationMethod(n_resamples=np.inf, random_state=rng)
        result = sps.pearsonr(actual, predicted, method=method)
    elif method == "Monte Carlos":
        rng = np.random.default_rng()
        method = sps.MonteCarloMethod(n_resamples=np.inf, random_state=rng)
        result = sps.pearsonr(actual, predicted, method=method)

    if R_only:
        return result.statistic  # The correlation coefficient
    else:
        # Full result contains:
        # result.statistic - correlation coefficient
        # result.pvalue - p-value for testing non-correlation
        result_dict = {
            "R": result.statistic,
            "pvalue": result.pvalue,
        }
        return result_dict
>>>>>>> c80f4457 (First commit)
# Non-parametric (categorical data)
def spearman_correlation(actual, predicted):
    """ Spearman rank correlation coefficient """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return sps.spearmanr(actual, predicted)

def kendall_tau(actual, predicted):
    """ Kendall Tau correlation coefficient """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    return sps.kendalltau(actual, predicted)

import matplotlib.pyplot as plt
# ------------------------- composite index ------------------------ #
def DISO(actual, predicted,check = False):
    """ DISO index (2007)"""
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
    if all(element == 0 for element in actual) or all(element == 0 for element in predicted):
        print("All actual or predicted values are zero")
        return np.nan
    if all(element == 0 for element in predicted): print("check predicted")
<<<<<<< HEAD
    R=pearson_correlation(actual, predicted)
    nmse=_nmse(actual, predicted)
    nrmse=_nrmse(actual, predicted)
    nmae=_nmae(actual, predicted)
    DISO=np.sqrt((R-1)**2+nmse**2+nrmse**2+nmae**2)
=======
    R=pearson_correlation(actual, predicted, R_only=True)
    nrmse=_nrmse(actual, predicted)
    nmae=_nmae(actual, predicted)
    DISO=np.sqrt((R-1)**2+nrmse**2+nmae**2)
>>>>>>> c80f4457 (First commit)
    return DISO

def Taylor_score(actual, predicted):
    """ Taylor Skill score """
    actual,predicted=non_nan(check_dim(actual),check_dim(predicted))
<<<<<<< HEAD
    R=pearson_correlation(actual, predicted)
=======
    R=pearson_correlation(actual, predicted, R_only=True)
>>>>>>> c80f4457 (First commit)
    std_ac_pred=std(actual)/std(predicted)
    return (1+R)*2/((std_ac_pred+1/std_ac_pred)**2)

# -------------------- Dictionary of the metrics ------------------- #
METRICS_SINGULAR = {
    'kurtosis': {'function': kurtosis, 'full_name': "Kurtosis"},
    'skewness': {'function': skewness, 'full_name': "Skewness"},
    'std': {'function': std, 'full_name': "Standard Deviation"},
<<<<<<< HEAD
    'manndall_kendall': {'function': mann_kendall, 'full_name': "Mann-Kendall Test"},
}

METRICS = {
    "per_error": {'function': _percentage_error, 'full_name':"Percentage Error or bias"},
    'mse': {'function': _mse, 'full_name':"Mean Squared Error"},
    'rmse': {'function': _rmse, 'full_name':"Root Mean Squared Error"},
    'nrmse': {'function': _nrmse, 'full_name':"Normalized Root Mean Squared Error"},
    'me': {'function': _me, 'full_name':"Mean Error"},
    'nme': {'function': _nme, 'full_name':"Normalized Mean Error"},
    'mae': {'function': _mae, 'full_name':"Mean Absolute Error"},
    'nmae': {'function': _nmae, 'full_name':"Normalized Mean Absolute Error"},
    'mad': {'function': _mad, 'full_name':"Mean Absolute Deviation"},
    'gmae': {'function': _gmae, 'full_name':"Geometric Mean Absolute Error"},
    'mdae': {'function': _mdae, 'full_name':"Median Absolute Error"},
    'mpe': {'function': _mpe, 'full_name':"Mean Percentage Error"},
    'mape': {'function': _mape, 'full_name':"Mean Absolute Percentage Error"},
    'mdape': {'function': _mdape, 'full_name':"Median Absolute Percentage Error"},
    'smape': {'function': _smape, 'full_name':"Symmetric Mean Absolute Percentage Error"},
    'smdape': {'function': _smdape, 'full_name':"Symmetric Median Absolute Percentage Error"},
    'maape': {'function': _maape, 'full_name':"Mean Arctangent Absolute Percentage Error"},
    'mase': {'function': _mase, 'full_name':"Mean Absolute Scaled Error"},
    'std_ae': {'function': _std_ae, 'full_name':"Normalized Absolute Error"},
    'std_ape': {'function': _std_ape, 'full_name':"Normalized Absolute Percentage Error"},
    'rmspe': {'function': _rmspe, 'full_name':"Root Mean Squared Percentage Error"},
    'rmdspe': {'function': _rmdspe, 'full_name':"Root Median Squared Percentage Error"},
    'rmsse': {'function': _rmsse, 'full_name':"Root Mean Squared Scaled Error"},
    'inrse': {'function': _inrse, 'full_name':"Integral Normalized Root Squared Error"},
    'rrse': {'function': _rrse, 'full_name':"Root Relative Squared Error"},
    'mre': {'function': _mre, 'full_name':"Mean Relative Error"},
    'rae': {'function': _rae, 'full_name':"Relative Absolute Error"},
    'mrae': {'function': _mrae, 'full_name':"Mean Relative Absolute Error"},
    'mdrae': {'function': _mdrae, 'full_name':"Median Relative Absolute Error"},
    'gmrae': {'function': _gmrae, 'full_name':"Geometric Mean Relative Absolute Error"},
    'mbrae': {'function': _mbrae, 'full_name':"Mean Bounded Relative Absolute Error"},
    'umbrae': {'function': _umbrae, 'full_name':"Unscaled Mean Bounded Relative Absolute Error"},
    'mda': {'function': _mda, 'full_name':"Mean Directional Accuracy"},
    'linregress': {'function': _linregress, 'full_name':"Linear Regression"},
    'R': {'function': pearson_correlation, 'full_name':"Pearson Correlation"},
    'spearman_correlation': {'function': spearman_correlation, 'full_name':"Spearman Correlation"},
    'kendall_tau': {'function': kendall_tau, 'full_name':"Kendall Tau"},
    'DISO': {'function': DISO, 'full_name':"DISO index"},
    'Taylor_score': {'function': Taylor_score, 'full_name':"Taylor Skill score"},
}

=======
    'mann_kendall': {'function': mann_kendall, 'full_name': "Mann-Kendall Test"},
}

METRICS = {
    "per_error": {'function': _percentage_error, 'full_name': "Percentage Error or bias", 'short_name': "PE", 'units': "%"},
    'mse': {'function': _mse, 'full_name': "Mean Squared Error", 'short_name': "MSE", 'units': "units^2"},
    'rmse': {'function': _rmse, 'full_name': "Root Mean Squared Error", 'short_name': "RMSE", 'units': "units"},
    'nrmse': {'function': _nrmse, 'full_name': "Normalized Root Mean Squared Error", 'short_name': "NRMSE", 'units': ""},
    'me': {'function': _me, 'full_name': "Mean Error", 'short_name': "ME", 'units': "units"},
    'nme': {'function': _nme, 'full_name': "Normalized Mean Error", 'short_name': "NME", 'units': ""},
    'mae': {'function': _mae, 'full_name': "Mean Absolute Error", 'short_name': "MAE", 'units': "units"},
    'nmae': {'function': _nmae, 'full_name': "Normalized Mean Absolute Error", 'short_name': "NMAE", 'units': ""},
    'mad': {'function': _mad, 'full_name': "Mean Absolute Deviation", 'short_name': "MAD", 'units': "units"},
    'gmae': {'function': _gmae, 'full_name': "Geometric Mean Absolute Error", 'short_name': "GMAE", 'units': "units"},
    'mdae': {'function': _mdae, 'full_name': "Median Absolute Error", 'short_name': "MDAE", 'units': "units"},
    'mpe': {'function': _mpe, 'full_name': "Mean Percentage Error", 'short_name': "MPE", 'units': "%"},
    'mape': {'function': _mape, 'full_name': "Mean Absolute Percentage Error", 'short_name': "MAPE", 'units': "%"},
    'mdape': {'function': _mdape, 'full_name': "Median Absolute Percentage Error", 'short_name': "MDAPE", 'units': "%"},
    'smape': {'function': _smape, 'full_name': "Symmetric Mean Absolute Percentage Error", 'short_name': "SMAPE", 'units': "%"},
    'smdape': {'function': _smdape, 'full_name': "Symmetric Median Absolute Percentage Error", 'short_name': "SMDAPE", 'units': "%"},
    'maape': {'function': _maape, 'full_name': "Mean Arctangent Absolute Percentage Error", 'short_name': "MAAPE", 'units': ""},
    'mase': {'function': _mase, 'full_name': "Mean Absolute Scaled Error", 'short_name': "MASE", 'units': ""},
    'std_ae': {'function': _std_ae, 'full_name': "Normalized Absolute Error", 'short_name': "StdAE", 'units': ""},
    'std_ape': {'function': _std_ape, 'full_name': "Normalized Absolute Percentage Error", 'short_name': "StdAPE", 'units': "%"},
    'rmspe': {'function': _rmspe, 'full_name': "Root Mean Squared Percentage Error", 'short_name': "RMSPE", 'units': "%"},
    'rmdspe': {'function': _rmdspe, 'full_name': "Root Median Squared Percentage Error", 'short_name': "RMDSPE", 'units': "%"},
    'rmsse': {'function': _rmsse, 'full_name': "Root Mean Squared Scaled Error", 'short_name': "RMSSE", 'units': ""},
    'inrse': {'function': _inrse, 'full_name': "Integral Normalized Root Squared Error", 'short_name': "INRSE", 'units': ""},
    'rrse': {'function': _rrse, 'full_name': "Root Relative Squared Error", 'short_name': "RRSE", 'units': ""},
    'mre': {'function': _mre, 'full_name': "Mean Relative Error", 'short_name': "MRE", 'units': ""},
    'rae': {'function': _rae, 'full_name': "Relative Absolute Error", 'short_name': "RAE", 'units': ""},
    'mrae': {'function': _mrae, 'full_name': "Mean Relative Absolute Error", 'short_name': "MRAE", 'units': ""},
    'mdrae': {'function': _mdrae, 'full_name': "Median Relative Absolute Error", 'short_name': "MDRAE", 'units': ""},
    'gmrae': {'function': _gmrae, 'full_name': "Geometric Mean Relative Absolute Error", 'short_name': "GMRAE", 'units': ""},
    'mbrae': {'function': _mbrae, 'full_name': "Mean Bounded Relative Absolute Error", 'short_name': "MBRAE", 'units': ""},
    'umbrae': {'function': _umbrae, 'full_name': "Unscaled Mean Bounded Relative Absolute Error", 'short_name': "UMBRAE", 'units': ""},
    'mda': {'function': _mda, 'full_name': "Mean Directional Accuracy", 'short_name': "MDA", 'units': "%"},
    'linregress': {'function': _linregress, 'full_name': "Linear Regression", 'short_name': "LinReg", 'units': ""},
    'R': {'function': pearson_correlation, 'full_name': "Pearson Correlation", 'short_name': "R", 'units': ""},
    'spearman_correlation': {'function': spearman_correlation, 'full_name': "Spearman Correlation", 'short_name': "Spearman", 'units': ""},
    'kendall_tau': {'function': kendall_tau, 'full_name': "Kendall Tau", 'short_name': "Kendall", 'units': ""},
    'DISO': {'function': DISO, 'full_name': "DISO index", 'short_name': "DISO", 'units': ""},
    'Taylor_score': {'function': Taylor_score, 'full_name': "Taylor Skill score", 'short_name': "Taylor", 'units': ""},
}

METRICS_COMPOSITE = {
    'mann_kendall': ['trend', 'h', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept'],
    'R': ['R', 'pvalue'],}

>>>>>>> c80f4457 (First commit)
# ----------------------- Evaluation function ---------------------- #

# ----------------------- Evaluation function ---------------------- #

def evaluate_single(actual, metrics):
    """
    Evaluates the given metrics on the actual data.

    Parameters:
    actual (array-like): The actual data.
    metrics (list): A list of metric names to evaluate.

    Returns:
    results (dict): A dictionary of metric names and their corresponding results.

    `METRICS_SINGULAR` is a dictionary that maps metric names to their corresponding functions.

    Here are the keys and their corresponding values:

    - 'kurtosis': kurtosis - This key corresponds to the kurtosis function, which measures the "tailedness" of the probability distribution of a real-valued random variable.
    - 'skewness': skewness - This key corresponds to the skewness function, which measures the asymmetry of the probability distribution of a real-valued random variable about its mean.
    - 'std': std - This key corresponds to the standard deviation function, which measures the amount of variation or dispersion of a set of values.
    - 'mandall_kernel': mann_kandall - This key corresponds to the Mann-Kendall test function, which is a non-parametric test for identifying trends in time series data.
    """

    actual=check_dim(actual)
    results = {}

    if type(metrics) is str:
        metrics=[metrics]
    elif type(metrics) is list:
        pass
    else:
        raise ValueError("Invalid metrics")
    for name in metrics:
        try:
<<<<<<< HEAD
            if name == 'manndall_kendall':
                sub_results = METRICS_SINGULAR[name]['function'](check_dim(actual))
=======
            metric_results = METRICS_SINGULAR[name]['function'](check_dim(actual))
            if isinstance(metric_results, dict):
                # If the result is a dictionary, we need to extract the values
                sub_results = metric_results
>>>>>>> c80f4457 (First commit)
                for key, value in sub_results.items():
                    if type(value) is str:
                        results[key] = value
                    else:
                        results[key] = round(float(value),4)
            else:
<<<<<<< HEAD
                results[name] = round(METRICS_SINGULAR[name]['function'](check_dim(actual)),4)
        except Exception as err:
            results = np.nan
            print('Unable to compute metric {0}: {1}'.format(name, err))
=======
                results[name] = round(float(metric_results),4)
        except Exception as err:
            results = np.nan
            raise ValueError('Unable to compute metric {0}: {1}'.format(name, err))
            # Stop the evaluation if an error occurs
            # raise ValueError("Unable to compute metric {0}: {1}".format(name,

    if len(metrics) == 1:
        results = results[metrics[0]]
>>>>>>> c80f4457 (First commit)
    return results

def evaluate_compare(actual, predicted, metrics):
    """
    Compares the actual and predicted data using the given metrics.

    Parameters:
    actual (array-like): The actual data.
    predicted (array-like, optional): The predicted data. If None, only single metrics are evaluated.
    metrics (list): A list of metric names to evaluate.

    Returns:
    results (dict): A dictionary of metric names and their corresponding results.

    METRICS is a dictionary that maps metric names to their corresponding functions and full names.

    Here are the keys and their corresponding values:

    - 'per_error': {'function': _percentage_error, 'full_name':"Percentage Error"}
    - 'mse': {'function': _mse, 'full_name':"Mean Squared Error"}
    - 'rmse': {'function': _rmse, 'full_name':"Root Mean Squared Error"}
    - 'nrmse': {'function': _nrmse, 'full_name':"Normalized Root Mean Squared Error"}
    - 'me': {'function': _me, 'full_name':"Mean Error"}
    - 'mae': {'function': _mae, 'full_name':"Mean Absolute Error"}
    - 'nmae': {'function': _nmae, 'full_name':"Normalized Mean Absolute Error"}
    - 'mad': {'function': _mad, 'full_name':"Mean Absolute Deviation"}
    - 'gmae': {'function': _gmae, 'full_name':"Geometric Mean Absolute Error"}
    - 'mdae': {'function': _mdae, 'full_name':"Median Absolute Error"}
    - 'mpe': {'function': _mpe, 'full_name':"Mean Percentage Error"}
    - 'mape': {'function': _mape, 'full_name':"Mean Absolute Percentage Error"}
    - 'mdape': {'function': _mdape, 'full_name':"Median Absolute Percentage Error"}
    - 'smape': {'function': _smape, 'full_name':"Symmetric Mean Absolute Percentage Error"}
    - 'smdape': {'function': _smdape, 'full_name':"Symmetric Median Absolute Percentage Error"}
    - 'maape': {'function': _maape, 'full_name':"Mean Arctangent Absolute Percentage Error"}
    - 'mase': {'function': _mase, 'full_name':"Mean Absolute Scaled Error"}
    - 'std_ae': {'function': _std_ae, 'full_name':"Normalized Absolute Error"}
    - 'std_ape': {'function': _std_ape, 'full_name':"Normalized Absolute Percentage Error"}
    - 'rmspe': {'function': _rmspe, 'full_name':"Root Mean Squared Percentage Error"}
    - 'rmdspe': {'function': _rmdspe, 'full_name':"Root Median Squared Percentage Error"}
    - 'rmsse': {'function': _rmsse, 'full_name':"Root Mean Squared Scaled Error"}
    - 'inrse': {'function': _inrse, 'full_name':"Integral Normalized Root Squared Error"}
    - 'rrse': {'function': _rrse, 'full_name':"Root Relative Squared Error"}
    - 'mre': {'function': _mre, 'full_name':"Mean Relative Error"}
    - 'rae': {'function': _rae, 'full_name':"Relative Absolute Error"}
    - 'mrae': {'function': _mrae, 'full_name':"Mean Relative Absolute Error"}
    - 'mdrae': {'function': _mdrae, 'full_name':"Median Relative Absolute Error"}
    - 'gmrae': {'function': _gmrae, 'full_name':"Geometric Mean Relative Absolute Error"}
    - 'mbrae': {'function': _mbrae, 'full_name':"Mean Bounded Relative Absolute Error"}
    - 'umbrae': {'function': _umbrae, 'full_name':"Unscaled Mean Bounded Relative Absolute Error"}
    - 'mda': {'function': _mda, 'full_name':"Mean Directional Accuracy"}
    - 'linregress': {'function': _linregress, 'full_name':"Linear Regression"}
    - 'R': {'function': pearson_correlation, 'full_name':"Pearson Correlation"}
    - 'spearman_correlation': {'function': spearman_correlation, 'full_name':"Spearman Correlation"}
    - 'kendall_tau': {'function': kendall_tau, 'full_name':"Kendall Tau"}
    - 'DISO': {'function': DISO, 'full_name':"DISO index"}
    - 'Taylor_score': {'function': Taylor_score, 'full_name':"Taylor Skill score"}

    """

    actual,predicted=check_dim(actual),check_dim(predicted)
    results = {}

    if type(metrics) is str:
        metrics=[metrics]
    elif type(metrics) is list:
        pass
    else:
        raise ValueError("Invalid metrics")

    for name in metrics:
        try:
            if name in METRICS:
<<<<<<< HEAD
                results[name] = round(METRICS[name]['function'](actual, predicted),4)
        except Exception as e:
            results[name] = np.nan
            print('Unable to compute metric {0}: {1}'.format(name, e))
=======
                metric_results = METRICS[name]['function'](actual, predicted)
                if isinstance(metric_results, dict):
                    # If the result is a dictionary, we need to extract the values
                    sub_results = metric_results
                    for key, value in sub_results.items():
                        if type(value) is str:
                            results[key] = value
                        else:
                            results[key] = round(float(value),4)
                else:
                    # If the result is a single value, we round it
                    results[name] = round(float(metric_results),4)
        except Exception as e:
            results[name] = np.nan
            raise ValueError('Unable to compute metric {0}: {1}'.format(name, e))

    if len(results) == 1:
        results = results[metrics[0]]
>>>>>>> c80f4457 (First commit)
    return results

def evaluate_single_all(actual):
    """
    Evaluates all single metrics on the actual data.

    Parameters:
    actual (array-like): The actual data.

    Returns:
    results (dict): A dictionary of metric names and their corresponding results.
    """
    result=evaluate_single(actual, metrics=list(METRICS_SINGULAR.keys()))
    return result

def evaluate_all(actual, predicted):
    """
    Compares the actual and predicted data using all available metrics.

    Parameters:
    actual (array-like): The actual data.
    predicted (array-like): The predicted data.

    Returns:
    results (dict): A dictionary of metric names and their corresponding results.
    """
    return evaluate_compare(actual, predicted, metrics=list(METRICS.keys()))

# ----------------------- Help function ---------------------- #
def trend_help(option:Optional[Literal['METRICS','METRIC-sing']]=None):
    if option == 'METRICS':
        for key, value in METRICS.items():
            print(f"{key}: {value['full_name']}")
    elif option == 'METRIC-sing':
        print(list(METRICS_SINGULAR.keys))
    else:
        print("Invalid option")