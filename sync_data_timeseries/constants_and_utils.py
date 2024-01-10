import datetime
import numpy as np
import warnings

def apply_smoothing(ts, num_before=3, num_after=3):
    """
    Return smoothed timeseries where the entry at time t is the average value 
    from t-num_before to t+num_after (inclusive).
    """
    assert num_before >= 0
    assert num_after >= 0
    smoothed_ts = np.zeros(len(ts)) * np.nan
    for i in range(len(ts)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            smoothed_ts[i] = np.nanmean(ts[i-num_before:i+num_after+1])
    return smoothed_ts