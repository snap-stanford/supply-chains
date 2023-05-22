import datetime
import numpy as np
import psutil
from psutil._common import bytes2human

PRIMARY_KEY = 'date, supplier_id, buyer_id, hs_code, quantity, weight, price, amount'
BATTERY_PARTS = [
            '850760', '280440', '281700', '282110', '282200', '382490', 
            '382499', '390210', '392020', '392051', '392099', '392119', 
            '392310', '392690', '401140', '401390', '401699', '420212', 
            '721240', '722230', '722699', '730120', '730690', '730711', 
            '730890', '731100', '731816', '732599', '732619', '732620', 
            '732690', '740822', '740919', '740921', '741011', '741021', 
            '741022', '741220', '741529', '741533', '741999', '750522', 
            '750610', '750620', '760612', '760719', '760720', '761699', 
            '790700', '810590', '830230', '830249', '831110', '831120', 
            '831190', '848049', '848280', '850110', '850120', '850440', 
            '850450', '850590', '850640', '850650', '850660', '850680', 
            '850730', '850780', '850790', '851830', '851890', '853222', 
            '853223', '853340', '853610', '853630', '853641', '854190', 
            '854290', '854370', '854411', '854442', '854519', '854720', 
            '860900'
        ]
BATTERY = '850760'

def check_memory_usage(print_message=True):
    """
    Checks current memory usage.
    """
    virtual_memory = psutil.virtual_memory()
    total_memory = getattr(virtual_memory, 'total')
    available_memory = getattr(virtual_memory, 'available')
    free_memory = getattr(virtual_memory, 'free')
    available_memory_percentage = 100. * available_memory / total_memory
    if print_message:
        print('Total memory: %s; free memory: %s; available memory %s; available memory %2.3f%%' % (
            bytes2human(total_memory),
            bytes2human(free_memory),
            bytes2human(available_memory),
            available_memory_percentage))
    return available_memory

def parse_battery_bom():
    """
    Parse battery_bom.txt to return dictionary.
    """
    fn = './battery_bom.txt'
    with open(fn, 'r') as f:
        content = f.readlines()
    bom = {}
    for l in content:
        if ' - ' in l:
            part, codes = l.split(' - ')
            codes = codes.split('[', 1)[1]
            codes = codes.rsplit(']', 1)[0]
            bom[part] = codes.replace('\'', '').split(',')
    return bom

def apply_smoothing(ts, num_before=3, num_after=3):
    """
    Return smoothed timeseries where the entry at time t is the average value 
    from t-num_before to t+num_after (inclusive).
    """
    assert num_before >= 1
    assert num_after >= 1
    smoothed_ts = np.zeros(len(ts)) * np.nan
    for i in range(len(ts)):
        smoothed_ts[i] = np.nanmean(ts[i-num_before:i+num_after+1])
    return smoothed_ts

def extract_daily_timeseries(pd_series, min_datetime, max_datetime, verbose=True):
    """
    Returns a value per date in range min_datetime, max_datetime (inclusive).  
    Assumes pd_series is a pandas Series with datetime as index. If date is missing
    from pd_series, then np.nan is included.
    
    Example
    pd_series
    2022/5/1   10
    2022/5/3    3
    2022/5 4    4
    min_datetime = 2022/5/1
    max_datetime = 2022/5/5
    
    Returns:
    [10, nan, 3, 4, nan]
    """
    dates = []
    vals = []
    dt = min_datetime
    while dt <= max_datetime:
        dates.append(dt)
        if dt in pd_series.index:  # check if we have data for this date
            vals.append(pd_series.loc[dt])
        else:
            vals.append(np.nan)
        dt += datetime.timedelta(days=1)
    vals = np.array(vals)
    if verbose:
        print('%d/%d days have data' % (len(dates)-np.isnan(vals).sum(), len(dates)))
    return dates, vals