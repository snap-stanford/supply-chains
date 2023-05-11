import psutil
from psutil._common import bytes2human

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