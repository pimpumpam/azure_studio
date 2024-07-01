import itertools
import numpy as np

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'

def find_item(dictionary, key):
    """
    Dictionary 내 key에 해당하는 value 값 탐색
    
    Argument
    ----------
    dictionary(dict): 탐색 대상이 되는 dictionary
    key(str): Dictionary에서 조회하고자 하는 value에 해당하는 key 값
    
    return
    ----------
    item(str, int, ...): 주어진 key에 해당하는 value 값
    
    """
    if key in dictionary:
        return dictionary[key]
    
    for k, v in dictionary.items():
        if isinstance(v, dict):
            item = find_item(v, key)
            
            if item is not None:
                return item
            
def combination_hyperparams(hyp_dict):
    
    for key, val in hyp_dict.items():
        if isinstance(val, list):
            continue
        else:
            hyp_dict[key] = [val]
            
            
    keys, values = zip(*hyp_dict.items())
    permutations = [dict(zip(keys, val)) for val in itertools.product(*values)]
    
    return permutations


def denormalization_pred(normalized_val, scaler, framework='pandas', column='rx_byte'):
    scaler_kind = scaler.scaler_kind
    
    if scaler_kind == 'MinMaxScaler':
        if framework == 'pyspark':
            max_val = scaler.max_dict[column]
            min_val = scaler.min_dict[column]
            
        elif framework == 'pandas':
            max_val = scaler.scaler.data_max_.item()
            min_val = scaler.scaler.data_min_.item()
            
        return normalized_val*(max_val-min_val) + min_val
    
    elif scaler_kind == 'StandardScaler':
        if framework == 'pyspark':
            print("기능 추가 필요")
            
        elif framework == 'pandas':
            mean_val = scaler.scaler.mean_.item()
            std_val = np.sqrt(scaler.scaler.var_.item())
            
        return (normalized_val*std_val) + mean_val