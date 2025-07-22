def _init():
    global _global_score_dict
    _global_score_dict = {}

def set_val(key, value):
    _global_score_dict[key] = value

def get_val(key):
    try:
        return _global_score_dict[key]
    except:
        return None