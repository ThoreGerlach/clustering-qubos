import warnings


def fill_dict(given_dict, compare_dict) -> dict:
    if given_dict is None:
        return_dict = compare_dict
    else:
        return_dict = {}
        default_keys = compare_dict.keys()
        logging_keys = given_dict.keys()
        for key in logging_keys:
            if key in default_keys:
                return_dict[key] = given_dict[key]
            else:
                warnings.warn(key + ' is not an intended logging string!')
        for key in default_keys:
            if key not in logging_keys:
                return_dict[key] = compare_dict[key]
    return return_dict
