def flat_dict(cfg_dict: dict):
    res = {}
    for item, value in cfg_dict.items():
        res.update(_unfold(item, value))
    return res


def _unfold(parent, element):
    res = {}
    if isinstance(element, dict):
        for item, value in element.items():
            res.update(_unfold(f"{parent}.{item}", value))
    else:
        res[parent] = element
    return res
