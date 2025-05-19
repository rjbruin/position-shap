def nested_attribute(key):
    # Given a dot-separated string describing a potentionally nested class attribute, return the attribute
    def get_attr(obj):
        for attr in key.split('.'):
            obj = getattr(obj, attr)
        return obj
    return get_attr

def collect_by_substring(key, exclude=[]):
    def collect(model):
        # Collect all parameters from the model's layers that contain a particular substring
        sources = []
        for name, param in model.named_parameters():
            if key in name and not any([e in name for e in exclude]):
                sources.append(param)
        return sources
    return collect

def get_attribute_from_name_or_callable(model, attribute):
    """
    Get an attribute from a model, either by name (if attribute is a string) or
    by calling the attribute (if attribute is a callable).
    """
    if isinstance(attribute, str) and hasattr(model, attribute):
        return getattr(model, attribute)
    elif callable(attribute):
        return attribute(model)
    else:
        # DEBUG
        return None
        # raise NotImplementedError(f"Attribute {attribute} is neither a string "
        #                           f"nor a callable.")