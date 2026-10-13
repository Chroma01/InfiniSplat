"""Restore losslessly deduplicated inference tensors before strict loading."""


def expand_aliases(state, aliases):
    if not isinstance(aliases, dict):
        raise ValueError('tensor_aliases must be a mapping')
    stored = set(state)
    for name, target in aliases.items():
        if name in stored or target not in stored or name == target:
            raise ValueError('Invalid tensor alias: ' + str(name))
        state[name] = state[target]
    return state
