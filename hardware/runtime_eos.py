"""Honor the installed model's full native EOS set consistently across runtimes."""


def normalize_eos_ids(value):
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return tuple(dict.fromkeys(int(token) for token in value))


def model_eos_ids(model, tokenizer):
    config = getattr(model, "generation_config", None)
    value = getattr(config, "eos_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    return normalize_eos_ids(value)


def tokenizer_eos_ids(tokenizer):
    return normalize_eos_ids(getattr(tokenizer, "specstop_eos_ids", tokenizer.eos_token_id))
