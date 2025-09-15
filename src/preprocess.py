import re
import numpy as np

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
MULTISPACE_PATTERN = re.compile(r"\s+")

def basic_clean(text: str) -> str:
    """Lowercase, strip URLs/mentions, collapse whitespace. Keep hashtag words."""
    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    # keep the hashtag token but drop the '#'
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()

def pad_sequences(seqs, maxlen, pad_value=0):
    """Simple left-trunc, right-pad (like Keras default)."""
    out = np.full((len(seqs), maxlen), pad_value, dtype=np.int32)
    for i, s in enumerate(seqs):
        s = np.asarray(s, dtype=np.int64)
        if len(s) > maxlen:
            s = s[-maxlen:]
        out[i, :len(s)] = s
    return out
