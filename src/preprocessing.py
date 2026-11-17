import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stemmer = PorterStemmer() # convert words to their root form
token_re = re.compile(r"[a-z0-9]+") # regular expression to match lowercase alphanumeric tokens

def preprocess(text):
    """Lowercase, tokenize, remove stopwords and 1-char tokens, stem."""
    tokens = token_re.findall(text.lower())
    return [stemmer.stem(t) for t in tokens
            if t not in ENGLISH_STOP_WORDS and len(t) > 1]

