# src/persona_helpers.py
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

def top_keywords(texts: List[str], n: int = 6) -> List[str]:
    """
    Return top-n keywords across `texts` using TF-IDF (excludes simple empty inputs).
    """
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < 3:
        return []
    vec = TfidfVectorizer(max_features=300, stop_words="english")
    X = vec.fit_transform(texts)
    scores = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    top = [terms[i] for i in scores.argsort()[::-1][:n]]
    return top
