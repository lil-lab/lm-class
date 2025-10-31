from features import BagOfWords


def test_bow_simple():
    text = "I love this movie"
    features = BagOfWords.featurize(text)
    assert features == {"bow/love": 1.0, "bow/movie": 1.0}


def test_bow_ignore_stop_words():
    text = "I"
    features = BagOfWords.featurize(text)
    assert features == {}


def test_bow_ignore_repeat():
    text = "I don't know if I love this movie or that movie"
    features = BagOfWords.featurize(text)
    assert features == {
        "bow/love": 1.0,
        "bow/movie": 1.0,
        "bow/know": 1.0,
    }


def test_bow_ignore_capital():
    text = "I LoVe THIs mOVIe"
    features = BagOfWords.featurize(text)
    assert features == {"bow/love": 1.0, "bow/movie": 1.0}


def test_bow_ignore_space():
    text = "   I love   this movie    "
    features = BagOfWords.featurize(text)
    assert features == {"bow/love": 1.0, "bow/movie": 1.0}
