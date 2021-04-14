from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from modules.preprocessing import clean_text
from stopwordsiso import stopwords as stopwords

class Vectorizer:
    train = None
    test = None

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def count_vectorizer(self):
        vectorizer = CountVectorizer(
            preprocessor = clean_text,
            stop_words = stopwords("ny")
        )
        train_features = vectorizer.fit_transform(self.train)
        test_features = vectorizer.transform(self.test)

        return train_features, test_features

    def tfidf_vectorizer(self):
        vectorizer = TfidfVectorizer(preprocessor = clean_text)
        train_features = vectorizer.fit_transform(self.train)
        test_features = vectorizer.transform(self.test)
        return train_features, test_features

    def get_vectorized_features(self, type = 'count'):
        if type == 'count':
            print('Getting count vectorized features...\n')
            return self.count_vectorizer()
        elif type == 'tfidf':
            print('Getting tfidf vectorized features...\n')
            return self.tfidf_vectorizer()
        else:
            print('Getting count vectorized features...\n')
            return self.count_vectorizer()
        