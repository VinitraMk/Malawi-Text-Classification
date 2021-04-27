from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from modules.preprocessing import clean_text
from stopwordsiso import stopwords as stopwords
from nltk.tokenize import sent_tokenize

class Vectorizer:
    train = None
    test = None

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def count_vectorizer(self):
        vectorizer = CountVectorizer(
            preprocessor = clean_text,
            stop_words = stopwords("ny"),
            ngram_range = (1, 2),
            min_df = 0.05,
        )
        train_features = vectorizer.fit_transform(self.train)
        test_features = vectorizer.transform(self.test)

        return train_features, test_features


    def tfidf_vectorizer(self):
        vectorizer = TfidfVectorizer(preprocessor = clean_text, stop_words=stopwords("ny"), ngram_range=(1,2), sublinear_tf=True, min_df=0.05, norm='12')
        train_features = vectorizer.fit_transform(self.train)
        test_features = vectorizer.transform(self.test)
        return train_features, test_features

    def tfidf_transformer(self):
        vectorizer = CountVectorizer(
            preprocessor = clean_text,
            stop_words = stopwords("ny"),
            ngram_range = (1, 4),
            min_df = 0.05,
        )
        train_features = vectorizer.fit_transform(self.train)
        test_features = vectorizer.transform(self.test)
        transformer = get_extractors('tfidf-transformer')
        train_features = transformer.fit_transform(train_features)
        test_features = transformer.transform(test_features)
        return train_features, test_features

    def get_vectorized_features(self, type = 'count'):
        if type == 'count':
            print('Getting count vectorized features...\n')
            return self.count_vectorizer()
        elif type == 'tfidf':
            print('Getting tfidf vectorized features...\n')
            return self.tfidf_vectorizer()
        elif type == 'tfidf-transformer':
            print('Getting tfidf transformed features...\n')
            return self.tfidf_transformer()
        else:
            print('Getting count vectorized features...\n')
            return self.count_vectorizer()


def get_extractors(extractor_type = 'count'):
    if extractor_type == 'count':
        transformer = CountVectorizer(
            preprocessor = clean_text,
            stop_words = stopwords('ny'),
            lowercase = True
        )
    elif extractor_type == 'tfidf':
        transformer = TfidfVectorizer(preprocessor = clean_text, stop_words = stopwords("ny"), ngram_range=(1,2))
    elif extractor_type == 'tfidf-transformer':
        transformer = TfidfTransformer(use_idf = False)
    else:
        transformer = CountVectorizer(
            preprocessor = clean_text,
            stop_words = stopwords('ny')
        )
    return transformer 
        

def sentence_tokenizer(text):
    return sent_tokenize(text)
