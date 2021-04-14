from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from modules.vectorizer import get_extractors

def get_gdsearch(text_clf, type = 'Gaussian'):
    parameters = {
        'vect__ngram_range': [(1,1), (1,2), (1,4)],
        'tfidf__use_idf': (True, False),
    }

    if type != 'Gaussian':
        parameters['clf__alpha'] = (1e-1, 1e-2, 1e-3)

    gs_clf = GridSearchCV(text_clf, parameters, cv = 5, n_jobs = -1)

    return gs_clf

def get_pipeline(clf):
    text_clf = Pipeline([
        ('vect',get_extractors()),
        ('tfidf', get_extractors('tfidf-transformer')),
        ('clf', clf)
    ])
    return text_clf