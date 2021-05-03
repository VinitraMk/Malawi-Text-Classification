from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from modules.vectorizer import get_extractors
from sklearn.feature_selection import chi2, SelectKBest

def get_gdsearch(text_clf, model_type = 'Gaussian', skip_vectorizer = False):
    parameters = {
        #'vect__ngram_range': [(1,1), (1,2), (1,4)],
        #'tfidf__use_idf': (True, False)
    }

    if model_type == 'Multinom':
        parameters['clf__alpha'] = [1e-3]
        parameters['vect__ngram_range'] = [(1,4)]
        parameters['vect__max_features'] = [250]
        parameters['tfidf__use_idf'] = [True]

    if model_type == 'XGBoost':
        parameters['clf__n_estimators'] = [30]
        parameters['clf__subsample'] = [0.5]
    
    if model_type == 'SVC':
        parameters['clf__break_ties'] = (True, False)
        parameters['clf__class_weight'] = ['balanced']
        parameters['clf__degree'] = [3, 4]

    if model_type == 'LSVM':
        parameters['clf__C'] = [1]

    if model_type == 'RandomForest':
        parameters['clf__n_estimators'] = [50]

    if model_type == 'Logistic':
        parameters['clf__max_iter'] = [10]
        if not(skip_vectorizer):
            parameters['vect__ngram_range'] = [(1,2)]
            parameters['tfidf__use_idf'] = [False]

    if model_type == 'DTree':
        parameters['clf__max_features'] = ['log2']

    print('Final paramgrid:', parameters,'\n')
    gs_clf = GridSearchCV(text_clf, parameters, cv = 5, n_jobs = -1, scoring='roc_auc_ovr')

    return gs_clf

def get_pipeline(clf, extractor_type = 'count', use_vectorizer = False):
    if not(use_vectorizer):
        text_clf = Pipeline([
            ('clf', clf)
        ])
    else:
        text_clf = Pipeline([
            ('vect', get_extractors('count')),
            ('tfidf', get_extractors('tfidf-transformer')),
            ('clf', clf)
        ])

    return text_clf

def select_kbest(k):
    kbest = SelectKBest(chi2, k)
    return kbest
