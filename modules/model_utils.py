from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from modules.vectorizer import get_extractors
from sklearn.feature_selection import chi2, SelectKBest

def get_gdsearch(text_clf, model_type = 'Gaussian'):
    parameters = {
        #'vect__ngram_range': [(1,1), (1,2), (1,4)],
        #'tfidf__use_idf': (True, False)
    }

    if model_type == 'Multinom':
        parameters['clf__alpha'] = (1e-1, 1e-2, 1e-3)

    if model_type == 'XGBoost':
        parameters['clf__n_estimators'] = [85]
        #parameters['clf__max_depth'] = [35]
        parameters['clf__subsample'] = [0.8]
    
    if model_type == 'SVC':
        parameters['clf__break_ties'] = (True, False)
        parameters['clf__class_weight'] = ['balanced']
        parameters['clf__degree'] = [3, 4]

    if model_type == 'LSVM':
        parameters['clf__C'] = [1]

    print('Final paramgrid:', parameters,'\n')
    gs_clf = GridSearchCV(text_clf, parameters, cv = 5, n_jobs = -1, scoring='roc_auc_ovr')

    return gs_clf

def get_pipeline(clf, extractor_type = 'count'):
    text_clf = Pipeline([
        #('tfidf', get_extractors('tfidf-transformer')),
        ('clf', clf)
    ])
    return text_clf

def select_kbest(k):
    kbest = SelectKBest(chi2, k)
    return kbest
