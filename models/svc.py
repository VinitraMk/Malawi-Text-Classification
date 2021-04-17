import pandas as pd
import numpy as np
from config.config import Config
from sklearn import svm
from modules.utils import get_filename
from config.config import Config
from sklearn.svm import LinearSVC
from modules.utils import get_filename
from modules.model_utils import get_pipeline, get_gdsearch

class SVC:
    X = None
    y = None
    test_ids = None
    linear_model = None
    rbf_model = None
    poly_model = None
    sigmoid_model = None
    label_encoder = None

    def __init__(self, X, y, test_ids, label_encoder):
        self.X = X
        self.y = y
        self.test_ids = test_ids
        self.label_encoder = label_encoder

    def train_models(self):
        pipeline = get_pipeline(svm.SVC(kernel='linear', decision_function_shape='ovr', C=1, max_iter = -1, random_state = 42), 'tfidf')
        self.linear_model = get_gdsearch(pipeline)
        self.linear_model.fit(self.X, self.y)
        pipeline = get_pipeline(svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1, gamma=1, max_iter = -1, random_state = 42))
        self.rbf_model = get_gdsearch(pipeline)
        self.rbf_model.fit(self.X, self.y)
        pipeline = get_pipeline(svm.SVC(kernel='sigmoid', decision_function_shape='ovr', C=1, gamma=1, max_iter = -1, random_state = 42))
        self.sigmoid_model = get_gdsearch(pipeline)
        self.sigmoid_model.fit(self.X, self.y)
        pipeline = get_pipeline(svm.SVC(kernel='poly', decision_function_shape='ovr', C=1, gamma=1, max_iter = -1, random_state = 42))
        self.poly_model = get_gdsearch(pipeline)
        self.poly_model.fit(self.X, self.y)
        
    def predict_and_save_csv(self, test_features):
        print('Training models...\n')
        self.train_models()
        print('Saving predictions to csv...\n')
        titles = ['linear_svc', 'rbf_svc', 'sigmoid_svc', 'poly_svc']
        output_directory = Config().get_config()['output_directory']
        for i, clf in enumerate((self.linear_model, self.rbf_model, self.sigmoid_model, self.poly_model)):
            y_preds = clf.predict(test_features)
            y_ids = pd.DataFrame(self.test_ids, columns=['ID'])
            y_preds_df = pd.DataFrame(y_preds, columns=['Label'])
            predictions = y_ids.join(y_preds_df)
            predictions = self.label_encoder.decode(predictions)
            predictions.to_csv(f'{output_directory}/{get_filename(titles[i])}.csv', index=False)