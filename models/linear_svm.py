import pandas as pd
import numpy as np
from config.config import Config
from sklearn.svm import LinearSVC
from modules.utils import get_filename
from modules.model_utils import get_pipeline, get_gdsearch

class LinearSVM:
    X = None
    y = None
    test_ids = None
    model = None
    label_encoder = None

    def __init__(self, X, y, test_ids, label_encoder):
        self.X = X
        self.y = y
        self.test_ids = test_ids
        self.label_encoder = label_encoder

    def train_models(self):
        pipeline = get_pipeline(LinearSVC(C=1, multi_class='ovr',class_weight='balanced', max_iter = 1000000, dual = True, tol=1e-5), 'tfidf')
        self.model = get_gdsearch(pipeline, 'LSVM').fit(self.X, self.y)
        print('Best estimator params: ', self.model.best_params_)
        print('All model params: ', self.model.get_params(True),'\n')

    def predict_and_save_csv(self, test_features):
        print('Training Linear SVM Model...\n')
        self.train_models()
        print('Saving predictions to csv...\n')
        title = get_filename('linear_svm')
        output_directory = Config().get_config()['output_directory']
        y_preds = self.model.predict(test_features)
        y_ids = pd.DataFrame(self.test_ids, columns=['ID'])
        y_preds_df = pd.DataFrame(y_preds, columns=['Label_Id'])
        predictions = y_ids.join(y_preds_df)
        predictions = self.label_encoder.decode(predictions)
        predictions.to_csv(f'{output_directory}/{title}.csv', index=False)
