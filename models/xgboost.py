from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from modules.model_utils import get_pipeline, get_gdsearch
from modules.utils import get_filename
from config.config import Config
import pandas as pd

class XGBoost:
    X = None
    y = None
    test_ids = None
    label_encoder = None

    def __init__(self, X, y, test_ids, label_encoder):
        self.X = X
        self.y = y
        self.test_ids = test_ids
        self.label_encoder = label_encoder

    def train_model(self, skip_pipeline = False):
        if not(skip_pipeline):
            pipeline = get_pipeline(XGBClassifier(random_state = 42, seed = 2, objective='multi:softmax', eval_metric = 'merror', use_label_encoder = False, learning_rate = 0.1, n_jobs = -1, colsample_bytree = 1.0), 'tfidf')
            self.model = get_gdsearch(pipeline, 'XGBoost')
            self.model = self.model.fit(self.X, self.y)
            print('Best model params:', self.model.best_params_)
            print('All model params: ', self.model.get_params(True),'\n')
        else:
            self.model = XGBClassifier(random_state = 42, seed = 2, objective = 'multi:softmax', eval_metric = 'merror', use_label_encoder = False, n_estimators = 30, colsample_bytree = 0.8, subsample = 1.0)
            self.model.fit(self.X, self.y)

    def predict_and_save_csv(self, test_features, skip_pipeline = False):
        print('Training XGBoost Classifier...\n')
        self.train_model(skip_pipeline)
        print('Saving predictions to csv...\n')
        title = get_filename('xgboost')
        output_directory = Config().get_config()['output_directory']
        y_preds = self.model.predict(test_features)
        y_ids = pd.DataFrame(self.test_ids, columns=['ID'])
        y_preds_df = pd.DataFrame(y_preds, columns=['Label_Id'])
        predictions = y_ids.join(y_preds_df)
        predictions = self.label_encoder.decode(predictions)
        predictions.to_csv(f'{output_directory}/{title}.csv', index=False)
