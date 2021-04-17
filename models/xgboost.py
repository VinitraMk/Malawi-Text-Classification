from xgboost import XGBClassifier
from modules.model_utils import get_pipeline, get_gdsearch
from modules.utils import get_filename
from config.config import Config
import pandas as pd

class XGBoost:
    X = None
    y = None
    test_ids = None

    def __init__(self, X, y, test_ids):
        self.X = X
        self.y = y
        self.test_ids = test_ids

    def train_model(self):
        pipeline = get_pipeline(XGBClassifier(random_state = 42, seed = 2, colsample_bytree = 0.6, subsample = 0.7), 'tfidf')
        self.model = get_gdsearch(pipeline).fit(self.X, self.y)
        print('Model best params:', self.model.best_params_)

    def predict_and_save_csv(self, test_features):
        print('Training XGBoost Classifier...\n')
        self.train_model()
        print('Saving predictions to csv...\n')
        title = get_filename('xgboost')
        output_directory = Config().get_config()['output_directory']
        y_preds = self.model.predict(test_features)
        y_ids = pd.DataFrame(self.test_ids, columns=['ID'])
        y_preds_df = pd.DataFrame(y_preds, columns=['Label'])
        predictions = y_ids.join(y_preds_df)
        predictions.to_csv(f'{output_directory}/{title}.csv', index=False)
