from sklearn.naive_bayes import MultinomialNB, GaussianNB
from modules.utils import get_filename
from config.config import Config
import pandas as pd
from modules.model_utils import get_gdsearch, get_pipeline
from modules.vectorizer import Vectorizer

class NaiveBayesian:
    X = None
    y = None
    model = None
    test_ids = None
    test_features = None

    def __init__(self, X, y, test_ids):
        self.X = X
        self.y = y
        self.test_ids = test_ids

    def train_model(self, type = 'Gaussian'):
        if type == 'Gaussian':
            vectorizer = Vectorizer(self.X, self.y)
            train_features, test_features = vectorizer.get_vectorized_features(type='count')
            self.test_features = test_features
            self.model = GaussianNB().fit(train_features.toarray(), self.y)
        else:
            self.model = get_gdsearch(get_pipeline(MultinomialNB()), type).fit(self.X, self.y)


    def predict_and_save_csv(self, test_features, model_type = 'Gaussian'):
        print('Training NaiveBayesian model...\n')
        self.train_model(model_type)
        print('Saving predictions to csv...\n')
        title = get_filename('naive_bayesian_multinom')
        output_directory = Config().get_config()['output_directory']
        y_preds = None
        if model_type == 'Gaussian':
            y_preds = self.model.predict(self.test_features.toarray())
        else:
            y_preds = self.model.predict(test_features)
        y_ids = pd.DataFrame(self.test_ids, columns=['ID'])
        y_preds_df = pd.DataFrame(y_preds, columns=['Label'])
        predictions = y_ids.join(y_preds_df)
        predictions.to_csv(f'{output_directory}/{title}.csv', index=False)

