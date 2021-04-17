import re
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    text = re.sub(r'<.*?>','',text)
    text = re.sub(r'\\','',text)
    text = re.sub(r"\ '",'',text)
    text = re.sub(r"\"","",text)

    text = text.strip().lower()
    
    filters = '|"\`#$%&()*+,-./:;<=>?@[\\]^_`{!}~\t\n'
    translate_dict = dict((c," ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    return text

class LabelEncoding:

    le = None
    all_labels = None

    def __init__(self, all_labels):
        self.le = LabelEncoder()
        self.le.fit(all_labels)

    def encode(self, train):
        trfs = self.le.transform(train['Label'])
        train['Label_Id'] = trfs
        return train

    def decode(self, test):
        trfs = self.le.inverse_transform(test['Label_Id'])
        test['Label'] = trfs
        test = test.drop(columns=['Label_Id'], axis=1)
        return test
