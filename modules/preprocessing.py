import re
from sklearn.preprocessing import LabelEncoder
from config.config import Config
import os

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

def build_vocabulary():
    input_file_path = os.path.join(Config().get_config()['input_directory'],'all.txt')
    fp = open(input_file_path)
    lines = fp.readlines()
    vocabulary = {}
    words = []
    c = 0
    
    for i in range(len(lines)):
        line = re.sub(r"<\n>",'', lines[i])
        line = line.strip().lower()
        if line not in vocabulary:
            vocabulary[line] = c
            words.append(line)
            c = c + 1

    print(f'Built vocabulary of size {len(words)}\n')

    return words


class LabelEncoding:

    le = None
    all_labels = None
    encoded_labels = None

    def __init__(self, all_labels):
        self.le = LabelEncoder()
        self.encoded_labels = self.le.fit_transform(all_labels)
        self.encoded_labels = dict(zip(self.encoded_labels, all_labels))

    def encode(self, train):
        trfs = self.le.transform(train['Label'])
        train['Label_Id'] = trfs
        return train, self.encoded_labels

    def decode(self, test):
        trfs = self.le.inverse_transform(test['Label_Id'])
        test['Label'] = trfs
        test = test.drop(columns=['Label_Id'], axis=1)
        return test
