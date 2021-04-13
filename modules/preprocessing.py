import re

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
