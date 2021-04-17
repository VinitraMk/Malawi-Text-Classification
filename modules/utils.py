from datetime import datetime
import re

def get_filename(filename):
    now = datetime.now().strftime('%d%m-%H%M%Y')
    return f'{filename}_{now}'
