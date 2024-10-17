import pandas as pd
import numpy as np
import joblib
from tqdm.notebook import tqdm
import pickle
from glob import glob
from unicodedata import normalize
from nltk.tokenize import word_tokenize
from os import path
import re



def clean_text(txt):
    txt_res = normalize("NFKD", txt).replace('\xa0', ' ')
    txt_res = txt_res.replace("\\", "").replace('\\xa0', '')
    return txt_res



def get_score_ocr(path_name, word_set):
    data = []
    for doc in tqdm(glob(path_name)):
        
        doc_name = path.splitext(path.basename(doc))[0]
        print(doc_name)
        with open(doc, encoding="utf8") as file:
            text = str(file.read()).lower()
            text_clean = clean_text(text) # enleve les accents
            text_clean = re.sub(r'[^\w\s]', '', text_clean)  # garde seulement les suites de caractère
            
            current_words = [word for word in word_tokenize(text_clean)]
            valid_words = [word for word in current_words if word in word_set]
            invalid_words = [word for word in current_words if word not in word_set]

            for word in invalid_words:
                print(word)

            
            accuracy = len(valid_words) / len(current_words) if current_words else 0

            data.append([doc_name, accuracy])
    
    df = pd.DataFrame(data, columns=['doc_name', 'accuracy'])
    return df


if __name__ == '__main__':
    print
    set_of_tokens =  joblib.load("/data/jbarre/DICT_CHAPITRES.pkl")
    set_of_tokens.update(['œil', 'cœur', 'cœurs', 'sœur', 'sœur', 'nœud'])
    
    path_name = r'/home/jbarre/corpus_simsim_txt/*.txt'

    df = get_score_ocr(path_name, set_of_tokens)
    df.to_csv('df_WER_MAIN.csv', index=False)
