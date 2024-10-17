import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import pickle
import random
from collections import Counter
import json
from FlagEmbedding import FlagModel
from glob import glob
from unicodedata import normalize
from nltk.tokenize import word_tokenize
from os import path
import os
import shutil
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

model = FlagModel('BAAI/bge-m3',
                  query_instruction_for_retrieval="",
                  use_fp16=True) 

def get_embeddings(path_name, N_tirages=500, M_words_per_passages=512):
    index_tmp, sentences_tmp, novels_tmp = [], [], []
    for doc in tqdm(glob(path_name)):
        doc_name = path.splitext(path.basename(doc))[0]
        with open(doc, encoding="utf8") as file:
            text = str(file.read()).lower()
            text_clean = clean_text(text)
            text_clean = re.sub(r'[^\w\s]', '', text_clean)

        current_words = [word for word in word_tokenize(text_clean)]
        passages_indices = [random.randint(0, len(current_words) - (M_words_per_passages+1)) for _ in range(N_tirages)]
        passages = [" ".join(current_words[indice:indice+M_words_per_passages]) for indice in passages_indices]
        index_passages = [doc_name+"_passage-"+str(j)+"_sentence-"+str(passages_indices[j]) for j in range(N_tirages)]

        sentences_tmp += passages
        index_tmp += index_passages
        novels_tmp += [doc_name for _ in range(N_tirages)]
    
    print(len(sentences_tmp), len(index_tmp))
    print("GET PASSAGES OK")

    embeddings_raw = model.encode(sentences_tmp)
    embeddings = [elem.tolist() for elem in embeddings_raw]
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    df_tmp = pd.DataFrame({'novel_name': novels_tmp, 'index_name': index_tmp, 'embeddings': scaled_embeddings.tolist()})
    df_tmp['embeddings'] = df_tmp['embeddings'].apply(np.array)
    
    num_dimensions = len(embeddings[0])
    embedding_columns_tmp = pd.DataFrame(df_tmp['embeddings'].tolist(), columns=[f'dim_{i}' for i in range(num_dimensions)])
    df_tmp['embeddings'] = embedding_columns_tmp.values.tolist()
    df_tmp['embeddings'] = df_tmp['embeddings'].apply(np.array)
    
    mean_embeddings = df_tmp.groupby('novel_name')['embeddings'].mean()
    unique_novel_names = list(df_tmp.groupby('novel_name').groups.keys())
    mean_embeddings_list = mean_embeddings.tolist()    
    df_mean_embeddings = pd.DataFrame({'novel_name': unique_novel_names, 'mean_embeddings': mean_embeddings_list})

    num_dimensions = len(mean_embeddings_list[0])
    embedding_columns = pd.DataFrame(df_mean_embeddings['mean_embeddings'].tolist(), columns=[f'dim_{i}' for i in range(num_dimensions)])
    df_mean_embeddings_final = pd.concat([df_mean_embeddings['novel_name'], embedding_columns], axis=1)

    return df_mean_embeddings_final, novels_tmp

def clean_text(txt):
    txt_res = normalize("NFKD", txt).replace('\xa0', ' ')
    txt_res = txt_res.replace("\\", "").replace('\\xa0', '')
    return txt_res

def calculate_cosine_distance(df):
    embedding_vectors = df.values
    cosine_similarity_matrix = cosine_similarity(embedding_vectors)
    cosine_distance_matrix = cosine_similarity_matrix
    cosine_distance_df = pd.DataFrame(cosine_distance_matrix, index=df.index, columns=df.index)
    return cosine_distance_df

def sanity_check(df, topN=5):
    good_accuracy, len_test = 0, 0

    for novel_name in df.index:
        most_similar = get_most_similar(df, novel_name, topN)
        most_similar_racine = ['_'.join(elem.split('_')[:-1]) if len(elem.split('_'))>4 else elem for elem in most_similar]
        roman_racine_courant = '_'.join(novel_name.split('_')[:-1]) if len(novel_name.split('_'))>4 else novel_name
            
        for novel_racine_similar in most_similar_racine:
            len_test += 1
            if novel_racine_similar == roman_racine_courant:
                good_accuracy += 1
                #print(novel_name)
    print(good_accuracy, len_test)
    print(good_accuracy/len_test)

    return good_accuracy/len_test 

def get_most_similar(df, novel_name, top_n=5):
    # Get the row corresponding to the novel_name
    row = df.loc[novel_name]
    # Sort the row in descending order and get the top_n indices
    most_similar = row.sort_values(ascending=False)[:top_n].index.tolist()
    return most_similar

# List of tirages to process
TIRAGES = [100, 50, 20, 10, 5, 4, 3, 2, 1]
TIRAGES_reversed = TIRAGES[::-1]
results_list = []

# Process each tirage
for tirage in TIRAGES_reversed:
    print(f"Processing N_tirages={tirage}...")
    
    # Get embeddings and novel names for the current tirage
    embeddings, novels_tmp = get_embeddings("/data/jbarre/selected_gallica_1000/*.txt", N_tirages=tirage)
    
    # Save novels_tmp for the current tirage
    with open(f"novels_tmp_tirage_{tirage}.pkl", "wb") as f:
        pickle.dump(novels_tmp, f)
    
    # Set the index and calculate cosine distance
    embeddings.set_index(["novel_name"], inplace=True)
    cosine_distance = calculate_cosine_distance(embeddings)
    
    # Run sanity check and store the result
    res = sanity_check(cosine_distance, 5)
    results_list.append(res)
    print(results_list)
