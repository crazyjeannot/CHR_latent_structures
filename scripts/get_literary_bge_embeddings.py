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
from os import path
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

list_of_tokens = ['/data/jbarre/lemmatization/TOKENS_GALLICA_1000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_2000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_3000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_4000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_5000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_6000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_7000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_8000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_9000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_10000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_11000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_12000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_13000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_14000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_GALLICA_15000.pkl',
                 '/data/jbarre/lemmatization/TOKENS_G.pkl']
                 
list_of_index = ['/data/jbarre/lemmatization/INDEX_GALLICA_1000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_2000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_3000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_4000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_5000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_6000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_7000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_8000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_9000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_10000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_11000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_12000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_13000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_14000.pkl',
                 '/data/jbarre/lemmatization/INDEX_GALLICA_15000.pkl',
                 '/data/jbarre/lemmatization/INDEX_G.pkl']

def get_embeddings(list_index, list_lemmas, df_BAD, N_tirages=1000, M_sentences_per_passages=20):
	conteur = 0

	for index_path, lemmas_path in tqdm(zip(list_index, list_lemmas), total=len(list_index)):
		conteur+=1
		index_tmp, sentences_tmp, novels_tmp = [], [], []

		index = joblib.load(index_path)
		lemmas = joblib.load(lemmas_path)
        
		assert len(index) == len(lemmas)

		for index_courant, lemmas_courant_raw in tqdm(zip(index, lemmas), total=len(index)):
           	# enleve les romans < 95% WER
			if index_courant in list(df_BAD_ocr["index_name"]):
				continue
            	# enleve les sentences vides ou inferieures à 20 char et enleve les 20 premières phrases et les 20 dernières
			lemmas_courant = [sentence for sentence in lemmas_courant_raw[20:-20] if len(sentence) >= 20] 
            	# passe les romans avec moins de 1000 phrases ()
			if len(lemmas_courant)<1000:
				continue
        
			passages_indices = [random.randint(0, len(lemmas_courant) - (M_sentences_per_passages+1)) for _ in range(N_tirages)]
            		# passages is list of N str (and str is joined M sentences)
			passages = [" ".join(lemmas_courant[indice:indice+M_sentences_per_passages]) for indice in passages_indices]
			index_passages = [index_courant+"_passage-"+str(j)+"_sentence-"+str(passages_indices[j]) for j in range(N_tirages)]
			index_romans = [index_courant for j in range(N_tirages)]

			sentences_tmp += passages
			index_tmp += index_passages
			novels_tmp += index_romans
            
		print(len(sentences_tmp), len(index_tmp))
		print("GET PASSAGES OK")
		embeddings_raw = model.encode(sentences_tmp, # list of str
                            batch_size=12, 
                            max_length=512,
                            )


		num_parts = 10
		part_size = len(embeddings_raw) // num_parts

		for I in range(num_parts):
			start = I * part_size
			end = (I + 1) * part_size if I < num_parts - 1 else None
			
			embeddings = embeddings_raw[start:end]
			embeddings = [elem.tolist() for elem in embeddings]
			scaler = StandardScaler()
			scaled_embeddings = scaler.fit_transform(embeddings)
			df_tmp = pd.DataFrame({'novel_name': novels_tmp[start:end], 'index_name': index_tmp[start:end], 'embeddings': scaled_embeddings.tolist()})        
			df_tmp['embeddings'] = df_tmp['embeddings'].apply(np.array)
			
			mean_embeddings = df_tmp.groupby('novel_name')['embeddings'].mean()
			unique_novel_names = list(df_tmp.groupby('novel_name').groups.keys())
    
			mean_embeddings_list = mean_embeddings.tolist()
			df_mean_embeddings = pd.DataFrame({'novel_name': unique_novel_names, 'mean_embeddings': mean_embeddings_list})


			num_dimensions = len(mean_embeddings_list[0])
			embedding_columns = pd.DataFrame(df_mean_embeddings['mean_embeddings'].tolist(), columns=[f'dim_{i}' for i in range(num_dimensions)])
			df_mean_embeddings_final = pd.concat([df_mean_embeddings['novel_name'], embedding_columns], axis=1)
			df_mean_embeddings_final.to_csv(f'EMBEDDINGS_BGE-BASE_INTERTEXTUALITY_1000_'+str(conteur)+'_PART_'+str(I)+'.csv', index=True, header=True)
	return df_means_embeddings_final





if __name__ == '__main__':

	model = FlagModel('crazyjeannot/fr_literary_bge_base',
		query_instruction_for_retrieval="",
		use_fp16=True) 
	df_BAD_ocr = pd.read_csv('../WER_GALLICA/df_BAD.csv') # sub 95% word error rate


	df_FINAL = get_embeddings(list_of_index, list_of_tokens, df_BAD_ocr)
