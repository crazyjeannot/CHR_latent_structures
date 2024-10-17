import numpy as np
import pandas as pd
import joblib
import pickle
from tqdm import tqdm
import random
import json

def propn_filter(pos_tag, lemma):
	if pos_tag == "PROPN":
		return "PROPN"
	return lemma

def get_filtered_dataset(pos_str, tokens_str):
	str_res = ""
	for pos, token in zip(pos_str.split(), tokens_str.split()):
		str_res += propn_filter(pos, token)+" "

	return str_res[:1100]# 1100 chars =< 512 tokens otherwise BGE get issues with the length of the input

def get_filtered_dataset_nested(pos_str_nested, tokens_str_nested):
	list_str_res = []
	for pos_paragraph, token_paragraph in zip(pos_str_nested, tokens_str_nested):
		list_str_res.append(get_filtered_dataset(pos_paragraph, token_paragraph))
	return list_str_res

def build_dataset(sentences, pos_tags, num_samples=400000, LENGHT_PARAPH=5, N_PARAPH=10):
	assert len(sentences) == len(pos_tags)
	assert len(sentences[10]) == len(pos_tags[10])

	dataset = []
	for _ in tqdm(range(num_samples)):
		book_idx = random.randint(0, len(sentences) - 1)
		if len(sentences[book_idx])<LENGHT_PARAPH*N_PARAPH:
			continue
		sentence_idx = random.randint(0, len(sentences[book_idx]) - LENGHT_PARAPH*N_PARAPH)

		query = ' '.join(sentences[book_idx][sentence_idx:sentence_idx+LENGHT_PARAPH])
		pos = [' '.join(sentences[book_idx][sentence_idx+i:sentence_idx+(i+LENGHT_PARAPH)]) for i in range(LENGHT_PARAPH, N_PARAPH*LENGHT_PARAPH, LENGHT_PARAPH)]
		
		neg_book_indices = [random.randint(0, len(sentences) - 1) for _ in range(N_PARAPH)]
		neg_sentences_indices = [random.randint(0, len(sentences[neg_book_indice])-LENGHT_PARAPH) for neg_book_indice in neg_book_indices]
		neg = [' '.join(sentences[book_idx][sentence_idx:sentence_idx+LENGHT_PARAPH]) for book_idx, sentence_idx in zip(neg_book_indices, neg_sentences_indices)]
		#neg = ''.join(str(e) for e in neg_sentences)

		pos_tags_query = ' '.join(pos_tags[book_idx][sentence_idx:sentence_idx+LENGHT_PARAPH])
		pos_tags_pos = [' '.join(pos_tags[book_idx][sentence_idx+i:sentence_idx+(i+LENGHT_PARAPH)]) for i in range(LENGHT_PARAPH, N_PARAPH*LENGHT_PARAPH, LENGHT_PARAPH)]
		pos_tags_neg = [' '.join(pos_tags[book_idx][sentence_idx:sentence_idx+LENGHT_PARAPH]) for book_idx, sentence_idx in zip(neg_book_indices, neg_sentences_indices)]
		#pos_tags_neg = ' '.join(pos_tags_neg_sentences)

		query_filtered = get_filtered_dataset(pos_tags_query, query)
		pos_filtered = get_filtered_dataset_nested(pos_tags_pos, pos)
		neg_filtered = get_filtered_dataset_nested(pos_tags_neg, neg)

		dataset.append({"query": query_filtered, "pos": pos_filtered, "neg": neg_filtered})
	return dataset


if __name__ == '__main__':

	print("LOAD CHAPITRES SENTENCES")
	#chapitres_lemmas_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_lemmas_stanza_sentences_chapitres.pkl')    
	chapitres_tokens_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_tokens_stanza_sentences_chapitres.pkl')
	chapitres_pos_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_pos_stanza_sentences_chapitres.pkl')
	#chapitres_index_sentences = joblib.load('/data/jbarre/lemmatization/main_list_index_stanza_sentences_chapitres.pkl')
	assert len(chapitres_tokens_sentences) == len(chapitres_pos_sentences)

	chapitres_tokens_sentences = [elem for elem in chapitres_tokens_sentences if len(elem) >= 50]
	chapitres_pos_sentences = [elem for elem in chapitres_pos_sentences if len(elem) >= 50]


	print("GET DATASET")
	data = build_dataset(chapitres_tokens_sentences, chapitres_pos_sentences)
	print(data[-1])	
	print("SAVE dataset")
	with open('PARAPH_DATASET_FT_400.jsonl', 'w', encoding='utf-8') as outfile:
		for entry in data:
		# Write each dictionary to the file as a JSON string, followed by a newline
			json.dump(entry, outfile, ensure_ascii=False)
			outfile.write('\n')
