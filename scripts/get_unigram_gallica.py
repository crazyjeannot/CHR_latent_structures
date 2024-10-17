import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import pickle
import random
from collections import Counter
import json
from glob import glob
from unicodedata import normalize
from nltk.tokenize import word_tokenize
from os import path
import re


list_of_lemmas = ['/data/jbarre/lemmatization/LEMMAS_GALLICA_1000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_2000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_3000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_4000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_5000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_6000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_7000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_8000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_9000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_10000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_11000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_12000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_13000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_14000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_GALLICA_15000.pkl',
                 '/data/jbarre/lemmatization/LEMMAS_G.pkl']
                 
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


list_lemma_select = ['le',
 'de',
 'un',
 'et',
 'il',
 'avoir',
 'à',
 'lui',
 'être',
 'son',
 'que',
 'l’',
 'ce',
 'je',
 'en',
 'd’',
 'qui',
 'pas',
 'ne',
 'vous',
 'se',
 'dans',
 'qu’',
 'tout',
 'pour',
 'faire',
 'dire',
 's’',
 'mon',
 'au',
 'éter',
 'n’',
 'sur',
 'plus',
 '-',
 'mais',
 'on',
 '–',
 'avec',
 'me',
 'par',
 'comme',
 'c’',
 'nous',
 'pouvoir',
 'si',
 'bien',
 'j’',
 'y',
 'voir',
 'aller',
 'même',
 'moi',
 'tu',
 'leur',
 'sans',
 'être',
 'm’',
 'vouloir',
 'savoir',
 'où',
 'venir',
 'homme',
 'autre',
 'ou',
 'à',
 'petit',
 'quelque',
 'prendre',
 'grand',
 'encore',
 'votre',
 'femme',
 'rien',
 'quand',
 'là',
 'main',
 'peu',
 'jour',
 'celui',
 'dont',
 'bon',
 'mettre',
 'aussi',
 'jeune',
 'heure',
 'cela',
 'devoir',
 'falloir',
 'été',
 'non',
 'croire',
 'temps',
 'oeil',
 'puis',
 'chose',
 'donc',
 'sous',
 'ça',
 'jamais',
 'Monsieur',
 'fois',
 'toujours',
 'passer',
 'notre',
 'après',
 'seul',
 'ton',
 'tête',
 'alors',
 'porte',
 'entendre',
 'étaier',
 'fille',
 'vers',
 'coup',
 'trouver',
 '-t',
 'point',
 'monsieur',
 'enfant',
 '-ce',
 'demander',
 'donner',
 'vie',
 'parler',
 'te',
 'devant',
 'oui',
 'car',
 'vieux',
 'tenir',
 'moment',
 'père',
 'trop',
 'attendre',
 "n'",
 'chez',
 'entre',
 'moins',
 'voix',
 'était',
 'mère',
 'm.',
 'laisser',
 'monde',
 'depuis',
 'quel',
 'nouveau',
 'air',
 'regarder',
 'ni',
 'maison',
 'mort',
 'reprendre',
 'nuit',
 'ainsi',
 'sortir',
 'mot',
 'ami',
 'cœur',
 'ici',
 'déjà',
 'avant',
 'lequel',
 'sentir',
 'jusqu’',
 'rendre',
 'fort',
 'pauvre',
 'répondit',
 'contre',
 'chambre',
 'an',
 '—',
 'long',
 'comprendre',
 'bras',
 'dieu',
 'personne',
 'eût',
 'perdre',
 'ouvrir',
 'devenir',
 'regard',
 'nom',
 'pied',
 'assez',
 'revenir',
 'pendant',
 'suivre',
 'enfin',
 'maintenant',
 'tant',
 'bel',
 'toi',
 'soir',
 'comment',
 'pourquoi',
 'connaître',
 'quoi',
 'penser',
 'voilà',
 'côté',
 'aimer',
 'amour',
 'très',
 'sembler',
 'premier',
 'mieux',
 'noir',
 'rire',
 'parce',
 'rue',
 'mal',
 'chercher',
 't’',
 'rester',
 'sourire',
 'aucun',
 'visage',
 'arriver',
 'terre',
 'doute',
 'partir',
 'trè',
 'place',
 'peine',
 'appeler',
 'vrai',
 'instant',
 'roi',
 'blanc',
 'presque',
 'gens',
 'entrer',
 'seulement',
 'haut',
 'eh',
 'mourir',
 'jeter',
 'porter',
 'beaucoup',
 'cependant',
 'écrier',
 'matin',
 'oh',
 'fils',
 'bas',
 'maître',
 'plein',
 'parole',
 'comte',
 'paraître',
 'pari',
 'finir',
 'raison',
 'asseoir',
 'chaque',
 'loin',
 'beau',
 'fait',
 'eau',
 'fond',
 'force',
 'bout',
 'autour',
 'cheval',
 'reste',
 'tour',
 'lettre',
 'lever',
 'tomber',
 'idée',
 "qu'",
 'lorsque',
 'mme',
 'longtemps',
 'or',
 'souvenir',
 'corps',
 'derrière',
 'affaire',
 'servir',
 'esprit',
 "l'",
 'table',
 'tel',
 'dernier',
 'continuer',
 'lieu',
 'gros',
 'milieu',
 'certain',
 'frère',
 'pourtant',
 'ailleurs',
 'sorte',
 'descendre',
 'aime',
 'près',
 'tendre',
 'silence',
 'besoin',
 'effet',
 'fin',
 'bruit',
 'dame',
 'lire',
 'part',
 'feu',
 'vivre',
 'heureux',
 'ciel',
 'minute',
 'lit',
 'cher',
 'apprendre',
 'âme',
 'pensée',
 'peur',
 'face',
 'abord',
 'bois',
 'droit',
 'montrer',
 'question',
 'route',
 'fenêtre',
 'cri',
 'chemin',
 'voiture',
 'ville',
 'année',
 'celui-ci',
 'mari',
 'histoire',
 'mois',
 'fût',
 'marquis',
 'premièr',
 'tard',
 'courir',
 'sang',
 'rouge',
 'livre',
 'murmurer',
 'argent',
 'conduire',
 'œil',
 'ordre',
 'pierre',
 'soleil',
 'famille',
 'passé',
 'épaule',
 'permettre',
 'malgré',
 'monter',
 'froid',
 'mauvais',
 'général',
 'suite',
 'bientôt',
 'souvent',
 'demande',
 'salle',
 'joie',
 'fou',
 'vite',
 'remettre',
 "c\\'est",
 'quitter',
 'pièce',
 'saint',
 'tandis',
 'lèvre',
 'grâce',
 'hôtel',
 'cheveu',
 'garde',
 'mouvement',
 'songer',
 'marcher',
 'geste',
 'êtes',
 'pays',
 'autant',
 'façon',
 'ajouter',
 'surtout',
 'tirer',
 'plaisir',
 'sûr',
 'tourner',
 'capitaine',
 'oncle',
 'crier',
 'aussitôt',
 'reconnaître',
 'second',
 'essayer',
 'joli',
 'cause',
 'front',
 'rappeler',
 'possible',
 'travers',
 'bonheur',
 'doux',
 'vue',
 'état',
 'compte',
 'lumière',
 'pousser',
 'plusieurs',
 'dormir',
 'mur',
 'travail',
 'garçon',
 'puisque',
 'chacun',
 'ombre',
 'voici',
 'agir',
 'larme',
 'plutôt',
 'oublier',
 'donné',
 'juste',
 'prince',
 'jouer',
 'jean',
 'figure',
 'bleu',
 'oreille',
 'bouche',
 'battre',
 'offrir',
 'mer',
 'honneur',
 'garder',
 'lorsqu’',
 'ancien',
 'chef',
 'pareil',
 'retourner',
 'bord',
 'doigt',
 'manger',
 'partie',
 'dernière',
 'rose',
 'cour',
 'sœur',
 '-vou',
 'coin',
 'demain',
 'docteur',
 'salon',
 'papier',
 'saisir',
 'souffrir',
 'profond',
 'êter',
 'prè',
 'terrible',
 'retrouver',
 'franc',
 'fleur',
 'aujourd’hui',
 'moyen',
 "qu\\'il",
 'signe',
 'dè',
 'arbre',
 'marier',
 'forme',
 'propre',
 'château',
 'vraiment',
 'france',
 'secret',
 'cas',
 'jardin',
 'craindre',
 'sir',
 'simple',
 'anglais',
 'vif',
 'calme',
 'sentiment',
 'parmi',
 'dû',
 'valoir',
 'boire',
 'jeanne',
 'trouve',
 'douleur',
 'triste',
 'sens',
 'sombre',
 'malheureux',
 'marche',
 'reine',
 'pâle',
 'donne',
 'service',
 'large',
 'rentrer',
 'lendemain',
 'trouvé',
 'clair',
 'frapper',
 'ensuite',
 'manière',
 'arrêter',
 'parle',
 'gauche',
 'au-dessus',
 'deux',
 'pleurer',
 'chien',
 'semble',
 'vent',
 'prier',
 'train',
 'pense',
 'poursuivre',
 'nature',
 'jambe',
 'droite',
 'taire',
 'âge',
 'arme',
 'revoir',
 'vérité',
 'impossible',
 'malheur',
 'promettre',
 'parfois',
 'journée',
 'approcher',
 'retenir',
 'escalier',
 'colère',
 'raconter',
 'brave',
 'expliquer',
 'attention',
 'joue',
 'soldat',
 'humain',
 'riche',
 'tuer',
 'verre',
 'ennemi',
 'guerre',
 'nez',
 'malade',
 'disparaître',
 'fer',
 'robe',
 'occuper',
 'court',
 'regarde',
 'émotion',
 'endroit',
 'quelqu’',
 'payer',
 'compter',
 'présence',
 'quant',
 'cacher',
 'habitude',
 'suffire',
 'don',
 'écouter',
 'genou',
 'baron',
 'chère',
 'voyage',
 'parvenir',
 'jusque',
 'fortune',
 'poser',
 'envoyer',
 'ensemble',
 'debout',
 'contraire',
 'poche',
 'rocambole',
 'surprendre',
 'rêve',
 'là.',
 'soin',
 'moindre',
 'effort',
 'aprè',
 'vide',
 'causer',
 'hasard',
 'toucher',
 'voisin',
 'duc',
 'bureau',
 'dos',
 'dehors',
 'recevoir',
 'désir',
 'laisse',
 'couleur',
 'appartement',
 'retour',
 'répondre',
 'semaine',
 'compagnon',
 'poitrine',
 'foi',
 'envie',
 'remplir',
 'arrivé',
 'midi',
 'dent',
 'manquer',
 'soudain',
 'laissé',
 'afin',
 'scène',
 'mariage',
 'gagner',
 'foule',
 'courage',
 'peau',
 'ligne',
 'vert',
 'amie',
 'rencontrer',
 'inconnu',
 'coucher',
 'meilleur',
 'libre',
 'diable',
 'changer',
 'maman',
 'aimé',
 'entrée',
 'arrive',
 'exemple',
 'seigneur',
 'oser',
 'sujet',
 'jacques',
 'choisir',
 'surprise',
 'passion',
 'intérieur',
 'cousin',
 'couvrir',
 'françai',
 'partout',
 'trait',
 'commencer',
 'existence',
 'atteindre',
 'étrange',
 'police',
 'glisser',
 'paraît',
 'tranquille',
 'vague',
 'soit',
 'vicomte',
 'objet',
 'ferme',
 'serrer',
 'médecin',
 'conversation',
 'officier',
 'fauteuil',
 'chapitre',
 'combien',
 'veille',
 'chapeau',
 'demeurer',
 'lentement',
 'bête',
 'doucement',
 'est',
 'grave',
 'charmant',
 'village',
 'visite',
 'faux',
 'jeu',
 'beauté',
 'apercevoir',
 'arrière',
 'maigret',
 'retirer',
 'première',
 'ouvert',
 'traverser',
 '*',
 'mesure',
 'travailler',
 'nul',
 'vivant',
 'mlle',
 'adresser',
 'guère',
 'relever',
 'passage',
 'cent',
 'fermer',
 'rejoindre',
 'chaise',
 'lourd',
 'chevalier',
 'goût',
 'vin',
 'courant',
 'brusquement',
 'interrompre',
 'champ',
 'conseil',
 'arrêta',
 "qu\\'elle",
 'noble',
 'sol',
 'intérêt',
 'crime',
 'confiance',
 'dessus',
 'palais',
 'situation',
 'immense',
 'prix',
 'baiser',
 'réponse',
 'danger',
 'gris',
 'pur',
 'parent',
 'autrefois',
 'sauver',
 'feuille',
 'répéter',
 'monseigneur',
 'occasion',
 'cou',
 'campagne',
 'éclat',
 'maîtresse',
 'difficile',
 'apporter',
 'départ',
 'appartenir',
 'écrit',
 'domestique',
 'épée',
 'hors',
 'faute',
 'diriger',
 'vieillard',
 'siècle',
 'cours',
 'hier',
 'simplement',
 'espèce',
 'personnage',
 'image',
 'refuser',
 'inutile',
 'arrivée',
 'moitié',
 'langue',
 'importe',
 'louis',
 'sauter',
 'auprès',
 'davantage',
 'expression',
 'juge',
 'demi',
 'peuple',
 'caractère',
 'comtesse',
 'adieu',
 'empêcher',
 'deviner',
 'douter',
 'immobile',
 'produire',
 'merci',
 'selon',
 'préser',
 'conscience',
 'époque',
 'naturel',
 'rouler',
 'robert',
 'embrasser',
 'échapper',
 'public',
 'détail',
 '�',
 'rencontre',
 'accepter',
 'compagnie',
 'soutenir',
 'belle-rose',
 'forêt',
 'nu',
 'journal',
 'trembler',
 'parti',
 'but',
 'vendre',
 'somme',
 'sommeil',
 'étranger',
 'aider',
 'oiseau',
 'espoir',
 'propos',
 'dîner',
 'volonté',
 'ceci',
 'mademoisell',
 'facile',
 'impression',
 'honnête',
 'léon',
 'fête',
 'contenir',
 'preuve',
 'fuir',
 'tombe',
 'juger',
 'type',
 'commissaire',
 'connaître',
 'genre',
 'apparaître',
 'animal',
 'amoureux',
 'camarade',
 'reçu',
 'dur',
 'phrase',
 'baccarat',
 'neuf',
 'avenir',
 'sien',
 'tellement',
 'cuisine',
 'œuvre',
 'sauvage',
 'taille',
 'rapide',
 'digne',
 'à-dire',
 'tromper',
 'aide',
 'faible',
 'curieux',
 'chaud',
 'amant',
 'capable',
 'cabinet',
 'montagne',
 'certainement',
 'commençer',
 'quelquefois',
 'armée',
 'événement',
 'toilette',
 'jeunesse',
 'aperçut',
 'ressembler',
 'beal',
 'promener',
 'luire',
 'valet',
 'obtenir',
 'parfait',
 'tort',
 'demandé',
 'côte',
 'billet',
 'projet',
 'jaune',
 'justice',
 'appuyer',
 'ah',
 'pencher',
 'semblable',
 'pluie',
 'présenter',
 'interroger',
 'quartier',
 'léger',
 'certes',
 'caroline',
 'écrire',
 'georges',
 'ignorer',
 'aventure',
 'parfaitement',
 'plan',
 'secouer',
 'vaste',
 'professeur',
 'quart',
 'remarquer',
 'mémoire',
 'indien',
 'déjeuner',
 'charles',
 'sac',
 'véritabl',
 'chanter',
 'baisser',
 'soirée',
 'environ',
 'philippe',
 'paul',
 'pardon',
 'tôt',
 'église',
 'désespoir',
 'rôle',
 'poste',
 '-vous',
 'vivement',
 'remercier',
 'former',
 'odeur',
 'pauline',
 'parlé',
 'pointe',
 'voyageur',
 'soi',
 'arracher',
 'boîte',
 'spectacle',
 'espérer',
 'nombre',
 'connaissance',
 'marchand',
 'pain',
 'emporter',
 'remonter',
 'imaginer',
 'avis',
 'éloigner',
 'nuage',
 'achever',
 'prison',
 'herbe',
 'art',
 'chance',
 'abandonner',
 'rougir',
 'drôle',
 "j\\'ai",
 'tué',
 'rare',
 'troupe',
 'distance',
 'innocent',
 'condition',
 'récit',
 'rapport',
 'plaire',
 'pitié',
 'mine',
 'convenir',
 'carte',
 'page',
 'sonner',
 'étier',
 'avouer',
 'endormir',
 'ministre',
 'ventre',
 'assurer',
 'prisonnier',
 'branche',
 'habit',
 'arrêté',
 'chair',
 'lueur',
 'glace',
 'colonel',
 'île',
 'énorme',
 'pont',
 'paysan']

def dict_freq_token(list_select, list_lemma):
    
    dict_result = dict.fromkeys(list_select)
    
    dict_temp = Counter(list_lemma)
        
    for key in dict_temp.keys():
        if key in dict_result.keys():
            dict_result[key] = dict_temp[key]/len(list_lemma)
    
    return dict_result


def get_counter_df(list_index, list_lemmas, list_lemma_select):
    dict_results = {}
    conteur = 0
    df_main = pd.DataFrame()

    for index_path, lemmas_path in tqdm(zip(list_index, list_lemmas), total=len(list_index)):
        conteur+=1
        index_tmp, sentences_tmp, novels_tmp = [], [], []

        index = joblib.load(index_path)
        lemmas = joblib.load(lemmas_path)
        
        assert len(index) == len(lemmas)
        
        for index_courant, lemmas_courant_raw in tqdm(zip(index, lemmas), total=len(index)):
            list_token_courant = []
            # enleve les sentences vides ou inferieures à 20 char et enleve les 20 premières phrases et les 20 dernières
            lemmas_courant = [sentence for sentence in lemmas_courant_raw[20:-20] if len(sentence) >= 20] 
            # passe les romans avec moins de 1000 phrases ()
            if len(lemmas_courant)<1000:
                continue

            for sentence in lemmas_courant:
                list_token_courant.extend(sentence.split())
                
            dict_results.update(dict_freq_token(list_lemma_select, list_token_courant))
            dict_results["index"] = index_courant
            df_temp = pd.DataFrame(dict_results, index=[0])

            df_main = pd.concat([df_main, df_temp], ignore_index=True) 
        df_main.to_csv('UNIGRAM_GALLICA'+str(conteur)+'.csv', index=True, header=True)
    df_main.set_index("index", inplace = True)
    return df_main

if __name__ == '__main__':

	df_main = get_counter_df(list_of_index, list_of_lemmas, list_lemma_select)
	df_main.to_csv('UNIGRAM_GALLICA_MAIN.csv', index=True, header=True)