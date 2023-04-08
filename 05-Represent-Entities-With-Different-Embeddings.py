# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Changes made by kts4 / noami2
# - `glove` was not available and so needed to install `mittens` instead
#   - `from mittens import GloVe as glove`
# - We do not have the `FastText` model and so all references to it and combined model have to be commented out
# - The calls to access the `Word2vec` and `FastText` models needed to be updated
#   - `w2vec[...]` changed to `w2vec.wv[...]` and `fasttext[...]` changed to `fasttext.wv[...]`
# - In Python 3 you can no longer convert from `map` to `np.array`
#   - So, `t = np.asarray(map(mean, zip(*avg)))` changed to `t = np.asarray(list(map(mean, zip(*avg))))`
# - Accessing `FastText` elements raised an `IndexError`
#   - These calls are now wraped in a `try` block
# - `new_word2vec_dict` was never initialised
#   - Should be a copy of `new_word2vec` with commonalities between it and `FastText` model removed
#   - Need to ensure the `diff` set is actually useful
# - Changes to newlines / spacings etc.

# %% tags=[]
import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec, FastText
from mittens import GloVe as glove
# import glove
# from glove import Corpus
import torch

import collections
import gc 

import warnings
warnings.filterwarnings('ignore')

# %% tags=[]
new_notes = pd.read_pickle("data/ner_df.p") # med7
w2vec = Word2Vec.load("embeddings/word2vec.model")
fasttext = FastText.load("embeddings/fasttext.model")

# %% tags=[]
from transformers import AutoTokenizer, AutoModel

bluebert_tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
bluebert_model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")

clinicalbert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinicalbert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# %% tags=[]
null_index_list = []
for i in new_notes.itertuples():
    
    if len(i.ner) == 0:
        null_index_list.append(i.Index)
new_notes.drop(null_index_list, inplace=True)

# %% tags=[]
med7_ner_data = {}

for ii in new_notes.itertuples():
    
    p_id = ii.SUBJECT_ID
    ind = ii.Index
    
    try:
        new_ner = new_notes.loc[ind].ner
    except:
        new_ner = []
            
    unique = set()
    new_temp = []
    
    for j in new_ner:
        for k in j:
            
            unique.add(k[0])
            new_temp.append(k)

    if p_id in med7_ner_data:
        for i in new_temp:
            med7_ner_data[p_id].append(i)
    else:
        med7_ner_data[p_id] = new_temp

# %% tags=[]
pd.to_pickle(med7_ner_data, "data/new_ner_word_dict.pkl")


# %% tags=[]
def mean(a):
    return sum(a) / len(a)


# %% tags=[]
data_types = [med7_ner_data]
data_names = ["new_ner"]

for data, names in zip(data_types, data_names):
   
     # print("w2vec starting..")

#     new_word2vec = {}
#     for k,v in data.items():

#         patient_temp = []
#         for i in v:
#             try:
#                 patient_temp.append(w2vec.wv[i[0]])
#             except:
#                 avg = []
#                 num = 0
#                 temp = []

#                 if len(i[0].split(" ")) > 1:
#                     for each_word in i[0].split(" "):
#                         try:
#                             temp = w2vec.wv[each_word]
#                             avg.append(temp)
#                             num += 1
#                         except:
#                             pass
#                     if num == 0: 
#                         continue
#                     avg = np.asarray(avg)
#                     t = np.asarray(list(map(mean, zip(*avg))))
#                     patient_temp.append(t)
#         if len(patient_temp) == 0: 
#             continue
#         new_word2vec[k] = patient_temp
        
#     print("w2vec finished")

#     ############################################################################
#     print("fasttext starting..")
        
#     new_fasttextvec = {}

#     for k,v in data.items():

#         patient_temp = []

#         for i in v:
#             try:
#                 patient_temp.append(fasttext.wv[i[0]])
#             except:
#                 pass
#         if len(patient_temp) == 0: continue
#         new_fasttextvec[k] = patient_temp

#     print("fasttext finished")
        
    #############################################################################   
    
    print("BlueBERT starting..")
    new_bluebert = {}

    for k,v in data.items():
        patient_temp = []
        for i in v:
            # try:
            input_ids = bluebert_tokenizer.encode(i[0], add_special_tokens=True)
            embeddings = bluebert_model(torch.tensor([input_ids]))[0][0][1:-1]
            patient_temp.append(embeddings)
            # except:
                # pass
        if len(patient_temp) == 0: 
            continue
        new_bluebert[k] = patient_temp
    
    print("BlueBERT finished")
    
    ############################################################################
    
    print("ClinicalBERT starting..")
    new_clinicalbert = {}

    for k,v in data.items():
        patient_temp = []
        for i in v:
            # try:
            input_ids = clinicalbert_tokenizer.encode(i[0], add_special_tokens=True)
            embeddings = clinicalbert_model(torch.tensor([input_ids]))[0][0][1:-1]
            patient_temp.append(embeddings)
            # except:
                # pass
        if len(patient_temp) == 0: 
            continue
        new_clinicalbert[k] = patient_temp

    print("ClinicalBERT finished")
    
    ############################################################################
        
#     print("combined starting..")
#     new_concatvec = {}

#     for k,v in data.items():
#         patient_temp = []
#     #     if k != 6: continue
#         for i in v:
#             w2vec_temp = []
#             try:
#                 w2vec_temp = w2vec.wv[i[0]]
#             except:
#                 avg = []
#                 num = 0
#                 temp = []

#                 if len(i[0].split(" ")) > 1:
#                     for each_word in i[0].split(" "):
#                         try:
#                             temp = w2vec.wv[each_word]
#                             avg.append(temp)
#                             num += 1
#                         except:
#                             pass
#                     if num == 0: 
#                         w2vec_temp = [0] * 100
#                     else:
#                         avg = np.asarray(avg)
#                         w2vec_temp = np.asarray(list(map(mean, zip(*avg))))
#                 else:
#                     w2vec_temp = [0] * 100
            
#             try:
#                 fasttemp = fasttext.wv[i[0]]
#                 appended = np.append(fasttemp, w2vec_temp, 0)
#             except:
#                 appended = np.append(w2vec_temp, 0)
            
#             # appended = np.append(fasttemp, w2vec_temp, 0)
#             patient_temp.append(appended)
#         if len(patient_temp) == 0: continue
#         new_concatvec[k] = patient_temp
    
#     print("combined finished")

#     print(len(new_word2vec), len(new_fasttextvec), len(new_concatvec), len(new_bluebert), len(new_clinicalbert))
#     pd.to_pickle(new_word2vec, "data/"+names+"_word2vec_dict.pkl")
#     pd.to_pickle(new_fasttextvec, "data/"+names+"_fasttext_dict.pkl")
#     pd.to_pickle(new_concatvec, "data/"+names+"_combined_dict.pkl")
#     pd.to_pickle(new_bluebert, "data/"+names+"_bluebert_dict.pkl")
#     pd.to_pickle(new_clinicalbert, "data/"+names+"_clinicalbert_dict.pkl")

# %%
print(len(new_bluebert), len(new_clinicalbert))

# %%
new_fasttext_dict = new_fasttextvec.copy()
new_word2vec_dict = new_word2vec.copy()
new_combined_dict = new_concatvec.copy()

diff = set(new_fasttext_dict.keys()).difference(set(new_word2vec_dict))
for i in diff:
    del new_fasttext_dict[i]
    del new_combined_dict[i]
print (len(new_word2vec_dict), len(new_fasttext_dict), len(new_combined_dict))


pd.to_pickle(new_word2vec_dict, "data/new_ner_word2vec_limited_dict.pkl")
pd.to_pickle(new_fasttext_dict, "data/"+"new_ner"+"_fasttext_limited_dict.pkl")
pd.to_pickle(new_combined_dict, "data/"+"new_ner"+"_combined_limited_dict.pkl")

# %% tags=[]
diff

# %%
