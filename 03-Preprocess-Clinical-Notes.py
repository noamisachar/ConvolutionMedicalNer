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
# - `punkt` was not available and so needed to add code to download
#   - `import nltk`
#   - `nltk.download('punkt')`

# %%
import pandas as pd
import os
import numpy as np
import re
import preprocess

# %%
import nltk
nltk.download('punkt')

# %%
PREPROCESS = "data/"

# %%
clinical_notes = pd.read_pickle(os.path.join(PREPROCESS, "sub_notes.p"))
clinical_notes.shape

# %%
sub_notes = clinical_notes[clinical_notes.SUBJECT_ID.notnull()]
sub_notes = sub_notes[sub_notes.CHARTTIME.notnull()]
sub_notes = sub_notes[sub_notes.TEXT.notnull()]

# %%
sub_notes.shape

# %%
sub_notes = sub_notes[['SUBJECT_ID', 'HADM_ID_y', 'CHARTTIME', 'TEXT']]

# %%
sub_notes['preprocessed_text'] = None

# %%
for each_note in sub_notes.itertuples():
    text = each_note.TEXT
    sub_notes.at[each_note.Index, 'preprocessed_text'] = preprocess.getSentences(text)

# %% [markdown]
# ### Save notes

# %%
pd.to_pickle(sub_notes, os.path.join(PREPROCESS, "preprocessed_notes.p"))

# %% [markdown]
# ### Additional preprocessing

# %%
# sub_notes = pd.read_pickle(os.path.join(PREPROCESS, "preprocessed_notes.p"))

# def preprocess1(x):
#     y=re.sub('\\[(.*?)\\]','',x) #remove de-identified brackets
#     y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
#     y=re.sub('dr\.','doctor',y)
#     y=re.sub('m\.d\.','md',y)
#     y=re.sub('admission date:','',y)
#     y=re.sub('discharge date:','',y)
#     y=re.sub('--|__|==','',y)
#     return y

# def preprocessing(df_less_n): 
#     df_less_n['preprocessed_text_v2']=df_less_n['preprocessed_text'].fillna(' ')
#     df_less_n['preprocessed_text_v2']=df_less_n['preprocessed_text_v2'].str.replace('\n',' ')
#     #df_less_n['preprocessed_text_v2']=df_less_n['preprocessed_text_v2'].str.replace('\r',' ')
#     #df_less_n['preprocessed_text_v2']=df_less_n['preprocessed_text_v2'].apply(str.strip)
#     #df_less_n['preprocessed_text_v2']=df_less_n['preprocessed_text_v2'].str.lower()

#     df_less_n['preprocessed_text_v2']=df_less_n['preprocessed_text_v2'].apply(lambda x: preprocess1(x))
    
# sub_notes = preprocessing(sub_notes)
