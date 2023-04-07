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
# - `en_core_med7_lg` was not available and so needed to pip install
#   - `!pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl`

# %%
# !pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl

# %%
import pandas as pd
import spacy

med7 = spacy.load("en_core_med7_lg")

# %%
preprocessed_df = pd.read_pickle("data/preprocessed_notes.p")

# %%
preprocessed_df['ner'] = None

# %%
count = 0
preprocessed_index = {}
for i in preprocessed_df.itertuples():
    
    if count % 1000 == 0:
        print(count)

    count += 1
    ind = i.Index
    text = i.preprocessed_text
    
    all_pred = []
    for each_sent in text:
        try:
            doc = med7(each_sent)
            result = ([(ent.text, ent.label_) for ent in doc.ents])
            if len(result) == 0: continue
            all_pred.append(result)
        except:
            print("error..")
            continue
    preprocessed_df.at[ind, 'ner'] = all_pred

# %%
pd.to_pickle(preprocessed_df, "data/ner_df.p")
