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

# %%
import pandas as pd
import numpy as np

# %%
target_problems = ['mort_hosp', 'mort_icu', 'los_3', 'los_7']

unit_size = 256
layers = "GRU"
embedding_type = 'word2vec'


# %%
GRU = {
    'mort_hosp':{'auc': 0.8504, 'auprc': 0.5215, 'F1': 0.4229},
    'mort_icu': {'auc': 0.8632, 'auprc': 0.4651, 'F1': 0.3630},
    'los_3':    {'auc': 0.6740, 'auprc': 0.6017, 'F1': 0.5336},
    'los_7':    {'auc': 0.7054, 'auprc': 0.1625, 'F1': 0.0233},
}

word2vec_avg = {
    'mort_hosp':{'auc': 0.8642, 'auprc': 0.5422, 'F1': 0.4542},
    'mort_icu': {'auc': 0.8717, 'auprc': 0.4847, 'F1': 0.4230},
    'los_3':    {'auc': 0.6863, 'auprc': 0.6181, 'F1': 0.5419},
    'los_7':    {'auc': 0.7159, 'auprc': 0.1791, 'F1': 0.0135},
}

word2vec_proposed = {
    'mort_hosp':{'auc': 0.8755, 'auprc': 0.5587, 'F1': 0.4723},
    'mort_icu': {'auc': 0.8835, 'auprc': 0.4923, 'F1': 0.4302},
    'los_3':    {'auc': 0.6954, 'auprc': 0.6268, 'F1': 0.5504},
    'los_7':    {'auc': 0.7255, 'auprc': 0.1878, 'F1': 0.0158},
}


# %%
def print_results(model, target_problem, auc, auprc, F1):
    
    print("Target Problem: {}".format(target_problem))
    print("----------------------------")
    print("AUROC: {:.4f} {} {:.3f}   (Paper: {:.4f}, Abs Diff: {:.4f} ({:.1f} %))".format(
        np.mean(auc),   
        u"\u00B1", 
        np.std(auc),
        model[target_problem]['auc'],
        np.abs(np.round(np.mean(auc),4) - model[target_problem]['auc']),
        np.abs(np.round(np.mean(auc),4) - model[target_problem]['auc']) / model[target_problem]['auc'] * 100
    ))
    
    print("AUPRC: {:.4f} {} {:.3f}   (Paper: {:.4f}, Abs Diff: {:.4f} ({:.1f} %))".format(
        np.mean(auprc), 
        u"\u00B1", 
        np.std(auprc),
        model[target_problem]['auprc'],
        np.abs(np.round(np.mean(auprc),4) - model[target_problem]['auprc']),
        np.abs(np.round(np.mean(auprc),4) - model[target_problem]['auprc']) / model[target_problem]['auprc'] * 100
    ))
    
    print("F1:    {:.4f} {} {:.3f}   (Paper: {:.4f}, Abs Diff: {:.4f} ({:.1f} %))".format(
        np.mean(F1),    
        u"\u00B1", 
        np.std(F1),
        model[target_problem]['F1'],
        np.abs(np.round(np.mean(F1),4) - model[target_problem]['F1']),
        np.abs(np.round(np.mean(F1),4) - model[target_problem]['F1']) / model[target_problem]['F1'] * 100
    ))
    print("")


# %% [markdown]
# ## GRU

# %%
for target_problem in target_problems:
    auc   = []
    auprc = []
    acc   = []
    F1    = []
    
    for run in range(1,11):
        data = pd.read_pickle("results/GRU/{}-{}-{}-{}-new.p".format(
            unit_size, 
            layers, 
            target_problem, 
            run))

        auc.append(data['auc'])
        auprc.append(data['auprc'])
        acc.append(data['acc'])
        F1.append(data['F1'])
        
    print_results(GRU, target_problem, auc, auprc, F1)

# %% [markdown]
# ## Word2Vec multimodal

# %%
for target_problem in target_problems:
    auc   = []
    auprc = []
    acc   = []
    F1    = []
    
    for run in range(1,11):
        data = pd.read_pickle("results/word2vec_avg/{}-{}-{}-{}-{}-new-avg-.p".format(
            layers, 
            unit_size, 
            embedding_type, 
            target_problem, 
            run))

        auc.append(data['auc'])
        auprc.append(data['auprc'])
        acc.append(data['acc'])
        F1.append(data['F1'])
        
    print_results(word2vec_avg, target_problem, auc, auprc, F1)

# %% [markdown]
# ## Proposed Model: word2vec

# %%
for target_problem in target_problems:
    auc   = []
    auprc = []
    acc   = []
    F1    = []    
    
    for run in range(1,11):
        data = pd.read_pickle("results/cnn/{}-{}-{}-{}-{}-new-cnn-.p".format(
            layers, 
            unit_size, 
            embedding_type, 
            target_problem, 
            run))
        
        auc.append(data['auc'])
        auprc.append(data['auprc'])
        acc.append(data['acc'])
        F1.append(data['F1'])
        
    print_results(word2vec_proposed, target_problem, auc, auprc, F1)

# %%
