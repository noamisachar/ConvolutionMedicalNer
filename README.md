# Implementation for Improving Clinical Outcome Predictions Using Convolution over Medical Entities with Multimodal Learning

Modified by Kevin Sweeney and Noam Isachar - students at University of Illinois, Urbana Champaign, Spring 2023, CS-498 Deep Learning for Healthcare.

[Paper link](https://arxiv.org/abs/2011.12349)
[Original Repository](https://github.com/tanlab/ConvolutionMedicalNer)

## Python Dependencies

Python library dependancies are defined in `environment.yml`. (uses Python version: `3.8.16`)

## Data & Pre-Trained Models

- [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/)
- [med7 pre-trained model](https://github.com/kormilitzin/med7)
- [Word2Vec](https://github.com/kexinhuang12345/clinicalBERT) and [FastText](https://drive.google.com/drive/folders/1bcR6ThMEPhguU9T4qPcPaZJ3GQzhLKlz?usp=sharing) pre-trained models

## Usage

### Step 1: Setup Required Code

1. Clone the code to local.

    ```shell
    git clone https://github.com/noamisachar/ConvolutionMedicalNer.git
    cd ConvolutionMedicalNer
    ```

2. Create and activate the conda environment from the available `YAML` file

    ```shell
    conda env create -f environment.yml
    conda activate DLH_project_py38
    ```

### Step 2: Get Required Data

1. Request access to [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) dataset.
   - Note: This may take several days

2. Run MIMIC-Extract Pipeline as explained in [MIMIC_Extract](https://github.com/MLforHealth/MIMIC_Extract) repo.
   - Note: The required data may be available from `gcp` as specified in the `Pre-processed Output` section of the repository.

3. Copy the output file of [MIMIC_Extract](https://github.com/MLforHealth/MIMIC_Extract) Pipeline, named `all_hourly_data.h5`, to `data` folder.

4. Also need these three data files from [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/). Place them in the same `data` folder.
     - `ADMISSIONS.csv`
     - `ICUSTAYS.csv`
     - `NOTEEVENTS.csv`

5. Download pretrained embeddings ([Word2Vec](https://github.com/kexinhuang12345/clinicalBERT) and [FastText](https://drive.google.com/drive/folders/1bcR6ThMEPhguU9T4qPcPaZJ3GQzhLKlz?usp=sharing)) into `embeddings` folder.

### Step 3: Run Code (Jupyter Notebooks)

1. Run `01-Extract-Timseries-Features.ipnyb`
   - to extract the first 24 hours of timeseries features from the [MIMIC_Extract](https://github.com/MLforHealth/MIMIC_Extract) raw data.

2. Run `02-Select-SubClinicalNotes.ipynb`
   - to select the subnotes from `NOTEEVENTS.csv` based on criteria.

3. Run `03-Preprocess-Clinical-Notes.ipynb`
   - to prepocess the clinical notes.

4. Run `04-Apply-med7-on-Clinical-Notes.ipynb`
   - to extract medical entities using the [med7 pre-trained model](https://github.com/kormilitzin/med7).

5. Run `05-Represent-Entities-With-Different-Embeddings.ipynb`
    - to convert medical entities into word representations using the available embedding techniques:
      - [Word2Vec](https://github.com/kexinhuang12345/clinicalBERT)
      - [FastText](https://drive.google.com/drive/folders/1bcR6ThMEPhguU9T4qPcPaZJ3GQzhLKlz?usp=sharing)
      - [ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) (ablation)
      - [BlueBERT](https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12) (ablation)

6. Run `06-Create-Timeseries-Data.ipynb`
   - to prepare the timeseries data to be fed through a GRU.

7. Run `07-Timeseries-Baseline.ipynb`
   - to run timeseries baseline model to predict 4 different clinical tasks.
   - This generates the results for the baseline GRU model.
   - Note: Notebooks 7-9 can be run in parallel.

8. Run `08-Multimodal-Baseline.ipynb`
   - to run multimodal baseline to predict 4 different clinical tasks.
   - This generates the results for the multimodal model: GRU in combination with Word2Vec, FastText, ClinicalBERT and/or BlueBERT.

9. Run `09-Proposed-Model.ipynb`
   - to run proposed model to predict 4 different clinical tasks.
   - This generates the results for the proposed model: GRU in combination with Word2Vec, FastText, ClinicalBERT and/or BlueBERT.
   - Can also run the ablation to amend the CNN model architecture

10. Run `load_print_results.ipynb`
    - to calculate and print out results from notebooks 7-9

## Results

The table below presents the findings from the reproduction of the results from the analysed paper.

- For each of the four (4) analysed tasks, the results from the best baseline model is compared to those from the proposed and ablation models.
- The `clinicalBERT` and `blueBERT` results are from the usage of these new embedding techniques
- The `(a)` following the `word2vec`, `fasttext` and `concat` represent the ablations due to the alternation of the CNN models for the proposed embedding techniques.
- Rankings for each scoring metric are provided

<style>
    .heatMap th {background: #DEDEDE;}
    .heatMap tr:nth-child(10) { background: #DEDEDE; }
    .heatMap tr:nth-child(20) { background: #DEDEDE; }
    .heatMap tr:nth-child(30) { background: #DEDEDE; }
</style>

<div class="heatMap">

| Task          | Model | Embedding | AUC           |     | AUPRC       |     | F1              |     |
|-----------------------|----------------|--------------------|--------------------------|-----|--------------------------|-----|--------------------------|-----|
|                       | Best Baseline  | -                  | 88.26 +/- 0.289          | (1) | 58.91 +/- 0.679          | (1) | 48.02 +/- 1.286          | (1) |
|                       |                | blueBERT (a)       | 88.04 +/- 0.252          | (6) | 57.84 +/- 0.434          | (3) | 44.87 +/- 2.467          | (8) |
|                       |                | clinicalBERT (a)   | 88.12 +/- 0.175          | (4) | 57.65 +/- 0.397          | (6) | 44.90 +/- 1.595          | (7) |
|                       |                | concat             | 88.14 +/- 0.170          | (3) | 57.68 +/- 0.386          | (4) | 45.69 +/- 1.620          | (6) |
| In-Hospital Mortality | Proposed Model | concat (a)         | 88.16 +/- 0.266          | (2) | 57.52 +/- 0.635          | (8) | 47.13 +/- 1.046          | (3) |
|                       |                | fasttext           | 87.89 +/- 0.329          | (9) | 57.24 +/- 0.646          | (9) | 44.49 +/- 1.630          | (9) |
|                       |                | fasttext (a)       | 88.09 +/- 0.142          | (5) | 57.55 +/- 0.438          | (7) | 46.60 +/- 0.896          | (4) |
|                       |                | word2vec           | 88.03 +/- 0.244          | (7) | 58.09 +/- 0.578          | (2) | 46.50 +/- 1.738          | (5) |
|                       |                | word2vec (a)       | 88.02 +/- 0.209          | (8) | 57.67 +/- 0.466          | (5) | 47.45 +/- 1.069          | (2) |
||||||||||
|                       | Best Baseline  | -                  | 89.17 +/- 0.176          | (1) | 53.55 +/- 0.414          | (1) | 46.13 +/- 1.631          | (3) |
|                       |                | blueBERT (a)       | 89.05 +/- 0.160          | (3) | 52.58 +/- 0.350          | (2) | 45.56 +/- 0.781          | (5) |
|                       |                | clinicalBERT (a)   | 89.10 +/- 0.103          | (2) | 52.16 +/- 0.427          | (6) | 44.34 +/- 1.005          | (6) |
|                       |                | concat             | 88.72 +/- 0.192          | (7) | 52.19 +/- 0.753          | (5) | 42.59 +/- 2.028          | (9) |
| In-ICU Mortality      | Proposed Model | concat (a)         | 88.99 +/- 0.180          | (4) | 52.20 +/- 0.575          | (4) | 47.00 +/- 0.942          | (1) |
|                       |                | fasttext           | 88.54 +/- 0.253          | (9) | 51.71 +/- 0.353          | (9) | 43.29 +/- 1.560          | (8) |
|                       |                | fasttext (a)       | 88.87 +/- 0.255          | (5) | 51.94 +/- 0.381          | (8) | 45.78 +/- 1.259          | (4) |
|                       |                | word2vec           | 88.69 +/- 0.350          | (8) | 52.25 +/- 0.678          | (3) | 43.74 +/- 2.809          | (7) |
|                       |                | word2vec (a)       | 88.86 +/- 0.249          | (6) | 52.06 +/- 0.417          | (7) | 46.49 +/- 1.362          | (2) |
||||||||||
|                       | Best Baseline  | -                  | 70.49 +/- 0.184          | (1) | 64.11 +/- 0.262          | (2) | 55.62 +/- 1.260          | (2) |
|                       |                | blueBERT (a)       | 70.23 +/- 0.180          | (7) | 63.86 +/- 0.246          | (8) | 55.15 +/- 0.679          | (6) |
|                       |                | clinicalBERT (a)   | 70.29 +/- 0.160          | (3) | 63.99 +/- 0.239          | (5) | 55.62 +/- 0.619          | (3) |
|                       |                | concat             | 70.14 +/- 0.326          | (8) | 63.91 +/- 0.366          | (7) | 55.46 +/- 1.292          | (4) |
| LOS $>$ 3 Days        | Proposed Model | concat (a)         | 70.28 +/- 0.163          | (5) | 63.97 +/- 0.166          | (6) | 55.02 +/- 0.885          | (7) |
|                       |                | fasttext           | 69.71 +/- 0.132          | (9) | 63.65 +/- 0.340          | (9) | 54.90 +/- 1.391          | (9) |
|                       |                | fasttext (a)       | 70.28 +/- 0.252          | (4) | 64.10 +/- 0.249          | (3) | 54.92 +/- 0.761          | (8) |
|                       |                | word2vec           | 70.38 +/- 0.168          | (2) | 64.30 +/- 0.273          | (1) | 55.66 +/- 0.773          | (1) |
|                       |                | word2vec (a)       | 70.26 +/- 0.162          | (6) | 64.02 +/- 0.248          | (4) | 55.30 +/- 0.658          | (5) |
||||||||||
|                       | Best Baseline  | -                  | 73.72 +/- 0.664          | (1) | 22.71 +/- 0.582          | (1) | 4.58 +/- 1.901           | (2) |
|                       |                | blueBERT (a)       | 73.66 +/- 0.394          | (2) | 22.47 +/- 0.440          | (4) | 1.66 +/- 0.957           | (8) |
|                       |                | clinicalBERT (a)   | 73.62 +/- 0.271          | (3) | 22.46 +/- 0.318          | (5) | 1.66 +/- 0.670           | (9) |
|                       |                | concat             | 73.29 +/- 0.427          | (4) | 22.19 +/- 0.548          | (7) | 1.67 +/- 0.780           | (7) |
| LOS $>$ 7 Days        | Proposed Model | concat (a)         | 73.27 +/- 0.279          | (5) | 22.45 +/- 0.307          | (6) | 5.46 +/- 1.430           | (1) |
|                       |                | fasttext           | 73.21 +/- 0.417          | (7) | 21.75 +/- 0.630          | (9) | 2.10 +/- 1.555           | (5) |
|                       |                | fasttext (a)       | 73.17 +/- 0.343          | (8) | 22.15 +/- 0.633          | (8) | 4.03 +/- 1.644           | (3) |
|                       |                | word2vec           | 73.13 +/- 0.534          | (9) | 22.56 +/- 0.387          | (3) | 1.94 +/- 0.972           | (6) |
|                       |                | word2vec (a)       | 73.22 +/- 0.352          | (6) | 22.62 +/- 0.618          | (2) | 3.69 +/- 1.207           | (4) |
</div>

## References

- Download the MIMIC-III dataset via <https://mimic.physionet.org/>

- MIMIC-Extract implementation: <https://github.com/MLforHealth/MIMIC_Extract>

- med7 implementation: <https://github.com/kormilitzin/med7>

- Download Pre-trained Word2Vec & FastText embeddings: <https://github.com/kexinhuang12345/clinicalBERT>

- Preprocessing Script: <https://github.com/kaggarwal/ClinicalNotesICU>
