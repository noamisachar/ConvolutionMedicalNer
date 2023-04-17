# Implementation for Improving Clinical Outcome Predictions Using Convolution over Medical Entities with Multimodal Learning

Modified by Kevin Sweeney and Noam Isachar - students at University of Illinois, Urbana Champaign, Spring 2023, CS-498 Deep Learning for Healthcare.

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
    https://github.com/tanlab/ConvolutionMedicalNer.git
    cd ConvolutionMedicalNer
    ```

2. Create and activate the conda environment

    ```shell
    conda env create -f environment.yml
    conda activate DLH_project_py38
    ```

### Step 2: Get Required Data

1. Get access to [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) dataset.
   - Note: This may take several days

2. Run MIMIC-Extract Pipeline as explained in <https://github.com/MLforHealth/MIMIC_Extract>.
   - Note: The required data may be available from `gcp` as specified in the `Pre-processed Output` section of the repository.

3. Copy the output file of MIMIC-Extract Pipeline named `all_hourly_data.h5` to `data` folder.

4. Also need these three data files from [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/). Place them in the same `data` folder.
     - `ADMISSIONS.csv`
     - `ICUSTAYS.csv`
     - `NOTEEVENTS.csv`

5. Download pretrained embeddings ([Word2Vec](https://github.com/kexinhuang12345/clinicalBERT) and [FastText](https://drive.google.com/drive/folders/1bcR6ThMEPhguU9T4qPcPaZJ3GQzhLKlz?usp=sharing)) into `embeddings` folder.

### Step 3: Run Code (Jupyter Notebooks)

1. Run `01-Extract-Timseries-Features.ipnyb`
   - to extract the first 24 hours of timeseries features from the MIMIC-Extract raw data.

2. Run `02-Select-SubClinicalNotes.ipynb`
   - to select the subnotes from `NOTEEVENTS.csv` based on criteria.

3. Run `03-Preprocess-Clinical-Notes.ipynb`
   - to prepocess the clinical notes.

4. Run `04-Apply-med7-on-Clinical-Notes.ipynb`
   - to extract medical entities using the [med7 pre-trained model](https://github.com/kormilitzin/med7).

5. Run `05-Represent-Entities-With-Different-Embeddings.ipynb`
    - to convert medical entities into word representations using [Word2Vec](https://github.com/kexinhuang12345/clinicalBERT) and/or [FastText](https://drive.google.com/drive/folders/1bcR6ThMEPhguU9T4qPcPaZJ3GQzhLKlz?usp=sharing) models.

6. Run `06-Create-Timeseries-Data.ipynb`
   - to prepare the timeseries data to be fed through a GRU.

7. Run `07-Timeseries-Baseline.ipynb`
   - to run timeseries baseline model to predict 4 different clinical tasks.
   - This generates the results for the baseline GRU model.
   - Note: Notebooks 7-9 can be run in parallel.

8. Run `08-Multimodal-Baseline.ipynb`
   - to run multimodal baseline to predict 4 different clinical tasks.
   - This generates the results for the multimodal model: GRU in combination with Word2Vec and/or FastText.

9. Run `09-Proposed-Model.ipynb`
   - to run proposed model to predict 4 different clinical tasks.
   - This generates the results for the proposed model: GRU in combination with Word2Vec and/or FastText using CNN.

10. Run `load_print_results.ipynb`
    - to calculate and print out results from notebooks 7-9

## References

- Download the MIMIC-III dataset via <https://mimic.physionet.org/>

- MIMIC-Extract implementation: <https://github.com/MLforHealth/MIMIC_Extract>

- med7 implementation: <https://github.com/kormilitzin/med7>

- Download Pre-trained Word2Vec & FastText embeddings: <https://github.com/kexinhuang12345/clinicalBERT>

- Preprocessing Script: <https://github.com/kaggarwal/ClinicalNotesICU>
