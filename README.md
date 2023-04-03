# Implementation for Improving Clinical Outcome Predictions Using Convolution over Medical Entities with Multimodal Learning

## Usage

1. Clone the code to local.

    ```shell
    https://github.com/tanlab/ConvolutionMedicalNer.git
    cd ConvolutionMedicalNer
    ```

2. Run MIMIC-Extract Pipeline as explained in <https://github.com/MLforHealth/MIMIC_Extract>.

3. Copy the output file of MIMIC-Extract Pipeline named `all_hourly_data.h5` to `data` folder.
   - Also need the three data files from <https://physionet.org/content/mimiciii/1.4/>. Place them in the same folder
     - `ADMISSIONS.csv`
     - `ICUSTAYS.csv`
     - `NOTEEVENTS.csv`

4. Create and activate the conda environment

    ```shell
    conda env create -f environment.yml
    conda activate DLH_project_py38
    ```

5. Run `01-Extract-Timseries-Features.ipnyb` to extract first 24 hours timeseries features from MIMIC-Extract raw data.

6. Copy the `ADMISSIONS.csv`, `NOTEEVENTS.csv`, `ICUSTAYS.csv` files into `data` folder.

7. Run `02-Select-SubClinicalNotes.ipynb` to select subnotes based on criteria from all MIMIC-III Notes.

8. Run `03-Prprocess-Clinical-Notes.ipnyb` to prepocessing notes.

9. Run `04-Apply-med7-on-Clinical-Notes.ipynb` to extract medical entities.

10. Download pretrained embeddings into `embeddings` folder via link in given References section.

11. Run `05-Represent-Entities-With-Different-Embeddings.ipynb` to convert medical entities into word representations.

12. Run `06-Create-Timeseries-Data.ipynb` to prepare the timeseries data to fed through GRU / LSTM.

13. Run `07-Timeseries-Baseline.ipynb` to run timeseries baseline model to predict 4 different clinical tasks.

14. Run `08-Multimodal-Baseline.ipynb` to run multimodal baseline to predict 4 different clinical tasks.

15. Run `09-Proposed-Model.ipynb` to run proposed model to predict 4 different clinical tasks.

## References

- Download the MIMIC-III dataset via <https://mimic.physionet.org/>

- MIMIC-Extract implementation: <https://github.com/MLforHealth/MIMIC_Extract>

- med7 implementation: <https://github.com/kormilitzin/med7>

- Download Pre-trained Word2Vec & FastText embeddings: <https://github.com/kexinhuang12345/clinicalBERT>

- Preprocessing Script: <https://github.com/kaggarwal/ClinicalNotesICU>
