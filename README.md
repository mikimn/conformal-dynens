
# Ensemble Conformal Prediction for Model Consistency

> This is a final project submission for the Seminar on Reliability in ML \
> Spring 2021 semester, Technion.

---

> **Abstract**: Recent work discusses the trustworthiness of AI systems which evolve over time and may become inconsistent in their predictions, when they are retrained. Ensemble models were shown to be effective in this domain, boosting the consistency of such models and increasing their trust. However, relying on ensembles of simple classification models provides no measure for the reliability of said models. We propose an adaptation of a novel ensemble method for increased consistency, where conformal classifiers are used to create ensembles. We propose a method for generating ensemble prediction sets, and experiment with a popular dataset for text classification. Our results suggest that conformal predictors can also benefit from such ensembling, and remain very consistent, while improving overall coverage. We investigate the effect of our ensembling method and discuss limitations and possible directions for future work.

# Table of Contents

* [Setup](#setup)
* [Reproducing the Results](#reproducing-the-results)

## Setup

To setup your environment, we recommend that you use either Anaconda/Miniconda 
or venv. This code was tested using `Cuda v11.0`.

To setup all requirements, from the root directory run
```shell
pip install -r requirements
```

## Reproducing the Results

To train a model, use the relevant script in the root directory. For example, to train the DynSnap model (`snapshotA`) for a single run, do:
```
python fasttext_yahoo_snapshotA_cp.py \
    --do_train \
    --run 1 \
    --datafile DS1
```

### Generate DynSnap model

You can prune the trained snapshot models with DynSnap using the following script (be sure to specify the dataset file):

```shell
ds=DS3;
python prune_avg_cp.py \
    -i "output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/run_1/index_cp.csv;output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/run_2/index_cp.csv;output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/run_3/index_cp.csv;output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/run_4/index_cp.csv;output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/run_5/index_cp.csv" \
    -n 10 \
    -o output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/index_cp.csv
```

### Combine results for all datasets

Use the following script to combine results into 

```shell
datasets=(DS1 DS2 DS3)
for ds in ${datasets[@]}; do 
    python combination_cp.py --workdir output/yahoo_answers_csv_imbalance3/${ds}/snapshot/run_1 --alpha 0.1
done
```

### Obtain results for two datasets

```shell
python get_results.py \
    --predfile1 output/yahoo_answers_csv_imbalance3/DS1/snapshot/run_1/prediction_set_conformal_threshold_selection.csv \
    --predfile2 output/yahoo_answers_csv_imbalance3/DS2/snapshot/run_1/prediction_set_conformal_threshold_selection.csv \
    --groundtruth output/yahoo_answers_csv_imbalance3/DS1/snapshot/run_1/target.csv
```
