
# Ensemble Conformal Prediction for Model Consistency

> This is a final project submission for the Seminar on Reliability in ML \
> Spring 2021 semester, Technion.

---

> **Abstract**: TBD

# Table of Contents

* [Setup](#setup)
* [Reproducing the Results](#reproducing-the-results)

## Setup

To setup your environment, we recommend that you use either Anaconda/Miniconda 
or venv. This code was tested using `Cuda v11.0`.

### Anaconda/Miniconda

*TBD*

### Other Environments

To setup all requirements, from the root directory run
```shell
pip install -r requirements
```

## Reproducing the Results

*TBD*

### Generate DynSnap model

```shell
ds=DS3; python prune_avg_cp.py -i "output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/run_1/index_cp.csv;output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/run_2/index_cp.csv;output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/run_3/index_cp.csv;output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/run_4/index_cp.csv;output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/run_5/index_cp.csv" -n 10 -o output/yahoo_answers_csv_imbalance3/${ds}/snapshotA/index_cp.csv
```

### Combine results for all datasets

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

## Contributions

*TBD*

## License

*TBD*