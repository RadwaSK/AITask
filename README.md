# AITask
This repo contains code of a classification task of different car types (ex. trucks, sedan...etc.)

You will find two branches for different models trained and tested on given datasets

### Files inside the repo:
* ```data_analysis_ipynb```: notebook for data analysis with insights
* ```dataloader.py```: has function that prepares dataloaders for the training/test
* ```prep_csv.py```: preparing csv files from dataset
* ```test.py```: testing file for dataset
* ```train.py```: training file
* ```TrucksDataset.py```: Dataset file
* ```utils.py```: has some utily functions

### Folders:
* dataset (contains train test folders)
* plots (plots drawn during training)
* saved_models (dvc files of saved models)
* trial_runs (text files of outputs of training/testing trials)

---------------------------

## Setups used in this repo:
* DVC for dataset shared on a drive link, to pull data:
  * ```dvc fetch```
  * ```dvc pull```
* MLFLow for models, to run the MLFLow UI
  * run ```mlflow ui --port 8800``` in command (supposedly, will be run by default inside the docker contained)
  * open ```localhost:8800``` in your browser
* Docker image (to do), to run it:
  * docker ...

----------------------

## To prepare data:
1. Move dataset into folder ```dataset```
2. Either dvc pull data or run ```prep_csv.py```

