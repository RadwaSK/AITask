# AITask
This repo contains code of a classification task of different car types (ex. trucks, sedan...etc.)

You will find two branches for different models trained and tested on given datasets

Files inside the repo:
* dataset (contains train test folders)
* ```dataloader.py```: has function that prepares dataloaders for the training/test


Setups used in this repo:
* DVC for dataset shared on a drive link, to pull data:
  * ```dvc fetch```
  * ```dvc pull```
* MLFLow for models, to run the MLFLow UI
  * run ```mlflow ui --port 8800``` in command (supposedly, will be run by default inside the docker contained)
  * open ```localhost:8800``` in your browser
* Docker image (to do), to run it:
  * docker ...