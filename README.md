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
* ```data_analysis.pdf```: same jupyter notebook file but in pdf

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

## To start the project:
1. ```git clone https://github.com/RadwaSK/AITask.git```
2. ```cd AITask```
3. Move dataset into folder ```dataset``` (there should be dataset/train and dataset/test, and you would find the csv files cloned with the repo)
4. Pull docker container ```docker pull radwask/aitask```
5. Download the attached model ResNet_0 and move it into a folder "saved_models" in the repo folder
6. To run test script:
   * ```sudo docker run -v <path to repo>/AITask/dataset:/AITask/dataset -v <path to repo>/AITask/saved_models:/AITask/saved_models --gpus device=0 radwask/aitask python3 test.py -m ResNet_0 -b 4```

### P.S. THIS IS ASSUMING YOU HAVE CUDA ON YOUR DEVICE ! The models are trained with CUDA available.

