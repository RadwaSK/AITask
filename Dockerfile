FROM python:3.9-slim-buster

WORKDIR /AITask

COPY requirements.txt ./

COPY dataloader.py ./dataloader.py
COPY prep_csv.py ./prep_csv.py
COPY train.py ./train.py
COPY test.py ./test.py
COPY TrucksDataset.py ./TrucksDataset.py
COPY utils.py ./utils.py

COPY dataset/train.csv ./dataset/train.csv
COPY dataset/val.csv ./dataset/val.csv
COPY dataset/test.csv ./dataset/test.csv

RUN pip install --no-cache-dir -r requirements.txt

CMD python3 test.py -m ResNet_0
