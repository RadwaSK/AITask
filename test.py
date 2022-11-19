from os.path import join
import argparse
import torch
from dataloader import get_test_dataloader
import numpy as np
from utils import get_vgg_based_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size for data')
parser.add_argument('-mn', '--model_name', type=str, required=True, help='enter name of model to test from models in saved_models')

options = parser.parse_args()

dataloader = get_test_dataloader(batch_size=options.batch_size)

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
print('CUDA available:', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

test_data_count = len(dataloader.dataset)
print("Number of data:", test_data_count)

model_path = 'saved_models'
model = get_vgg_based_model(CUDA=use_cuda).to(device)
model_name = join(model_path, options.model_name)
checkpoint = torch.load(model_name)
model.load_state_dict(checkpoint['model_state'])
model.eval()

y_trues = np.empty([0])
y_preds = np.empty([0])

model.eval()

for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.long().squeeze().to(device)

    outputs = model(inputs).squeeze()

    preds = torch.max(outputs, dim=-1)

    y_trues = np.append(y_trues, labels.data.cpu().numpy())
    y_preds = np.append(y_preds, preds.indices.cpu())


acc = accuracy_score(y_trues, y_preds)
f1 = f1_score(y_trues, y_preds, average='weighted')
recall = recall_score(y_trues, y_preds, average='weighted')
precision = precision_score(y_trues, y_preds, average='weighted')

print('\nF1 Score:\t' + str(f1))
print('\nRecall:\t' + str(recall))
print('\nPrecision:\t' + str(precision))
print('\nAccuracy:\t' + str(acc))
print('\nConfusion Matrix of classes: \n', confusion_matrix(y_trues, y_preds))

