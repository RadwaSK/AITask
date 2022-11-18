import argparse
import os
from pandas import __version__ as pd_ver
import mlflow as ml
from os.path import join
import torch
from torchvision import __version__ as tvs_ver
from torch import optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import __version__ as mp_ver
from utils import get_resnet_based_model, EarlyStopping, get_classes_dict, save_model
from dataloader import get_train_val_dataloader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn import __version__ as sk_ver

print('Torch V:', torch.__version__)
print('TorchVision version:', tvs_ver)
print('NumPy V:', np.__version__)
print('MLFlow V:', ml.__version__)
print('Matplotlib V:', mp_ver)
print('Pandas V:', pd_ver)
print('Sklearn V:', sk_ver)

os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_number', type=int, default=len(os.listdir('saved_models')), help='Run number to give unique value to each run')
parser.add_argument('-d', '--data_path', type=str, default='dataset/train', help='path for training dataset')
parser.add_argument('-m', '--model_name', type=str, default='saved_models', help='name of the model to be saved.')
parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size for data')
parser.add_argument('-s', '--st_epoch', type=int, default=0, help='start epoch number')
parser.add_argument('-n', '--n_epochs', type=int, default=20, help='number of epochs')
parser.add_argument('-l', '--load', type=str, default='n', help='enter y to load saved model, n for not')
parser.add_argument('-rn', '--run_name', type=str, default='ResNet', help='Enter run name for MLFlow')

options = vars(parser.parse_args())

print('\n======================================================================\n')
print(options)
print('\n======================================================================\n')

dataloaders = get_train_val_dataloader(batch_size=options['batch_size'])
dataset_count = {x: len(dataloaders[x].dataset) for x in dataloaders}
print(dataset_count)

use_cuda = torch.cuda.is_available()
print('CUDA available:', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
models_path = 'saved_path'

model = get_resnet_based_model(freeze_resnet=False, CUDA=use_cuda)

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)

if options['load'] == 'y':
    model_name = join(models_path, options['model_name'])
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])

criterion = nn.CrossEntropyLoss().to(device)
softmax = nn.Softmax(dim=-1)

train_loss = []
val_loss = []
train_f1 = []
val_f1 = []
train_rec = []
val_rec = []
train_prec = []
val_prec = []
train_acc = []
val_acc = []

early_stopping = EarlyStopping(tolerance=5, min_delta=0.2)

classes_dict = get_classes_dict()
n_classes = len(classes_dict)

experiment_name = options['run_name'] + '_' + str(options['run_number'])
plots_folder = join('plots', experiment_name)
os.makedirs(plots_folder, exist_ok=True)
model_name = join(options['model_name'], experiment_name)

experiment_ID = ml.create_experiment(name=experiment_name)
finish = False

with ml.start_run(experiment_id=experiment_ID) as r:
    ml.log_params(options)
    ml.log_param('optimizer', 'Adam')
    ml.log_param('Loss', 'CrossEntropy')
    ml.log_param('Using_CUDA', use_cuda)
    ml.log_param('Continue_Learning', (options['load'] == 'y'))
    ml.log_artifacts(plots_folder)

    print('Running', experiment_name, 'with MLFlow')
    min_loss = 10 ** 6

    for epoch in range(options['st_epoch'], options['n_epochs']):
        for phase in ['train', 'validation']:
            running_loss = .0
            y_trues = np.empty([0])
            y_preds = np.empty([0])

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.squeeze().to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()

                    loss = torch.tensor([0]).to(device)
                    calc_bef = False

                    for c in classes_dict:
                        id = classes_dict[c]
                        indices = (labels == id).nonzero(as_tuple=False)
                        if indices.numel():
                            real = torch.squeeze(labels[indices], dim=1)
                            predicted = torch.squeeze(outputs[indices], dim=1)
                            if calc_bef:
                                loss += torch.mul(criterion(predicted, real), len(real) / n_classes)
                            else:
                                loss = torch.mul(criterion(predicted, real), len(real) / n_classes)
                                calc_bef = True

                    if phase == 'train' and calc_bef:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                preds = torch.max(outputs, dim=-1)

                y_trues = np.append(y_trues, labels.data.cpu().numpy())
                y_preds = np.append(y_preds, preds.indices.cpu())

            epoch_loss = running_loss / dataset_count[phase]
            acc = accuracy_score(y_trues, y_preds)
            f1 = f1_score(y_trues, y_preds, average='weighted')
            recall = recall_score(y_trues, y_preds, average='weighted')
            precision = precision_score(y_trues, y_preds, average='weighted')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_f1.append(f1)
                train_rec.append(recall)
                train_prec.append(precision)
                train_acc.append(acc)
                last_train_loss = epoch_loss

            else:
                val_loss.append(epoch_loss)
                val_f1.append(f1)
                val_rec.append(recall)
                val_prec.append(precision)
                val_acc.append(acc)
                if epoch_loss < min_loss:
                    print('\n\n<<<Saving model>>>\n')
                    save_model(model, optimizer, model_name)
                    min_loss = epoch_loss

                early_stopping(last_train_loss, epoch_loss)
                finish = early_stopping.early_stop

            print("[{}] Epoch: {}/{} Loss: {}".format(phase, epoch + 1, options['n_epochs'], epoch_loss))
            print('\nF1 Score:\t' + str(f1))
            print('\nRecall:\t' + str(recall))
            print('\nPrecision:\t' + str(precision))
            print('\nAccuracy:\t' + str(acc))
            print('\nConfusion Matrix of classes: \n', confusion_matrix(y_trues, y_preds))
            print('\n================================================================================\n')

        if finish:
            break

    ml.log_param('Last_Epoch', epoch + 1)

    # plotting
    plt.plot(range(options['st_epoch'] + 1, options['st_epoch'] + epoch + 2), train_loss, label='train loss')
    plt.plot(range(options['st_epoch'] + 1, options['st_epoch'] + epoch + 2), val_loss, label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.savefig(join(plots_folder, 'loss.png'))
    plt.clf()

    plt.plot(range(options['st_epoch'] + 1, options['st_epoch'] + epoch + 2), train_f1, label='train F1')
    plt.plot(range(options['st_epoch'] + 1, options['st_epoch'] + epoch + 2), val_f1, label='validation F1')
    plt.xlabel('epochs')
    plt.ylabel('F1 Scores')
    plt.legend()
    plt.savefig(join(plots_folder, 'f1_scores.png'))
    plt.clf()

    plt.plot(range(options['st_epoch'] + 1, options['st_epoch'] + epoch + 2), train_rec, label='train recall')
    plt.plot(range(options['st_epoch'] + 1, options['st_epoch'] + epoch + 2), val_rec, label='validation recall')
    plt.xlabel('epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig(join(plots_folder, 'recall.png'))
    plt.clf()

    plt.plot(range(options['st_epoch'] + 1, options['st_epoch'] + epoch + 2), train_prec, label='train precision')
    plt.plot(range(options['st_epoch'] + 1, options['st_epoch'] + epoch + 2), val_prec, label='validation precision')
    plt.xlabel('epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(join(plots_folder, 'precision.png'))
    plt.clf()

    plt.plot(range(options['st_epoch'] + 1, options['st_epoch'] + epoch + 2), train_acc, label='train accuracy')
    plt.plot(range(options['st_epoch'] + 1, options['st_epoch'] + epoch + 2), val_acc, label='validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuracies')
    plt.legend()
    plt.savefig(join(plots_folder, 'accuracy_score.png'))
    plt.clf()



