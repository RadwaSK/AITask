import torch
import os
from mlflow.pyfunc import log_model
from mlflow.utils import PYTHON_VERSION
import mlflow
import torchvision
import torch.nn as nn
from mlflow.pyfunc import PythonModel

def get_resnet_based_model(freeze_resnet=False, CUDA=True, num_classes=8):
    model = torchvision.models.resnet50(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = not freeze_resnet

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda:0" if CUDA else "cpu")
    model = model.to(device)

    return model


class ResNetModelWrapper(PythonModel):

    def __init__(self):
        self.model = get_resnet_based_model()

    def load_context(self, context):
        checkpoint = torch.load(context.artifacts['model_path'])
        self.model.load_state_dict(checkpoint['model_state'])

    def predict(self, context, input_video_path):
        # TO DO !!!!
        return None


def get_classes_names(data_path='dataset/train'):
    classes = os.listdir(data_path)
    return classes


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.1):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0


def save_model(model, optim, path):
    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
            'python={}'.format(PYTHON_VERSION),
            'pip',
            {
                'pip': [
                    'mlflow=={}'.format(mlflow.__version__),
                    'torch=={}'.format(torch.__version__),
                ],
            },
        ],
        'name': 'mlflow_env'
    }
    torch.save({'model_state': model.state_dict(), 'optim_state': optim.state_dict()}, path)
    artifacts = {"model_path": path}
    log_model('model', python_model=ResNetModelWrapper(), conda_env=conda_env, artifacts=artifacts)
