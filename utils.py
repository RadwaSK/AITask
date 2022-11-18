import torch
import os
from mlflow.pyfunc import log_model
from mlflow.utils import PYTHON_VERSION
import mlflow


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
