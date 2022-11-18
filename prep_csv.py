import pandas as pd
import os

train_path = 'dataset/train'
test_path = 'dataset/test'

classes = os.listdir(train_path)

classes_dist = {'bus': 77, 'crossover': 480, 'hatchback': 419, 'motorcycle': 95, 'pickup-truck': 357, 'sedan': 1581, 'truck': 258, 'van': 418}
train_dist = {}
val_dist = {}
for c in classes_dist:
    train_dist[c] = int(0.8 * classes_dist[c])
    val_dist[c] = classes_dist[c] - train_dist[c]

train_data = []
val_data = []

for c in classes:
    path = os.path.join(train_path, c)
    files = os.listdir(path)
    counter = 0
    for i, f in enumerate(files):
        file_path = os.path.join(path, f)
        if counter < train_dist[c]:
            train_data.append([file_path, c])
        else:
            val_data.append([file_path, c])
        counter += 1

train_csv = pd.DataFrame(train_data, columns=['path', 'label']).sample(frac=1)
val_csv = pd.DataFrame(val_data, columns=['path', 'label']).sample(frac=1)

train_csv.to_csv('dataset/train.csv', index=False)
val_csv.to_csv('dataset/val.csv', index=False)

test_data = []

for c in classes:
    path = os.path.join(test_path, c)
    files = os.listdir(path)
    for i, f in enumerate(files):
        file_path = os.path.join(path, f)
        test_data.append([file_path, c])

test_csv = pd.DataFrame(test_data, columns=['path', 'label']).sample(frac=1)
test_csv.to_csv('dataset/test.csv', index=False)
