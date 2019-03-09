import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import argparse
import os


parser = argparse.ArgumentParser(description='Supervised model script')
parser.add_argument('--data', type=str, default='../datasets/supervised_data', metavar='D',
                    help="folder where data is located.")
parser.add_argument('--type', type=str, default='consecutive', metavar='D',
                    help="subfolder where data is located.")
parser.add_argument('--model', type=str, default='knn', metavar='D',
                    help="subfolder where data is located.")

args = parser.parse_args()

path_to_data = args.data

X_train = np.load(os.path.join(path_to_data, args.type, 'features_train.npy'))
y_train = np.load(os.path.join(path_to_data, args.type, 'labels_train.npy'))

X_test = np.load(os.path.join(path_to_data, args.type, 'features_test.npy'))
y_test = np.load(os.path.join(path_to_data, args.type, 'labels_test.npy'))
vid_id_test = np.load(os.path.join(path_to_data, args.type, 'video_id_test.npy'))
print(len(X_train))
print(len(y_train))

print(len(X_test))
print(len(y_test))

model_name = args.model

if model_name == 'logreg':
    model = LogisticRegression()
elif model_name == 'knn':
    model = KNeighborsClassifier()
elif model_name == 'rf':
    model = RandomForestClassifier(n_estimators=400)
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))


y_pred = model.predict_proba(X_test)
df = pd.DataFrame(y_pred)
df.columns = list(range(101))
df['video_id'] = vid_id_test
dir_name = os.path.join('../', args.model, args.type)
#os.mkdir(dir_name)
#df.to_csv(os.path.join(dir_name, 'predictions.csv'))
df = df.groupby('video_id').sum()
df['pred_label'] = df.idxmax(axis=1)
y_pred = df['pred_label'].values
df2 = pd.DataFrame()
df2['video_id'] = vid_id_test
df2['labels'] = y_test
df2 = df2.groupby('video_id').mean()
y_real = df2['labels']
print(y_pred)
print(y_real)
print('Final accuracy for model {} : {}'.format(model_name, accuracy_score(y_real, y_pred)))

with open(os.path.join(dir_name, "model.pickle"), "w") as input_file:
    pickle.dump(model, input_file)