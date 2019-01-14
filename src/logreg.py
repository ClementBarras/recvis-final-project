import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

X_train = np.load('../datasets/supervised_data/random/features_train01.npy')
y_train = np.load('../datasets/supervised_data/random/labels_train01.npy')

X_test = np.load('../datasets/supervised_data/random/features_test01.npy')
y_test = np.load('../datasets/supervised_data/random/labels_test01.npy')
vid_id_test = np.load('../datasets/supervised_data/random/video_id_test01.npy')
print(len(X_train))
print(len(y_train))

print(len(X_test))
print(len(y_test))

model = LogisticRegression(n_jobs=2)
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))


y_pred = model.predict_proba(X_test)
df = pd.DataFrame(y_pred)
df.columns = list(range(101))
df['video_id'] = vid_id_test
df.to_csv('logreg_random.csv')
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
print(accuracy_score(y_real, y_pred))

with open("log_reg_random.pickle", "w") as input_file:
    pickle.dump(model, input_file)