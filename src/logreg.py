import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

X_train = np.load('precalculated_features_train1.npy')
y_train = np.load('labels_train1.npy')

X_test = np.load('precalculated_features_val1.npy')
y_test = np.load('labels_val1.npy')
vid_id_test = np.load('video_id_val1.npy')
print(len(X_train))
print(len(y_train))

print(len(X_test))
print(len(y_test))

model = LogisticRegression(multi_class='multinomial', solver='saga', n_jobs=2)
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))


y_pred = model.predict_proba(X_test)
df = pd.DataFrame(y_pred)
df.columns = list(range(101))
df['video_id'] = vid_id_test
df = df.groupby('video_id').sum()
df['pred_label'] = df.idxmax(axis=1)
y_pred = df['pred_label'].values
y_real = df.index.values
print(accuracy_score(y_real, y_pred))

with open("log_reg_constrained.pickle", "w") as input_file:
    pickle.dump('model', input_file)