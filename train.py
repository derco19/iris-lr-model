import pandas as pd
import os
from joblib import dump

RAW_DATA_DIR = os.environ["RAW_DATA_DIR"]
RAW_DATA_FILE = os.environ["RAW_DATA_FILE"]
raw_data_path = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE)

# Read data
dataset = pd.read_csv(raw_data_path, sep=",")

dataset.head()

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class='ovr', random_state = 0)
logit_model=classifier.fit(x_train, y_train)

MODEL_DIR = os.environ["MODEL_DIR"]
model_name = 'logit_model.joblib'
model_path = os.path.join(MODEL_DIR, model_name)


# Serialize and save model
dump(logit_model, model_path)

PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
x_test_path = os.path.join(PROCESSED_DATA_DIR, 'x_test.csv')
y_test_path = os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')
x_test=pd.DataFrame(x_test)
y_test=pd.DataFrame(y_test)
x_test.to_csv(x_test_path, index=False)
y_test.to_csv(y_test_path,  index=False)