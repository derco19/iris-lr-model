import os
from joblib import load
import os
import pandas as pd
import json


MODEL_DIR = os.environ["MODEL_DIR"]
model_file = 'logit_model.joblib'
model_path = os.path.join(MODEL_DIR, model_file)

logit_model = load(model_path)

PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
x_test = 'x_test.csv'
x_test_path = os.path.join(PROCESSED_DATA_DIR, x_test)
y_test = 'y_test.csv'
y_test_path = os.path.join(PROCESSED_DATA_DIR, y_test)

df = pd.read_csv(x_test_path, sep=",")
x_test=df
df = pd.read_csv(y_test_path, sep=",")
y_test=df


from sklearn.metrics import confusion_matrix, accuracy_score
predictions = logit_model.predict(x_test)
cm = confusion_matrix(predictions, y_test)
print(cm)
test_logit = accuracy_score(predictions, y_test)

test_metadata = {
    'test_acc': test_logit
}

RESULTS_DIR = os.environ["RESULTS_DIR"]
test_results_file = 'test_metadata.json'
results_path = os.path.join(RESULTS_DIR, test_results_file)

with open(results_path, 'w') as outfile:
    json.dump(test_metadata, outfile)