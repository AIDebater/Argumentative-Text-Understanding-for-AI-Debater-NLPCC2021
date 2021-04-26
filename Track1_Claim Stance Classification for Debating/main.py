from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.metrics import classification_report
import logging
import csv
import numpy as np

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Define metric
def clf_report(labels, preds):
    return classification_report(labels, preds, output_dict=True)


# evaluate on test set
eval_df = pd.read_csv('data/test.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
eval_df.columns = ['text_a', 'text_b', 'labels']
model = ClassificationModel('bert', 'outputs/best_model/')
result, model_outputs, wrong_predictions = model.eval_model(eval_df, clf_report=clf_report)

preds = list(np.argmax(model_outputs, axis=-1))
label_map = {0: 'Support', 1: 'Against', 2: 'Neutral'}
preds = [label_map[x] for x in preds]

with open('outputs/submission.csv', 'w') as f:
	for x in preds:
	    out_f.write(x+'\n')






