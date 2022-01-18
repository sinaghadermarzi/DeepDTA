#!/usr/bin/env python
import sys
from train_model import build_combined_categorical
from config import get_config
from dataset import dta_dataset
import numpy as np
if len(sys.argv) != 2:
    print("\nUsage: predict.py <path to input csv>\n")
    exit(1)



def save_predictions(pids,cids,predicted_labels,csv_path):
    with open(csv_path, "w") as csvf:
        for i in range(len(pids)):
            pair_str = pids[i]+"_"+cids[i]
            pred = str(predicted_labels[i])
            line = pair_str+","+pred+"\n"
            csvf.writelines(line)



input_csv_path = sys.argv[1]
output_csv_path = input_csv_path.replace(".csv", "_predicted.csv")

conf = get_config()
with open("saved_model/model_hyperparams.txt") as parf:
    spl= parf.read().rstrip().split("\t")
    params = [int(x) for x in spl]

test_dataset = dta_dataset(conf,input_csv_path,unlabeled=True)
n_test = test_dataset.get_num_pairs()
test_inds = list(range(n_test))
test_X_d , test_X_t, test_Y,cids, pids = test_dataset.get_objects(test_inds)
[p1, p2,p3 ] =  params
model = build_combined_categorical(conf, p1,p2,p3)
model.load_weights("saved_model/model_weights")

predicted_labels = model.predict([np.array(test_X_d), np.array(test_X_t)]).squeeze()
save_predictions(pids,cids,predicted_labels,output_csv_path)