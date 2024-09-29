import pandas as pd
import json

def save_as_json(preds, ids, out_path_dict):
  preds_dict = dict()
  for idx, x in enumerate(preds):
    pred_array = int(x) 
    preds_dict[ids[idx]] = pred_array
  #print(preds_dict)
    
  with open(out_path_dict, 'w+') as fp:
        json.dump(preds_dict, fp)