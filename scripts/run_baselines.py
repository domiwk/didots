import pandas as pd
import pickle
import json
import argparse
import pathlib

from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve,accuracy_score,f1_score,recall_score,precision_score,roc_auc_score

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import matthews_corrcoef

from dataset_utils import save_as_json

parser = argparse.ArgumentParser(description='Train and evaluate baseline model on datasets.')

parser.add_argument('--datapath',action='store', type=str, help='Path to dataset.')
parser.add_argument('--test_set_path',default = None, type=str, help='Path to dataset.')
parser.add_argument('--datapath_og',default = None, type=str, help='Whether or not to add the og dataset.')
parser.add_argument('--model',action='store', type=str, help='Model type (SVM, RTF).')
parser.add_argument('--feature', type=str, default = "TFIDF",help='Feature type (BOW, TFIDF).')
parser.add_argument('--dataset_name',action='store', type=str, help='Name of dataset.')
parser.add_argument('--path_to_model_save',action='store', type=str, help='Path where to save model.')
parser.add_argument('--path_to_results_dir',action='store', type=str, help='Path where to save predictions and metrics.')
parser.add_argument('-i','--Label', type=str, default = "User_id", help='Path where to save predictions and metrics.')
parser.add_argument('-c_d_p','--is_custom_data_path', type=bool, default=False,help='Whether or not load custom data files to paraphrase')

def run_baseline(train,test,model_name = "SVM",dataset="ADReSS",feature = "TFIDF", report = False, task = "classification", seed = 42, save_model_to = None,idx2Label={0:'cc',1:"cd"}):
  
  classifiers = {'SVM':svm.SVC, 'RF':RandomForestClassifier}

  models = classifiers[model_name]

  weights_list = []

  if feature == "BOW":
    vectorizer = CountVectorizer(analyzer = "word") 
    x_train = vectorizer.fit_transform(train['Text'])
    x_val = vectorizer.transform(test['Text'])

    if save_model_to:
      filename = f'{save_model_to}/{dataset}_{feature}_vectorizer.pkl'
      pickle.dump(vectorizer, open(filename, 'wb'))
      weights_list.append(filename)

  elif feature == "TFIDF":
    tfvectorizer = TfidfVectorizer()
    x_train = tfvectorizer.fit_transform(train['Text'])
    x_val = tfvectorizer.transform(test['Text'])

    if save_model_to:
      filename = f'{save_model_to}/{dataset}_{feature}_vectorizer.pkl'
      pickle.dump(tfvectorizer, open(filename, 'wb'))
      weights_list.append(filename)

  if feature == "BOW":
              params = {
                  'kernel': 'linear', 'C': 0.016763776584841045, 'gamma': 6.150033282302239e-05,"random_state" :seed
              }
  else:
    params = {"random_state" :seed}

  if task == "classification":
    model = models(**params)
    model.fit(x_train, train['Label'])
  
    # save the model to disk
    if save_model_to:
      filename = f'{save_model_to}/{dataset}_{model_name}_{feature}.pkl'
      pickle.dump(model, open(filename, 'wb'))
      weights_list.append(filename)

    pred = model.predict(x_val)
    
    preds = [idx2Label[enum] for enum in pred]
    real_trait_labels = [idx2Label[enum] for enum in test['Label']]
    ## metrics

    if report:
      print(classification_report(real_trait_labels, preds))
    report = classification_report(real_trait_labels, preds,output_dict=True)

    if len(idx2Label.keys()) == 2:
      res = {"accuracy":accuracy_score(test['Label'], pred),
            "mcc" : matthews_corrcoef(test['Label'], pred),
            "recall":recall_score(test['Label'], pred),
            "recall_0": recall_score(test['Label'], pred, pos_label =0),
            'precision':precision_score(test['Label'], pred),
            'precision_0':precision_score(test['Label'], pred, pos_label =0),
            'f1':f1_score(test['Label'], pred),
            'f1_0':f1_score(test['Label'], pred,pos_label =0)
      }
    else:
      res = {"accuracy":accuracy_score(test['Label'], pred),
          "mcc" : matthews_corrcoef(test['Label'], pred),
          "recall":recall_score(test['Label'], pred, average ='weighted'),
          "recall_0": 0,
          'precision':precision_score(test['Label'], pred,average ='weighted'),
          'precision_0':0,
          'f1':f1_score(test['Label'], pred,average ='weighted'),
          'f1_0':0
    }

  return res,pred,test['Label']


def main(args):
    feature = args.feature

    train_data = pd.read_csv(pathlib.Path(f"{args.datapath}"))
    test_data = pd.read_csv(pathlib.Path(f"{args.test_set_path}"))

    train_data = train_data.dropna()
    test_data = test_data.dropna()

    if "AdvText" in train_data.columns:
      train_data = train_data.drop(columns = ["Text"])
      train_data = train_data.rename(columns ={"AdvText":"Text"})

    if "AdvText" in test_data.columns:
      test_data = test_data.drop(columns = ["Text"])
      test_data = test_data.rename(columns ={"AdvText":"Text"})

    idx2Label = {0:'cc', 1:'cd'}
    report, pred, real_trg = run_baseline(train_data,test_data, model_name=args.model, feature = args.feature,dataset=args.dataset_name, idx2Label= idx2Label, save_model_to=args.path_to_model_save, report = True)

    #print(f'{args.path_to_results_dir}/predictions.json')
    save_as_json(pred, list(range(len(pred))), f'{args.path_to_results_dir}/predictions.json')

    columns = ['Model', "Features", "Accuracy","MCC","1-Recall","0-Recall","1-Precision","0-Precision","1-F1","0-F1"] 

    df = pd.DataFrame([[args.model,feature,report['accuracy'],report['mcc'],report['recall'],report['recall_0'],report['precision'],report['precision_0'], report['f1'],report['f1_0']]], columns = columns)
    df.to_csv(f"{args.path_to_results_dir}/{args.dataset_name}_{args.model}_{feature}_metrics.csv",index=False,mode='w+' )

    dall = {}
    d = vars(args)
    dall.update(d)
    with open(f"{args.path_to_model_save}/{args.dataset_name}_{args.model}_{feature}_config.json", 'w+') as fp:
        json.dump(dall, fp)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
