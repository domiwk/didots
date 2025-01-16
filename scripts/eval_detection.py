import numpy as np
import pandas as pd
import json
import pickle
import os
from pathlib import Path
from tqdm import tqdm

import torch

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from AD_Dataset import AD_Dataset

from collections import defaultdict

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score

from torch.utils.data import DataLoader

import nltk
nltk.download('wordnet', quiet=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tqdm.pandas()

def eval_against_baseline(path_to_model_save, test_data, real_targets, idx2label,report = False):
    model = pickle.load(open(path_to_model_save, 'rb'))

    pred = model.predict(test_data)
    #idx2label={0:'cc',1:"cd"}
    label2idx={v: k for k, v in idx2label.items()}

    preds = [idx2label[enum] for enum in pred]
    try:
        real_trait_labels = [idx2label[enum] for enum in real_targets]
    except:
        real_trait_labels = real_targets
        real_targets = [label2idx[enum] for enum in real_targets]

    if report:
      print(classification_report(real_trait_labels, preds))

    report = classification_report(real_trait_labels, preds,output_dict=True)

    res = {"accuracy":accuracy_score(real_targets, pred),
        "recall":recall_score(real_targets, pred),
        "recall_0": recall_score(real_targets, pred, pos_label =0),
        'precision':precision_score(real_targets, pred),
        'precision_0':precision_score(real_targets, pred, pos_label =0),
        'f1':f1_score(real_targets, pred),
        'f1_0':f1_score(real_targets, pred,pos_label =0)
    }


    return res,pred,real_targets

def evaluate_bert_model(model, val_dl, device = "cpu"):
    
    model.to(device)
    model.eval()
    
    pred_trait_labels = []
    real_trait_labels = []
    logits = []

    for batch in val_dl:
        #batch = tuple(b.to(device) for b in batch)

        input_ids = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)

        trait_labels = batch['labels']
        traits_logits = model(input_ids, masks).logits

        pred_traits = traits_logits.cpu().detach().numpy()
        pred_traits = np.argmax(pred_traits, axis=1).flatten()

        pred_trait_labels.append(pred_traits)
        real_trait_labels.append(trait_labels.cpu().numpy())

        logits.append(traits_logits[0].cpu().detach().numpy()[1])

    pred = np.concatenate(pred_trait_labels, axis=0)
    labels = np.concatenate(real_trait_labels, axis=0)


    res = {"accuracy":accuracy_score(labels, pred),
            "recall":recall_score(labels, pred),
            "recall_0": recall_score(labels, pred, pos_label =0),
            'precision':precision_score(labels, pred),
            'precision_0':precision_score(labels, pred, pos_label =0),
            'f1':f1_score(labels, pred),
            'f1_0':f1_score(labels, pred,pos_label =0)
            #'auc':roc_auc_score(labels, logits)
    }

    return res, pred, labels, logits

def update_results(model_name,df_all_res, clf, dataset,obfuscation, temperature,metrics,level, df_sys_res= None):
    results = defaultdict()
    results['Model'] = model_name
    results['Dataset'] = dataset
    results['Obfuscation'] = obfuscation
    results['Level'] = level
    results['Temperature'] = str(temperature)
    
    for m in metrics:
        results[m] = [metrics[m]]
        results[m] = [metrics[m]]

    df_res = pd.DataFrame(results)
    columns = [('Model', ''),('Dataset', ''),('Obfuscation', ''),('Level', ''),('Temperature', '')] + [(clf, m) for m in metrics]
    df_res.columns = pd.MultiIndex.from_tuples(columns)

    #update sys results df
    if df_sys_res is not None:
        if ((df_sys_res['Model'] == model_name) & (df_sys_res['Temperature'] == str(temperature))).any():
            for m in metrics:
                df_sys_res.loc[(df_sys_res['Model'] == model_name)& (df_sys_res['Temperature'] == str(temperature) ), (clf,m)] = metrics[m]
        else:
            df_sys_res = pd.concat([df_sys_res,df_res], ignore_index=True)

    #update all results df
    if ((df_all_res['Model'] == model_name) & (df_all_res['Temperature'] == str(temperature))).any():
        for m in metrics:
            df_all_res.loc[(df_all_res['Model'] == model_name)& (df_all_res['Temperature'] == str(temperature) ), (clf,m)] = metrics[m]
    else:
        df_all_res = pd.concat([df_all_res,df_res], ignore_index=True)
    
    return df_sys_res, df_all_res

def evaluate_detection(obf_model,obf_model_path,classifiers,dataset, all_results_csv_path, save_csv_path, path_prefix = None, temperatures = [0.5], device='cpu', level = 'sent', lang = None):
    
    #columns = ["Obfuscator",'Model', "Features","Temperature", "Accuracy","1-Recall","0-Recall","1-Precision","0-Precision","1-F1","0-F1","AUC","Dataset"] 
    idx2label = {0:'cc',1:'ad'}

    ### Prepare resulds dataframes ###
    
    #all_results_csv_path = f'{all_results_csv_path}/{dataset}/{dataset}_detection_results.csv'

    print(f"Dataset name: {dataset}")
    if not os.path.isfile(all_results_csv_path):
        df_all_res = pd.DataFrame(columns=[('Model', ''),('Dataset', ''),('Obfuscation', ''),('Temperature', '')])
        df_all_res.columns = pd.MultiIndex.from_tuples([(x,'') if 'Unnamed' in y else (x,y) for x,y in df_all_res.columns ])
    else:
        df_all_res = pd.read_csv(all_results_csv_path,header=[0, 1])
        df_all_res.columns = pd.MultiIndex.from_tuples([(x,'') if 'Unnamed' in y else (x,y) for x,y in df_all_res.columns ])

    if not os.path.isfile(save_csv_path):
        df_sys_res = pd.DataFrame(columns=[('Model', ''),('Dataset', ''),('Obfuscation', ''),('Temperature', '')])
        df_sys_res.columns = pd.MultiIndex.from_tuples([(x,'') if 'Unnamed' in y else (x,y) for x,y in df_sys_res.columns ])
    else:
        df_sys_res = pd.read_csv(save_csv_path,header=[0, 1])
        df_sys_res.columns = pd.MultiIndex.from_tuples([(x,'') if 'Unnamed' in y else (x,y) for x,y in df_sys_res.columns ])

    #Path(f"{save_dir}/{dataset}").mkdir( parents=True, exist_ok=True )
    obf_model_save_path =  Path(f"{obf_model_path}/{dataset}")
    obf_model_save_path.mkdir( parents=True, exist_ok=True )

    ### load samples ###
    if level =="sent": 
        og_samples_sent = pd.read_csv(f"{obf_model_save_path}/{dataset}_test_obfuscated_{temperatures[0]}_sentences.csv")
        og_samples_sent = og_samples_sent.drop(columns=["AdvText"])
        og_samples_sent = og_samples_sent.dropna()
        og_samples_sent = og_samples_sent[og_samples_sent['Text'].apply(lambda x: len(x.split()) > 3)]
        og_samples_sent = og_samples_sent.rename(columns={"Class": "Label"})

        label_columns = ['doc_ID','Label']
        og_samples_doc = og_samples_sent.groupby(label_columns, as_index = False).agg({'Text': '.'.join})

    else:
        if "knobs" in obf_model:
            t_0 = json.dumps(temperatures[0]).replace('"',"")
            t_0 = f'{t_0}'
        else:
            t_0 = temperatures[0]
        og_samples_doc = pd.read_csv(f"{obf_model_save_path}/{dataset}_test_obfuscated_{t_0}.csv")

    obf_samples_sent_dict = defaultdict()
    obf_samples_doc_dict = defaultdict()
    for t in temperatures:
        if level =='sent':
            obf_samples = pd.read_csv(f"{obf_model_save_path}/{dataset}_test_obfuscated_{t}_sentences.csv")
            obf_samples = obf_samples.dropna(subset=['Text'])
            og_samples_sent = og_samples_sent[og_samples_sent['Text'].apply(lambda x: len(x.split()) > 3)]

            obf_samples = obf_samples.drop(columns=["Text"])
            obf_samples = obf_samples.rename(columns={"AdvText": "Text","Class": "Label"})

            obf_samples = obf_samples.fillna('')
            obf_samples_sent_dict[str(t)] = obf_samples

            obf_samples_doc = obf_samples.groupby(label_columns, as_index = False).agg({'Text': '.'.join})
            obf_samples_doc_dict[str(t)] = obf_samples_doc
        
        else:
            if "knobs" in obf_model:
                t = json.dumps(t).replace('"',"")
                t = f'{t}'

            obf_samples_doc = pd.read_csv(f"{obf_model_save_path}/{dataset}_test_obfuscated_{t}.csv")
            obf_samples_doc = obf_samples_doc.dropna(subset=['Text'])

            obf_samples_doc = obf_samples_doc.drop(columns=["Text"])
            obf_samples_doc = obf_samples_doc.rename(columns={"AdvText": "Text","Class": "Label"})

            obf_samples_doc = obf_samples_doc.fillna('')
            obf_samples_doc_dict[str(t)] = obf_samples_doc

    for clf in classifiers:

        if 'SVM' in clf:

            if 'sent' in clf:
                type_suffixe = '_sent'
                og_samples = og_samples_sent
                obf_samples_dict = obf_samples_sent_dict
            else:
                type_suffixe = ''
                og_samples = og_samples_doc
                obf_samples_dict = obf_samples_doc_dict

            if 'BT' in clf:
                lang_suffixe = f'_{lang}'
            else:
                lang_suffixe =''
            
            clf_model_name = f'SVM{lang_suffixe}{type_suffixe}'

            # eval against classifier
            path_to_model_save= Path(f"{path_prefix}/Static/{dataset}/{dataset}_SVM_Utility{lang_suffixe}{type_suffixe}/Models/{dataset}_SVM_TFIDF.pkl")
            vectorizer_path = Path(f"{path_prefix}/Static/{dataset}/{dataset}_SVM_Utility{lang_suffixe}{type_suffixe}/Models/{dataset}_TFIDF_vectorizer.pkl")

            vectorizer = pickle.load(open(vectorizer_path, 'rb'))
            #og samples results
            new_samples = vectorizer.transform(og_samples['Text'].values.astype('U'))
            report, new_pred, real_trg = eval_against_baseline(path_to_model_save, new_samples, og_samples['Label'],idx2label, report = False)

            m_res = {'Accuracy': report['accuracy'],
                        'F1-score': report['f1']}
            
            df_sys_res, df_all_res = update_results(model_name = "Original",df_sys_res = df_sys_res,df_all_res = df_all_res , clf = clf_model_name, dataset = dataset,obfuscation = 'N', temperature = 'None',metrics=m_res,level= level)
            
            for t in temperatures:
                
                if "knobs" in obf_model:
                    t = json.dumps(t).replace('"',"")
                    t = f'{t}'

                obf_samples = obf_samples_dict[str(t)]

                new_samples = vectorizer.transform(obf_samples['Text'].values.astype('U'))
                report, new_pred, real_trg = eval_against_baseline(path_to_model_save, new_samples, obf_samples['Label'],idx2label, report = False)

                m_res = {'Accuracy': report['accuracy'],
                        'F1-score': report['f1']}  

                df_sys_res, df_all_res = update_results(model_name = obf_model,df_sys_res = df_sys_res,df_all_res = df_all_res , clf = clf_model_name, dataset = dataset,obfuscation = 'Y', temperature = t,metrics=m_res,level= level)

        if 'BERT' in clf:

            if 'sent' in clf:
                type_suffixe = '_sent'
                og_samples = og_samples_sent
                obf_samples_dict = obf_samples_sent_dict
            else:
                type_suffixe = ''
                og_samples = og_samples_doc
                obf_samples_dict = obf_samples_doc_dict

            if 'BT' in clf:
                lang_suffixe = f'_{lang}'
            else:
                lang_suffixe =''

            clf_model_name = f'BERT{lang_suffixe}{type_suffixe}'
        
            device = torch.device('mps') if torch.backends.mps.is_built() else torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
            print(f'Using device: {device}')

            folder = f"{path_prefix}/Static/{dataset}/{dataset}_Bert_Utility{lang_suffixe}{type_suffixe}/Models/"
            sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
            sub_folders.sort()

            #path_to_predictions_og=f"{path_prefix}/Static/{dataset}/{dataset}_Bert_utility/Results/predictions.json"
            #df_og = pd.read_csv(f"{path_prefix}/Static/{dataset}/{dataset}_Bert_utility/Results/{dataset}_Bert_config_1_metrics.csv")
            model_save = f"{path_prefix}/Static/{dataset}/{dataset}_Bert_Utility{lang_suffixe}{type_suffixe}/Models/{sub_folders[0]}"
                
            model = AutoModelForSequenceClassification.from_pretrained(model_save)
            tokenizer = AutoTokenizer.from_pretrained(model_save)
            special_tokens_dict = {'additional_special_tokens': ['...']}
            tokenizer.add_special_tokens(special_tokens_dict)

            model = model.to(device)

            test_data = AD_Dataset(tokenizer,og_samples, max_length = 60)
            test_dataloader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, collate_fn=test_data.collate_fn, num_workers = 2 )
            report, preds_og, real_trg, logits = evaluate_bert_model(model, test_dataloader, device)
            
            m_res = {'Accuracy': report['accuracy'],
                        'F1-score': report['f1']}
        

            df_sys_res, df_all_res = update_results(model_name = "Original",df_sys_res = df_sys_res,df_all_res = df_all_res , clf = clf_model_name, dataset = dataset,obfuscation = 'N', temperature = 'None',metrics=m_res,level= level)
                                
            for t in temperatures:
                if "knobs" in obf_model:
                    t = json.dumps(t).replace('"',"")
                    t = f'{t}'

                obf_samples = obf_samples_dict[str(t)]
                test_data = AD_Dataset(tokenizer,obf_samples, max_length = 60)
                test_dataloader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, collate_fn=test_data.collate_fn, num_workers = 2 )
                report, new_pred, real_trg, logits = evaluate_bert_model(model, test_dataloader)

                m_res = {'Accuracy': report['accuracy'],
                        'F1-score': report['f1']}

                df_sys_res, df_all_res = update_results(model_name = obf_model,df_sys_res = df_sys_res,df_all_res = df_all_res , clf = clf_model_name, dataset = dataset,obfuscation = 'Y', temperature = str(t),metrics=m_res,level= level)

    df_sys_res.to_csv(save_csv_path, index = False)
    df_all_res.to_csv(all_results_csv_path, index = False)

   