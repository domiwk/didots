import os
import argparse
import numpy as np
import pandas as pd
import pathlib

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve,accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import matthews_corrcoef

from AD_Dataset import AD_Dataset
from dataset_utils import save_as_json

parser = argparse.ArgumentParser(description='Train and evaluate Transformer model on datasets.')

parser.add_argument('--datapath',action='store', type=str, default="./recipes/classifiers/data/df_train.csv",help='Path to dataset.')
parser.add_argument('--test_set_path', type=str, default="./recipes/Classifiers/data/df_test.csv",help='Path to dataset.')
parser.add_argument('--model_name',action='store', type=str, default = 'Bert',help='Model type (Bert, Roberta)).')
parser.add_argument('--dataset_name',action='store', type=str, default = 'ADReSS',help='Name of dataset.')
parser.add_argument('--path_to_model_save',action='store', type=str, default = './experiments/classifiers/adaptive/Mistral_7B/ADReSS_Bert_Utility_zs_sent/Models',help='Path where to save model.')
parser.add_argument('--path_to_results_dir',action='store', type=str, default = "./experiments/classifiers/adaptive/Mistral_7B/ADReSS_Bert_Utility_zs_sent/Results",help='Path where to save predictions and metrics.')
parser.add_argument('--stage', default = 'train',help='Whether to train or eval.')
parser.add_argument('--seed', default = 42,help='Experiment seed.')
parser.add_argument('--input', default = 'Text',help='Experiment seed.')
parser.add_argument('--epochs', default = 10,type=int,help='Whether to train or eval.')
parser.add_argument('-c_d_p','--is_custom_data_path', type=bool, default=True,help='Whether or not load custom data files to paraphrase')


def evaluate_model(model, val_dl, device = "mps"):
    
    model.to(device)
    model.eval()
    
    pred_trait_labels = []
    real_trait_labels = []
    
    for batch in val_dl:
        input_ids = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)

        trait_labels = batch['labels'].to(device)

        out = model(input_ids, masks)
        traits_logits = out['logits']

        pred_traits = traits_logits.cpu().detach().numpy()
        pred_traits = np.argmax(pred_traits, axis=1).flatten()

        pred_trait_labels.append(pred_traits)
        real_trait_labels.append(trait_labels.cpu().numpy())
        

    pred = np.concatenate(pred_trait_labels, axis=0)
 
    labels = np.concatenate(real_trait_labels, axis=0)

    res = {"accuracy":accuracy_score(labels, pred),
            "mcc":matthews_corrcoef(labels, pred),
            "recall":recall_score(labels, pred),
            "recall_0": recall_score(labels, pred, pos_label =0),
            'precision':precision_score(labels, pred),
            'precision_0':precision_score(labels, pred, pos_label =0),
            'f1':f1_score(labels, pred),
            'f1_0':f1_score(labels, pred,pos_label =0)
      }

    return res, pred, labels

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    res = {"accuracy":accuracy_score(labels, predictions),
            "mcc":matthews_corrcoef(labels, predictions),
            "recall":recall_score(labels, predictions),
            "recall_0": recall_score(labels, predictions, pos_label =0),
            'precision':precision_score(labels, predictions),
            'precision_0':precision_score(labels, predictions, pos_label =0),
            'f1':f1_score(labels, predictions),
            'f1_0':f1_score(labels, predictions,pos_label =0)
      }


    return res

def train(train_data, model, tokenizer, seed = 42, epochs = 10, save_model_to = None):
    trainData = AD_Dataset(tokenizer,train_data, max_length = 256)

    training_args = TrainingArguments(
    output_dir= save_model_to,
    save_total_limit = 1,
    save_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    num_train_epochs = epochs,
    report_to='none', # disable wandb
    eval_strategy = 'no',
    seed = int(seed)
    )

    trainer = Trainer(
        model = model,
        args=training_args,
        train_dataset=trainData,   
        data_collator = trainData.collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer, model

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    print(f'Training with seed {args.seed}')

    pretrained_dict = {'BERT':'bert-base-uncased', 'BERT_large':'bert-large-uncased','RoBERTa':'roberta-base', 'MobileBERT':'google/mobilebert-uncased', 'Electra':'google/electra-base-discriminator'}

    model_name = args.model_name
    input = args.input

    config_name = 'config_1'

    tokenizer = AutoTokenizer.from_pretrained(pretrained_dict[model_name], use_fast=True)
    special_tokens_dict = {'additional_special_tokens': ['...']}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_dict[model_name], num_labels=2)

    train_data = pd.read_csv(pathlib.Path(f"{args.datapath}"))
    test_data = pd.read_csv(pathlib.Path(f"{args.test_set_path}"))

    if input == 'Text':
        if "AdvText" in train_data.columns:
            train_data = train_data.drop(columns = ["Text"])
            train_data = train_data.rename(columns ={"AdvText":"Text"})

        if "AdvText" in test_data.columns:
            test_data = test_data.drop(columns = ["Text"])
            test_data = test_data.rename(columns ={"AdvText":"Text"})

    train_data = train_data.dropna(subset=['Text'])
    test_data =  test_data.dropna(subset=['Text'])

    if args.stage == 'train':
        print(f"Training {args.model_name} on {args.dataset_name} dataset.")
        trainer, model = train(train_data, model, tokenizer, save_model_to=args.path_to_model_save, epochs =args.epochs, seed = args.seed)
    
        test_data_class = AD_Dataset(tokenizer,test_data, max_length = 256)
        test_predictions = trainer.predict(test_data_class)

        test_predictions_argmax = np.argmax(test_predictions[0], axis=1)
        #test_references = np.array(test_data["Label"])

        new_pred = test_predictions_argmax
        report = test_predictions[2]
        keys = [k for k in report.keys()]
        for k in keys:
            new_key = k.replace('test_','')
            report[new_key] = report.pop(k)
        print(report)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.path_to_model_save)
        tokenizer = AutoTokenizer.from_pretrained(args.path_to_model_save)
        special_tokens_dict = {'additional_special_tokens': ['...']}
        tokenizer.add_special_tokens(special_tokens_dict)

        model = model.to(device)
        test_data_class = AD_Dataset(tokenizer,test_data, max_length = 256)
        testDataLoader = DataLoader(dataset=test_data_class, batch_size=8, shuffle=False, collate_fn=test_data_class.collate_fn, num_workers = 2 )
        report, new_pred, real_trg = evaluate_model(model, testDataLoader, device)

    save_as_json(new_pred, list(range(len(new_pred))) , f'{args.path_to_results_dir}/predictions.json')

    columns = ['Model', "Config", "Accuracy","MCC", "1-Recall","0-Recall","1-Precision","0-Precision","1-F1","0-F1"]
    df = pd.DataFrame([[args.model_name, config_name,report['accuracy'],report["mcc"],report['recall'],report['recall_0'],report['precision'],report['precision_0'], report['f1'],report['f1_0']]], columns = columns)
    df.to_csv(f"{args.path_to_results_dir}/{args.dataset_name}_{args.model_name}_{config_name}_metrics.csv", index=False, mode='w+' )

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
