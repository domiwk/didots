import os
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from typing import Any, List, Union
import random

import Levenshtein
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from jiwer import transforms as tr
from jiwer.transformations import wer_default
from jiwer.process import process_words, process_characters,_word2char,_apply_transform
from jiwer.measures import _deprecate_truth
import textdescriptives as td

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lmppl
from transformers import BartTokenizer, BartConfig
from sentence_transformers import SentenceTransformer,util
import evaluate

from parabart import ParaBart

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tqdm.pandas()

rank  = random.randint(0,25)
rouge = evaluate.load("rouge",experiment_id = rank)
meteor = evaluate.load('meteor',experiment_id = rank)
bleu = evaluate.load("bleu",experiment_id = rank)

def wer(
    reference: Union[str, List[str]] = None,
    hypothesis: Union[str, List[str]] = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    truth: Union[str, List[str]] = None,
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = None,
) -> float:
    
    (
        reference,
        hypothesis,
        reference_transform,
        hypothesis_transform,
    ) = _deprecate_truth(
        reference=reference,
        hypothesis=hypothesis,
        truth=truth,
        reference_transform=reference_transform,
        truth_transform=truth_transform,
        hypothesis_transform=hypothesis_transform,
    )

    output = process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    )
    return output.wer,output.substitutions, output.insertions, output.deletions
def get_formality(samples,model,tokenizer):
    batch_size = 32
    text = samples

    n = math.ceil(len(text)/batch_size)
    batches = np.array_split(text, n)

    print(f"The batch size is {batch_size} and number of samples is {len(text)}.")
    print(f"The number of batches is {n}")

    batched = []
    for batch_id, elements in enumerate(tqdm(batches)):
        sents = elements.to_list()
        batch = tokenizer(sents,truncation=True, padding='longest',max_length=model.config.max_length, return_tensors="pt").to("cpu")
        out = model(**batch)['logits']
        pred = torch.softmax(out, dim = 1)
        formality = pred[:,1].detach().numpy()
        batched.extend(formality)

    return batched

def load_parabart_model(path_prefix):
    cache_path = f"./models/ParaBART"
    torch_device = 'cuda' if torch.cuda.is_available() else  'mps' if torch.backends.mps.is_available() else 'cpu'
    #torch_device = "mps"
    model_path = f"./models/ParaBART/model.pt"

    config = BartConfig.from_pretrained('facebook/bart-base', cache_dir=cache_path)
    model = ParaBart(config)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir=cache_path)
    model.load_state_dict(torch.load(model_path, map_location=torch_device))

    model = model.to(torch_device)

    config.word_dropout = 0.2
    config.max_sent_len = 40
    config.max_synt_len = 160

    return model,tokenizer

def load_formality_model():
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker")
    return model, tokenizer

def build_embeddings(model, tokenizer, sents, name):
    torch_device = 'cuda' if torch.cuda.is_available() else  'mps' if torch.backends.mps.is_available() else 'cpu'
    batch_size = 32
    model.eval()
    embeddings = torch.ones((len(sents), model.config.d_model))

    n = math.ceil(len(sents)/batch_size)
    batches = np.array_split(sents, n)

    #print(f"The batch size is {batch_size} and number of samples is {len(sents)}.")
    #print(f"The number of batches is {n}")
    batched = []

    embeddings = []
    for batch_id, elements in enumerate(batches):
        elements = elements.tolist()
        elements = [str(e) for e in elements]
        sent_inputs = tokenizer(elements, return_tensors="pt", padding='longest',truncation= True, max_length = 1024)
        sent_token_ids = sent_inputs['input_ids']
        sent_embed = model.encoder.embed(sent_token_ids.to(torch_device)) 
        embeddings.extend(sent_embed.squeeze(1).detach().cpu().clone())
    
    return torch.stack(embeddings)

def get_similarity_mean(train_data, aug_data, model):
  sentences1 = train_data.values
  sentences2 = aug_data.values
  embeddings1 = model.encode(sentences1, convert_to_tensor=True)
  embeddings2 = model.encode(sentences2, convert_to_tensor=True)

  cosine_scores = util.cos_sim(embeddings1, embeddings2)
  cos_list=[]

  #Output the pairs with their score
  for i in range(len(sentences1)):
      #print("{} /t/t {} /t/t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
      cos_list.append(cosine_scores[i][i].cpu())

  return cos_list,np.mean(cos_list),np.std(cos_list)

def compute_set_ratio(og_sample,obf_sample):
    tokens1 = [word for word in word_tokenize(og_sample.lower()) if word.isalpha()]
    tokens2 = [word for word in word_tokenize(obf_sample.lower()) if word.isalpha()]

    stop_words = set(stopwords.words('english'))
    tokens1 = [word for word in tokens1 if word not in stop_words]
    tokens2 = [word for word in tokens2 if word not in stop_words]

    set_ratio = Levenshtein.setratio(tokens1, tokens2)
    return set_ratio

def get_metric_scores(obf_samples, og_samples, metric , simi_model = None, tokenizer = None,formality_model = None, formality_tokenizer = None , sentence_const = None):
    obf_texts = obf_samples
    og_texts = og_samples

    metrics_scores = []
    #names = []
    metrics = []

    if metric == 'SUB/ADD/DEL':
        sub_list = []
        add_list = []
        del_list = []

        if sentence_const is None:
            texts_mt = obf_texts
            texts_mt_og = og_texts
        else:
            sentence_const['AdvText'] = ["" if x is None else x for x in sentence_const['AdvText']]
            sentence_const_tmp= sentence_const.dropna()
            texts_mt = sentence_const_tmp['AdvText']
        for para,og in zip(texts_mt,texts_mt_og):    
            ref_transformed = _apply_transform(
                    [og], wer_default, is_reference=True
                )
            hyp_transformed = _apply_transform(
                    [para], wer_default, is_reference=False
                )
            ref_as_chars, hyp_as_chars = _word2char(ref_transformed, hyp_transformed)    
            total_len_words = len(ref_as_chars[0])
            wer_score,substitutions, insertions,deletions  = wer(hypothesis = para,reference = og)
            sub_list.append(substitutions/total_len_words)
            add_list.append(insertions/total_len_words)
            del_list.append(deletions/total_len_words)
        return np.mean(sub_list), np.mean(add_list),np.mean(del_list)
    
    if metric == "Semantics" :
        # load models
        sent_t_simi_model = SentenceTransformer('all-mpnet-base-v2')
        scores, mean, std = get_similarity_mean(og_texts,obf_texts,sent_t_simi_model)
        print(f"Semantics score: {mean}")
        return scores,mean,std
        #names.extend(['Similarity Mean',"Simialrity SD"])
        
    if metric == 'Formality':
        formality_scores = get_formality(obf_texts,formality_model, formality_tokenizer)
        return formality_scores, np.mean(formality_scores), np.std(formality_scores)

    if metric == "Lex_div":
        og_text_lex = og_texts.reset_index().drop(columns = ['index'])['Text'].to_list()
        obf_samples_lex = obf_samples.reset_index().drop(columns = ['index'])['AdvText'].to_list()

        lex_scores = [compute_set_ratio(og_text_lex[idx],x) for idx, x in enumerate(obf_samples_lex)]
        return lex_scores, np.mean(lex_scores), np.std(lex_scores)

    if metric == "ParaBART_Semantics":
        
        parabart_emb1 = build_embeddings(simi_model, tokenizer,og_texts , "sem_source")
        parabart_emb2 = build_embeddings(simi_model, tokenizer,obf_texts , "sem_para")
        cosine_scores_parabart = util.cos_sim(parabart_emb1, parabart_emb2)
        parabart_dist=[]
        #print('Computing Mean Similarity')
        for i in tqdm(range(len(og_texts))):
            parabart_dist.append(cosine_scores_parabart[i][i].cpu())

        mean, std = np.mean(parabart_dist),np.std(parabart_dist)
        print(f"ParaBart Semantics Doc score {mean}")
        #metrics_scores.extend([mean])
        return parabart_dist, mean,std
        #names.extend(['Similarity Mean',"Simialrity SD"])

    if metric in ['Syllables','Sentence_Length']: 
        #meteor_scorer = ev.load("meteor")
        meteors_total = []

        if sentence_const is None:
            texts_mt = obf_texts
            texts_mt_og = og_texts

        else:
            sentence_const['AdvText'] = ["" if x is None else x for x in sentence_const['AdvText']]
            sentence_const_tmp= sentence_const.dropna()

            texts_mt = sentence_const_tmp['AdvText']

        metrics_stats = td.extract_metrics(
        text=texts_mt,
        spacy_model="en_core_web_sm",
        metrics=["readability"],
        )

        metrics_stats_syllables = metrics_stats["syllables_per_token_mean"]
        metrics_stats_sent_length = metrics_stats["sentence_length_mean"]

        if metric == "Syllables":
            print(f"Syllables Mean: {np.mean(np.array(metrics_stats_syllables))}")
            return metrics_stats_syllables,np.mean(np.array(metrics_stats_syllables)),np.std(np.array(metrics_stats_syllables))

        if metric == 'Sentence_Length':
            print(f"Sentence_Length: {np.mean(np.array(metrics_stats_sent_length))}")
            return metrics_stats_sent_length,np.mean(np.array(metrics_stats_sent_length)),np.std(np.array(metrics_stats_sent_length))

    if metric == "METEOR": 
        #meteor_scorer = ev.load("meteor")
        meteors_total = []

        if sentence_const is None:

            texts_mt = obf_texts
            texts_mt_og = og_texts

        else:
            sentence_const['AdvText'] = ["" if x is None else x for x in sentence_const['AdvText']]
            sentence_const_tmp= sentence_const.dropna()

            texts_mt = sentence_const_tmp['AdvText']
            texts_mt_og = sentence_const_tmp['Text']

        meteor_scores = [meteor.compute(predictions = [para], references = [og])['meteor'] for para,og in zip(texts_mt,texts_mt_og)]
        meteor_score = np.mean(meteor_scores)

        print(f"METEOR: {meteor_score}")
        return meteor_scores,meteor_score,0

    if metric == 'Perplexity':
        scorer = lmppl.LM('gpt2')

        print("Computing perplexity")

        if sentence_const is None:
            #obf_texts_sent = [x.split(".") for x in obf_texts.to_list()]
            prlp = scorer.get_perplexity(obf_texts.to_list(),batch=32)
            mean_perplexity_obf =  np.mean(np.array(prlp))

        else:
            sentence_const['AdvText'] = ["" if x is None else x for x in sentence_const['AdvText']]

            sentence_const_tmp= sentence_const.dropna()
            sentence_const_tmp["Perplexity_obf"] = scorer.get_perplexity(sentence_const_tmp['AdvText'].to_list(),batch=32)

            mean_perplexity_obf_sent= sentence_const_tmp["Perplexity_obf"].mean()
            sentence_const_tmp  = sentence_const_tmp.groupby('ID').agg('mean')
            mean_perplexity_obf = np.mean(sentence_const_tmp['Perplexity_obf'].tolist())

        print(f"Perplexity OBF: {mean_perplexity_obf}")
        return prlp, mean_perplexity_obf, np.std(np.array(prlp))

def eval_quality(dataset_name,model_name,samples_path,results_csv_path, metrics, temperatures,path_prefix, level):
    # dataframes with results
    if os.path.isfile(results_csv_path):
        df_all_res = pd.read_csv(results_csv_path)
    else:
        df_all_res = pd.DataFrame(columns=['Model','Dataset','Temperature'])

    #load samples
    results = defaultdict()
    results['Model'] = model_name
    results['Dataset'] = dataset_name

    if "ParaBART_Semantics" in metrics:
        simi_model,tokenizer = load_parabart_model(path_prefix)
    else:
        simi_model, tokenizer = None, None

    if "Formality" in metrics:
        formality_model,formality_tokenizer = load_formality_model()
    else:
        formality_model, formality_tokenizer = None, None

    for t in temperatures:
        print(f'Evaluating Temperature = {t}')

        if level == "sent":
            df_test_reference = pd.read_csv(f'{samples_path}/{dataset_name}/{dataset_name}_test_obfuscated_{t}_sentences.csv')
            df_test_reference = df_test_reference.dropna(subset=['Text'])
            df_test_reference = df_test_reference.fillna("")
            df_test_reference = df_test_reference.loc[df_test_reference['Text'].apply(lambda x: len(x.split(" ")) > 3)]

        obf_samples, og_samples = df_test_reference['AdvText'], df_test_reference['Text']
        obf_samples = obf_samples.fillna('')


        for m in metrics:
            print(f'Evaluating {m}')
        
            if m == 'SUB/ADD/DEL':
                sub_score, add_score, del_score = get_metric_scores(metric = m,obf_samples = obf_samples,og_samples = og_samples,simi_model = simi_model, tokenizer = tokenizer, formality_model= formality_model, formality_tokenizer=formality_tokenizer)

                #save metrics
                results_metric = results.copy()
                results_metric['Temperature'] = [str(t)]
                results_metric['SUB'] = [sub_score]
                results_metric['ADD'] = [add_score]
                results_metric['DEL'] = [del_score]
                df_res = pd.DataFrame(results_metric)

                #update all results df
                if ((df_all_res['Model'] == model_name) & (df_all_res['Temperature'] == str(t))).any():
                    df_all_res.loc[(df_all_res['Model'] == model_name)& (df_all_res['Temperature'] == str(t)), "SUB"] = sub_score
                    df_all_res.loc[(df_all_res['Model'] == model_name)& (df_all_res['Temperature'] == str(t)), 'ADD'] = add_score
                    df_all_res.loc[(df_all_res['Model'] == model_name)& (df_all_res['Temperature'] == str(t)), 'DEL'] = del_score
                else:
                    df_all_res = pd.concat([df_all_res,df_res], ignore_index=True)

                df_all_res.to_csv(results_csv_path,index = False)

            else:
                scores, mean, std = get_metric_scores(metric = m,obf_samples = obf_samples,og_samples = og_samples,simi_model = simi_model, tokenizer = tokenizer, formality_model= formality_model, formality_tokenizer=formality_tokenizer)

                mean_score_cc = np.nanmean([x if df_test_reference['Label'].iloc[idx] == 0 else np.nan for idx,x in enumerate(scores)])
                mean_score_ad = np.nanmean([x if df_test_reference['Label'].iloc[idx] == 1 else np.nan for idx,x in enumerate(scores)])

                #save metrics
                results_metric = results.copy()
                results_metric['Temperature'] = [str(t)]
                results_metric[m] = [mean]
                results_metric[f'{m}_CC'] = [mean_score_cc]
                results_metric[f'{m}_AD'] = [mean_score_ad]
                df_res = pd.DataFrame(results_metric)
                #df_res.to_csv(f"{output_path}/{dataset_name}_quality_{m}_eval.csv", index = False)

                #update all results df
                if ((df_all_res['Model'] == model_name) & (df_all_res['Temperature'] == str(t))).any():
                    df_all_res.loc[(df_all_res['Model'] == model_name)& (df_all_res['Temperature'] == str(t)), m] = mean
                    df_all_res.loc[(df_all_res['Model'] == model_name)& (df_all_res['Temperature'] == str(t)), f'{m}_CC'] = mean_score_cc
                    df_all_res.loc[(df_all_res['Model'] == model_name)& (df_all_res['Temperature'] == str(t)), f'{m}_AD'] = mean_score_ad
                else:
                    df_all_res = pd.concat([df_all_res,df_res], ignore_index=True)

                df_all_res.to_csv(results_csv_path,index = False)
            print(f'Evaluating {m} Done')

    