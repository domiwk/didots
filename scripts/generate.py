import os, argparse,sys
import numpy as np
import torch
import pandas as pd
import json

from tqdm import tqdm
from pprint import pprint
import math
from Levenshtein import setratio,seqratio
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from peft import LoraModel, LoraConfig,PeftModelForSeq2SeqLM, get_peft_config,BOFTConfig, IA3Config,get_peft_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,PegasusForConditionalGeneration,BartForConditionalGeneration
from safetensors.torch import load_model, save_model,load_file

parser = argparse.ArgumentParser()

parser.add_argument('-d_n','--dataset_name',type=str, default= "ADReSS",help='Name of dataset.')
parser.add_argument('-d_p','--dataset_path', type=str,default= f"./datasets/ADReSS/ADReSS_test_original_sent_const.csv",help='Path_to_dataset.')
parser.add_argument('-d_t','--dataset_type', type=str,default = "test",help='Path_to_dataset.')
parser.add_argument('-m_n','--model_name', type=str,default = 'facebook/bart-base',help='Model name')
parser.add_argument('-m_p','--model_path',type=str, default ="/experiments/DiDOTS/ADV_AD/PAR3_mistral_ZS_gt_LORA/Models/latest",help='Path of model to load')
parser.add_argument('-c_p','--cache_path', type=str,default = "/experiments/DiDOTS/ADV_AD/PAR3_mistral_ZS_gt_LORA/Models/latest",help='Path_to_cache.')
parser.add_argument('-s_d','--save_dir', type=str, default= f"/experiments//DiDOTS/mistral_gt_LORA/Results/ADReSS/",help='Path where to save results.')
parser.add_argument('-c','--column', type=str, default="Text",help='Target parse column name')
parser.add_argument('-stp','--step', type=str, default="paraphrase",help='Generation stage. Paraphrase.')
parser.add_argument('-t','--temperature', type=str, default = "None",help='Control knob for paraphrasing.')
parser.add_argument('-ts','--temperatures', nargs='+', required=False,default=[1.5],help='List of temperatures for paraphrasing.')
parser.add_argument('-bydoc','--by_document', type=str, default = "False",help='Whether or not Generate paraphrase on the whole document.')
parser.add_argument('--base_model', type=str, default = None,help='Whether or not Generate paraphrase on the whole document.')
parser.add_argument('--seed', type=int, default = 66,help='Seed to fix.')
parser.add_argument('--batch_size', type=int, default =32,help='Seed to fix.')

def model_inference(model, tokenizer, sents, num_beams, torch_device, sampling_t = None):
    model.eval()
    sent_txt = sents['Text'].to_list()

    sents = sent_txt

    batch = tokenizer(sents,truncation=True, padding='longest',max_length=256, return_tensors="pt").to(torch_device)

    if sampling_t is None:
        outputs = model.generate(**batch,max_length=256,num_beams=num_beams, num_return_sequences=1,length_penalty = 2.0)
    else:
        outputs = model.generate(**batch, max_length=256, do_sample=True, temperature =sampling_t,num_return_sequences=1,early_stopping= True, 
                    no_repeat_ngram_size = 2,length_penalty = 2.0)
        
    tgt_text = tokenizer.batch_decode(outputs, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    tgt_text = [t.split("<sep>")[-1] for t in tgt_text]
    
    return tgt_text

def batch_sentence_paraphrase(df, model,tokenizer,torch_device,t_sampling=None,batch_size = 32):
  
  text = df['Text']
  df['Text'] = df['Text'].astype("string")

  n = math.ceil(len(df['Text'])/batch_size)
  batches = np.array_split(df, n)
  print(f"The number of batches is {n}, the batch size is {batch_size} and number of samples is {len(text)}.")
  batched = []

  for batch_id, elements in enumerate(tqdm(batches)):
    if t_sampling:
        elements['AdvText'] = model_inference(model, tokenizer, elements,num_beams = 4, torch_device = torch_device, sampling_t=t_sampling)
    else:
        elements['AdvText'] = model_inference(model, tokenizer, elements,num_beams = 4, torch_device = torch_device, sampling_t=t_sampling)
                                    
    batched.append(elements)

  df_concat = pd.concat(batched, axis=0,ignore_index=True)
  return df_concat

def paraphrase(data_df,model,column,tokenizer,temperature,temperatures,torch_device,batch_size = 32,by_documents= None):
        data_df = data_df.dropna()

        t_sampling = None
        if temperatures:
            t_sampling = temperature    

        if by_documents:
            if 'doc_ID' in data_df.columns:
                column = 'doc_ID'
                intent_columns = ['doc_ID','Label']
                data_df =  data_df.groupby(intent_columns,as_index = False).agg({'Text':''.join})  

            para_df = batch_sentence_paraphrase(df=data_df, model = model, tokenizer = tokenizer, torch_device = torch_device,t_sampling=t_sampling, batch_size = batch_size)
            data_df['AdvText'] = para_df['AdvText']
            temperature = temperature.replace('"','')

            print('Saving doc only')
            data_df.to_csv(f"{save_dir}/{dataset_name}_{dataset_type}_obfuscated_{temperature}.csv",index=False)
    
        else:
            para_df = batch_sentence_paraphrase(df=data_df, model = model, tokenizer = tokenizer, torch_device = torch_device,batch_size = batch_size, t_sampling=t_sampling)

            column = 'doc_ID'
            intent_columns = ['doc_ID','Label']

            doc_df = data_df.groupby(intent_columns, as_index = False).agg({'Text': '.'.join})
            res = para_df.groupby([column], as_index = False).agg({'AdvText': '.'.join})
            doc_df['AdvText'] = res['AdvText']
            doc_df = doc_df.drop(column,axis=1)

            print("==== generate paraphrases ====")
            print('Saving sentences and doc')
            para_df.to_csv(f"{save_dir}/{dataset_name}_{dataset_type}_obfuscated_{temperature}_sentences.csv",index=False)
            doc_df.to_csv(f"{save_dir}/{dataset_name}_{dataset_type}_obfuscated_{temperature}.csv",index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    #pprint(vars(args))
    #print()

    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    torch_device = 'cpu'
    #torch_device = torch.device('mps') if torch.backends.mps.is_built() else torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    
    print(f'Device:{torch_device}')
    print("==== loading models ====")

    #load model
    model_name = args.model_name
    model_path = args.model_path

    if "pt" in model_path:
        model = torch.load(model_path,map_location=torch.device('cpu'))
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        model_name = model_path
        cache_dir = args.cache_path

        print(f"Loading model from {model_path}")

        if args.base_model is None:
            model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')
            tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        if 'LORA' in model_path or 'BOFT' in model_path or 'IA3' in model_path:
            if 'LORA' in model_path:
                peft_config = LoraConfig(
                task_type="SEQ_2_SEQ_LM",
                r=16,
                lora_alpha=32,
                lora_dropout=0.01,
                use_rslora = True
                )
            if 'BOFT' in model_path:
                peft_config = BOFTConfig( boft_block_size=4,
                                boft_dropout=0.1, 
                                bias="none")
                
            if 'IA3' in model_path:
                peft_config = IA3Config(
                    peft_type="IA3",
                    task_type="SEQ_2_SEQ_LM",
                )
            
            model =  get_peft_model(model,peft_config)

            if os.path.exists(f"{model_path}/pytorch_model.bin"):
                state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location=torch.device('cpu'))
                state_dict = {f"model.{k}":v for k,v in state_dict.items()}

                tokenizer.add_tokens(['<','<sep>'])
                model.resize_token_embeddings(len(tokenizer))
                model.load_state_dict(state_dict)

            elif os.path.isfile(f"{model_path}"):
                model = torch.load(model_path,map_location=torch.device('cpu'))

            else:
                model  = BartForConditionalGeneration.from_pretrained(model_path,ignore_mismatched_sizes=True)

    cache_dir = args.cache_path
    model = model.to(torch_device)

    print("==== load data ====")

    dataset_path = args.dataset_path
    data_df = pd.read_csv(dataset_path)

    dataset_name = args.dataset_name
    save_dir = args.save_dir
    dataset_type = args.dataset_type
    temperature = args.temperature
    step = args.step
    column = args.column

    by_documents = True if args.by_document == 'True' else False

    print(f'Torch device {torch_device}')
    print(f'Model name {model_name}')
    print(f'Document level? {by_documents}')

    print(f'Saving to {save_dir}')

    if step =="paraphrase":
        paraphrase(data_df=data_df, model = model, column = column, tokenizer = tokenizer,temperature = temperature, torch_device = torch_device,temperatures=None, batch_size = args.batch_size,by_documents = by_documents)

