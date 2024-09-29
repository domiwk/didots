import os
import re
import pandas as pd

def extract_first_sentence(text):
    # Define a regular expression pattern to identify the first sentence
    sentence_endings = re.compile(r'([.!?])\s')
    
    # Search for the first occurrence of sentence-ending punctuation followed by a space
    match = sentence_endings.search(text)
    
    # If a match is found, return the text up to and including the matched punctuation
    if match:
        return text[:match.end()]
    
    # If no match is found, return the entire text (it's a single sentence)
    return text.rstrip()

def clean_generated_text(text, system, setting):
    text = text.replace('"','').rstrip()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^)]*\]', '', text)
    text = text.replace("*","")
    text = text.replace("\n","")
    adv_text = text
    if text!='':
        if system == 'phi3:instruct':
            if setting == 'ZS':
                if 'ealthy:' in text:
                    adv_text = text.split("ealthy:")[-1].split('.')[0]
                else:
                    adv_text = text.split('.')[0]
            
            if setting =='FS':
                adv_text = text.split("ealthy:")[-1].split('.')[0]

            if setting =='KB':
                if 'utput sentence:' in text:
                    adv_text = text.split('utput sentence:')[1].split('.')[0]
                elif 'utput:' in text:
                    adv_text = text.split('utput:')[1].split('.')[0]
                
            if ';' in adv_text:
                adv_text = adv_text.split(';')[0]

        if system =='mistral:instruct':
            if setting =='ZS':
                if text.strip().split(" ")[0] == "Dementia:":
                    if 'ealthy:' in text:
                        adv_text = text.split('ealthy:')[1]
                        adv_text = extract_first_sentence(adv_text)
                    else:
                        adv_text = text.strip().split('ementia:')[1]
                        adv_text = extract_first_sentence(adv_text)

            elif setting == 'FS':
                #print("setting is fs")
                if text.strip().split(" ")[0] == "Dementia:":
                    if 'ealthy:' in text:
                        #print("healthy in text")
                        adv_text = text.split('ealthy:')[1].split('.')[0]
                        #print(adv_text)
                    else:
                        adv_text = text.split('ementia:')[1].split('.')[0]

                elif 'ealthy:' in text:
                    adv_text = text.split("ealthy:")[-1].split('.')[0]
            else:
                adv_text = text.split('.')[0]

        if system == 'gemma:2b-instruct':
            if setting == 'FS' or setting == 'ZS':
                    adv_text  = ''

            if setting =='KB':
                if'ewrite:' in text:
                    adv_text = text.split('ewrite:')[1].split('.')[0]
                elif 'utput sentence:' in text:
                    adv_text = text.split('utput sentence:')[1].split('.')[0]
                elif 'ewritten sentence:' in text:
                    adv_text = text.split('ewritten sentence:')[1].split('.')[0]
                elif 'utput:' in text:
                    adv_text = text.split('utput:')[1].split('.')[0]
                elif 'ource:' in text:
                    adv_text = text.split('ource:')[1].split('.')[0]


        if system == 'llama3:instruct':
            if setting == 'KB' or setting == 'FS':
                if 'ealthy:' in text:
                        adv_text = text.split('ealthy:')[1].split('.')[0]
                elif 'ource:' in text:
                        adv_text = text.split('ource:')[1].split('.')[0]

            if setting == 'ZS' or setting == 'FS':
                adv_text  = ''

            if setting == 'KB':
                if 'utput:' in text:
                        adv_text = text.split('utput:')[1].split('.')[0]
        
        adv_text = extract_first_sentence(adv_text)
        try:
            if adv_text[0] == "'" and adv_text[-1]!="'":
                adv_text = adv_text+"'"
        except:
            pass
        
    return adv_text

def clean_generated_outputs(datasets,data_set,settings,systems,results_dir):
    for dataset in datasets:
        print(f"Cleaning for {dataset}.")
        for setting in settings:
            for system in systems:
                for set in data_set:
                    if os.path.exists(f"{results_dir[system]}/{dataset}/{dataset}_{set}_obfuscated_{setting}_sentences_raw.csv"):
                        df = pd.read_csv(f"{results_dir[system]}/{dataset}/{dataset}_{set}_obfuscated_{setting}_sentences_raw.csv")
                    else:
                        df = pd.read_csv(f"{results_dir[system]}/{dataset}/{dataset}_{set}_obfuscated_{setting}_sentences.csv")
                    df = df.fillna('')
                    df.to_csv(f"{results_dir[system]}/{dataset}/{dataset}_{set}_obfuscated_{setting}_sentences_raw.csv", index = False)

                    df['AdvText'] = df['AdvText'].apply(lambda x : clean_generated_text(x,system,setting))
                    df.to_csv(f"{results_dir[system]}/{dataset}/{dataset}_{set}_obfuscated_{setting}_sentences.csv", index = False)
        print(f"Cleaning for {dataset} done.")