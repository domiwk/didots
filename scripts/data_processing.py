
import pandas as pd
import numpy as np
import argparse
import os

from nltk.tree import Tree

import scripts.utils as d_ut

parser = argparse.ArgumentParser(description='Train and evaluate baseline model on datasets.')

parser.add_argument('-dps','--data_paths', type=str,nargs='+', help='List of datasets paths to process')
parser.add_argument('-d_ns','--dataset_names', type=str,nargs='+',default = ["ADReSS"], help='List of dataset names corresponding to datapaths.')
parser.add_argument('-sps','--save_paths', type=str,nargs='+', help='Paths to save folder.')
parser.add_argument('-t','--type', type=str,default = True, help='Type of samples. Obfuscated or not')
parser.add_argument('-tk','--task', type=str,default = "doc_constituency", help='Task to process.')
parser.add_argument('-c','--column', type=str,default = "GeneratedConstituency", help='Task to process.')
parser.add_argument('-tc','--target_column', type=str,default = "doc_constituency", help='Task to process.')

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

def string_comma(string):
    start = 0
    new_string = ''
    while start < len(string):
        if string[start:].find(",") == -1:
            new_string += string[start:]
            break
        else:
            index = string[start:].find(",")
            if string[start - 2] != "(":
                new_string += string[start:start + index]
                new_string += " "
            else:
                new_string = new_string[:start - 1] + ", "
            start = start + index + 1
    return new_string

def clean_tuple_str(tuple_str):
    new_str_ls = []
    if len(tuple_str) == 1:
        new_str_ls.append(tuple_str[0])
    else:
        for i in str(tuple_str).split(", "):
            if i.count("'") == 2:
                new_str_ls.append(i.replace("'", ""))
            elif i.count("'") == 1:
                new_str_ls.append(i.replace("\"", ""))
    str_join = ' '.join(ele for ele in new_str_ls)
    return string_comma(str_join)

def trim_tree_nltk(root, height):
    try:
        root.label()
    except AttributeError:
        return

    if height < 1:
        return
    all_child_state = []
    #     print(root.label())
    all_child_state.append(root.label())

    if len(root) >= 1:
        for child_index in range(len(root)):
            child = root[child_index]
            if trim_tree_nltk(child, height - 1):
                all_child_state.append(trim_tree_nltk(child, height - 1))
    #                 print(all_child_state)
    return all_child_state

def get_syntax_templates(parse_str):

    #parse_str.replace('""','')
    parses = clean_tuple_str(to_tuple(trim_tree_nltk(Tree.fromstring(parse_str), 4)))
    return parses


def get_h4_parse(dataset_path,save_path, column,newcolumn ):
    file_path = dataset_path[0]
    file_name = os.path.basename(file_path).split('.')[0]
    df = pd.read_csv(file_path)

    df[column] = df[column].apply(get_syntax_templates)

    df.to_csv(f"{save_path[0]}/{file_name}_h4.csv", index=False)

    return df


def process_constituency(dataset_paths, dataset_names, save_paths, obfuscated,sent = False):
    if obfuscated == "True":
        type_name = "obfuscated"
    else:
        type_name = "original" 
        
    for idx,data_path in enumerate(dataset_paths):
    
        test_data = pd.read_csv(data_path)
        #test_data = test_data.rename(columns={"Class": "Intent","AdvText": "Text"})

        if sent:
            test_data = d_ut.compute_sent_constituency(test_data)
        else:  
            test_data = d_ut.compute_doc_constituency(test_data)

        if not os.path.exists(save_paths[idx]):
            print(f'{save_paths[idx]} does not exist, creating new file')
        else:
            print("Dir exists")

        #test_data.to_csv(f"{save_paths[idx]}/{dataset_names[idx]}_test_{type_name}_const.csv", index = False)
        test_data.to_csv(f"{save_paths[idx]}", index = False)
            


def split_into_sentences(dataset_paths, dataset_names, save_paths, obfuscated):

    if obfuscated == "True":
        type_name = "obfuscated"
    else:
        type_name = "original" 

    print(type_name)

    for idx,data_path in enumerate(dataset_paths):
        if type_name == "original":
            if dataset_names[idx] == "ADReSS":
                file_path = f"{data_path}/adress_test_full_clean.csv"
                save_path = f"{data_path}/ADReSS_test_original_sent_const.csv"

            elif dataset_names[idx] == "DementiaBank":
                file_path = f"{data_path}/test_dataset.csv"
                save_path = f"{data_path}/DementiaBank_test_original_sent_const.csv"
    
        test_data = pd.read_csv(file_path)
        #test_data = test_data.rename(columns={"Class": "Intent","AdvText": "Text"})

        test_data = d_ut.split_into_sentences_and_constituency(test_data)

        print(test_data)

        print(len(test_data))
    
        if not os.path.exists(save_paths[idx]):
            print(f'{save_paths[idx]} does not exist')
        else:
            print("Dir exists")

        #test_data.to_csv(f"{save_paths[idx]}/{dataset_names[idx]}_test_{type_name}_const.csv", index = False)
        test_data.to_csv(f"{save_path}", index = False)


if __name__ == "__main__":
    args = parser.parse_args()

    #task = args.task
    task = "sent_constituency"

    if task == "doc_constituency":
        process_constituency(dataset_paths = args.data_paths, dataset_names = args.dataset_names, save_paths = args.save_paths, obfuscated = args.type, sent = False  )

    elif task == "sent_constituency":
        process_constituency(dataset_paths = args.data_paths, dataset_names = args.dataset_names, save_paths = args.save_paths, obfuscated = args.type , sent = True )

    elif task == "split_into_sentences":
        split_into_sentences(dataset_paths = args.data_paths, dataset_names = args.dataset_names, save_paths = args.save_paths, obfuscated = args.type  )

    elif task == "trim_parse":
        get_h4_parse(dataset_paths = args.data_paths,save_paths = args.save_paths, column = args.column, target_column = args.target_column )


    else:
        print('No Correct Task given')
    