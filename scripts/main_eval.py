import pathlib
import os
import json
import shutil
import argparse

import pandas as pd
import numpy as np

from eval_detection import evaluate_detection, update_results
from eval_quality import eval_quality

def parse_args():
    parser = argparse.ArgumentParser(description="Description of your script.")
    
    # Directories
    parser.add_argument('--os', type=str, default="HPC",
                        help="Path to the directory for processing scripts.")
    parser.add_argument('--path_prefix', type=str, default=".",
                        help="Global directory")
    parser.add_argument('--project_dir', type=str, default=".",
                        help="Path to the directory for project")
    parser.add_argument('--python_handle', type=str, default="python",
                        help="python handle")
    parser.add_argument('--device', type=str, default="mps",
                        help="device to perfom operation on.")
    
    # Datasets
    parser.add_argument('--datasets_paths', type=str, default=None,
                        help="Dict with source datasets and their paths")

    parser.add_argument('--datasets', nargs='+', default=['MockUp'],
                        help="List of datasets to use.")
    # Systems
    parser.add_argument('--system_paths', type=str, default=None,
                        help="Dict with systems paths and configs")
    parser.add_argument('--systems', nargs='+',default=['DiDOTS_BART_MISTRAL_KB'],help="List of systems to use.")
    # static classifiers
    parser.add_argument('--static', action='store_true', default=False,
                        help="Enable static parameter.")
    parser.add_argument('--static_classifiers', nargs='+', default=['BERT','SVM'],
                        help="List of classifiers for static training.")

    # Sampling
    parser.add_argument('--sample', action='store_true', default=True,
                        help="Enable sampling from models.")
    # Adaptive Training
    parser.add_argument('--adaptive', action='store_true', default=False,
                        help="Enable adaptive training.")
    parser.add_argument('--ada_classifiers', nargs='+', default=['SVM_sent',"BERT_sent"],
                        help="List of classifiers for adaptive training.")
    parser.add_argument('--doc_type', nargs='+', default=['sent'],
                        help="List of document types for adaptive training.")

    # Detection Evaluation
    parser.add_argument('--evaluate_detection', action='store_true', default=False,
                        help="Enable detection evaluation.")
    parser.add_argument('--classifiers', nargs='+', default=['BERT_sent','SVM_sent'], #'BERT_sent','SVM_sent'
                        help="List of classifiers for detection evaluation.")

    # Sample Quality Evaluation
    parser.add_argument('--evaluate_quality', action='store_true', default=False,
                        help="Enable sample quality evaluation.")
    parser.add_argument('--quality_metrics', nargs='+', default=['Semantics', 'ParaBART_Semantics','Formality', 'METEOR', 'Lex_div','Perplexity', 'SUB/ADD/DEL'],
                        help="List of quality metrics for evaluation.")
    parser.add_argument('--eval_all_dir', type=str, default="./experiments/evaluations/all",
                        help="Directory for evaluation results.")
    
    return parser.parse_args()

# Function to train a static or adaptive classifier
def train_classifier(model_name, script_name, scripts_dir, dataset_name, train_data_path, test_data_path, task, recipe, project_dir, python_handle, seed=None, epochs=None):
    print(f'Training {model_name} on {dataset_name}')

    project_name = f"{dataset_name}_{model_name}_{task}_sent"
    experiment_dir = f"{project_dir}/experiments/{recipe}/{project_name}"
    models_dir = f"{experiment_dir}/models"
    results_dir = f"{experiment_dir}/results"
    
    # Create directories
    pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    if model_name == 'BERT':
        config_dir = f"{experiment_dir}/models/configs/config_1.json"
        print('train BERT')
        os.system(
            f"{python_handle} {scripts_dir}/{script_name} --stage 'train' "
            f"--datapath {train_data_path} --test_set_path {test_data_path} "
            f"--model_name {model_name} --dataset_name {dataset_name} "
            f"--path_to_model_save {models_dir} "
            f"--path_to_results_dir {results_dir} -c_d_p 'True' "
            f"--epochs {epochs} --seed {seed}"
        )
    else:
        os.system(
            f"{python_handle} {scripts_dir}/{script_name} "
            f"--datapath {train_data_path} --test_set_path {test_data_path} "
            f"--model {model_name} --dataset_name {dataset_name} "
            f"--path_to_model_save {models_dir} --path_to_results_dir {results_dir} "
            f"-c_d_p 'True'"
        )

# Function for static training
def run_static_training():
    ######## STATIC TRAINING for CLASSIFIERS FOR EVALUATION ########
    if not STATIC:
        return
    
    print('Running static training.')
    print("---------------------------")

    task = 'Utility'
    scripts_dir = f"{PROJECT_DIR}/scripts"
    for d in DATASETS:
        print(f'Training Static classifiers on {d}')
        recipe = f"classifiers/static/{d}"
        train_data_path = f'{DATASETS_PATHS[d]}/{d}_train_original_sentences.csv'
        test_data_path = f'{DATASETS_PATHS[d]}/{d}_test_original_sentences.csv'

        if "BERT_sent" in STATIC_CLASSIFIERS:
            train_classifier('BERT', 'train_bert.py', scripts_dir, d, train_data_path, test_data_path, task, recipe, PROJECT_DIR, python_handle, SEED, epochs=8)

        if "SVM_sent" in STATIC_CLASSIFIERS:
            train_classifier('SVM', 'run_baselines.py', scripts_dir, d, train_data_path, test_data_path, task, recipe, PROJECT_DIR, python_handle)

    print("---------------------------")
    print()

# Function for adaptive training
def run_adaptive_training(args, system):
    if not args.adaptive:
        return

    print('Running adaptive training.')
    print("---------------------------")

    task = 'Utility'
    obfuscator = system  # Assuming only one system
    recipe = f"classifiers/adaptive/{obfuscator}"
    scripts_dir = f"{PROJECT_DIR}/scripts"

    #temp folder
    pathlib.Path(f"{PROJECT_DIR}/experiments/{recipe}/data").mkdir(parents=True, exist_ok=True)

    for d in args.datasets:
        eval_all_dir = f"{EVAL_ALL_DIR}/{d}"
        pathlib.Path(eval_all_dir).mkdir(parents=True, exist_ok=True)
        all_ada_results_csv_path = f'{EVAL_ALL_DIR}/{d}/{d}_detection_adaptive_results.csv'
        train_data_path = f"{DATASETS_PATHS[d]}/{d}_train_original_sentences.csv"
        test_data_path = f"{DATASETS_PATHS[d]}/{d}_test_original_sentences.csv"

        for t in SYSTEMS_PATHS[system]['temperatures']:
            train_data_path_obf = f"{SYSTEMS_PATHS[system]['project_dir']}/{d}/{d}_train_obfuscated_{t}_sentences.csv"
            test_data_path_obf = f"{SYSTEMS_PATHS[system]['project_dir']}/{d}/{d}_test_obfuscated_{t}_sentences.csv"
            train_obf_df = pd.read_csv(train_data_path_obf).dropna(subset=['Text'])
            test_obf_df = pd.read_csv(test_data_path_obf).dropna(subset=['Text'])

            # Append obfuscated data to original
            df_train = pd.read_csv(train_data_path)
            df_train = df_train.loc[df_train['Text'].apply(lambda x: len(x.split(" ")) > 3)]
            df_train = df_train.dropna(subset=['Text'])
            df_train = df_train.append(train_obf_df, ignore_index=True)

            df_train.to_csv(f"{PROJECT_DIR}/experiments/{recipe}/data/df_train.csv", index=False)
            test_obf_df.to_csv(f"{PROJECT_DIR}/experiments/{recipe}/data/df_test.csv", index=False)

            train_data_path = f"{PROJECT_DIR}/experiments/{recipe}/data/df_train.csv"
            test_data_path = f"{PROJECT_DIR}/experiments/{recipe}/data/df_test.csv"

            if "BERT_sent" in ADA_CLASSIFIERS:
                train_classifier('BERT', 'train_bert.py', scripts_dir, d, train_data_path, test_data_path, task, f"{recipe}/{d}", PROJECT_DIR, python_handle, SEED, epochs=8)

            if "SVM_sent" in ADA_CLASSIFIERS:
                train_classifier('SVM', 'run_baselines.py', scripts_dir, d, train_data_path, test_data_path, task, f"{recipe}/{d}", PROJECT_DIR, python_handle)

        # Update results
        if not os.path.isfile(all_ada_results_csv_path):
            all_res = pd.DataFrame(columns=[('Model', ''),('Dataset', ''),('Obfuscation', ''),('Level', ''),('Temperature', '')])
            all_res.columns = pd.MultiIndex.from_tuples([(x,'') if 'Unnamed' in y else (x,y) for x,y in all_res.columns ])
        else:
            all_res = pd.read_csv(all_ada_results_csv_path,header=[0, 1])
            all_res.columns = pd.MultiIndex.from_tuples([(x,'') if 'Unnamed' in y else (x,y) for x,y in all_res.columns ])

        # SVM
        if 'SVM_sent' in ADA_CLASSIFIERS:
            clf = 'ADA_SVM_sent'

            project_name = f"{d}/{d}_SVM_{task}_sent"
            experiment_dir = f"{PROJECT_DIR}/experiments/{recipe}/{project_name}"
            models_dir = f"{experiment_dir}/models"
            results_dir = f"{experiment_dir}/results"

            #ADA_RESULTS_DIR_SVM_SENT = f"{project_dir}/experiments/{recipe}/{project_name}"
            ada_res = pd.read_csv(f'{results_dir}/{d}_SVM_TFIDF_metrics.csv')
            metrics = {'Accuracy':ada_res['Accuracy'].iloc[0],
                        'F1-score':ada_res['1-F1'].iloc[0]}  
            
            _,all_res = update_results(model_name =obfuscator, df_all_res = all_res,clf =clf,dataset = d, obfuscation ='Y', temperature= t,metrics = metrics,level = level, df_sys_res= None)
            all_res.to_csv(all_ada_results_csv_path, index = False)
            
            #rm files to save space
            shutil.rmtree(models_dir, ignore_errors=True)

        # BERT  
        if 'BERT_sent' in ADA_CLASSIFIERS:
            clf = 'ADA_BERT_sent' 
            project_name = f"{d}/{d}_BERT_{task}_sent"
            experiment_dir = f"{PROJECT_DIR}/experiments/{recipe}/{project_name}"
            models_dir = f"{experiment_dir}/models"
            results_dir = f"{experiment_dir}/results"

            ada_res = pd.read_csv(f'{results_dir}/{d}_Bert_config_1_metrics.csv')
            metrics = {'Accuracy':ada_res['Accuracy'].iloc[0],
                        'F1-score':ada_res['1-F1'].iloc[0]}  
            _,all_res = update_results(model_name =obfuscator,df_all_res = all_res, clf =clf,dataset = d, obfuscation ='Y', temperature= t,metrics = metrics,level = level, df_sys_res= None)
            all_res.to_csv(all_ada_results_csv_path, index = False)
            
            #rm files to save space
            shutil.rmtree(models_dir, ignore_errors=True)

    print('Training of Adaptive models done')       
    print("---------------------------")
    print()

def run_sampling(system, system_dir, DATASETS, SYSTEMS_PATHS, temps, scripts_dir, python_handle):
    if not args.sample:
        return
    
    print('Running Sampling')
    print("---------------------------")
    
    for d in DATASETS:
        output_path = f'{system_dir}/{d}'
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        
        for t in temps:
            for dataset_type in ['test', 'train']:
                data_path = f'{DATASETS_PATHS[d]}/{d}_{dataset_type}_original_sentences.csv'
                by_document = SYSTEMS_PATHS[system]['by_document']
                compute_const = SYSTEMS_PATHS[system]['compute_const']
                base_model = SYSTEMS_PATHS[system].get('base_model', '')
                
                run_system_script(system,d, data_path, output_path, dataset_type, t, by_document, compute_const, base_model, scripts_dir, python_handle)
    
def run_system_script(system, dataset_name, data_path, output_path, dataset_type, temp, by_document, compute_const, base_model, scripts_dir, python_handle):
    if 'DiDOTS' in system:
        model_ckp = SYSTEMS_PATHS[system]['model_ckp']
        ckt = SYSTEMS_PATHS[system]['ckt']
        os.system(
            f"{python_handle} {scripts_dir}/generate.py -d_n {dataset_name} -d_p {data_path} -d_t {dataset_type} "
            f"-m_p {model_ckp} -c_p {model_ckp} -s_d {output_path} -c 'Text' -t '{temp}' "
            f"--step 'paraphrase' -bydoc {by_document} --base_model {base_model}"
        )
    if compute_const:
        process_constituency(system, dataset_type, temp, output_path)

def process_constituency(dataset_type, temp, system_dir):
    obfuscated_samples_path = f"{system_dir}/{dataset_type}_obfuscated_{temp}_sentences.csv"
    save_path = f"{system_dir}/{dataset_type}_obfuscated_{temp}_sentences_const.csv"
    os.system(f"{python_handle} ./data_processing.py -dps {obfuscated_samples_path} -sps {save_path} -t true -tk 'sent_constituency'")

if __name__ == '__main__':
    args = parse_args()
    OS = args.os
    print(OS)
    SEED = 666
    PROJECT_DIR= args.project_dir
    PATH_PREFIX = args.path_prefix
    DEVICE = args.device
    python_handle = args.python_handle

    #path prefix to saved models
    #Saved model should be saved as experiments/classifiers/static/{dataset_name}_{model_name}/models/{model_file}
    pp_clf =  f"{PROJECT_DIR}/experiments/classifiers"

    # DATA
    DATASETS_PATHS = {
            'MockUp':f"{PATH_PREFIX}/datasets/MockUp",
        }

    if args.datasets_paths:
        print('Adding dataset paths')
        print(args.datasets_paths)
        sys_dict = json.loads(args.system_paths)
        SYSTEMS_PATHS = {**DATASETS_PATHS, **sys_dict}
        print()

    # paths to different systems output folder and where to save evaluation results
    # temperature iterates over variation of output samples
    SYSTEMS_PATHS = {}

    #Alternatively, fill in systems paths manually
    """
    SYSTEMS_PATHS = {
            'DiDOTS_BART_MISTRAL_KB':{
            'project_dir':f'{PROJECT_DIR}/experiments/DiDOTS/None/None_BART_MISTRAL_7B_KB/results',
            'temperatures':['None'],
            'target_dir':f'{PROJECT_DIR}/experiments/evaluations/DiDOTS/None/None_BART_MISTRAL_7B_KB',
            'sampling_script':f"{PROJECT_DIR}/scripts",
            'ckt':'None',
            "model_ckp":f"{PROJECT_DIR}/experiments/DiDOTS/None/None_BART_MISTRAL_7B_KB/models/latest",
            'obfuscator':"DiDOTS/None_BART_MISTRAL_KB",
            'compute_const': False,
            "level":'sent',
            "by_document":False,
            'base_model':'facebook/bart-base'
        }
    }
    """

    if args.system_paths:
        print('Adding system paths')
        print(args.system_paths)
        sys_dict = json.loads(args.system_paths)
        SYSTEMS_PATHS = {**SYSTEMS_PATHS, **sys_dict}
        print()

    SYSTEMS = args.systems
    DATASETS = args.datasets
    STATIC = args.static
    STATIC_CLASSIFIERS = args.static_classifiers
    SAMPLE = args.sample
    ADAPTIVE = args.adaptive
    ADA_CLASSIFIERS = args.ada_classifiers
    doc_type = args.doc_type
    EVALUATE_DETECTION = args.evaluate_detection
    CLASSIFIERS = args.classifiers
    EVALUATE_QUALITY = args.evaluate_quality
    quality_metrics = args.quality_metrics
    EVAL_ALL_DIR = f"{PROJECT_DIR}/experiments/evaluations/all"

   # Run static training if applicable
    run_static_training()

    for system in SYSTEMS:
        print(f'Evaluating {system}')
        print("####################")

        system_dir = SYSTEMS_PATHS[system]['project_dir']
        output_eval_dir = SYSTEMS_PATHS[system]['target_dir']
        temps = SYSTEMS_PATHS[system]['temperatures']
        scripts_dir = SYSTEMS_PATHS[system]['sampling_script']
        level = SYSTEMS_PATHS[system]['level']

        pathlib.Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

        # Run sampling if enabled
        run_sampling(system, system_dir, DATASETS, SYSTEMS_PATHS, temps, scripts_dir, python_handle)

        # Run adaptive training if enabled
        run_adaptive_training(args,system)

        ######## EVALUATION ########
        if EVALUATE_DETECTION:
            # evaluate the samples synthesized by the systems against dementia classifiers.
            print('Evaluating Detection.')
            print("---------------------------")

            assert not (level=='doc' and ('BERT_sent' in CLASSIFIERS or 'SVM_sent' in CLASSIFIERS)), "Cannot evaluate on sentences with doc level"

            for d in DATASETS:
                pathlib.Path(f'{EVAL_ALL_DIR}/{d}').mkdir(parents=True, exist_ok=True)
                all_results_csv_path = f'{EVAL_ALL_DIR}/{d}/{d}_detection_results_sent.csv'
                pathlib.Path(f"{output_eval_dir}/{d}").mkdir( parents=True, exist_ok=True )
                save_csv_path = f"{output_eval_dir}/{d}/{d}_obfuscation_results_sent.csv"

                evaluate_detection(obf_model = system, obf_model_path = system_dir, classifiers = CLASSIFIERS,save_csv_path=save_csv_path,all_results_csv_path = all_results_csv_path, temperatures = temps, dataset = d, path_prefix = pp_clf, device=DEVICE, level = level)
            
            print()
            print('Detection Evaluation done.')
            print("---------------------------")
            print()

        if EVALUATE_QUALITY:
            print('Evaluating Samples Quality')

            for d in DATASETS:

                pathlib.Path(f"{output_eval_dir}/{d}").mkdir( parents=True, exist_ok=True )
                all_results_csv_path = f'{EVAL_ALL_DIR}/{d}/{d}_quality_results_sent.csv'

                eval_quality(d,system,system_dir,all_results_csv_path, quality_metrics, temps,PATH_PREFIX, level = level)
                
            print()
            print('Quality Evaluation done.')
            print("---------------------------")
            print()
