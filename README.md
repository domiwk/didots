# **DiDOTS: Knowledge Distillation from Large-Language-Models for Dementia Obfuscation in Transcribed Speech**
## **Introduction**

<p align="center">
<img width="667" alt="image" src="https://github.com/user-attachments/assets/10e11a59-4faa-4afc-a707-221c9f18178d">
</p>

This repository contains the code for the paper  **"DiDOTS: Knowledge Distillation from Large-Language-Models for Dementia Obfuscation in Transcribed Speech"** accepted at PETS 25'. 

In this paper, we investigate LLMs for the task of dementia obfuscation in text and propose a light weight obfuscator through knowledge distillation.

**Authors:** Dominika Woszczyk and Soteris Demetriou

## **Dataset**
This work was performed with the ADReSS and ADReSSo datasets that are available upon request at https://dementia.talkbank.org/.

In this repository we provide code to replicate the results with a mock dataset.

## **Installation**
### **DiDOTS Dependencies**


**This project was built using python 3.9**

To create a new environment for training and evaluating DiDOTS run

```
git clone https://github.com/domiwk/didots.git
cd didots

conda create --name didots python=3.9
python -m pip install -r requirements.txt
```

To simply install dependencies, use the following command:

```
git clone https://github.com/domiwk/didots.git

cd didots
pip install -r requirements.txt
```

For the evaluation, please download the model.pt file from [here](https://drive.google.com/file/d/1V5RnfPOSYumqPQElC80ZwKCWP4E_MpSm/view?usp=drive_link) and place it under "./models/ParaBART".


### **LLMs Inference Dependencies**
To run **LLMs inference (Mistral, LLama3, etc... )** you need to install **ollama** from [here](https://ollama.com).

Then install the python package
```
pip install ollama
```

To download and run models, open a terminal and run the following and replacing `model_name` with a LLM from their repository. In this work we used `mistral:instruct`,`llama3:instruct`,`phi3:instruct` and `gemma:2b-instruct`. Replace "path/to/download/dir" with the desired location for storing the downloaded model.

```
OLLAMA_MODELS="path/to/download/dir" ollama run <model_name>
```

Once the server is alive, you can run the inference code `./LLM_inference/main.py`

## **Usage**

### **LLM Inference**

After installing ollama and starting the server, you can generate synthetic pairs using LLMs for your dataset. 

For this, change the variables accordingly in the script `./LLM_inference/main.py` and run it with

```
python ./LLM_inference/main.py
```
**The cleaning script might need adjusting given the dataset.**

### **Training & Evaluation**

You can train a model by running the recipes in the `recipes` folder.

For example to run DiDOTS trained on synthetic datasets built with Mistral run `/recipes/distill_llm_BART_Mistral_7B.sh`

You can change parameters inside the bash file to run different systems. (e.g  set PATH_PREFIX to your working dir)

Some bash files are already prepared for 

- Training T5 instead of BART: `recipes/distill_llm_T5_Mistral_7B.sh`
- Training on different synthetic dataset: 
    - `recipes/distill_llm_BART_Phi3.sh`
    - `recipes/distill_llm_BART_Llama3.sh`

### **Evaluating Multiple Systems**

If you have already trained models and want to evaluate multiple systems fill in the `SYSTEMS_PATHS` in `./scripts/main_eval.py` with an entry for each system. 

For example:
```
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
```

and run 
```
python3.9 ./scripts/main_eval.py --os ${OS} --python_handle python3.9 --device ${DEVICE} --datasets MockUp --systems DIDOTS_BART_MISTRAL_KB  --static --sample --adaptive --eval_detection --eval_quality
```

### **Ablation Studies and Results from the Paper**

Run the following recipes

**DiDOTS experiments for tables 1,4, and ablations on tables 6 and 7**:
- Different prompt settings and PEFT methods: `/recipes/distill_llm_BART_Mistral_7B.sh`
- Training T5 instead of BART: `recipes/distill_llm_T5_Mistral_7B.sh`
- Training on different synthetic dataset: 
    - `recipes/distill_llm_BART_Phi3.sh`
    - `recipes/distill_llm_BART_Llama3.sh`

**LLMs experiments for tables 1,3 and 4**: See recipe `recipes/eval_llms.sh`
