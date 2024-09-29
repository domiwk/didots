# Artifact Appendix

Paper title: **DiDOTS: Knowledge Distillation from Large-Language-Models for
Dementia Obfuscation in Transcribed Speech**

Artifacts HotCRP Id: **#21**

Requested Badge: Either **Available**, **Functional**

## Description
We release the code to train and evaluate our proposed system DiDOTS. We also include code to generate synthetic datasets with LLM through different prompts.

### Security/Privacy Issues and Ethical Concerns (All badges)
This code does not hold any risk for the privacy of the reviewer's machine. As we do not release the original dataset with dementia samples, we believe that there is no ethical concerns linked to this artifact.

## Basic Requirements (Only for Functional and Reproduced badges)
A laptop with 8gb RAM can run this code, although training and inference can be slow on cpu only. A GPU is preferred (cuda or mps).

### Software Requirements
This code should run on any OS with python 3.9, altough it was only tested on Linux and MacOS. For environments and packages see below or README.md

### Estimated Time and Storage Consumption
**LLM Inference**
Storage need for LLM inference: Phi3 and Gemma 2 need ~2GB of memory each, LLama3 and Mistral take up ~4-5GB each.
Inference can takes from 30 min per dataset and prompt to several hours (especially in few-shot setting). We suggest to only generate data with one LLM to check that the code is running and use either 'ZS' or "KB" setting to gain time.
We provide some mockup data for training DiDOTS. 

**DiDOTS**
Time to run training a BART model should be around 10min on gpu and take roughly 500mb of space. T5 models takes ~800mb.
Training static adversaries should be ~15min.
The whole evaluation of one system should be take about 30-60min for inference, adaptive adversaries training and evaluation.

## Environment 
### Accessibility (All badges)
The code is accessible at https://github.com/domiwk/didots with latest commit.

### Set up the environment (Only for Functional and Reproduced badges)
#### **DiDOTS Dependencies**


**This project was built using python 3.9**

To create a new environment for training and evaluating DiDOTS run

```
git clone https://github.com/domiwk/didots.git
cd didots

conda create --name didots python=3.9
conda activate didots
pip install -r requirements.txt
```

To simply install dependencies, use the following command:

```
git clone https://github.com/domiwk/didots.git

cd didots
pip install -r requirements.txt
```

#### **LLMs Inference Dependencies**
To run **LLMs inference (Mistral, LLama3, etc... )** you will need to install **ollama** from [here](https://ollama.com).

Then install the python package
```
pip install ollama
```

To download and run models, open a terminal and run the following and replacing `model_name` with a LLM from their repository. In this work we used `mistral:instruct`,`llama3:instruct`,`phi3:instruct` and `gemma:2b-instruct`. 

```
OLLAMA_MODELS="PATH/to/download/dir" ollama run <model_name>
```

Once the server is alive, you can run the inference code `./LLM_inference/main.py`

### Testing the Environment (Only for Functional and Reproduced badges)

### **Training & Evaluation**

You can train a model by running the recipes in the `recipes` folder.

For example to run DiDOTS trained on synthetic datasets built with Mistral run `/recipes/distill_llm_BART_Mistral_7B.sh`

You can change parameters inside the bash file to run different systems.

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
            'DIDOTS_BART_MISTRAL_KB':{
            'project_dir':f'{PROJECT_DIR}/experiments/DIDOTS/None_BART_MISTRAL_7B_KB/results',
            'temperatures':['None'],
            'target_dir':f'{PROJECT_DIR}/experiments/evaluations/DiDOTS/None/None_BART_MISTRAL_7B_KB',
            'sampling_script':f"{PROJECT_DIR}/scripts",
            'ckt':'None',
            "model_ckp":f"{PROJECT_DIR}/experiments/DiDOTS/None/None_BART_MISTRAL_7B_KB/models/latest",
            'obfuscator':"DiDOTS/None_BART_MISTRAL_KB",
            'compute_const': False,
            "level":'sent',
            "by_document":False,
        }
}
```

and run 
```
python ./scripts/main_eval.py
```

## Artifact Evaluation (Only for Functional and Reproduced badges)

### **Ablation Studies and Results from the Paper**

Run the following recipes

**DiDOTS experiments for tables 1,4, and ablations on tables 6 and 7**:
- Different prompt settings and PEFT methods: `/recipes/distill_llm_BART_Mistral_7B.sh`
- Training T5 instead of BART: `recipes/distill_llm_T5_Mistral_7B.sh`
- Training on different synthetic dataset: 
    - `recipes/distill_llm_BART_Phi3.sh`
    - `recipes/distill_llm_BART_Llama3.sh`

**LLMs experiments for tables 1,3 and 4**: See recipe `recipes/eval_llms.sh`


## Limitations (Only for Functional and Reproduced badges)
We do not provide the datasets are they required a license and contain sensitive information. Hence, the results of our papaer are not directly reproductible. We also do not provide weights for trained models as it might leak information about the source data it was trained on.

## Notes on Reusability (Only for Functional and Reproduced badges)
This code can be adapted to create various synthetic text datasets and to train smaller efficient models for tasks such as disfluency correction, style transfer, translation, etc...

The LLMs, dataset and pretrained-language models used can be interchanged with other models. 
