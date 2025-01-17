OS='Local'
PROJECT_DIR="/Users/domiceli/Documents/PhD/DiDOTS"
python_handle='python'
DEVICE='mps' # change accordingly (cpu/cuda/mps)

scripts_dir="${PROJECT_DIR}/scripts"
SEED=666

#LLM_BASES=( Mistral_7B LLama3 Phi3 Gemma2B )
LLM_BASES=( Mistral_7B )
BASE_MODEL='None'
#SETTINGS= '["FS" "ZS" "KB"]'
SETTINGS='["KB"]'

for LLM_BASE in ${LLM_BASES[@]}; do
    echo LLM: ${LLM_BASE}

    #### EVAL LLMS #####
    EVAL_DATASETS="MockUp" # ADReSS ADReSSo
    SYSTEM="${LLM_BASE}"
    
    # sampling code for LLMS is separate
    # See main.py in LLM_inference folder
    SAMPLE=''

    #whether to train static classifier or not, run only once.
    STATIC='--static'
    #STATIC=''
    STATIC_CLASSIFIERS="BERT_sent SVM_sent"

    ADAPTIVE="--adaptive"
    #ADAPTIVE=''

    EVAL_ALL_DIR="${PROJECT_DIR}/experiments/evaluations/all"
    ADA_CLASSIFIERS="BERT_sent SVM_sent"
    
    EVAL_DETECTION="--evaluate_detection"
    #EVAL_DETECTION=""

    EVAL_QUALITTY="--evaluate_quality"
    #EVAL_QUALITTY=""
    
    QUALITY_METRICS="Semantics ParaBART_Semantics Formality METEOR Lex_div Perplexity"
    EVAL_CLASSIFIERS="SVM_sent BERT_sent"

    SYSTEM_PATH='{"'"${SYSTEM}"'": {"project_dir": "'"${PROJECT_DIR}/experiments/${LLM_BASE}/results"'", "temperatures": '${SETTINGS}', "target_dir": "'"${PROJECT_DIR}/experiments/evaluations/${LLM_BASE}"'", "sampling_script": "None", "ckt": "None", "model_ckp": "None", "obfuscator": "'"${SYSTEM}"'", "compute_const": false, "level": "sent", "by_document": false,"base_model":"'"${BASE_MODEL}"'"}}'

    echo "----- STARTING EVALUATION SCRIPT ---"
    ${python_handle} ${scripts_dir}/main_eval.py --os ${OS} --path_prefix ${PROJECT_DIR} --project_dir ${PROJECT_DIR} --python_handle ${python_handle} --device ${DEVICE} --datasets ${EVAL_DATASETS} --systems "${SYSTEM}" ${STATIC} --static_classifiers ${STATIC_CLASSIFIERS} ${SAMPLE} ${ADAPTIVE} --ada_classifiers ${ADA_CLASSIFIERS} --doc_type doc sent ${EVAL_DETECTION} --classifiers ${EVAL_CLASSIFIERS} ${EVAL_QUALITTY} --quality_metrics ${QUALITY_METRICS} --eval_all_dir ${EVAL_ALL_DIR} --system_paths "${SYSTEM_PATH}"""
    echo "----- EVALUATION DONE ---"#
    echo
done