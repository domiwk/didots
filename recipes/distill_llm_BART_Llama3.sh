OS='Local'
PATH_PREFIX='/Users/domiceli/Documents/PhD' # change accordingly 
PROJECT_DIR="${PATH_PREFIX}/DiDOTS"
python_handle='python'
DEVICE='mps' # change accordingly (cpu/cuda/mps)

scripts_dir="${PROJECT_DIR}/scripts"
SEED=666

##### TRAIN DiDOTS #####
PRETRAIN_DATASET='None'
TRAIN_DATASET='MockUp'
MODEL_NAME='facebook/bart-base' # facebook/bart-base' t5-base'
MODEL='BART' #T5
NUM_EPOCHS=1
LR='1e-6'
LLM_BASE=Llama3

PEFT=('None')
SETTINGS=( 'KB' )
#PEFT=( 'None' 'LORA' 'IA3' 'BOFT' )
#SETTINGS=( 'FS' 'ZS' 'KB_v2' )
#PEFT=( LORA 'IA3' )
#SETTINGS=( 'ZS')

if [ "$PRETRAIN_DATASET" == 'None' ]; then
    BASE_MODEL=${MODEL_NAME}
fi
if [ "$PRETRAIN_DATASET" == 'ParaNMT' ]; then
    BASE_MODEL='/Users/domiceli/Documents/PhD/AdvStyle/Experiments/SNLI_BART/ParaNMT_enc_dec/paraphrase/Models/Scratch/checkpoint-91500'
fi

for SETTING in ${SETTINGS[@]}; do
    echo Training on ${SETTING} dataset
    for LORA in ${PEFT[@]}; do
        echo Lora: ${LORA}
        LORA_SUFFIX=''
        if [ "$LORA" == 'LORA' ]; then
            LORA_SUFFIX='_LORA'
        fi
        if [ "$LORA" == 'BOFT' ]; then
            LORA_SUFFIX='_BOFT'
        fi
        if [ "$LORA" == 'IA3' ]; then
            LORA_SUFFIX='_IA3'
        fi

        SAVE_DIR="${PROJECT_DIR}/experiments/DiDOTS/${PRETRAIN_DATASET}/${PRETRAIN_DATASET}_${MODEL}_${LLM_BASE}_${SETTING}${LORA_SUFFIX}/models/latest"

        DATASET_TRAIN_PATH="${PROJECT_DIR}/experiments/${LLM_BASE}/results/MockUp/MockUp_train_obfuscated_${SETTING}_sentences.csv"
        DATASET_TEST_PATH="${PROJECT_DIR}/experiments/${LLM_BASE}/results/MockUp/MockUp_test_obfuscated_${SETTING}_sentences.csv"
        DATASET_VAL_PATH="${PROJECT_DIR}/experiments/${LLM_BASE}/results/MockUp/MockUp_val_obfuscated_${SETTING}_sentences.csv"

        #### EVAL DiDOTS #####
        EVAL_DATASETS="MockUp" # ADReSS ADReSSo
        SYSTEM="DiDOTS_${MODEL}_${PRETRAIN_DATASET}_${LLM_BASE}_${SETTING}${LORA_SUFFIX}"
        #SAMPLE='--sample' 
        SAMPLE=''

        #whether to train static classifier or not
        #STATIC='--static'
        STATIC=''
        STATIC_CLASSIFIERS="BERT_sent SVM_sent"

        ADAPTIVE="--adaptive"
        #ADAPTIVE=''
        EVAL_ALL_DIR="${PROJECT_DIR}/experiments/evaluations/all"
        ADA_CLASSIFIERS="SVM_sent"
        
        EVAL_DETECTION="--evaluate_detection"
        #EVAL_DETECTION=""

        EVAL_QUALITTY="--evaluate_quality"
        #EVAL_QUALITTY=""
        
        QUALITY_METRICS="Semantics ParaBART_Semantics Formality METEOR Lex_div Perplexity"
        EVAL_CLASSIFIERS="SVM_sent BERT_sent"

        SYSTEM_PATH='{"'"${SYSTEM}"'": {"project_dir": "'"${PROJECT_DIR}/experiments/DiDOTS/${PRETRAIN_DATASET}/${PRETRAIN_DATASET}_${MODEL}_${LLM_BASE}_${SETTING}${LORA_SUFFIX}/results"'", "temperatures": ["None"], "target_dir": "'"${PROJECT_DIR}/experiments/evaluations/DiDOTS/${PRETRAIN_DATASET}/${PRETRAIN_DATASET}_${MODEL}_${LLM_BASE}_${SETTING}${LORA_SUFFIX}"'", "sampling_script": "'"${PROJECT_DIR}/scripts"'", "ckt": "None", "model_ckp": "'"${PROJECT_DIR}/experiments/DiDOTS/${PRETRAIN_DATASET}/${PRETRAIN_DATASET}_${MODEL}_${LLM_BASE}_${SETTING}${LORA_SUFFIX}/models/latest"'", "obfuscator": "'"DiDOTS/${PRETRAIN_DATASET}_${MODEL}_${LLM_BASE}_${SETTING}${LORA_SUFFIX}"'", "compute_const": false, "level": "sent", "by_document": false,"base_model":"'"${BASE_MODEL}"'"}}'

        echo "----- STARTING TRAINING ---"
        ${python_handle} ${scripts_dir}/train.py --disable_tqdm --dataset ${TRAIN_DATASET} --path_prefix ${PROJECT_DIR} --save_dir ${SAVE_DIR} --model_name ${MODEL_NAME} --model_type ${MODEL} --num_epochs ${NUM_EPOCHS} --early_stopping_patience 3 --top_n_models_to_save 2 --gradient_clip 1.0 --log_interval 100 --lr ${LR} --setting ${SETTING} --para_dataset ${PRETRAIN_DATASET} --lora ${LORA} --train_dataset_path ${DATASET_TRAIN_PATH} --test_dataset_path ${DATASET_TEST_PATH} --val_dataset_path ${DATASET_VAL_PATH}
        echo "----- TRAINING DONE ---"
        echo ''

        echo "----- STARTING EVALUATION SCRIPT ---"
        ${python_handle} ${scripts_dir}/main_eval.py --os ${OS} --path_prefix ${PATH_PREFIX} --project_dir ${PROJECT_DIR} --python_handle ${python_handle} --device ${DEVICE} --datasets ${EVAL_DATASETS} --systems "${SYSTEM}" ${STATIC} --static_classifiers ${STATIC_CLASSIFIERS} ${SAMPLE} ${ADAPTIVE} --ada_classifiers ${ADA_CLASSIFIERS} --doc_type doc sent ${EVAL_DETECTION} --classifiers ${EVAL_CLASSIFIERS} ${EVAL_QUALITTY} --quality_metrics ${QUALITY_METRICS} --eval_all_dir ${EVAL_ALL_DIR} --system_paths "${SYSTEM_PATH}"""
        echo "----- EVALUATION DONE ---"#
    done
done