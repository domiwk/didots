from cleaning_up_utils import clean_generated_outputs
from generate_with_llms import LLM_inference


if __name__ == '__main__':

    STEPS=['inference','cleaning']

    llm_names = {'phi3:instruct':'Phi3','llama3:instruct':'LLama3', 'gemma:2b-instruct':'Gemma_2B','mistral:instruct':'Mistral_7B'}
    datapaths = {'MockUp':"/Users/domiceli/Documents/PhD/DiDOTS/datasets/MockUp"}

    model_names = ['phi3:instruct']
    datasets = ['MockUp']
    #settings =['KB','ZS','FS']
    settings =['KB']
    data_set = ['test','train']

    results_dir = {'phi3:instruct':"./experiments/Phi3/results",
                'mistral:instruct':"./experiments/Mistral_7B/results",
                    'gemma:2b':"./experiments/Gemma_2B/Results",
                    'llama3:instruct':"./experiments/Llama3/Results",
                }

    #inference
    if 'inference' in STEPS:
        print('Starting LLM inference...')
        LLM_inference(model_names,datapaths,llm_names,settings,datasets,results_dir)

    #clean
    if 'cleaning' in STEPS:
        print('Starting LLM output cleaning...')
        clean_generated_outputs(datasets,data_set,settings,model_names,results_dir)