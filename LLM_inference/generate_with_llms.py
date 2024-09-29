import pandas as pd
from tqdm import tqdm
import pathlib

import ollama
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, csv_file, setting):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['Text'].apply(lambda x: len(x.split()) >= 3)]
        self.setting = setting

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Text']
        
        if self.setting == 'KB':
            text = f"I want you to rewrite a sentence. Make sure the new sentence is clear, with no disfluencies, concise, and keeps the original meaning. Do not ask questions, add explanation or comments, just perform the task. Strictly follow the following format. Source: 'input_sentence'. Output: 'output_sentence'. Rewrite this sentence. Source: {text} Output:"
        elif self.setting == 'FS':
            text = f"Here are examples of Healthy samples: the mother's standing in the middle of the water; The little girl is laughing at her brother who's taking a cookie out_of the cookie jar.; this boy is about to fall off of the stool.; The wind is blowing the curtains.; the boy is falling off of the stool stealing the cookies out of the cookie jar.\n Here are some examples of Dementia samples: I really can't see what she's doing. ;I can't really pick it out but ... oh and there's a little girl here talking and a little boy I assume on this side here .; She's doing dishes and the sink's running over and wetting her feet .; And the kid's in the cookie jars.; Curtains at the windows. Stricly follow the following format and keep the meaning the same. Do not add verbose or explanation. Only give the answer in the following format: Dementia:'text'. Healthy:'text'. Here is a sentence. Dementia: {text}. Healthy:"
        else:
            # zs
            text = f"I want you to replace a sentence transcribed from dementia speech with a healthy one. Do not ask questions just perform the task. Stricly follow the following format and keep the meaning the same. Do not add verbose or explanation. Only give the answer in the following format: Dementia:'text'. Healthy:'text'. Here is a sentence. Dementia: {text}. Healthy:"
        return {'input_text': text}

def LLM_inference(model_names,datapaths,llm_paths,settings,datasets,results_dir):
    for dataset_name, data_path_dir in datapaths.items():
        if dataset_name in datasets:
            print(f'Dataset: {dataset_name}')
            print("----------------------------")
            for model_name in model_names:
                for setting in settings:
                    print(f'Inference with {model_name} and {setting} prompt.')
                    for type in ['test','train']:
                        print(f'Starting inference on {type}')
                        data_path = f"{data_path_dir}/{dataset_name}_{type}_original_sentences.csv"
                        dataset = CustomDataset(data_path,setting)
                        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

                        paraphrases = []
                        for batch in tqdm(data_loader):
                            input_texts = batch['input_text'][0]                        
                            outputs = ollama.generate(model=model_name, prompt=input_texts)['response']
                            paraphrases.append(outputs)

                        adv_text = [t.replace('"','') for t in paraphrases]

                        # Save paraphrases to a new CSV file
                        out_dir = f"{results_dir[model_name]}/{dataset_name}"
                        pathlib.Path(out_dir).mkdir(parents = True, exist_ok = True)
                        og_data = pd.read_csv(f"{data_path_dir}/{dataset_name}_{type}_original_sentences.csv")
                        og_data = og_data[og_data['Text'].apply(lambda x: len(x.split()) >= 3)]
                        og_data['AdvText'] = adv_text
                        og_data.to_csv(f"{out_dir}/{dataset_name}_{type}_obfuscated_{setting}_sentences.csv", index=False)
                        print(f'Inference on {type} done')