import argparse
import pathlib
from tqdm import tqdm
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, IA3Config, BOFTConfig

parser = argparse.ArgumentParser()

# General settings
parser.add_argument('--disable_tqdm', action='store_true', help='Disable tqdm')
parser.add_argument('--dataset', type=str, default='ADReSSo', help='Dataset name')
parser.add_argument('--train_dataset_path', type=str, help='Path to training file.')
parser.add_argument('--test_dataset_path', type=str, help='Path to test file.')
parser.add_argument('--val_dataset_path', type=str, help='Path to val file.')
parser.add_argument('--path_prefix', type=str, default=".", help='Path prefix')
parser.add_argument('--save_dir', type=str, default='./experiments/DiDOTS/None/None_T5_MISTRAL_GT_ZS', help='Save directory')
parser.add_argument('--model_name', type=str, default='bart-base', help='Model name')
parser.add_argument('--model_type', type=str, default='BART', help='Model architecture type')
parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
parser.add_argument('--validate_every_n_steps', type=int, default=40, help='Validation frequency in steps')
parser.add_argument('--setting', type = str,default ='ZS', help='Enable training')
parser.add_argument('--llm_base', type = str,default ='Mistral_7B', help='Enable training')
parser.add_argument('--eval_steps', type = int,default =50, help='Evaluate every x steps')
parser.add_argument('--device', type = str,default ='cpu', help='mps, cpu or cuda')

# Early stopping parameters
parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience')
parser.add_argument('--top_n_models_to_save', type=int, default=2, help='Top N models to save')
parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')
parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lora', type=str, default='BOFT',help='Enable LORA')
parser.add_argument('--para_dataset', type=str, default='None',help='Parpahrase dataset base')

args = parser.parse_args()

dataset = args.dataset
path_prefix= args.path_prefix
model_type = args.model_type
save_dir = args.save_dir
setting = args.setting

num_epochs = args.num_epochs
early_stopping_patience = args.early_stopping_patience
gradient_clip = args.gradient_clip
log_interval = args.log_interval
lr = args.lr
use_lora = args.lora
disable_tqdm = args.disable_tqdm
validate_every_n_steps = args.validate_every_n_steps
train_path = args.train_dataset_path
test_path = args.test_dataset_path
val_path = args.val_dataset_path
eval_steps = args.eval_steps
#device = "cpu"
device = args.device

if 'rds' in path_prefix:
    data_path_prefix = '/rds/general/user/dcw19/home'
else:
    data_path_prefix = path_prefix

class ParaphraseDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, labels, tokenizer, max_length=100, sample_size =1):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_size = sample_size

        self.source_sentences = self.source_sentences[:int(len(self.source_sentences)*sample_size)]
        self.target_sentences = self.target_sentences[:int(len(self.target_sentences)*sample_size)]
        self.labels = labels[:int(len(labels)*sample_size)]
        self.target_sentences = self.target_sentences.fillna('')

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        source = self.source_sentences[idx]
        target = self.target_sentences[idx]
        source_encoding = self.tokenizer(source, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        target_encoding = self.tokenizer(target, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {"source_input_ids": source_encoding["input_ids"].squeeze(),
                "source_attention_mask": source_encoding["attention_mask"].squeeze(),
                "target_input_ids": target_encoding["input_ids"].squeeze(),
                "label": float(self.labels[idx]),
                "target_attention_mask": target_encoding["attention_mask"].squeeze()}

seq_len = 100
para_dataset = args.para_dataset

if model_type == 'BART':
    if para_dataset == 'ParaNMT':
        model_path = f'{path_prefix}/Experiments/SNLI_BART/ParaNMT_enc_dec/BART/Model'
    else:
        model_path = 'facebook/bart-base'

if model_type == 'T5':
    if para_dataset == 'PAR3':
        model_path = f'{path_prefix}/Experiments/CTS/PAR3/PAR3_T5/Models/best/best_metric_model'
    else:
        model_path = 't5-base'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

if model_type == 'BART':
            lr = 1e-5
elif model_type == 'T5':
            lr = 4e-4

peft_suffix =''
if use_lora and use_lora!='None':
    if use_lora =='LORA':
        peftconfig = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        base_model_name_or_path = model_path,
        r=16,
        lora_alpha=32,
        lora_dropout=0.01,
        use_rslora = True
        )
        peft_suffix = '_LORA'

        if model_type == 'BART':
            lr = 4e-4
        elif model_type == 'T5':
            lr = 1e-4

    elif use_lora =='IA3':
        peftconfig = IA3Config(
            peft_type="IA3",
            task_type="SEQ_2_SEQ_LM",
        )
        peft_suffix = '_IA3'

        if model_type == 'BART':
            lr = 5e-3
        elif model_type == 'T5':
            lr = 4e-3

    elif use_lora =='BOFT':
        peftconfig = BOFTConfig( boft_block_size=4,
                                boft_dropout=0.1, 
                                bias="none")
        peft_suffix = '_BOFT'
        if model_type == 'BART':
            lr = 5e-3
        elif model_type == 'T5':
            lr = 1e-5

    model = get_peft_model(model, peftconfig)

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

X_train,y_train, y_labels = train_df['Text'], train_df['AdvText'], train_df['Label']
X_val,y_val, val_labels = val_df['Text'], val_df['AdvText'], val_df['Label']

dataset_train = ParaphraseDataset(X_train, y_train, y_labels,tokenizer,sample_size = 1)
dataset_val = ParaphraseDataset(X_val, y_val, val_labels,tokenizer,sample_size = 1)

# Training loop with validation
model.to(device)
model.train()

#base
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
train_loader = DataLoader(dataset_train, batch_size=2, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=2, shuffle=False)

total_steps = 0

pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

best_val_loss = float('inf')
no_improvement = 0

for epoch in range(num_epochs):
    # Training
    total_train_loss = 0

    for batch in tqdm(train_loader):
        model.train()
        source_input_ids = batch["source_input_ids"].to(device)
        source_attention_mask = batch["source_attention_mask"].to(device)
        target_input_ids = batch["target_input_ids"].to(device)
        target_attention_mask = batch["target_attention_mask"].to(device)

        optimizer.zero_grad()
        out = model(input_ids=source_input_ids, attention_mask=source_attention_mask, labels=target_input_ids)

        loss = out.loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
            
        total_steps += 1
        if total_steps % validate_every_n_steps == 0:
            # Validation
            total_val_loss = 0

            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    source_input_ids = val_batch["source_input_ids"].to(device)
                    source_attention_mask = val_batch["source_attention_mask"].to(device)
                    target_input_ids = val_batch["target_input_ids"].to(device)
                    target_attention_mask = val_batch["target_attention_mask"].to(device)
                    d_labels = val_batch["label"].type(torch.FloatTensor).to(device)

                    outputs = model.generate(input_ids=source_input_ids, attention_mask=source_attention_mask, max_length=60, num_beams=2, early_stopping=True)
                    paraphrases = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    print("Example Paraphrases:")
                    for i in range(len(paraphrases)):
                        source_text = tokenizer.decode(val_batch["source_input_ids"][i], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                        target_text = tokenizer.decode(val_batch["target_input_ids"][i], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                        print(f"Source: {source_text[:100]}") 
                        print(f"-> Target: {target_text[:100]}")
                        print(f"-> Gen. Paraphrase: {paraphrases[i][:100]}")
                        print()
                    break  # Print only for the first batch

                for val_batch in val_loader:
                    source_input_ids = val_batch["source_input_ids"].to(device)
                    source_attention_mask = val_batch["source_attention_mask"].to(device)
                    target_input_ids = val_batch["target_input_ids"].to(device)
                    target_attention_mask = val_batch["target_attention_mask"].to(device)

                    outputs = model(input_ids=source_input_ids, attention_mask=source_attention_mask,labels=target_input_ids)

                    loss = outputs.loss
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f"Epoch:{epoch}, Step {total_steps}, Train Loss: {total_train_loss / eval_steps:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement = 0
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
            else:
                no_improvement += 1

            total_train_loss = 0
            model.train()

            # Early stopping condition
            if no_improvement >= early_stopping_patience:
                print(f"No improvement for {early_stopping_patience} epochs. Early stopping...")
                break