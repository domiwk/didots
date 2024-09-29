import torch

class AD_Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, dataset_df,  max_length = 50):

        self.tokenizer = tokenizer
        self.texts = dataset_df
        self.max_length = max_length

    def collate_fn(self, batch):
        texts = []
        labels = []
        device_labels = []

        for b in batch:
            if isinstance(b['input_ids'], float):
                b['input_ids'] = '' 
            texts.append(b['input_ids'])
            labels.append(b['labels'])

        encodings = self.tokenizer(texts, return_tensors='pt', add_special_tokens = True, padding=True, truncation=True, max_length = self.max_length)

        labels =  torch.tensor(labels)

        encodings['labels'] = labels

        return encodings
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {'input_ids': self.texts['Text'].iloc[idx],
                'labels': int(self.texts['Label'].iloc[idx])}
        return item


    
