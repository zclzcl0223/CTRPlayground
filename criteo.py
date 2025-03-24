import shutil
import os
import hashlib
import zipfile
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from tqdm import tqdm

cache_dir = 'dataset/cache'
os.makedirs(cache_dir, exist_ok=True)

class Tokenizer:
    def __init__(self):
        self.tokenizer = {}

    def process_chunk(self, chunk, n_sparse, min_categr_count):
        column_value_counts = {}
        
        # Fill missing values
        for col in chunk.columns[1:]:
            if chunk[col].apply(lambda x: isinstance(x, str)).any():
                chunk[col] = chunk[col].fillna('')
            else:
                chunk[col] = chunk[col].fillna('')
                chunk[col] = chunk[col].map(lambda x: int(np.floor(np.log(x) ** 2)) if x != '' and x > 2 else (x if x == '' else int(x)))
                chunk[col] = chunk[col].astype(str)

        # Calculate value counts for relevant columns
        for col in chunk.columns[1:]:
            value_counts = chunk[col].value_counts()
            for val, count in value_counts.items():
                if count >= min_categr_count and val != '':
                    if col not in column_value_counts:
                        column_value_counts[col] = {}
                    if val not in column_value_counts[col]:
                        column_value_counts[col][val] = 0
                    column_value_counts[col][val] += count

        return column_value_counts

    def fit(self, path:str, n_sparse:int, min_categr_count:int, chunk_size:int, num_workers:int=5):
        column_value_counts = {}

        with mp.Pool(num_workers) as pool:
            results = []
            for chunk in tqdm(pd.read_csv(path, chunksize=chunk_size), desc="Fitting tokenizer"):
                results.append(pool.apply_async(self.process_chunk, (chunk, n_sparse, min_categr_count)))

            for result in results:
                column_value_counts.update(result.get())

        # Update tokenizer dictionary
        for col, counts in column_value_counts.items():
            self.tokenizer[col] = {'': 0, '<unk>': len(counts) + 1}
            for i, key in enumerate(counts.keys()):
                self.tokenizer[col][key] = i + 1

    def encode(self, df):
        df_encoded = df.copy()
        for col in df_encoded.columns:
            if col in self.tokenizer:
                df_encoded[col] = df_encoded[col].map(lambda x: self.tokenizer[col].get(x, self.tokenizer[col]['<unk>']))
        return df_encoded


def process_chunk(chunk, tokenizer):
    return tokenizer.encode(chunk)

def save_dataset_in_chunks(dataset, split, tokenizer, save_dir="dataset/criteo", chunk_size=100000, num_workers=4):
    os.makedirs(save_dir, exist_ok=True)

    with mp.Pool(num_workers) as pool:
        encode_func = partial(process_chunk, tokenizer=tokenizer)
        
        results = []
        for chunk in tqdm(pd.read_csv(dataset, chunksize=chunk_size), desc=f"Processing {split} set"):
            for col in chunk.columns[1:]:
                if chunk[col].apply(lambda x: isinstance(x, str)).any():
                    chunk[col] = chunk[col].fillna('')
                else:
                    chunk[col] = chunk[col].fillna('')
                    chunk[col] = chunk[col].map(lambda x: int(np.floor(np.log(x) ** 2)) if x != '' and x > 2 else (x if x == '' else int(x)))
                    chunk[col] = chunk[col].astype(str)
            encoded_chunk = tokenizer.encode(chunk)
            results.append(encoded_chunk)
        
        for idx, chunk in enumerate(results):
            file_name = os.path.join(save_dir, f"{split}_{idx}.npy")
            np.save(file_name, chunk.values)
            print(f"Saved: {file_name}")

def preprocess(root:str='dataset', name:str='Criteo_x4', 
                   n_dense:int=13, n_sparse:int=26, min_categr_count:int=2):
    name = name.lower()
    path = os.path.join(root, name)
    tokenizer_path = 'dataset/criteo/tokenizer.json'
    split = {'test':0, 'train':1,  'valid':2}

    if name in ['criteo_x4']:
        extract_path = 'dataset/criteo'
        os.makedirs(extract_path, exist_ok=True)
        file_path = os.listdir(extract_path)
        if not file_path:
            file_path = hf_hub_download(
                repo_id='reczoo/Criteo_x4',
                filename='Criteo_x4.zip',
                repo_type='dataset',
                cache_dir=cache_dir
            )
            print(f'Downloaded file is at: {file_path}')
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        shutil.rmtree(cache_dir)
    else:
        # TODO
        extract_path = 'dataset/xx'
        os.makedirs(extract_path, exist_ok=True)
        file_path = os.listdir(extract_path)

    print("loading dataset...")

    # fit tokenizer
    tokenizer = Tokenizer()
    #with open(tokenizer_path, 'r') as f:
    #    tokenizer.tokenizer = json.load(f)
    tokenizer.fit(os.path.join(extract_path, file_path[split['train']]), n_sparse, min_categr_count, chunk_size=1000000, num_workers=5)

    with open('dataset/criteo/tokenizer.json', "w") as json_file:
        json.dump(tokenizer.tokenizer, json_file, indent=4)

    print('Encoding train set...')
    save_dataset_in_chunks(os.path.join(extract_path, file_path[split['train']]), 'train', tokenizer, chunk_size=1000000, num_workers=5)
    
    print('Encoding valid set...')
    save_dataset_in_chunks(os.path.join(extract_path, file_path[split['valid']]), 'valid', tokenizer, chunk_size=1000000, num_workers=5)
    
    print('Encoding test set...')
    save_dataset_in_chunks(os.path.join(extract_path, file_path[split['test']]), 'test', tokenizer, chunk_size=1000000, num_workers=5)
    

if __name__ == '__main__':
    preprocess()
