import os
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

def load_samples(filename):
    nps = np.load(filename)
    pts = torch.tensor(nps, dtype=torch.long)
    return pts

class CTRStreamDataset(IterableDataset):
    def __init__(self, data_path:str, split:str, batch_size:int):
        super().__init__()
        self.data_path = data_path
        self.shards = [file for file in os.listdir(data_path) if split in file and 'npy' in file]
        self.split = split
        self.batch_size = batch_size
        self.rng = np.random.default_rng(114515)
        self.reset()

    def load_shard(self, filename):
        samples = load_samples(os.path.join(self.data_path, filename))
        if self.split == 'train':
            idx = np.arange(len(samples))
            self.rng.shuffle(idx)
            samples = samples[idx]
        return samples

    def reset(self):
        if self.split == 'train':
            self.rng.shuffle(self.shards)
    
    def sample_batches(self, samples):
        total_samples = len(samples)
        idx = 0
        while idx < total_samples:
            yx = samples[idx]
            y = yx[:1]
            x = yx[1:]
            yield x, y
            idx += 1

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            shard_indices = range(len(self.shards))
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            shard_indices = range(worker_id, len(self.shards), num_workers)
        
        for i in shard_indices:
            samples = self.load_shard(self.shards[i])
            yield from self.sample_batches(samples)
        self.reset()

def get_dataloader(data_path:str, batch_size:int):
    train_dataset = CTRStreamDataset(data_path, 'train', batch_size)
    valid_dataset = CTRStreamDataset(data_path, 'valid', batch_size)
    test_dataset = CTRStreamDataset(data_path, 'test', batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=False)

    return train_loader, valid_loader, test_loader
