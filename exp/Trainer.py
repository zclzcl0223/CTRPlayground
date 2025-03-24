import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from tqdm import trange, tqdm
from utils import AverageMeter, get_lr_lambda, get_optimizer
from sklearn.metrics import roc_auc_score

class Trainer:
    def __init__(self, model, device, tokenizer, config,
                 train_loader, valid_loader=None, test_loader=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.decay_expand_rate = config.decay_expand_rate
        self.max_grad_norm = config.max_grad_norm
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
    
    def train(self):
        learning_rate = self.config.learning_rate
        epochs = self.config.epochs
        optimizer = get_optimizer(self.model, self.config)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: get_lr_lambda(it, 0, epochs * self.decay_expand_rate, 
                                                                                                    learning_rate, learning_rate*0.1))
        bce_loss = nn.BCEWithLogitsLoss()
        train_loader = tqdm(self.train_loader, disable=False)
        for epoch in trange(epochs):
            logloss_total = AverageMeter()
            iter_count = 0
            tqdm.write(f'Epoch: {epoch}:')
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device).float()
                # mixed precision
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    y_hat = self.model(x)
                    loss = bce_loss(y_hat, y)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                logloss_total.update(loss.item(), x.shape[0])
                iter_count += 1
                
                if iter_count % 250 == 0 and self.valid_loader is not None:
                    valid_logloss, auc = self.eval()
                    tqdm.write(f'   Train iter: {iter_count}, Train Loss: {logloss_total.avg:.4f}, Valid Loss: {valid_logloss.avg:.4f}, Valid AUC: {auc:.4f}')
                    self.model.train()
    
    def eval(self, eval_iter=100):
        self.model.eval()
        logloss_total = AverageMeter()
        iter_count = 0
        bce_loss = nn.BCEWithLogitsLoss()
        pred = []
        true = []
        for x, y in self.valid_loader:
            x, y = x.to(self.device), y.to(self.device).float()
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    y_hat = self.model(x)
                    loss = bce_loss(y_hat, y)
            pred.extend(F.sigmoid(y_hat.float()).cpu().detach().numpy())
            true.extend(y.cpu().detach().numpy())
            logloss_total.update(loss.item(), x.shape[0])
            iter_count += 1
            if iter_count == eval_iter:
                break
        auc = roc_auc_score(true, pred)
        return logloss_total, auc
