import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_psnr_min = 0
        self.delta = delta
        self.checkpoint_perf = []

    def __call__(self, g, d, train_psnr, val_psnr):

        score = val_psnr
        self.early_stop = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(g, d, val_psnr)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.counter = 0
                self.best_score = None
                self.val_psnr_min = 0
        else:
            self.best_score = score
            self.save_checkpoint(g, d, val_psnr)
            self.counter = 0
            self.checkpoint_perf = [train_psnr, val_psnr]
        return self.checkpoint_perf

    def save_checkpoint(self, g, d, val_psnr):
        self.val_psnr_min = val_psnr
        if self.verbose:
            print(f'Validation PSNR increased ({self.val_psnr_min:.6f} --> {val_psnr:.6f}).  Saving model ...')
            torch.save(g.state_dict(), 'Generator.pth')
            torch.save(d.state_dict(), 'Discriminator.pth')
        else:
            torch.save(g.state_dict(), 'Generator.pth')
            torch.save(d.state_dict(), 'Discriminator.pth')
