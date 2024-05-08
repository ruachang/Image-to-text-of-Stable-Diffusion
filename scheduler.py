from torch.optim.lr_scheduler import _LRScheduler 
import torch
import numpy as np
class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, warmup_init_lr, max_lr, min_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = (self.max_lr - self.warmup_start_lr) / self.warmup_steps * self.last_epoch + \
                self.warmup_start_lr
        else:
            cos_decay = 0.5 * (1 + np.cos(np.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            lr = (self.max_lr - self.min_lr) * cos_decay + self.min_lr
        return [lr for _ in self.base_lrs]