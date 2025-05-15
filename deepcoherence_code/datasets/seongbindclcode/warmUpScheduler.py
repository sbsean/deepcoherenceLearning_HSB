import torch

class FixedWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    수정된 워밍업 스케줄러:
    - 최소 학습률(min_lr)에서 시작하여 워밍업 단계 동안 선형으로 목표 학습률(target_lr)까지 증가
    - 워밍업 종료 후에는 after_scheduler(예, CosineAnnealingWarmRestarts)를 적용
    """

    def __init__(
        self, optimizer, warmup_steps, min_lr, target_lr, after_scheduler=None
    ):
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step >= self.warmup_steps:
            if self.after_scheduler and not self.finished:
                self.finished = True
                self.after_scheduler.base_lrs = [self.target_lr for _ in self.base_lrs]
                return self.after_scheduler.get_lr()
            return [self.target_lr for _ in self.base_lrs]

        factor = float(self.last_step) / float(max(1, self.warmup_steps))
        return [
            self.min_lr + factor * (self.target_lr - self.min_lr) for _ in self.base_lrs
        ]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        if step >= self.warmup_steps:
            if self.after_scheduler and not self.finished:
                self.after_scheduler.step(step - self.warmup_steps)
            else:
                super(FixedWarmupScheduler, self).step(step)
        else:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr

        return self.get_lr()
