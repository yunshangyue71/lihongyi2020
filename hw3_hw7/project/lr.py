from torch.optim import lr_scheduler
class LR():
    def __init__(self):
        self.config = {"init": 0.001,
                   "lr_policy": "multi_stage",

                   "multi_stage": {"milestones": [20, 40, 60], "gamma": 0.3},
                   "step": {"step_size": 20, "gamma": 0.3}
                   }
    def get_lr_office(self, optimizer):
        """Return a learning rate scheduler
            Parameters:
            optimizer -- 网络优化器
            opt.lr_policy -- 学习率scheduler的名称: linear | step | plateau | cosine
            how to use: 先optimizer.step(),然后再scheduler.step(),
        """
        if self.config["lr_policy"] == 'linear':
            def lambda_rule(epoch):
                # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                # return lr_l
                return 1
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        elif self.config["lr_policy"] == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config["step"]["step_size"], gamma=self.config["step"]["gamma"])
            """
            功能： 等间隔调整学习率，调整倍数为gamma倍，调整间隔为step_size。间隔单位是step。需要注意的是，step通常是指epoch，不要弄成iteration了。
            参数：
            step_size(int) - 学习率下降间隔数，若为30，则会在30、60、90......个step时，将学习率调整为lr * gamma。
            gamma(float) - 学习率调整倍数，默认为0.1 倍，即下降10倍。
            last_epoch(int) - 上一个epoch数，这个变量用来指示学习率是否需要调整。当last_epoch符合设定的间隔时，就会对学习率进行调整。当为 - 1
            时，学习率设置为初始值。
            """

        elif self.config["lr_policy"] == 'multi_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, self.lr["muli_step"]["milestones"], gamma=self.lr["muli_step"]["gamma"], last_epoch=-1)
            """
            功能： 按设定的间隔调整学习率。这个方法适合后期调试使用，观察loss曲线，为每个实验定制学习率调整时机。
            参数：
            milestones(list) - 一个list，每一个元素代表何时调整学习率，list元素必须是递增的。如
            milestones = [30, 80, 120]
            gamma(float) - 学习率调整倍数，默认为0  .1  倍，即下降10倍。
            last_epoch(int) - 上一个epoch数，这个变量用来指示学习率是否需要调整。当last_epoch符合设定的间隔时，就会对学习率进行调整。当为 - 1
            时，学习率设置为初始值。
            """
        elif self.config["lr_policy"] == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)


        elif self.lr["lr_policy"] == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)

        else:
            return NotImplementedError('learning rate policy [%s] is not implemented',self.lr["lr_policy"])
        return scheduler
