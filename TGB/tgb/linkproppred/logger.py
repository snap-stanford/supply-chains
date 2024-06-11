from abc import ABC, abstractmethod
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import wandb

class LoggerMixin(ABC):
    def __init__(self):
        self.logged_metrics = ["loss", "logits_loss", "inv_loss", "perf_metric_val"]

    @abstractmethod
    def log(self, **kwargs):
        pass

    @abstractmethod
    def __del__(self):
        pass

class TensorboardLogger(LoggerMixin):
    def __init__(self):
        self.writer = SummaryWriter()
        self.logged_metrics = ["loss", "logits_loss", "inv_loss", "perf_metric_val"]
        self._n_iter = 0

    def log(self, metrics):
        for metric in self.logged_metrics:
            self.writer.add_scalar(f"{metric}",  metrics[metric], self._n_iter)
        self._n_iter += 1

    def __del__(self):
        self.writer.close()
        

class WandbLogger(LoggerMixin):
    def __init__(self, num_neighbors=10, model_name='tgnpl', wandb_project="model-experiments", wandb_team="supply-chains", *args):
        super().__init__()
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_project,
            entity=wandb_team,
            resume="allow",
            # track hyperparameters and run metadata
            config=args
        )
        self.config = wandb.config
        wandb.summary["num_neighbors"] = num_neighbors
        wandb.summary["model_name"] = model_name

    def log(self, mode='training', **kwargs):
        wandb.log({f"{mode}_{metric}": kwargs[metric] for metric in self.logged_metrics})

    def __del__(self):
        wandb.finish()
