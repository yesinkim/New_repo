import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..utils.utils import count_parameters
from ..utils.weight_initialization import select_weight_initialize_method
from .base import AbstracTools


class Trainer(AbstracTools):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.init_optimizer = None

    def train(self):
        model = self.get_model()
        model.train()
        print(f"The model {count_parameters(model)} trainerble parameters.")

        select_weight_initialize_method(
            method=self.arg.model.weight_init,
            distribution=self.arg.model.weight_distribution,
            model=model,
        )

    # TODO: model과 optimizer 선언을 initilize 할 때 해주는 방법도 있기 때문에 고민해볼 것!
    def init_optimizer(self, model: nn.Module) -> None:
        optimizer_type = self.arg.trainer.optimizer
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        elif optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        else:
            raise ValueError("trainer param 'optimizer' must be one of [Adam, AdamW].")
