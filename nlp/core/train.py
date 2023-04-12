import os

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from ..utils.utils import count_parameters
from ..utils.weight_initialization import select_weight_initialize_method
from .base import AbstractTools


class Trainer(AbstractTools):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.model = self.get_model()
        self.model.train()
        self.optimizer = self.init_optimizer()

        select_weight_initialize_method(
            method=self.arg.model.weight_init,
            distribution=self.arg.model.weight_distribution,
            model=self.model,
        )

        self.train_loader, self.valid_loader = self.get_loader()
        self.loss_funtion = nn.CrossEntropyLoss(
            ignore_index=self.arg.pad_id,
            label_smoothing=self.arg.trainer.label_smoothing_value,
        )

    def train(self):
        print(f"The model {count_parameters(self.model)} trainerble parameters.")

        epoch_step = len(self.train_loader) + 1  # 한 epoch 스텝수
        total_step = self.arg.trainer.epochs * epoch_step
        step = 0

        for epoch in range(self.arg.trainer.epochs):
            for idx, data in enumerate(self.train_loader, 1):
                try:
                    self.optimizer.zero_grad()
                    src_input, trg_input, trg_output = data
                    output = self.model(src_input, trg_input)
                    loss = self.calculate_loss(output, trg_output)

                    if step % self.arg.trainer.print_train_step == 0:
                        print(
                            f"[Train] epoch {epoch:2d} iter: {epoch_step:4d}/{step:4d} step: {step:6d}/{total_step:6d} => loss: {loss.items():10f}"
                        )

                    if step % self.arg.trainer.print_valid_step == 0:
                        val_loss = self.valid()
                        print(
                            f"[Train] epoch {epoch:2d} iter: {epoch_step:4d}/{step:4d} step: {step:6d}/{total_step:6d} => loss: {val_loss:10f}"
                        )

                    if step % self.arg.trainer.svae_step == 0:
                        self.save_model(epoch, step)

                    loss.backward()
                    self.optimizer.step()
                    step += 1

                except Exception as e:
                    self.save_model(epoch, step)
                    raise e

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

        return optimizer_type

    def calculate_loss(self, predict: Tensor, target: Tensor) -> Tensor:
        """_summary_

        Args:
            predict (Tensor): [batch_size, max_seq_size, vocab_size]
            target (Tensor): [batch_size, max_seq_size]

        Returns:
            Tensor: _description_
        """
        predict = predict.transpose(1, 2)
        return self.loss_function(predict, target)

    def save_model(self, epoch: int, step: int) -> None:
        model_name = f"{str(step).zfill(6)}_{self.arg.model.model_type}.pth"
        model_path = os.path.join(self.arg.data.model_path, model_name)
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "data": self.arg.data,  # data, model, trainer는 보통 Option
                "model": self.arg.model,
                "trainer": self.arg.trainer,
                "model_state_dict": self.model.state_dict(),
            },
            model_path,
        )

    def valid(self) -> float:
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data in self.valid_loader:
                src_input, trg_input, trg_output = data
                output = self.model(src_input, trg_input)
                loss = self.calculate_loss(output, trg_output)
                total_loss += loss.items()

        # validation sample 확인
        input_sentence = self.tensor2sentence(src_input[0].tolist(), self.src_vocab)
        predict_sentence = self.tensor2sentence(
            output.topk(1)[1].squeeze()[0, :].tolist(), self.trg_vocab
        )
        target_sentence = self.tensor2sentence(trg_input[0].tolist(), self.trg_vocab)
        self.print_result(input_sentence, predict_sentence, target_sentence)
        self.model.train()

        return total_loss / len(self.valid_loader)
