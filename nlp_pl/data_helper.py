import lightning.pytorch as pl
import sentencepiece as spm
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from omegaconf import DictConfig
from src.datasets.data_helper import TrainDataset
from torch.utils.data import DataLoader, RandomSampler


class TranslationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        arg_data: DictConfig,
        src_vocab: spm.SentencePieceProcessor,
        trg_vocab: spm.SentencePieceProcessor,
        max_seq_len: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.arg_data = arg_data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def prepare_data(
        self,
    ) -> None:  # download, split data. only called on 1 GPU/TPU in distributed
        return super().prepare_data()

    def setup(self, stage: str) -> None:  # make assignments here (train/val/test split)
        self.train_dataset = TrainDataset(
            x_path=self.arg_data.src_train_path,
            src_vocab=self.src_vocab,
            y_path=self.arg_data.trg_train_path,
            trg_vocab=self.trg_vocab,
            max_sequence_size=self.max_seq_len,
        )

        self.valid_dataset = TrainDataset(
            x_path=self.arg_data.src_valid_path,
            src_vocab=self.src_vocab,
            y_path=self.arg_data.trg_valid_path,
            trg_vocab=self.trg_vocab,
            max_sequence_size=self.max_seq_len,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_sampler = RandomSampler(self.train_dataset)
        return DataLoader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_sampler = RandomSampler(self.valid_dataset)
        return DataLoader(
            dataset=self.valid_dataset,
            sampler=valid_sampler,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()

    def teardown(self, stage: str) -> None:  # clean up after fit or test
        return super().teardown(stage)
