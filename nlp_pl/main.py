import os

import hydra
from hydra.utils import get_original_cwd
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from nlp_pl.data_helper import TranslationDataModule
from nlp_pl.train import TranslationModel


def make_config(cfg: DictConfig) -> dict[list]:
    result = {}
    result.update(dict(cfg.data))
    result.update(dict(cfg.model))
    result.update(dict(cfg.trainer))

    return result


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    model = TranslationModel(cfg)
    datamodule = TranslationDataModule(
        arg_data=cfg.data,
        src_vocab=model.src_vocab,
        trg_vocab=model.trg_vocab,
        max_seq_len=cfg.model.max_seq_len,
        batch_size=cfg.trainer.batch_size,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(get_original_cwd(), "./SavedModel/"),
        filename=cfg.data.foldername,
        verbose=True,
        save_top_k=5,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.trainer.early_stopping,
        verbose=True,
        mode="min",
    )
    wandb_logger = WandbLogger(
        project="Transformer pytorch lightning", name=cfg.data.folder_name
    )
    wandb_logger.log_hyperparams(make_config(cfg))
    trainer = Trainer(devices="auto", accelerator="auto", max_epochs=cfg.trainer.epochs)
    trainer.fit(
        model=model,
        datamodule=datamodule,
        callbacks=[checkpoint_callback, early_stop_callback],
    )


if __name__ == "__main__":
    train()
