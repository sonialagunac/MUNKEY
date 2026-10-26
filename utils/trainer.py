from pathlib import Path
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf
from utils.model import get_checkpoint_dir


def configure_logger(cfg):
    if cfg.logging.use_wandb:
        return WandbLogger(
            name=cfg.logging.wandb_run_name,
            save_dir=cfg.logging.logdir,
            project=cfg.logging.wandb_project_name,
            entity=cfg.logging.wandb_entity,
            offline=cfg.logging.wandb_offline,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    logger = CSVLogger(cfg.logging.logdir, name="metrics")
    Path(logger.log_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(logger.log_dir) / "config.yaml", resolve=True)
    return logger


def make_trainer(cfg, logger, training=True):
    use_gpu = torch.cuda.is_available() and not cfg.training.use_cpu
    precision = "32-true"
    if use_gpu and cfg.training.fp16:
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    callbacks = []
    if training:
        callbacks = [
            ModelCheckpoint(
                dirpath=get_checkpoint_dir(cfg), filename="model",
                save_last=True, monitor="val_val/base/loss", mode="min",
                save_top_k=cfg.logging.keep_last_n_checkpoints,
                every_n_epochs=cfg.training.eval_every_n_epochs,
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]
    return Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.training.num_gpus if use_gpu else 1,
        num_nodes=cfg.training.num_nodes,
        precision=precision,
        max_epochs=cfg.training.epochs,
        check_val_every_n_epoch=cfg.training.eval_every_n_epochs,
        accumulate_grad_batches=cfg.training.accum_batches,
        num_sanity_val_steps=0 if cfg.training.skip_sanity_checks else 2,
        log_every_n_steps=1 if cfg.debug else 50,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=training,
        default_root_dir=cfg.logging.logdir,
    )
