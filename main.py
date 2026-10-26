from pathlib import Path
import hydra
import torch
from lightning.pytorch import seed_everything
from configs.config import TrainConfig, register_conf
from experiments.mul import MUL
from utils.model import get_checkpoint_dir
from utils.trainer import configure_logger, make_trainer


torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="configs", config_name="cifar10")
def main(cfg: TrainConfig):
    seed_everything(cfg.seed, workers=True)
    checkpoint = cfg.training.resume_training_from
    if checkpoint is None and cfg.training.load_previous_checkpoint:
        candidate = Path(get_checkpoint_dir(cfg)) / "last.ckpt"
        if candidate.is_file():
            checkpoint = str(candidate)
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    if cfg.training.resume_eval_only and checkpoint is None:
        raise ValueError("Evaluation-only mode requires training.resume_training_from or an existing last.ckpt")

    logger = configure_logger(cfg)
    experiment = MUL(cfg)
    try:
        if cfg.training.resume_eval_only:
            # Checkpoints contain the saved Hydra config as well as model weights.
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            experiment.on_load_checkpoint(saved)
            experiment.load_state_dict(saved["state_dict"], strict=True)
        else:
            trainer = make_trainer(cfg, logger)
            trainer.fit(experiment, ckpt_path=checkpoint, weights_only=False)
            # Always leave a final checkpoint, even between validation epochs.
            trainer.save_checkpoint(str(Path(get_checkpoint_dir(cfg)) / "last.ckpt"))
            del trainer

        trainer = make_trainer(cfg, logger, training=False)
        experiment.test_unlearn = False
        trainer.test(experiment)
        if not experiment.retrain:
            experiment.test_unlearn = True
            trainer.test(experiment)
    finally:
        if cfg.logging.use_wandb:
            import wandb
            wandb.finish()
        else:
            logger.finalize("success")


if __name__ == "__main__":
    register_conf()
    main()
