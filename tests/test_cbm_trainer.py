from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.paths import add_repo_root_to_sys_path

add_repo_root_to_sys_path()
from src.trainers.cbm import CBMTrainer, CBMTrainerConfig, ConceptDataset


def test_cbm_trainer_runs_one_epoch():
    x = torch.randn(8, 4)
    y = (torch.rand(8, 3) > 0.5).float()
    train_ds = ConceptDataset(x, y)
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)

    cfg = CBMTrainerConfig(
        input_dim=4,
        num_labels=3,
        device=torch.device("cpu"),
    )
    trainer = CBMTrainer(cfg)
    history, best_f1, best_state = trainer.train(loader, loader, epochs=1)
    assert history
    assert best_state is not None
    assert isinstance(best_f1, float)
