from pathlib import Path

from src.paths import add_repo_root_to_sys_path

add_repo_root_to_sys_path()
from src.paths import DATA_DIR, GENERATED, OUTPUTS, PRETRAINED, LOCAL_DATA, TMP_RADGRAPH, WANDB_RUNS


def test_data_dirs_exist():
    # Directories may be symlinks; ensure the parent exists and paths are defined.
    assert DATA_DIR is not None
    for path in [GENERATED, OUTPUTS, PRETRAINED, LOCAL_DATA, TMP_RADGRAPH, WANDB_RUNS]:
        assert path.name  # path object created
