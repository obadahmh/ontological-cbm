from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.paths import add_repo_root_to_sys_path

add_repo_root_to_sys_path()
from src.trainers.concept_classifier import ConceptSample, ConceptImageDataset, build_model, split_samples, build_transforms  # noqa: E402


def test_concept_classifier_split_and_model():
    samples = [
        ConceptSample(image_path=Path("img1"), study_id="s1", targets=torch.tensor([1., 0.])),
        ConceptSample(image_path=Path("img2"), study_id="s2", targets=torch.tensor([0., 1.])),
        ConceptSample(image_path=Path("img3"), study_id="s3", targets=torch.tensor([1., 1.])),
    ]
    train, test = split_samples(samples, train_frac=0.67, seed=0)
    assert train and test

    # Dataset can be constructed (doesn't load images here)
    train_tf, eval_tf = build_transforms(image_size=64)
    ds = ConceptImageDataset(train, transform=train_tf)
    assert len(ds) == len(train)

    model = build_model("simple_cnn", num_outputs=2, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape[-1] == 2
