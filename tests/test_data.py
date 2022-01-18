import os
import torch
from src.data.make_dataset import DataParser
import pytest
from tests import _PATH_DATA

@pytest.mark.skipif(
    not os.path.exists(f"{_PATH_DATA}/processed/train.pth"),
    reason="Data files not found",
)
def test_load_training_data():
    train_dataset = torch.load(f"{_PATH_DATA}/processed/train.pth")
    assert (len(train_dataset) == 40000), "Dataset did not have the correct number of samples"
    train_labels = []
    for image, label in iter(train_dataset):
        assert image.shape == torch.Size([28, 28]), "Expected each sample to have shape [28, 28]"
        train_labels.append(label.item())
    assert list(sorted(set(train_labels))) == [i for i in range(10)], "All labels must be represented"
