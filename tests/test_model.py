import torch

from src.model import create_resnet50


def test_resnet50_binary_output_shape():
    model = create_resnet50(pretrained=False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, 1)

