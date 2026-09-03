import torch

from src.model import create_resnet50


def test_resnet50_multiclass_output_shape():
    model = create_resnet50(pretrained=False, num_classes=17)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 17)
