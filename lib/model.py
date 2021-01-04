from torch import nn
import torch.nn.functional as F
import torchvision


class DenseNetUprightAdjustment(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNetUprightAdjustment, self).__init__()
        densenet = torchvision.models.densenet121(pretrained=pretrained)
        self.features = densenet.features
        self.n_features = densenet.classifier.in_features
        self.regressor = nn.Linear(93184, 3)
        self.parameters_list = list(self.parameters())

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x).contiguous()
        x = x.view(batch_size, -1)
        x = self.regressor(x)
        x = F.normalize(x)
        return x


def get_model(pretrained=True):
    model = DenseNetUprightAdjustment(pretrained=pretrained)

    return model
