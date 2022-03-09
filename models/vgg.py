import torch
import torch.nn as nn
import torchvision

cfg = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, with_latent=False):
        out = self.features(x)
        latent = out.view(out.size(0), -1)
        out = self.classifier(latent)
        if with_latent:
            return out, latent
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class InputNormalize(nn.Module):
    """
    A module (custom layer) for normalizing the input to have a fixed
    mean and standard deviation (user-specified).
    """

    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean) / self.new_std
        return x_normalized


class ModelwithInputNormalization(torch.nn.Module):
    def __init__(self, net, mean, std):
        super(ModelwithInputNormalization, self).__init__()
        self.normalizer = InputNormalize(mean, std)
        self.net = net

    def forward(self, inp):
        normalized_inp = self.normalizer(inp)
        output = self.net(normalized_inp)
        return output


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3, 1, 1)
std = torch.tensor(cifar10_std).view(3, 1, 1)

upper_limit = (1 - mu) / std
lower_limit = (0 - mu) / std


def vgg11(**kwargs):
    return VGG("vgg11", **kwargs)


def vgg13(**kwargs):
    return VGG("vgg13", **kwargs)


def vgg16(**kwargs):
    return VGG("vgg16", **kwargs)


def vgg19(**kwargs):
    return VGG("vgg19", **kwargs)


def vgg11_adv(**kwargs):
    model = vgg11(**kwargs)
    model = ModelwithInputNormalization(model, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    return model


def vgg11_nat(**kwargs):
    model = torchvision.models.vgg11_bn(**kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier[0] = nn.Linear(512 * 1 * 1, 4096)
    model = ModelwithInputNormalization(model, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    return model
