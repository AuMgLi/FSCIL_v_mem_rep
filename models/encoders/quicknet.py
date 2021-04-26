import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickNet(nn.Module):

    def __init__(self, n_classes=None):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.fc1 = nn.Linear(64 * 3 * 3, 100)

        self.n_classes = n_classes
        if n_classes is not None:
            self.classifier = nn.Linear(100, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        if self.n_classes is not None:
            x = self.classifier(x)

        return x


def debug():
    model = QuickNet(n_classes=60)
    data = torch.randn(2, 3, 32, 32)
    model = model.cuda()
    data = data.cuda()
    x = model(data)  # [2, 64, 3, 3]
    print('x:', x.shape, )


if __name__ == '__main__':
    debug()
