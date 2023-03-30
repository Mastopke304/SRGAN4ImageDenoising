import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, inC, outC):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(inC, outC, kernel_size=3, stride=1, padding=1, bias=False), 
                                    nn.BatchNorm2d(outC), 
                                    nn.PReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(outC, outC, kernel_size=3, stride=1, padding=1, bias=False), 
                                    nn.BatchNorm2d(outC))

    def forward(self, x):
        resudial = x

        out = self.layer1(x)
        out = self.layer2(out)
        out = out + resudial

        return out


class Generator(nn.Module):
    def __init__(self, n_blocks):
        super(Generator, self).__init__()
        self.convlayer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, bias=False),
                                        nn.PReLU())

        self.ResBlocks = nn.ModuleList([ResBlock(64, 64) for _ in range(n_blocks)])

        self.convlayer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), 
                                        nn.BatchNorm2d(64))

        self.convout = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=False)

    def forward(self, x):
        out = self.convlayer1(x)
        residual = out

        for block in self.ResBlocks:
            out = block(out)

        out = self.convlayer2(out)
        out = out + residual

        out = self.convout(out)

        return out

class DownSample(nn.Module):
    def __init__(self, input_channel, output_channel,  stride, kernel_size=3, padding=1):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
                                   nn.BatchNorm2d(output_channel),
                                   nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                   nn.LeakyReLU(inplace=True))

        self.down = nn.Sequential(DownSample(64, 64, stride=2, padding=1),
                                  DownSample(64, 128, stride=1, padding=1),
                                  DownSample(128, 128, stride=2, padding=1),
                                  DownSample(128, 256, stride=1, padding=1),
                                  DownSample(256, 256, stride=2, padding=1),
                                  DownSample(256, 512, stride=1, padding=1),
                                  DownSample(512, 512, stride=2, padding=1))

        self.dense = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(512, 1024, 1),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(1024, 1, 1),
                                   nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.down(x)
        x = self.dense(x)
        return x