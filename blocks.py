import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SELayer, self).__init__()
        self.SE = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(in_channels // reduction),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                )
        

    def forward(self, x):
        atten = self.SE(x)
        atten = torch.clamp(atten + 3, 0, 6) / 6
        return x * atten


class HS(nn.Module):

	def __init__(self):
		super(HS, self).__init__()

	def forward(self, inputs):
		clip = torch.clamp(inputs + 3, 0, 6) / 6
		return inputs * clip



class Blcok(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels, ksize, stride, activation, useSE):
        super(Shufflenet, self).__init__()

        conv1 = [
            # pw
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation
            ]
        conv2 = [
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=ksize, stride=stride, padding=ksize//2, groups = mid_channels,bias=False),
            nn.BatchNorm2d(mid_channels),
            activation
        ]
        conv3 = [
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
            
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)
        self.conv3 = nn.Sequential(*conv3)

        self.shortcut = (stride == 1 and in_channels == out_channels)

        self.SE = None
        if useSE:
            self.SE = nn.Sequential(SELayer(mid_channels))


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.SE is not None:
            out = self.SE(out)

        out = self.conv3(out)

        if self.shortcut:
            out = out + x
        return out
