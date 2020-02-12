import torch
import torch.nn as nn
from blocks import Blcok, HS, SELayer

class MobileNetV3(nn.Module):
    def __init__(self, input_size=224, n_class=1000, model_size='Large'):
        super(MobileNetV3, self).__init__()

        print('model size is ', model_size)

        ReLU = nn.ReLU(inplace=True)
        
        if model_size == 'Large':
            block_num = 15
            self.out  = [16,16,24,24,40,40,40,80,80,80,80,112,112,160,160,160,960,1280]
            mid  = [-1,16,64,72,72,120,120,240,200,184,184,480,672,672,960,960,-1]
            stride = [2,1]*3+[1,2]+[1]*5+[2]+[1]*4
            ks = [3]*4+[5]*3+[3]*6+[5]*3+[1]*2
            useSE = [False]*4+[True]*3+[False]*4+[True]*5+[False]*2
            activation = [HS()]+[ReLU]*6+[HS()]*11
        elif model_size == 'Small':
            block_num = 11
            self.out  = [16,16,24,24,40,40,40,48,48,96,96,96,576,1024]
            mid  = [-1,16,72,88,96,240,240,120,144,288,576,576,-1]
            stride = [2]*3+[1,2]+[1]*4+[2]+[1]*4
            ks = [3]*4+[5]*8+[1]*2
            useSE = [False]+[True]+[False]*2+[True]*9+[False]
            activation = [HS()]+[ReLU]*3+[HS()]*10
            
        else:
            raise NotImplementedError


        # building first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.out[0], kernel_size=ks[0], stride=stride[0], padding=ks[0]//2, bias=False),
            nn.BatchNorm2d(self.out[0]),
            activation[0]
        )

        self.features = []
        for i in range(block_num):
            self.features.append(Shufflenet(in_channels=self.out[i], out_channels=self.out[i+1], mid_channels = mid[i+1], ksize=ks[i+1], stride=stride[i+1], activation=activation[i+1], useSE=useSE[i+1]))
        self.features = nn.Sequential(*self.features)

        self.conv_last1 = nn.Sequential(
            nn.Conv2d(in_channels=self.out[-3], out_channels=self.out[-2], kernel_size=ks[-2], stride=stride[-2], padding=ks[-2]//2, bias=False),
            nn.BatchNorm2d(self.out[-2]),
            activation[-2]
        )
        self.globalpool = nn.AvgPool2d(7)
        self.conv_last2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out[-2], out_channels=self.out[-1], kernel_size=ks[-1], stride=stride[-1], padding=ks[-1]//2, bias=False),
            nn.BatchNorm2d(self.out[-1]),
            activation[-1]
        )
        self.classifier = nn.Sequential(nn.Linear(self.out[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)

        x = self.features(x)
      
        x = self.conv_last1(x)
        x = self.globalpool(x)
        x = self.conv_last2(x)
        x = x.contiguous().view(-1, self.out[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

