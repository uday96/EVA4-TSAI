import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSample(nn.Sequential):
    def __init__(self, skip_input, offset, output_features, dropout=0.0):
        super(UpSample, self).__init__()
        self.dropout = dropout
        self.up = nn.ConvTranspose2d(skip_input, skip_input, kernel_size=2, stride=2, padding=0)
        self.convA = nn.Conv2d(skip_input+offset, output_features, kernel_size=3, stride=1, padding=1)
        # self.bnA = nn.BatchNorm2d(output_features)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        # self.bnB = nn.BatchNorm2d(output_features)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        # up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        up_x = self.up(x)
        up_x = torch.cat([up_x, concat_with], dim=1)
        # up_x = self.leakyreluA(self.bnA(self.convA(up_x)))
        up_x = self.leakyreluA(self.convA(up_x))
        # up_x = F.dropout(up_x, p=0.1)
        # up_x = self.leakyreluB(self.bnB(self.convB(up_x)))
        up_x = self.leakyreluB(self.convB(up_x))
        # up_x = F.dropout(up_x, p=self.dropout)
        return up_x

class Decoder(nn.Module):
    def __init__(self, num_features=256, id_factor=0.5, dropout=0.0):
        super(Decoder, self).__init__()
        features = int(num_features * id_factor)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSample(skip_input=features//1, offset=128, output_features=features//2, dropout=dropout)
        self.up2 = UpSample(skip_input=features//2, offset=64, output_features=features//4, dropout=dropout)
        self.conv3 = nn.Conv2d(features//4, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, out_4, out_2, out_1):
        x_d0 = self.conv2(out_4)
        x_d1 = self.up1(x_d0, out_2)
        x_d2 = self.up2(x_d1, out_1)
        return self.conv3(x_d2)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.dropout = dropout

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.dropout(out, p=self.dropout)
        out = self.bn2(self.conv2(out))
        out = F.dropout(out, p=self.dropout)
        out += self.shortcut(x)
        out = F.relu(out)
        # out = F.dropout(out, p=self.dropout)
        return out

class Encoder(nn.Module):
    def __init__(self, dropout=0.0, block=BasicBlock):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, 32, 2, stride=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.layer2 = self._make_layer(block, 32, 32, 2, stride=1)
        self.layer3 = self._make_layer(block, 64, 128, 2, stride=2)
        self.layer4 = self._make_layer(block, 128, 256, 2, stride=2)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, out_planes, stride, dropout=self.dropout))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = x[:,:3,:,:]
        x2 = x[:,3:,:,:]
        out1 = F.relu(self.bn1(self.conv1(x1)))
        out1 = F.dropout(out1, p=self.dropout)
        out1_1 = self.layer1(out1)
        out2 = F.relu(self.bn2(self.conv2(x2)))
        out2 = F.dropout(out2, p=self.dropout)
        out2_1 = self.layer2(out2)
        out_1 = torch.cat([out1_1,out2_1], dim=1)   #64
        out_2 = self.layer3(out_1)  #128
        out_4 = self.layer4(out_2)  #256
        return out_4, out_2, out_1

class Net(nn.Module):
    def __init__(self, block, dropout=0.0):
        super(Net, self).__init__()
        self.dropout = dropout
        self.encoder = Encoder(dropout=dropout)
        self.decoder_m = Decoder(dropout=dropout, id_factor=0.25)
        self.decoder_d = Decoder(dropout=dropout, id_factor=0.5)

    def forward(self, x):
        out_4, out_2, out_1 = self.encoder(x)
        out_m = self.decoder_m(out_4, out_2, out_1)
        out_d = self.decoder_d(out_4, out_2, out_1)
        return out_m, out_d

def DNN(dropout=0.0):
    return Net(BasicBlock, dropout=dropout)

if __name__ == '__main__':
    net = DNN()
    ym, yd = net(torch.randn(1,6,224,224))
    print(ym.shape, yd.shape)
