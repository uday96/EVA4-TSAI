import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Sequential):
    def __init__(self, in_planes, out_planes, args):
        super(Block, self).__init__()

        self.x5_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout), 
        )
        self.x6_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout),  
        )
        self.x7_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Dropout(args.dropout), 
        )

    def forward(self, x4):
        x5 = self.x5_block(x4)
        x6 = self.x6_block(x4+x5)
        x7 = self.x7_block(x4+x5+x6)
        return x5, x6, x7


class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()

        self.x1_block = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout),  
        )
        self.x2_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout), 
        )
        self.x3_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Dropout(args.dropout),
        )
        self.x4_pool = nn.MaxPool2d(2, 2) 
        self.x5_6_7_block = Block(64, 64, args)
        self.x8_pool = nn.MaxPool2d(2, 2) 
        self.x9_10_11_block = Block(64, 64, args)

        self.idm1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        self.xm2_block = Block(64, 64, args)
        self.idm2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        self.xm4_block = Block(64, 64, args)
        self.idm4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        self.finm = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.idd1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        self.xd2_block = Block(64, 64, args)
        self.idd2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        self.xd4_block = Block(64, 64, args)
        self.idd4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        self.find = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x1 = self.x1_block(x)
        x2 = self.x2_block(x1)
        x3 = self.x3_block(x1+x2)
        x4 = self.x4_pool(x1+x2+x3)
        x5, x6, x7 = self.x5_6_7_block(x4)
        x8 = self.x8_pool(x5+x6+x7)
        x9, x10, x11 = self.x9_10_11_block(x8)
        
        outm = self.idm1(x11)
        outm = F.interpolate(outm, scale_factor=2, mode='bilinear')
        _, _, outm = self.xm2_block(outm + x7)
        outm = self.idm2(outm)
        outm = F.interpolate(outm, scale_factor=2, mode='bilinear')
        _, _, outm = self.xm4_block(outm + x3)
        outm = self.idm4(outm)
        outm = self.finm(outm)

        outd = self.idd1(x11)
        outd = F.interpolate(outd, scale_factor=2, mode='bilinear')
        _, _, outd = self.xd2_block(outd + x7)
        outd = self.idd2(outd)
        outd = F.interpolate(outd, scale_factor=2, mode='bilinear')
        _, _, outd = self.xd4_block(outd + x3)
        outd = self.idd4(outd)
        outd = self.finm(outd)
        return outm, outd

if __name__ == '__main__':
    class DNNArg:
        dropout = 0.0

    net = DNN(DNNArg())
    y = net(torch.randn(1,3,224,224))
    print(y.size())
