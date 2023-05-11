import torch
import torch.nn as nn

# class MS_CAM(nn.Module):
#     '''
#     单特征 进行通道加权,作用类似SE模块
#     '''
#
#     def __init__(self, channels=64, r=4):
#         super(MS_CAM, self).__init__()
#         inter_channels = int(channels // r)
#
#         self.local_att = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         xl = self.local_att(x)
#         xg = self.global_att(x)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)
#         return x * wei
import torch
import torch.nn as nn
from models.deformconv import DeformConv2d

class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)


        self.local_att = nn.Sequential(
            # nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            DeformConv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            DeformConv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            DeformConv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            DeformConv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

if __name__ == "__main__":
    img = torch.randn([8, 3, 256, 256])
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=5 // 2, stride=1, bias=False)
    conv1 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=7, padding= 7 // 2, stride=1, bias=False)

    img = conv(img)
    # print(img.shape)
    attmodel = MS_CAM(channels=64)
    res = attmodel(img)
    res = conv1(res)
    print(res.shape)
