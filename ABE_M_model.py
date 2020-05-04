#modified
# base network
"""google net in pytorch
[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, 
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

# reproduce model
"""ABE-M
[2] Wonsik Kim, Bhavya Goyal, Kunal Chawla, Jungmin Lee, Keunjoo Kwon
    Attention-based Ensemble for Deep Metric Learning
    2018 ECCV
"""

import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

#==================================ABE_M==================================================

class ABE_M(nn.Module):
    def __init__(self, M=4, total_len=512):
        super(ABE_M, self).__init__()
        self.M = M
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            #nn.Conv2d(64, 64, kernel_size=1, stride=1),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        #"""In general, an Inception network is a network consisting of
        #modules of the above type stacked upon each other, with occasional 
        #max-pooling layers with stride 2 to halve the resolution of the 
        #grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # attention module
        self.att_a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.att_b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.att_c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.att_d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.att_e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.att_conv_branches = []
        for i in range(self.M):
            self.att_conv_branches.append(nn.Sequential(
                nn.Conv2d(832, 480, kernel_size=1,stride=1),
                nn.BatchNorm2d(480),
                nn.ReLU(inplace=True)
            ))
        self.att_conv_branches = nn.ModuleList(self.att_conv_branches)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 7*7*1024
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, int(total_len/self.M))
    
    def forward(self, x):
        output = self.prelayer(x)
        output = self.a3(output)
        output = self.b3(output)
        
        output = self.maxpool(output)

        att_outputs = []
        for i in range(self.M):
            att_module_output = self.att_a4(output)
            att_module_output = self.att_b4(att_module_output)
            att_module_output = self.att_c4(att_module_output)
            att_module_output = self.att_d4(att_module_output)
            att_module_output = self.att_e4(att_module_output)
            att_mask = self.att_conv_branches[i](att_module_output)

            att_output = torch.mul(att_mask, output)

            att_output = self.a4(att_output)
            att_output = self.b4(att_output)
            att_output = self.c4(att_output)
            att_output = self.d4(att_output)
            att_output = self.e4(att_output)

            att_output = self.maxpool(att_output)

            att_output = self.a5(att_output)
            att_output = self.b5(att_output)

            #"""It was found that a move from fully connected layers to
            #average pooling improved the top-1 accuracy by about 0.6%, 
            #however the use of dropout remained essential even after 
            #removing the fully connected layers."""
            att_output = self.avgpool(att_output)
            att_output = self.dropout(att_output)
            att_output = att_output.view(att_output.size()[0], -1)
            att_output = self.linear(att_output)
            norm = att_output.norm(dim=1, p=2, keepdim=True)
            att_output = att_output.div(norm.expand_as(att_output))

            att_outputs.append(att_output)

        return att_outputs