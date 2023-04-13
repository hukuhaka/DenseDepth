import torch
from torch import nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode="bilinear", align_corners=True):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x, concat_data):

        concat_height, concat_width = concat_data.shape[2:]

        x = F.interpolate(
            x, size=[concat_height, concat_width], mode="bilinear", align_corners=True)
        x = torch.cat([x, concat_data], dim=1)

        x = self.upsample(x)

        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.model_name = args.encoder_model
        
        if self.model_name  == "densenet":
            num_features = 1664
            channels = [1280, 512, 256, 64, 3]
        elif self.model_name  == "efficientnet":
            num_features = 2048
            channels = [176, 64, 40, 24, 3]
        elif self.model_name  == "mobilevit":
            num_features = 640
            channels = [128, 96, 64, 32, 3]
        elif self.model_name  == "mobilevitv2":
            num_features = 1024
            channels = [768, 512, 256, 128, 3]
        
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=1)
        
        self.upsample1 = Upsample(num_features + channels[0], num_features // 2)
        self.upsample2 = Upsample(num_features // 2 + channels[1], num_features // 4)
        self.upsample3 = Upsample(num_features // 4 + channels[2], num_features // 8)
        self.upsample4 = Upsample(num_features // 8 + channels[3], num_features // 16)
        self.upsample5 = Upsample(num_features // 16 + channels[4], num_features // 32)
        
        self.conv3 = nn.Conv2d(num_features // 32, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, feature_map):
        
        if self.model_name  == "densenet":
            x0 = feature_map[0]
            x1 = feature_map[2]
            x2 = feature_map[4]
            x3 = feature_map[6]
            x4 = feature_map[8]
            x5 = feature_map[11]
        elif self.model_name  == "efficientnet":
            x0 = feature_map[0]
            x1 = feature_map[3]
            x2 = feature_map[4]
            x3 = feature_map[5]
            x4 = feature_map[7]
            x5 = feature_map[11]
        elif self.model_name  == "mobilevit":
            x0 = feature_map[0]
            x1 = feature_map[2]
            x2 = feature_map[3]
            x3 = feature_map[4]
            x4 = feature_map[5]
            x5 = feature_map[7]
        elif self.model_name  == "mobilevitv2":
            x0 = feature_map[0]
            x1 = feature_map[2]
            x2 = feature_map[3]
            x3 = feature_map[4]
            x4 = feature_map[5]
            x5 = feature_map[7]
        
        x = self.conv2(x5)
        x = self.upsample1(x, x4)
        x = self.upsample2(x, x3)
        x = self.upsample3(x, x2)
        x = self.upsample4(x, x1)
        x = self.upsample5(x, x0)
        x = self.conv3(x)
        
        return x