import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Upsample, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.convA = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_channels, output_channels, 3, 1, 1)

    def forward(self, x, concat_with):

        concat_h_dim = concat_with.shape[2]
        concat_w_dim = concat_with.shape[3]

        upsampled_x = F.interpolate(
            x, size=[concat_h_dim, concat_w_dim], mode="bilinear", align_corners=True
        )
        upsampled_x = torch.cat([upsampled_x, concat_with], dim=1)

        upsampled_x = self.convA(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)
        upsampled_x = self.convB(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)

        return upsampled_x


class Encoder(nn.Module):
    def __init__(self, encoder_pretrained=True, type="densenet"):
        super(Encoder, self).__init__()

        self.type = type

        if type == "densenet":
            if encoder_pretrained:
                self.backend = timm.create_model(
                    "densenet161", pretrained=True)
            else:
                self.backend = timm.create_model(
                    "densenet161", pretrained=False)
        elif type == "efficientnet":
            if encoder_pretrained:
                self.backend = torch.hub.load(
                    'rwightman/gen-efficientnet-pytorch', "tf_efficientnet_b5_ap", pretrained=True)
            else:
                self.backend = torch.hub.load(
                    'rwightman/gen-efficientnet-pytorch', "tf_efficientnet_b5_ap", pretrained=False)
        elif type == "mobilevit":
            self.backend = timm.create_model("mobilevit_s", pretrained=True)

    def forward(self, x):
        feature_maps = [x]
        if self.type == "densenet":
            for key, value in self.backend.features._modules.items():
                feature_maps.append(value(feature_maps[-1]))
        elif self.type == "efficientnet":
            for k, v in self.backend._modules.items():
                if (k == 'blocks'):
                    for ki, vi in v._modules.items():
                        feature_maps.append(vi(feature_maps[-1]))
                else:
                    feature_maps.append(v(feature_maps[-1]))
        elif self.type == "mobilevit":
            for key, value in self.backend._modules.items():
                if key == "stages":
                    for k, v in value._modules.items():
                        feature_maps.append(v(feature_maps[-1]))
                else:
                    feature_maps.append(value(feature_maps[-1]))

        return feature_maps


class Decoder(nn.Module):
    def __init__(self, decoder_width=0.5, scales=[1, 2, 4, 8], type="densenet"):
        super(Decoder, self).__init__()

        self.type = type
        if self.type == "densenet":
            num_features = 2208
            features = int(num_features * decoder_width)

            self.conv2 = nn.Conv2d(
                in_channels=num_features, out_channels=features,
                kernel_size=1, stride=1, padding=1)

            self.upsample1 = Upsample(
                features // scales[0] + 384, features // (scales[0] * 2))
            self.upsample2 = Upsample(
                features // scales[1] + 192, features // (scales[1] * 2))
            self.upsample3 = Upsample(
                features // scales[2] + 96, features // (scales[2] * 2))
            self.upsample4 = Upsample(
                features // scales[3] + 96, features // (scales[3] * 2))

            self.conv3 = nn.Conv2d(
                in_channels=features // (scales[3] * 2), out_channels=1,
                kernel_size=3, stride=1, padding=1)

        elif self.type == "efficientnet":
            num_features = 2048
            features = int(num_features * decoder_width)

            self.conv2 = nn.Conv2d(
                in_channels=num_features, out_channels=features,
                kernel_size=1, stride=1, padding=1)

            self.upsample1 = Upsample(
                features // scales[0] + 176, features // (scales[0] * 2))
            self.upsample2 = Upsample(
                features // scales[1] + 64, features // (scales[1] * 2))
            self.upsample3 = Upsample(
                features // scales[2] + 40, features // (scales[2] * 2))
            self.upsample4 = Upsample(
                features // scales[3] + 24, features // (scales[3] * 2))

            self.conv3 = nn.Conv2d(
                in_channels=features // (scales[3] * 2), out_channels=1,
                kernel_size=3, stride=1, padding=1)

        elif self.type == "mobilevit":
            num_features = 640
            features = int(num_features * decoder_width)
            
            self.conv2 = nn.Conv2d(
                in_channels=num_features, out_channels=features,
                kernel_size=1, stride=1, padding=1)

            self.upsample1 = Upsample(
                features // scales[0] + 128, features // (scales[0] * 2))
            self.upsample2 = Upsample(
                features // scales[1] + 96, features // (scales[1] * 2))
            self.upsample3 = Upsample(
                features // scales[2] + 64, features // (scales[2] * 2))
            self.upsample4 = Upsample(
                features // scales[3] + 32, features // (scales[3] * 2))

            self.conv3 = nn.Conv2d(
                in_channels=features // (scales[3] * 2), out_channels=1,
                kernel_size=3, stride=1, padding=1)

    def forward(self, features):

        if self.type == "densenet":
            x_block0 = features[2]
            x_block1 = features[3]
            x_block2 = features[5]
            x_block3 = features[7]
            x_block4 = features[10]

            x = self.conv2(x_block4)
            x = self.upsample1(x, x_block3)
            x = self.upsample2(x, x_block2)
            x = self.upsample3(x, x_block1)
            x = self.upsample4(x, x_block0)

        elif self.type == "efficientnet":
            x_block0 = features[4]
            x_block1 = features[5]
            x_block2 = features[6]
            x_block3 = features[8]
            x_block4 = features[11]

            x = self.conv2(x_block4)
            x = self.upsample1(x, x_block3)
            x = self.upsample2(x, x_block2)
            x = self.upsample3(x, x_block1)
            x = self.upsample4(x, x_block0)

        elif self.type == "mobilevit":
            x_block0 = features[2]
            x_block1 = features[3]
            x_block2 = features[4]
            x_block3 = features[5]
            x_block4 = features[7]

            x = self.conv2(x_block4)
            x = self.upsample1(x, x_block3)
            x = self.upsample2(x, x_block2)
            x = self.upsample3(x, x_block1)
            x = self.upsample4(x, x_block0)

        return self.conv3(x)


class DenseDepth(nn.Module):
    def __init__(self, encoder_pretrained=True, type="efficientnet"):
        super(DenseDepth, self).__init__()

        self.encoder = Encoder(
            encoder_pretrained=encoder_pretrained, type=type)
        self.decoder = Decoder(type=type)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x
