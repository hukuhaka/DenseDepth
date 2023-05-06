from torch import nn

import timm


class Encoder(nn.Module):
    def __init__(self, encoder_model):
        super(Encoder, self).__init__()

        if encoder_model == "densenet":
            self.model = timm.create_model("densenet169", pretrained=True)
        elif encoder_model == "efficientnet":
            self.model = timm.create_model(
                "tf_efficientnet_b5_ap", pretrained=True)
        elif encoder_model == "mobilevit":
            self.model = timm.create_model("mobilevit_s", pretrained=True)
        elif encoder_model == "mobilevitv2":
            self.model = timm.create_model("mobilevitv2_200", pretrained=True)

        self.model_name = encoder_model

    def forward(self, x):

        feature_map = [x]

        if self.model_name == "densenet":
            for key, value in self.model.features._modules.items():
                feature_map.append(value(feature_map[-1]))

        elif self.model_name == "efficientnet":
            for key, value in self.model._modules.items():
                if key == "blocks":
                    for key2, value2 in value._modules.items():
                        feature_map.append(value2(feature_map[-1]))
                else:
                    feature_map.append(value(feature_map[-1]))

        elif self.model_name == "mobilevit":
            for key, value in self.model._modules.items():
                if key == "stages":
                    for key2, value2 in value._modules.items():
                        feature_map.append(value2(feature_map[-1]))
                else:
                    feature_map.append(value(feature_map[-1]))

        elif self.model_name == "mobilevitv2":
            for key, value in self.model._modules.items():
                if key == "stages":
                    for key2, value2 in value._modules.items():
                        feature_map.append(value2(feature_map[-1]))
                else:
                    feature_map.append(value(feature_map[-1]))

        return feature_map
