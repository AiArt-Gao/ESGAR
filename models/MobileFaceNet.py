'''
Adapted from https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/mobilefacenet.py
Original author cavalleria
'''

import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, functional as F
import torch


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c),
            PReLU(num_parameters=out_c)
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)


class DepthWise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        )

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(c, c, True, kernel, stride, padding, groups))
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        # self.layers = nn.Sequential(
        #     LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
        #     Flatten(),
        #     Linear(512 *8*8, embedding_size*8*8, bias=False),
        #     BatchNorm1d(embedding_size *8*8))
        self.a = LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.b = Flatten()
        self.c = Linear(512, embedding_size, bias=False)
        self.d = BatchNorm1d(embedding_size)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.a(x)
        x = self.pooling(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        return x


class MobileFaceNet(Module):
    def __init__(self, fp16=False, num_features=512, blocks=(1, 4, 6, 2), scale=2):
        super(MobileFaceNet, self).__init__()
        self.scale = scale
        self.fp16 = fp16
        self.layers = nn.ModuleList()
        self.layers.append(
            ConvBlock(3, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
            )
        else:
            self.layers.append(
                Residual(64 * self.scale, num_block=blocks[0], groups=128, kernel=(3, 3), stride=(1, 1),
                         padding=(1, 1)),
            )

        self.layers.extend(
            [
                DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128),
                Residual(64 * self.scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1),
                         padding=(1, 1)),
                DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256),
                Residual(128 * self.scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1),
                         padding=(1, 1)),
                DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512),
                Residual(128 * self.scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1),
                         padding=(1, 1)),
            ])

        self.conv_sep = ConvBlock(128 * self.scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.features = GDC(num_features)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            for func in self.layers:
                x = func(x)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def get_mbf(fp16, num_features, blocks=(1, 4, 6, 2), scale=2):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)


def get_mbf_large(fp16, num_features, blocks=(2, 8, 12, 4), scale=4):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.model = MobileFaceNet()

    def forward(self, image):
        return self.model(image)


"""
judge the attributes from the result of feature extraction
"""

class FeatureClassfier(nn.Module):
    def __init__(self):
        super(FeatureClassfier, self).__init__()

        num_features = 512
        self.featureClassfier_hair = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2))
        self.featureClassfier_hair_color = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5))
        self.featureClassfier_gender = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2))
        self.featureClassfier_earring = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2))
        self.featureClassfier_smile = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2))
        self.featureClassfier_frontal = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2))
        self.featureClassfier_style = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3))

        self.apply(self.all_init_weights)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return {
            'hair': self.featureClassfier_hair(x),
            'hair_color': self.featureClassfier_hair_color(x),
            'gender': self.featureClassfier_gender(x),
            'earring': self.featureClassfier_earring(x),
            'smile': self.featureClassfier_smile(x),
            'frontal_face': self.featureClassfier_frontal(x),
            'style': self.featureClassfier_style(x),
        }

    def all_init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)


"""
conbime the extraction and classfier
"""

class FaceAttrModel(nn.Module):
    def __init__(self, model_type, pretrained):
        super(FaceAttrModel, self).__init__()
        self.featureExtractor = FeatureExtraction(pretrained, model_type)
        self.featureClassfier = FeatureClassfier(model_type)

    def forward(self, image):

        features = self.featureExtractor(image)
        return self.featureClassfier(features)

    def get_loss(self, net_output, ground_truth, device):
        hair_loss = F.cross_entropy(net_output['hair'], ground_truth['hair'].to(device))
        hair_color_loss = F.cross_entropy(net_output['hair_color'], ground_truth['hair_color'].to(device))
        gender_loss = F.cross_entropy(net_output['gender'], ground_truth['gender'].to(device))
        earring_loss = F.cross_entropy(net_output['earring'], ground_truth['earring'].to(device))
        smile_loss = F.cross_entropy(net_output['smile'], ground_truth['smile'].to(device))
        frontal_face_loss = F.cross_entropy(net_output['frontal_face'], ground_truth['frontal_face'].to(device))
        style_loss = F.cross_entropy(net_output['style'], ground_truth['style'].to(device))
        loss = hair_loss + gender_loss + hair_color_loss + earring_loss + smile_loss + frontal_face_loss + style_loss

        return loss, {'hair': hair_loss,
                      'gender': gender_loss,
                      'hair_color': hair_color_loss,
                      'earring': earring_loss,
                      'smile': smile_loss,
                      'frontal_face': frontal_face_loss,
                      'style': style_loss
                      }


