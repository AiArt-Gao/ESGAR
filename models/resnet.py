import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, models

"""
Adopt the pretrained resnet model to extract feature of the feature
"""


class FeatureExtraction(nn.Module):
    def __init__(self, pretrained, model_type="Resnet18"):
        super(FeatureExtraction, self).__init__()
        if model_type == "Resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_type == "Resnet34":
            self.model = models.resnet34(pretrained=pretrained)

        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, image):
        return self.model(image)

class FeatureClassfier(nn.Module):
    def __init__(self, model_type='Resnet18'):
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
    def __init__(self, model_type, pretrained, selected_attrs=None):
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
