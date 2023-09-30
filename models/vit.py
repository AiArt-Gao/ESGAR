import torch.nn.functional as F
import torch.nn as nn
import timm

"""
Adopt the pretrained resnet model to extract feature of the feature
"""


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)

        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, image):

        return self.model(image)


"""
judge the attributes from the result of feature extraction
"""

class FeatureClassfier(nn.Module):
    def __init__(self):
        super(FeatureClassfier, self).__init__()

        num_features = 768
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

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.apply(self.all_init_weights)

    def forward(self, x):
        x = self.avgpool(x.transpose(1, 2))
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

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 need_weights=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.need_weights = need_weights

    def forward(self, x, attention_mask=None):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # mask
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attn = attn + attention_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn




"""
conbime the extraction and classfier
"""
class FaceAttrModel(nn.Module):
    def __init__(self, ):
        super(FaceAttrModel, self).__init__()

        self.featureExtractor = FeatureExtraction()
        self.featureClassfier = FeatureClassfier()

    def forward(self, image):
        x = self.featureExtractor(image)
        x = self.featureClassfier(x)
        return x

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