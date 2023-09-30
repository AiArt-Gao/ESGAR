import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import timm

"""
Adopt the pretrained resnet model to extract feature of the feature
"""


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, image):
        return self.model(image)

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


class blocks(nn.Module):
    def __init__(self, num_features, num_cls):
        super(blocks, self).__init__()

        self.featureClassfier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_cls))

        self.sa = Attention(num_features, num_heads=12)
        self.norm = nn.LayerNorm(num_features)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.apply(self.all_init_weights)

    def forward(self, x, seg_mask):
        seg_mask = seg_mask.unsqueeze(-1)
        x = x * seg_mask

        x = self.sa(self.norm(x))[0] + x
        x = self.avgpool(x.transpose(1, 2))
        x = x.view(x.size(0), -1)  # flatten
        return self.featureClassfier(x)

    def all_init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)


"""
judge the attributes from the result of feature extraction
"""
class FeatureClassfier(nn.Module):
    def __init__(self):
        super(FeatureClassfier, self).__init__()

        num_features = 768
        self.featureClassfier_hair = blocks(num_features=num_features, num_cls=2)
        self.featureClassfier_hair_color = blocks(num_features=num_features, num_cls=5)
        self.featureClassfier_gender = blocks(num_features=num_features, num_cls=2)
        self.featureClassfier_earring = blocks(num_features=num_features, num_cls=2)
        self.featureClassfier_smile = blocks(num_features=num_features, num_cls=2)
        self.featureClassfier_frontal = blocks(num_features=num_features, num_cls=2)
        self.featureClassfier_style = blocks(num_features=num_features, num_cls=3)

        self.apply(self.all_init_weights)

    def forward(self, x, seg_mask):
        return {
            'hair': self.featureClassfier_hair(x, seg_mask['hair']),
            'hair_color': self.featureClassfier_hair_color(x, seg_mask['hair']),
            'gender': self.featureClassfier_gender(x, seg_mask['global']),
            'earring': self.featureClassfier_earring(x, seg_mask['earring']),
            'smile': self.featureClassfier_smile(x, seg_mask['smile']),
            'frontal_face': self.featureClassfier_frontal(x, seg_mask['global']),
            'style': self.featureClassfier_style(x, seg_mask['global']),
        }

    def all_init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)



"""
conbime the extraction and classfier
"""

class FaceAttrModel(nn.Module):
    def __init__(self):
        super(FaceAttrModel, self).__init__()
        self.featureExtractor = FeatureExtraction()
        self.featureClassfier = FeatureClassfier()

        self.hair_loss = BCE_loss(type='bal', cls_num=[1010, 48])
        self.hair_color_loss = BCE_loss(type='bal', cls_num=[288, 423, 60, 48, 239])
        self.gender_loss = BCE_loss(type='bal', cls_num=[574, 484])
        self.earring_loss = BCE_loss(type='bal', cls_num=[209, 849])
        self.smile_loss = BCE_loss(type='bal', cls_num=[645, 413])
        self.front_face_loss = BCE_loss(type='bal', cls_num=[917, 141])
        self.style_loss = BCE_loss(type='bal', cls_num=[357, 351, 350])

    def forward(self, image, ground_truth, device):
        features = self.featureExtractor(image)
        seg_mask = dict({
            'hair': ground_truth['seg_hair'].to(device),
            'global': ground_truth['seg_total'].to(device),
            'smile': ground_truth['seg_smile'].to(device),
            'earring': ground_truth['seg_earring'].to(device),
        })
        x = self.featureClassfier(features, seg_mask)
        return x

    def get_loss(self, net_output, ground_truth, device):
        hair_loss = self.hair_loss(net_output['hair'], ground_truth['hair'].to(device))
        hair_color_loss = self.hair_color_loss(net_output['hair_color'], ground_truth['hair_color'].to(device))
        gender_loss = self.gender_loss(net_output['gender'], ground_truth['gender'].to(device))
        earring_loss = self.earring_loss(net_output['earring'], ground_truth['earring'].to(device))
        smile_loss = self.smile_loss(net_output['smile'], ground_truth['smile'].to(device))
        frontal_face_loss = self.front_face_loss(net_output['frontal_face'], ground_truth['frontal_face'].to(device))
        style_loss = self.style_loss(net_output['style'], ground_truth['style'].to(device))

        loss = hair_loss + gender_loss + hair_color_loss + earring_loss + smile_loss + frontal_face_loss + style_loss
        return loss, {'hair': hair_loss,
                      'gender': gender_loss,
                      'hair_color': hair_color_loss,
                      'earring': earring_loss,
                      'smile': smile_loss,
                      'frontal_face': frontal_face_loss,
                      'style': style_loss
                      }


class BCE_loss(nn.Module):
    def __init__(self,
                target_threshold=None,
                type=None,
                cls_num=None,
                reduction='mean',
                pos_weight=None):
        super(BCE_loss, self).__init__()
        self.lam = 1.
        self.K = 1.
        self.smoothing = 0.
        self.target_threshold = target_threshold
        self.weight = None
        self.pi = None
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight)

        if type == 'Bal':
            self._cal_bal_pi(cls_num)

    def _cal_bal_pi(self, cls_num):
        cls_num = torch.Tensor(cls_num)
        self.pi = cls_num / torch.sum(cls_num)

    def _cal_cb_weight(self, args):
        eff_beta = 0.9999
        effective_num = 1.0 - np.power(eff_beta, args.cls_num)
        per_cls_weights = (1.0 - eff_beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(args.cls_num)
        self.weight = torch.FloatTensor(per_cls_weights).to(args.device)

    def _bal_sigmod_bias(self, x):
        pi = self.pi.to(x.device)
        bias = torch.log(pi) - torch.log(1-pi)
        x = x + self.K * bias
        return x

    def _neg_reg(self, labels, logits, weight=None):
        if weight == None:
            weight = torch.ones_like(labels).to(logits.device)
        pi = self.pi.to(logits.device)
        bias = torch.log(pi) - torch.log(1-pi)
        logits = logits * (1 - labels) * self.lam + logits * labels # neg + pos
        logits = logits + self.K * bias
        weight = weight / self.lam * (1 - labels) + weight * labels # neg + pos
        return logits, weight

    def _one_hot(self, x, target):
        num_classes = x.shape[-1]
        off_value = self.smoothing / num_classes
        on_value = 1. - self.smoothing + off_value
        target = target.long().view(-1, 1)
        target = torch.full((target.size()[0], num_classes),
            off_value, device=x.device,
            dtype=x.dtype).scatter_(1, target, on_value)
        return target

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            target = self._one_hot(x, target)
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        weight = self.weight
        if self.pi != None: x = self._bal_sigmod_bias(x)
        # if self.lam != None:
        #     x, weight = self._neg_reg(target, x)
        C = x.shape[-1] # + log C
        return C * F.binary_cross_entropy_with_logits(
                    x, target, weight, self.pos_weight,
                    reduction=self.reduction)

