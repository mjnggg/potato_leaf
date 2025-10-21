# model.py
import torch
import torch.nn as nn
import timm

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f = self.backbone(x)  # [B, feat_dim]
        return self.head(f)


class ViTClassifier(nn.Module):
    def __init__(self, num_classes, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        # timm model with zeroed classifier then add our own head
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.vit.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f = self.vit(x)  # [B, feat_dim]
        return self.head(f)


class HybridEfficientTransformer(nn.Module):
    """
    EfficientNet backbone -> conv projection -> flatten spatial tokens -> TransformerEncoder -> classifier
    """
    def __init__(self, num_classes, eff_name='efficientnet_b0', pretrained=True, proj_dim=256, nhead=8, num_layers=4):
        super().__init__()
        # EfficientNet features_only to get final feature map (C,H,W)
        self.backbone = timm.create_model(eff_name, pretrained=pretrained, features_only=True)
        # take last feature map channels
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # e.g. 1280
        self.proj = nn.Conv2d(self.feature_dim, proj_dim, kernel_size=1)  # reduce channels

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=nhead, dim_feedforward=proj_dim*2, dropout=0.1, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]            # [B, C, H, W]
        feats = self.proj(feats)               # [B, proj_dim, H, W]
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).permute(2, 0, 1)  # [S=H*W, B, proj_dim]
        tokens = self.transformer(tokens)           # [S, B, proj_dim]
        pooled = tokens.mean(0)                     # [B, proj_dim]
        return self.classifier(pooled)
