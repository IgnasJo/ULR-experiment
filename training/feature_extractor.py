# --- DEFINE FEATURE EXTRACTOR (AM-RADIO) ---
import torch
import torch.nn as nn
from torchvision.models import vgg19

class RADIOFeatureExtractor(nn.Module):
    def __init__(self):
        super(RADIOFeatureExtractor, self).__init__()
        print("Loading NVIDIA RADIOv2.5-g model... (this may take a moment)")
        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-g', progress=True)
        self.model.eval()
        
        # Freeze parameters to ensure it acts as a fixed feature extractor
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, no_grad=False):
        # RADIO expects normalized inputs, but usually handles standard ImageNet norm.
        # It returns a tuple, we usually want the patch embeddings or the summary token.
        # For perceptual loss, we use the intermediate features or the final projection.
        # Based on RADIO API, simple forward returns features.
        # 
        # no_grad=True for real images (don't need gradients)
        # no_grad=False for fake images (need gradients to flow back to generator)
        if no_grad:
            with torch.no_grad():
                _, features = self.model(x)
                return features
        else:
            # Allow gradients to flow through for generator training
            _, features = self.model(x)
            return features
        

class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        # Use a pre-trained VGG19 network 
        vgg = vgg19(pretrained=True).features
        
        # In ESRGAN, features are taken BEFORE activation [cite: 113]
        # Specifically, layers like VGG19-54 (before 5th pooling) are used 
        self.layers = nn.Sequential(*list(vgg.children())[:35]).eval()
        
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, input):
        vgg_features = self.layers(input)
        return vgg_features