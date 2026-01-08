# --- DEFINE FEATURE EXTRACTOR (AM-RADIO) ---
import torch
import torch.nn as nn

class RADIOFeatureExtractor(nn.Module):
    def __init__(self):
        super(RADIOFeatureExtractor, self).__init__()
        print("Loading NVIDIA RADIOv2.5-g model... (this may take a moment)")
        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-g', progress=True)
        self.model.eval()
        
        # Freeze parameters to ensure it acts as a fixed feature extractor
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # RADIO expects normalized inputs, but usually handles standard ImageNet norm.
        # It returns a tuple, we usually want the patch embeddings or the summary token.
        # For perceptual loss, we use the intermediate features or the final projection.
        # Based on RADIO API, simple forward returns features.
        with torch.no_grad():
             # Output is (summary, features)
            _, features = self.model(x) 
            return features