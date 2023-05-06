import torch

from .encoder import Encoder
from .decoder import Decoder

class DenseDepth(torch.nn.Module):
    def __init__(self, encoder_model):
        super(DenseDepth, self).__init__()
        self.encoder = Encoder(encoder_model)
        self.decoder = Decoder(encoder_model)
        
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)