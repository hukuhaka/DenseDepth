import torch

from .encoder import Encoder
from .decoder import Decoder

class DenseDepth(torch.nn.Module):
    def __init__(self, args):
        super(DenseDepth, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)