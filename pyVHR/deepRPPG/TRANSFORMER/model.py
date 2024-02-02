import torch
import torch.nn as nn

class rPPGTransformer(nn.Transformer):
    """Documentation for rPPGTransformer

    """
    def __init__(self, emb_dim, nhead, num_encoder_layers=12, num_decoder_layers=12):
        super(rPPGTransformer, self).__init__(emb_dim, nhead, num_decoder_layers, num_encoder_layers)


    def forward(self, X):
        Y = super(rPPGTransformer,self).forward(X,X)
        return torch.mean(Y, axis=1)
        
