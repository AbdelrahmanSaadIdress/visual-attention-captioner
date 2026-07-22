import torch.nn as nn

from imgcap.models.decoder import Decoder
from imgcap.models.encoder import ResNet50


class ImgCap(nn.Module):
    def __init__(self, feature_size: int = 2048, lstm_hidden_size: int = 512,
                 num_layers: int = 2, vocab_size: int = 50,
                 embedding_dim: int = 512, attention_dim: int = 512,
                 dropout: float = 0.5):
        super().__init__()
        self.encoder = ResNet50()
        self.decoder = Decoder(
            feature_size=feature_size,
            lstm_hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )

    def forward(self, images, captions, teacher_forcing_ratio: float = 0.5):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, teacher_forcing_ratio)
        return outputs
