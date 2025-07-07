import torch.nn as nn


class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, latent_dim=256, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.model = nn.Sequential(self.encoder, self.classifier)

    def forward(self, x):
        return self.model(x)

    def encode(self, x):
        return self.encoder(x)

    def classify(self, x):
        return self.classifier(x)
