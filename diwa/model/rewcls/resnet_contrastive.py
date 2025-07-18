import torch
import torch.nn as nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    def __init__(self, latent_dim=512, num_classes=2):
        super(ResNetClassifier, self).__init__()

        # Load a pretrained ResNet18
        base_resnet = models.resnet18(pretrained=True)

        # Freeze the ResNet parameters
        for param in base_resnet.parameters():
            param.requires_grad = False

        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(base_resnet.children())[:-1])  # Output: (batch_size, 512, 1, 1)

        # Add a new fully connected layer to reduce the output to the desired latent dimension (trainable)
        self.fc = nn.Linear(512 * 2, latent_dim)  # Since we concatenate two 512-d vectors

        # Final classification layer (trainable)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        """
        Forward pass for the ResNet classifier.
        :param x: Input tensor of shape (batch_size, 6, 224, 224)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        encoded_features = self.encode(x)
        output = self.classify(encoded_features)
        return output

    def encode(self, x):
        # x shape: (batch_size, 6, 224, 224)
        static_img = x[:, 0:3, :, :]  # First 3 channels
        gripper_img = x[:, 3:6, :, :]  # Last 3 channels

        # Extract features from both images
        static_feat = self.feature_extractor(static_img)  # Shape: (batch, 512, 1, 1)
        gripper_feat = self.feature_extractor(gripper_img)  # Shape: (batch, 512, 1, 1)

        # Flatten the features
        static_feat = static_feat.view(static_feat.size(0), -1)  # Shape: (batch, 512)
        gripper_feat = gripper_feat.view(gripper_feat.size(0), -1)

        # Concatenate features
        combined_feat = torch.cat((static_feat, gripper_feat), dim=1)  # Shape: (batch, 1024)

        # Pass through final FC layer
        out = self.fc(combined_feat)  # Shape: (batch, 512)

        return out

    def classify(self, x):
        """
        Classify the encoded features.
        :param x: Encoded features of shape (batch_size, 512)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        return self.classifier(x)
