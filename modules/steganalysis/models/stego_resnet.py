import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class StegoResNet(nn.Module):
    """ResNet model adapted for steganography detection"""
    
    def __init__(self, input_shape=(224, 224, 3), weights='imagenet'):
        """
        Constructs a ResNet50 model adapted for steganalysis
        
        Args:
            input_shape: Input shape (height, width, channels)
            weights: Pre-trained weights to use ('imagenet' or None)
        """
        super(StegoResNet, self).__init__()
        
        # Load pre-trained ResNet50 without fully connected layers
        if weights == 'imagenet':
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.base_model = models.resnet50(weights=None)
        
        # Freeze the initial layers to preserve low-level feature detection
        for i, param in enumerate(self.base_model.parameters()):
            if i < 50:  # Freeze first 50 layers
                param.requires_grad = False
        
        # Preprocessing layer crucial for steganalysis
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        
        # Feature dimension after global pooling
        self.feature_dim = self.base_model.fc.in_features
        
        # Replace the final fully connected layer
        self.base_model.fc = nn.Identity()
        
        # Additional layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),  # *2 because we concatenate features
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Direct path through ResNet
        direct_features = self.base_model(x)
        
        # Preprocessing path
        preprocess_features = self.preprocess(x)
        preprocess_features = self.base_model(preprocess_features)
        
        # Concatenate features from both paths
        combined_features = torch.cat((direct_features, preprocess_features), dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        return output

    @staticmethod
    def build_model(input_shape=(224, 224, 3), weights='imagenet'):
        """
        Factory method to build and configure the model
        
        Args:
            input_shape: Input shape (height, width, channels)
            weights: Pre-trained weights to use ('imagenet' or None)
            
        Returns:
            A configured StegoResNet model
        """
        model = StegoResNet(input_shape, weights)
        return model
    
    @staticmethod
    def prepare_image(img_array, target_size=(224, 224)):
        """
        Prepares an image for inference
        
        Args:
            img_array: Image as numpy array
            target_size: Target size (height, width)
            
        Returns:
            Tensor prepared for the model
        """
        import torch
        import numpy as np
        import torchvision.transforms as transforms
        
        # Ensure RGB format
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] == 1:  # Grayscale with channel
            img_array = np.concatenate((img_array,) * 3, axis=-1)
        
        # Create transformer for preprocessing
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),  # This normalizes to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Apply transformations
        tensor = transform(img_array.astype(np.uint8))
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
