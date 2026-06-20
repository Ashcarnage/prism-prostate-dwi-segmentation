import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
import math

class PatchSizeCalculator:
    """
    Calculates optimal patch sizes for anisotropic 3D medical images
    based on the PRISM search space formulation.
    """
    
    def __init__(self, stride_parameter: int = 2):
        self.stride_parameter = stride_parameter
        
    def calculate_patch_sizes(self, 
                            image_dimensions: Tuple[int, int, int, int],
                            search_factors: List[int] = [0, 1, 2, 3, 4]) -> Dict:
        """
        Calculate patch sizes using PRISM formulation:
        Patch Size H/W = max(H, W) / S^4 - S^4 * {0, 1, 2, 3, 4}
        Patch Size Depth = D / S^4 - S^4 * {0, 1, 2, 3, 4}
        """
        channels, depth, height, width = image_dimensions
        s4 = self.stride_parameter ** 4
        
        # Calculate base patch sizes
        hw_base = max(height, width) // s4
        depth_base = depth // s4
        
        # Generate search space options
        hw_options = [max(16, hw_base * s4 - s4 * factor) for factor in search_factors]
        depth_options = [max(8, depth_base * s4 - s4 * factor) for factor in search_factors]
        
        return {
            'height_width_options': hw_options,
            'depth_options': depth_options,
            'total_combinations': len(hw_options) * len(depth_options)
        }

class SearchSpaceEncoder:
    """
    Encodes the macro search space into discrete actions for the LSTM controller.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.action_space = self._build_action_space()
        self.total_actions = len(self.action_space)
        
    def _build_action_space(self) -> Dict:
        """Build the complete action space from configuration."""
        return {
            'patch_size_hw': list(range(5)),  # 5 options for height/width
            'patch_size_depth': list(range(5)),  # 5 options for depth
            'pooling_stride_stage3': [0, 1],  # stride 1 or 2
            'pooling_stride_stage4': [0, 1],  # stride 1 or 2
            'dilation_rate_stage2': [0, 1, 2],  # rates 1, 2, 3
            'dilation_rate_stage3': [0, 1, 2],  # rates 1, 2, 3
            'dilation_rate_stage4': [0, 1, 2],  # rates 1, 2, 3
            'activation_function': [0, 1, 2],  # relu, leaky_relu, elu
            'skip_connections': list(range(2**6)),  # 6 possible skip connections
            'drop_path_operations': list(range(2**4))  # 4 possible drop-path locations
        }
    
    def encode_architecture(self, actions: List[int]) -> Dict:
        """Encode a sequence of actions into an architecture configuration."""
        if len(actions) != self.get_num_decisions():
            raise ValueError(f"Expected {self.get_num_decisions()} actions, got {len(actions)}")
        
        config = {}
        action_idx = 0
        
        # Patch size decisions
        config['patch_size_hw_factor'] = actions[action_idx]
        action_idx += 1
        config['patch_size_depth_factor'] = actions[action_idx]
        action_idx += 1
        
        # Pooling stride decisions
        config['pooling_stride_stage3'] = [1, 2][actions[action_idx]]
        action_idx += 1
        config['pooling_stride_stage4'] = [1, 2][actions[action_idx]]
        action_idx += 1
        
        # Dilation rate decisions
        config['dilation_rate_stage2'] = [1, 2, 3][actions[action_idx]]
        action_idx += 1
        config['dilation_rate_stage3'] = [1, 2, 3][actions[action_idx]]
        action_idx += 1
        config['dilation_rate_stage4'] = [1, 2, 3][actions[action_idx]]
        action_idx += 1
        
        # Activation function
        config['activation_function'] = ['relu', 'leaky_relu', 'elu'][actions[action_idx]]
        action_idx += 1
        
        # Skip connections (encoded as binary)
        skip_bits = actions[action_idx]
        config['skip_connections'] = [(skip_bits >> i) & 1 for i in range(6)]
        action_idx += 1
        
        # Drop path operations
        drop_bits = actions[action_idx]
        config['drop_path_operations'] = [(drop_bits >> i) & 1 for i in range(4)]
        
        return config
    
    def get_num_decisions(self) -> int:
        """Get the total number of architectural decisions."""
        return 10  # Total number of decisions the controller needs to make

class EfficientSkipConnection(nn.Module):
    """
    Implements memory-efficient skip connections using element-wise sum
    instead of concatenation, as specified in PRISM.
    """
    
    def __init__(self, 
                 encoder_channels: int,
                 decoder_channels: int,
                 use_1x1_conv: bool = True):
        super(EfficientSkipConnection, self).__init__()
        
        self.use_1x1_conv = use_1x1_conv
        
        if use_1x1_conv and encoder_channels != decoder_channels:
            # Use 1x1x1 convolution to match channel dimensions
            self.channel_matcher = nn.Conv3d(
                encoder_channels, 
                decoder_channels, 
                kernel_size=1,
                bias=False
            )
        else:
            self.channel_matcher = nn.Identity()
            
    def forward(self, encoder_features: torch.Tensor, 
                decoder_features: torch.Tensor) -> torch.Tensor:
        """Perform efficient skip connection using element-wise sum."""
        # Match spatial dimensions if necessary
        if encoder_features.shape[2:] != decoder_features.shape[2:]:
            encoder_features = self._resize_features(
                encoder_features, 
                decoder_features.shape[2:]
            )
        
        # Match channel dimensions
        encoder_features = self.channel_matcher(encoder_features)
        
        # Element-wise sum
        return encoder_features + decoder_features
    
    def _resize_features(self, 
                        features: torch.Tensor, 
                        target_size: Tuple[int, int, int]) -> torch.Tensor:
        """Resize features to match target spatial dimensions."""
        return nn.functional.interpolate(
            features,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

class MacroSearchSpace:
    """
    Implements the complete macro search space for PRISM.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.patch_calculator = PatchSizeCalculator()
        self.encoder = SearchSpaceEncoder(config)
        
    def sample_architecture(self, image_dims: Tuple[int, int, int, int]) -> Dict:
        """Sample a random architecture from the search space."""
        actions = []
        
        # Sample each decision
        for decision_type, options in self.encoder.action_space.items():
            if isinstance(options, list):
                actions.append(np.random.choice(options))
        
        # Encode architecture
        arch_config = self.encoder.encode_architecture(actions)
        
        # Calculate actual patch sizes
        patch_info = self.patch_calculator.calculate_patch_sizes(image_dims)
        hw_idx = arch_config['patch_size_hw_factor']
        depth_idx = arch_config['patch_size_depth_factor']
        
        arch_config['patch_size'] = (
            patch_info['depth_options'][depth_idx],
            patch_info['height_width_options'][hw_idx],
            patch_info['height_width_options'][hw_idx]
        )
        
        return arch_config
    
    def get_search_space_size(self) -> int:
        """Calculate the total size of the search space."""
        total_size = 1
        for options in self.encoder.action_space.values():
            total_size *= len(options)
        return total_size

def create_activation_function(activation_name: str) -> nn.Module:
    """Create activation function based on name."""
    activation_map = {
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.01, inplace=True),
        'elu': nn.ELU(inplace=True)
    }
    return activation_map.get(activation_name, nn.ReLU(inplace=True))

def calculate_feature_map_size(input_size: Tuple[int, int, int],
                              pooling_strides: List[int]) -> Tuple[int, int, int]:
    """Calculate feature map size after series of pooling operations."""
    d, h, w = input_size
    for stride in pooling_strides:
        d = d // stride
        h = h // stride
        w = w // stride
    return (d, h, w) 