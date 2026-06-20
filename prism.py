import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import wandb
from typing import List, Tuple, Dict, Optional
import random
import json
from tqdm import tqdm
from collections import OrderedDict
import math

# ============================================================================
# PRISM: Prostate Reinforcement Image Segmentation Model
# Based on: "Resource Optimized Neural Architecture Search for 3D Medical Image Segmentation" (MICCAI 2019)
# 
# ============================================================================

class PRISMConfig:
    """Configuration class for PRISM following configuration specifications"""
    def __init__(self):
        # Search space configuration (Macro search configured)
        self.patch_size_factors = [0, 1, 2, 3, 4]  # 5 options for patch size
        self.pooling_strides = [1, 2]  # 2 options
        self.dilation_rates = [1, 2, 3]  # 3 options  
        self.activation_functions = ['relu', 'leaky_relu', 'elu']  # 3 options (paper uses 3)
        
        # Controller configuration (LSTM-based RL)
        self.controller_lstm_size = 100  # Configured
        self.controller_lstm_layers = 2
        self.controller_lr = 0.001
        self.controller_entropy_weight = 0.0001
        
        # Training configuration (optimized for efficiency)
        self.episodes = 150  # Configured
        self.child_epochs_per_episode = 3  # Configured
        self.child_networks_per_episode = 20  # Configured
        self.child_lr = 0.001
        self.child_weight_decay = 0.00001
        
        # Architecture configuration
        self.base_channels = 32
        self.num_stages = 4  # Fixed 4 stages configured
        self.num_classes = 1  # Binary segmentation for prostate DWI
        
        # Memory optimization (key PRISM features)
        self.use_mixed_precision = True
        self.parameter_sharing = True  # Key PRISM innovation
        self.use_element_wise_skip = True  # Element-wise sum instead of concat
        
    def get_search_space_size(self):
        """Calculate total search space combinations"""
        # 5 patch sizes × 2 pooling × 3 dilation × 3 activation × skip choices
        return 5 * 5 * 2 * 2 * 3 * 3 * 3 * 3 * 64  # Approx macro search space

class PatchSizeCalculator:
    """Calculate optimal patch sizes for anisotropic 3D medical images following PRISM"""
    
    @staticmethod
    def calculate_patch_sizes(image_shapes: List[Tuple], stride: int = 2) -> Dict:
        """
        PRISM patch size calculation:
        Patch Size H/W = max(H, W) / S^4 - S^4 * {0, 1, 2, 3, 4}
        Patch Size Depth = D / S^4 - S^4 * {0, 1, 2, 3, 4}
        """
        # Get median dimensions for robust calculation
        depths = [shape[0] for shape in image_shapes]
        heights = [shape[1] for shape in image_shapes]
        widths = [shape[2] for shape in image_shapes]
        
        median_d = int(np.median(depths))
        median_h = int(np.median(heights))
        median_w = int(np.median(widths))
        
        # PRISM formulation
        S = stride
        S4 = S ** 4
        
        # Height/Width patch sizes (equal for square patches)
        max_hw = max(median_h, median_w)
        hw_base = max_hw // S4
        hw_sizes = [max(16, hw_base - S4 * i) for i in range(5)]
        
        # Depth patch sizes (can be different for anisotropic data)
        d_base = median_d // S4  
        d_sizes = [max(8, d_base - S4 * i) for i in range(5)]
        
        return {
            'hw_sizes': hw_sizes,
            'd_sizes': d_sizes,
            'base_channels': 32
        }

class ParameterSharingManager:
    """
    Implements parameter sharing across child networks - key PRISM innovation
    This avoids retraining from scratch and enables 1.39 day training time
    """
    def __init__(self, config: PRISMConfig, device: torch.device):
        self.config = config
        self.device = device
        self.shared_weights = {}
        self.weight_usage_count = defaultdict(int)
        
    def initialize_shared_weights(self, template_architecture: Dict):
        """Initialize shared weight bank"""
        # Create template network to get weight shapes
        template_net = ChildNetwork(
            self.config, template_architecture, 
            {'hw_sizes': [64], 'd_sizes': [32]}, 
            in_channels=1
        ).to(self.device)
        
        # Store weight templates
        for name, param in template_net.named_parameters():
            self.shared_weights[name] = param.data.clone()
            
    def apply_shared_weights(self, child_network: nn.Module):
        """Apply shared weights to child network"""
        for name, param in child_network.named_parameters():
            if name in self.shared_weights:
                # Check shape compatibility
                if param.shape == self.shared_weights[name].shape:
                    param.data.copy_(self.shared_weights[name])
                else:
                    # Handle shape mismatch with intelligent initialization
                    self._handle_shape_mismatch(param, self.shared_weights[name])
                    
    def update_shared_weights(self, child_network: nn.Module, weight: float = 0.1):
        """Update shared weights based on trained child (exponential moving average)"""
        for name, param in child_network.named_parameters():
            if name in self.shared_weights:
                if param.shape == self.shared_weights[name].shape:
                    self.shared_weights[name] = (
                        (1 - weight) * self.shared_weights[name] + 
                        weight * param.data.clone()
                    )
                    
    def _handle_shape_mismatch(self, target_param: torch.Tensor, source_param: torch.Tensor):
        """Handle parameter shape mismatches intelligently"""
        if target_param.ndim == source_param.ndim:
            # Copy compatible portions
            min_shape = tuple(min(t, s) for t, s in zip(target_param.shape, source_param.shape))
            slices = tuple(slice(0, dim) for dim in min_shape)
            target_param.data[slices].copy_(source_param[slices])
        
        # Initialize remaining with Xavier
        if len(target_param.shape) > 1:
            nn.init.xavier_uniform_(target_param.data)

class SearchSpace:
    """Macro search space for PRISM focusing on high-level architectural decisions"""
    
    def __init__(self, config: PRISMConfig):
        self.config = config
        self.action_space = self._build_action_space()
        
    def _build_action_space(self):
        """Build the action space for the controller"""
        actions = OrderedDict()
        
        # Patch size selection (2 actions: HW and D)
        actions['patch_hw'] = len(self.config.patch_size_factors)
        actions['patch_d'] = len(self.config.patch_size_factors)
        
        # Pooling decisions for stages 3 and 4 (stages 1,2 fixed configured)
        actions['pooling_stride_stage3'] = len(self.config.pooling_strides)
        actions['pooling_stride_stage4'] = len(self.config.pooling_strides)
        
        # Dilation rates for stages 2, 3, 4 (stage 1 fixed configured)
        actions['dilation_rate_stage2'] = len(self.config.dilation_rates)
        actions['dilation_rate_stage3'] = len(self.config.dilation_rates)
        actions['dilation_rate_stage4'] = len(self.config.dilation_rates)
        
        # Activation function choice
        actions['activation_function'] = len(self.config.activation_functions)
        
        # Skip connection decisions (6 possible connections in U-Net)
        actions['skip_connections'] = 64  # 2^6 combinations
                
        return actions
    
    def sample_architecture(self) -> Dict:
        """Sample a random architecture from the search space"""
        arch = {}
        for action_name, num_choices in self.action_space.items():
            arch[action_name] = random.randint(0, num_choices - 1)
        return arch
    
    def decode_architecture(self, actions: List[int]) -> Dict:
        """Decode controller actions into architecture specification"""
        arch = {}
        action_idx = 0
        
        action_names = list(self.action_space.keys())
        for i, action_name in enumerate(action_names):
            if action_idx < len(actions):
                arch[action_name] = actions[action_idx] % self.action_space[action_name]
                action_idx += 1
            else:
                arch[action_name] = 0  # Default value
                
        return arch

class LSTMController(nn.Module):
    """LSTM-based reinforcement learning controller for architecture generation"""
    
    def __init__(self, config: PRISMConfig, search_space: SearchSpace):
        super().__init__()
        self.config = config
        self.search_space = search_space
        self.num_actions = len(search_space.action_space)
        
        # LSTM controller
        self.embedding_dim = config.controller_lstm_size
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=config.controller_lstm_size,
            num_layers=config.controller_lstm_layers,
            batch_first=True
        )
        
        # Embeddings for each action type
        self.embeddings = nn.ModuleDict()
        for action_name, num_choices in search_space.action_space.items():
            self.embeddings[action_name] = nn.Embedding(
                num_choices + 1,  # +1 for start token
                self.embedding_dim
            )
        
        # Action prediction heads
        self.action_heads = nn.ModuleDict()
        for action_name, num_choices in search_space.action_space.items():
            self.action_heads[action_name] = nn.Linear(
                config.controller_lstm_size, num_choices
            )
        
        self.baseline = None
        
    def forward(self, batch_size: int = 1) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """Generate architecture and return actions, log_probs, entropies"""
        device = next(self.parameters()).device
        
        # Initialize hidden state
        h0 = torch.zeros(self.config.controller_lstm_layers, batch_size, 
                        self.config.controller_lstm_size, device=device)
        c0 = torch.zeros(self.config.controller_lstm_layers, batch_size,
                        self.config.controller_lstm_size, device=device)
        
        hidden = (h0, c0)
        actions = []
        log_probs = []
        entropies = []
        
        # Start token
        inputs = torch.zeros(batch_size, 1, self.embedding_dim, device=device)
        
        # Generate actions sequentially for each decision
        action_names = list(self.search_space.action_space.keys())
        
        for action_name in action_names:
            lstm_out, hidden = self.lstm(inputs, hidden)
            logits = self.action_heads[action_name](lstm_out.squeeze(1))
            
            # Apply temperature for exploration
            logits = torch.tanh(logits)  # Prevent extreme values
            
            # Sample action
            probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            
            actions.append(action.item())
            log_probs.append(action_dist.log_prob(action))
            entropies.append(action_dist.entropy())
            
            # Prepare next input
            next_embedding = self.embeddings[action_name](action)
            inputs = next_embedding.unsqueeze(1)
            
        return actions, torch.stack(log_probs), torch.stack(entropies)
    
    def update(self, rewards: List[float], log_probs: torch.Tensor, entropies: torch.Tensor):
        """Update controller using REINFORCE with baseline"""
        if self.baseline is None:
            self.baseline = np.mean(rewards)
        else:
            self.baseline = 0.95 * self.baseline + 0.05 * np.mean(rewards)
            
        # Calculate advantages
        advantages = torch.tensor([r - self.baseline for r in rewards], 
                                device=log_probs.device, dtype=torch.float32)
        
        # Policy loss (REINFORCE)
        policy_loss = -(log_probs.mean(0) * advantages.mean()).sum()
        
        # Entropy loss for exploration
        entropy_loss = -entropies.mean() * self.config.controller_entropy_weight
        
        total_loss = policy_loss + entropy_loss
        return total_loss, policy_loss.item(), entropy_loss.item()

class AdaptiveConv3D(nn.Module):
    """Adaptive 3D convolution with configurable activation and dilation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, activation: str = 'relu'):
        super().__init__()
        
        padding = dilation if kernel_size == 3 else 0
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                            padding=padding, dilation=dilation, bias=False)
        # Use Instance Norm as per PRISM paper for medical images
        self.norm = nn.InstanceNorm3d(out_channels)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01, inplace=True)  
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
            
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class EfficientSkipConnection(nn.Module):
    """
    PRISM efficient skip connection using element-wise sum for memory efficiency
    This is THE KEY difference from standard U-Net that enables PRISM efficiency
    """
    
    def __init__(self, encoder_channels: int, decoder_channels: int):
        super().__init__()
        
        self.match_channels = encoder_channels != decoder_channels
        if self.match_channels:
            # Use 1x1x1 conv to match channels (like DeepLabV3+)
            self.channel_matcher = nn.Conv3d(
                encoder_channels, decoder_channels, 
                kernel_size=1, bias=False
            )
        
    def forward(self, decoder_features: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
        """Perform ELEMENT-WISE SUM (not concatenation) for memory efficiency"""
        # Match channel dimensions if needed
        if self.match_channels:
            encoder_features = self.channel_matcher(encoder_features)
            
        # Resize if spatial dimensions don't match
        if decoder_features.shape[2:] != encoder_features.shape[2:]:
            encoder_features = F.interpolate(
                encoder_features, 
                size=decoder_features.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
            
        # ELEMENT-WISE SUM (key PRISM innovation)
        return decoder_features + encoder_features

class ChildNetwork(nn.Module):
    """
    PRISM Child network with proper element-wise skip connections and deep supervision
    """
    
    def __init__(self, config: PRISMConfig, architecture: Dict, 
                 patch_sizes: Dict, in_channels: int = 1):
        super().__init__()
        
        self.config = config
        self.architecture = architecture
        self.patch_sizes = patch_sizes
        
        # Build encoder
        self.encoder_stages = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        
        current_channels = in_channels
        self.encoder_channels = []
        
        for stage in range(config.num_stages):
            out_channels = config.base_channels * (2 ** stage)
            
            # Get stage-specific parameters
            if stage == 0:  # Stage 1 fixed configured
                dilation = 1
            else:
                dilation_key = f'dilation_rate_stage{stage+1}'
                if dilation_key in architecture:
                    dilation = config.dilation_rates[architecture[dilation_key]]
                else:
                    dilation = 1
                    
            activation = config.activation_functions[
                architecture.get('activation_function', 0)
            ]
            
            # Double convolution block (U-Net style)
            stage_layers = nn.Sequential(
                AdaptiveConv3D(current_channels, out_channels, 3, dilation, activation),
                AdaptiveConv3D(out_channels, out_channels, 3, 1, activation)
            )
            
            self.encoder_stages.append(stage_layers)
            self.encoder_channels.append(out_channels)
            
            # Pooling layer (except for last stage)
            if stage < config.num_stages - 1:
                if stage < 2:  # Stages 1,2 fixed stride 2
                    stride = 2
                else:  # Stages 3,4 configurable
                    stride_key = f'pooling_stride_stage{stage+1}'
                    stride = config.pooling_strides[
                        architecture.get(stride_key, 1)
                    ]
                self.pooling_layers.append(nn.MaxPool3d(kernel_size=stride, stride=stride))
                
            current_channels = out_channels
            
        # Build decoder with EFFICIENT skip connections
        self.decoder_stages = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for stage in range(config.num_stages - 2, -1, -1):
            in_channels = self.encoder_channels[stage + 1]
            out_channels = self.encoder_channels[stage]
            
            # Upsampling
            self.upsampling_layers.append(
                nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
            )
            
            # Skip connection (check if enabled)
            skip_bits = architecture.get('skip_connections', 63)  # Default all enabled
            skip_idx = config.num_stages - 2 - stage
            use_skip = bool((skip_bits >> skip_idx) & 1) if skip_idx < 6 else True
            
            if use_skip:
                self.skip_connections.append(
                    EfficientSkipConnection(out_channels, out_channels)
                )
            else:
                self.skip_connections.append(None)
                
            # Decoder block
            activation = config.activation_functions[
                architecture.get('activation_function', 0)
            ]
            decoder_block = nn.Sequential(
                AdaptiveConv3D(out_channels, out_channels, 3, 1, activation),
                AdaptiveConv3D(out_channels, out_channels, 3, 1, activation)
            )
            
            self.decoder_stages.append(decoder_block)
            
        # Final output layer
        self.final_conv = nn.Conv3d(self.encoder_channels[0], config.num_classes, 1)
        
        # Deep supervision heads (PRISM feature)
        self.deep_supervision_heads = nn.ModuleList()
        for i in range(len(self.decoder_stages) - 1):  # Exclude final stage
            channels = self.encoder_channels[config.num_stages - 2 - i]
            self.deep_supervision_heads.append(
                nn.Conv3d(channels, config.num_classes, 1)
            )
        
    def forward(self, x):
        original_size = x.shape[2:]
        
        # Encoder path
        encoder_features = []
        current = x
        
        for i, (encoder_stage, pooling) in enumerate(zip(self.encoder_stages[:-1], self.pooling_layers)):
            current = encoder_stage(current)
            encoder_features.append(current)
            current = pooling(current)
            
        # Last encoder stage (no pooling)
        current = self.encoder_stages[-1](current)
        
        # Decoder path with EFFICIENT skip connections
        deep_outputs = []
        
        for i, (upsampling, skip_conn, decoder_stage) in enumerate(
            zip(self.upsampling_layers, self.skip_connections, self.decoder_stages)
        ):
            # Upsampling
            current = upsampling(current)
            
            # Skip connection (ELEMENT-WISE SUM, not concatenation!)
            if skip_conn is not None:
                skip_feat = encoder_features[-(i+1)]  # Corresponding encoder feature
                current = skip_conn(current, skip_feat)
            
            # Decoder block
            current = decoder_stage(current)
            
            # Deep supervision output
            if i < len(self.deep_supervision_heads):
                deep_out = self.deep_supervision_heads[i](current)
                # Resize to original size
                deep_out = F.interpolate(
                    deep_out, size=original_size, 
                    mode='trilinear', align_corners=False
                )
                deep_outputs.append(deep_out)
                
        # Final output
        output = self.final_conv(current)
        output = F.interpolate(output, size=original_size, mode='trilinear', align_corners=False)
        
        if self.training and deep_outputs:
            return output, deep_outputs
        else:
            return output

class DiceLoss(nn.Module):
    """Dice loss for medical image segmentation - PRISM implementation"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors for each sample in batch
        batch_size = inputs.size(0)
        inputs_flat = inputs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Calculate intersection and union for each sample
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        total = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        # Calculate dice coefficient for each sample
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        
        # Return average dice loss across batch
        return 1 - dice.mean()

class PRISMTrainer:
    """
    Main training pipeline for PRISM with parameter sharing
    Implements the key innovation that enables 1.39 day training time
    """
    
    def __init__(self, config: PRISMConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize search space and controller
        self.search_space = SearchSpace(config)
        self.controller = LSTMController(config, self.search_space).to(device)
        
        # Parameter sharing manager (KEY PRISM INNOVATION)
        self.param_manager = ParameterSharingManager(config, device)
        
        # Controller optimizer
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(), 
            lr=config.controller_lr
        )
        
        # Loss function
        self.criterion = DiceLoss()
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Training history
        self.history = {
            'rewards': [],
            'entropies': [],
            'controller_losses': [],
            'best_architectures': []
        }
        
        # Initialize parameter sharing
        self._initialize_parameter_sharing()
        
    def _initialize_parameter_sharing(self):
        """Initialize the parameter sharing system"""
        # Create a default architecture for initialization
        default_arch = {
            'patch_hw': 2, 'patch_d': 2,
            'pooling_stride_stage3': 1, 'pooling_stride_stage4': 1,
            'dilation_rate_stage2': 0, 'dilation_rate_stage3': 0, 'dilation_rate_stage4': 0,
            'activation_function': 0, 'skip_connections': 63
        }
        
        self.param_manager.initialize_shared_weights(default_arch)
        print("✓ Parameter sharing initialized")
        
    def calculate_patch_sizes_from_data(self, data_loader):
        """Calculate patch sizes from actual data"""
        shapes = []
        for batch_idx, batch in enumerate(data_loader):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            # Assuming shape is [B, C, D, H, W]
            for i in range(x.shape[0]):
                shapes.append((x.shape[2], x.shape[3], x.shape[4]))
            if len(shapes) >= 50:  # Sample enough for good statistics
                break
                
        return PatchSizeCalculator.calculate_patch_sizes(shapes)
    
    def train_child_network(self, architecture: Dict, train_loader, val_loader, 
                           patch_sizes: Dict, epochs: int = None) -> float:
        """
        Train a child network using PARAMETER SHARING - key PRISM innovation
        This avoids training from scratch and enables efficiency
        """
        if epochs is None:
            epochs = self.config.child_epochs_per_episode
            
        # Create child network
        child_net = ChildNetwork(
            self.config, architecture, patch_sizes, 
            in_channels=1
        ).to(self.device)
        
        # APPLY SHARED WEIGHTS (key PRISM step)
        self.param_manager.apply_shared_weights(child_net)
        
        # Optimizer for child network
        optimizer = torch.optim.Adam(
            child_net.parameters(),
            lr=self.config.child_lr,
            weight_decay=self.config.child_weight_decay
        )
        
        # Training loop
        child_net.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Adapt to patch size if needed
                data, target = self._adapt_to_patch_size(data, target, architecture, patch_sizes)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if self.config.use_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = child_net(data)
                        loss = self._calculate_loss_with_deep_supervision(outputs, target)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = child_net(data)
                    loss = self._calculate_loss_with_deep_supervision(outputs, target)
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Limit training batches for efficiency (PRISM constraint)
                if batch_idx >= 10:  # Train on limited batches per episode
                    break
        
        # UPDATE SHARED WEIGHTS (key PRISM step)
        self.param_manager.update_shared_weights(child_net)
        
        # Validation
        val_score = self.evaluate_child_network(child_net, val_loader, patch_sizes, architecture)
        return val_score
    
    def _adapt_to_patch_size(self, data: torch.Tensor, target: torch.Tensor, 
                           architecture: Dict, patch_sizes: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adapt data to match architecture patch size"""
        target_size = (
            patch_sizes['d_sizes'][architecture['patch_d']],
            patch_sizes['hw_sizes'][architecture['patch_hw']], 
            patch_sizes['hw_sizes'][architecture['patch_hw']]
        )
        
        if data.shape[2:] != target_size:
            data = F.interpolate(data, size=target_size, mode='trilinear', align_corners=False)
            target = F.interpolate(target.float(), size=target_size, mode='trilinear', align_corners=False)
            
        return data, target
    
    def _calculate_loss_with_deep_supervision(self, outputs, target):
        """Calculate loss including deep supervision as per PRISM"""
        if isinstance(outputs, tuple):
            main_output, deep_outputs = outputs
            
            # Main loss
            main_loss = self.criterion(main_output, target)
            
            # Deep supervision losses (weighted at 0.4 as per common practice)
            deep_loss = 0.0
            for deep_out in deep_outputs:
                # Resize target to match deep supervision output if needed
                if deep_out.shape[2:] != target.shape[2:]:
                    deep_target = F.interpolate(
                        target, size=deep_out.shape[2:], 
                        mode='trilinear', align_corners=False
                    )
                else:
                    deep_target = target
                deep_loss += self.criterion(deep_out, deep_target)
            
            total_loss = main_loss + 0.4 * deep_loss
        else:
            total_loss = self.criterion(outputs, target)
            
        return total_loss
    
    def evaluate_child_network(self, child_net: nn.Module, val_loader, 
                              patch_sizes: Dict, architecture: Dict) -> float:
        """Evaluate child network and return dice score"""
        child_net.eval()
        total_dice = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Adapt to patch size
                data, target = self._adapt_to_patch_size(data, target, architecture, patch_sizes)
                
                # Forward pass
                outputs = child_net(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Use main output for evaluation
                
                # Calculate dice score for each sample in batch
                batch_dice = self._calculate_batch_dice_score(outputs, target)
                total_dice += batch_dice.sum().item()
                num_samples += data.size(0)
                
                # Limit validation for efficiency
                if batch_idx >= 5:
                    break
        
        return total_dice / num_samples if num_samples > 0 else 0.0
    
    def _calculate_batch_dice_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate dice score for each sample in the batch"""
        predictions = torch.sigmoid(predictions)
        pred_binary = (predictions > 0.5).float()
        target_binary = (targets > 0.5).float()
        
        batch_size = predictions.size(0)
        dice_scores = []
        
        for i in range(batch_size):
            pred_flat = pred_binary[i].view(-1)
            target_flat = target_binary[i].view(-1)
            
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            
            if union > 0:
                dice = (2.0 * intersection) / union
            else:
                dice = torch.tensor(1.0, device=predictions.device)  # Perfect score for empty regions
                
            dice_scores.append(dice)
        
        return torch.stack(dice_scores)
    
    def search(self, train_loader, val_loader) -> Dict:
        """
        Main architecture search loop implementing PRISM algorithm
        """
        print("=" * 60)
        print("🧠 PRISM Architecture Search Starting...")
        print(f"📊 Episodes: {self.config.episodes}")
        print(f"🏗️  Child networks per episode: {self.config.child_networks_per_episode}")
        print(f"🎯 Search space size: {self.config.get_search_space_size():,}")
        print("=" * 60)
        
        # We need to calculate patch sizes once before the search
        patch_sizes = self.config.patch_sizes
        print(f"   📐 Using optimized patch sizes from config: HW={self.config.patch_sizes['hw_sizes']}, D={self.config.patch_sizes['d_sizes']}")

        self.best_reward = -float('inf')
        self.best_architecture = None

        # Main search loop
        for episode in range(1, self.config.episodes + 1):
            print(f"\n🔄 Episode {episode}/{self.config.episodes}")
            
            episode_rewards = []
            episode_log_probs = []
            episode_entropies = []

            # Train child networks for one episode
            for child_idx in range(self.config.child_networks_per_episode):
                try:
                    # Sample one architecture
                    actions, log_probs, entropies = self.controller()
                    architecture = self.search_space.decode_architecture(actions)

                    print(f"  🏗️  Child {child_idx + 1}: {architecture}")
                    
                    # Train and evaluate child network
                    reward = self.train_child_network(
                        architecture, train_loader, val_loader, patch_sizes
                    )
                    
                    episode_rewards.append(reward)
                    episode_log_probs.append(log_probs)
                    episode_entropies.append(entropies)
                    
                    print(f"     📊 Dice Score: {reward:.4f}")
                    
                    # Track best architecture
                    if reward > self.best_reward:
                        self.best_reward = reward
                        self.best_architecture = architecture.copy()
                        print(f"     🏆 New best! Dice: {self.best_reward:.4f}")
                        
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    # Use dummy values for failed attempts to keep lists aligned
                    episode_rewards.append(0.0)
                    dummy_log_probs = torch.zeros(len(self.search_space.action_space), device=self.device)
                    dummy_entropies = torch.zeros(len(self.search_space.action_space), device=self.device)
                    episode_log_probs.append(dummy_log_probs)
                    episode_entropies.append(dummy_entropies)

            # Update controller with REINFORCE
            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                print(f"  📈 Episode average reward: {avg_reward:.4f}")

                # Stack tensors for batch update
                all_log_probs = torch.stack(episode_log_probs).to(self.device)
                all_entropies = torch.stack(episode_entropies).to(self.device)
                
                # Update controller
                self.controller_optimizer.zero_grad()
                controller_loss, policy_loss, entropy_loss = self.controller.update(
                    episode_rewards, all_log_probs, all_entropies
                )
                controller_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 5.0)
                
                self.controller_optimizer.step()
                
                # Log metrics
                self.history['rewards'].append(avg_reward)
                self.history['entropies'].append(torch.mean(all_entropies).item())
                self.history['controller_losses'].append(controller_loss.item())
                
                print(f"  🎛️  Controller loss: {controller_loss.item():.4f}")
                print(f"  🏆 Best reward so far: {self.best_reward:.4f}")
        
        print("\n" + "=" * 60)
        print(f"🎉 Search completed! Best dice score: {self.best_reward:.4f}")
        print("=" * 60)
        
        return {
            'best_architecture': self.best_architecture,
            'best_reward': self.best_reward,
            'patch_sizes': patch_sizes,
            'history': self.history
        }

# Example usage and testing
if __name__ == "__main__":
    print("PRISM - Resource Optimized Neural Architecture Search for 3D Medical Image Segmentation")
    print("Following the MICCAI 2019 paper implementation")
    
    # Test configuration
    config = PRISMConfig()
    print(f"✓ Configuration loaded with {len(config.patch_size_factors)} patch size factors")
    print(f"✓ Parameter sharing: {config.parameter_sharing}")
    print(f"✓ Element-wise skip connections: {config.use_element_wise_skip}")
    
    # Test search space
    search_space = SearchSpace(config)
    sample_arch = search_space.sample_architecture()
    print(f"✓ Sample architecture: {sample_arch}")
    
    # Test controller
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = LSTMController(config, search_space).to(device)
    
    with torch.no_grad():
        decisions, log_prob, entropy = controller()
        print(f"✓ Controller decisions: {decisions}")
        print(f"✓ Log probability: {log_prob.sum().item():.4f}")
        print(f"✓ Entropy: {entropy.sum().item():.4f}")
    
    # Test child network with corrected architecture
    patch_sizes = {'hw_sizes': [64, 48, 32, 16, 8], 'd_sizes': [32, 24, 16, 8, 4]}
    child_net = ChildNetwork(config, sample_arch, patch_sizes, in_channels=1)
    dummy_input = torch.randn(1, 1, 32, 64, 64)
    
    try:
        output = child_net(dummy_input)
        if isinstance(output, tuple):
            main_out, deep_outs = output
            print(f"✓ Child network output shape: {main_out.shape}")
            print(f"✓ Deep supervision outputs: {len(deep_outs)}")
        else:
            print(f"✓ Child network output shape: {output.shape}")
        print("✓ All PRISM components working correctly!")
    except Exception as e:
        print(f"❌ Error in child network: {e}") 
        import traceback
        traceback.print_exc() 