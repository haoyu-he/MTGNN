import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
import copy


class MTGNNExplainer:
    """
    MTGNN Explainer for explaining single-step predictions with timestamp-specific adjacency support.
    
    This explainer learns:
    1. Adjacency masks: Importance masks for graph edges with 5 shared masks across timestamp groups (num_nodes, num_nodes, 5)
       Each mask is shared across seq_length/5 timestamps, reducing complexity while maintaining temporal patterns
    2. Feature masks: Importance masks per timestamp (batch, channels, nodes, timesteps)
       These are truly timestamp-specific
    
    The goal is to minimize prediction difference to show what's important FOR the prediction.
    
    Requirements:
    - Use gtnet_timestamp model for true timestamp-specific adjacency explanations
    - Falls back to averaged adjacency masks with standard gtnet model
    """
    
    def __init__(self, model: nn.Module, epochs: int = 100, lr: float = 0.01, 
                 device: str = 'cpu', mask_features: bool = True, num_mask_groups: int = 5, **kwargs):
        """
        Initialize MTGNN Explainer.
        
        Args:
            model: Trained MTGNN model
            epochs: Number of training epochs for mask optimization
            lr: Learning rate for mask optimization
            device: Device to run on
            mask_features: Whether to apply feature masks (True) or only adjacency masks (False)
            num_mask_groups: Number of mask groups to share across timestamps (int). Values outside [1, seq_length]
                              are treated as invalid and we fall back to per-timestamp masks.
            **kwargs: Additional hyper-parameters for regularization coefficients
        """
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.mask_features = mask_features  # Option to mask features or not
        self.num_mask_groups = num_mask_groups  # Number of mask groups (validated per input by _effective_num_mask_groups)
        
        # Regularization coefficients
        self.coeffs = {
            'edge_size': 0.005,
            'edge_reduction': 'sum',
            'node_feat_size': 1.0,
            'node_feat_reduction': 'mean',
            # 'edge_ent': 1.0,        # Commented out entropy regularization
            # 'node_feat_ent': 0.1,   # Commented out entropy regularization
            'EPS': 1e-15,
        }
        self.coeffs.update(kwargs)
        
        # Will store the adjacency matrix used for masking
        self.adp = None
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()

    def explain(self, input_tensor: torch.Tensor, target_node: Optional[int] = None,
                target_timestep: int = 0, idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Explain a single-step prediction.
        
        Args:
            input_tensor: Input tensor of shape (batch, channels, nodes, timesteps)
            target_node: Specific node to explain (if None, explains all nodes)
            target_timestep: Which timestep in the prediction to explain (usually 0 for single-step)
            idx: Node indices for the model (if using subgraph sampling)
            
        Returns:
            Dictionary containing:
            - 'adjacency_masks': Adjacency importance masks (num_nodes, num_nodes)
            - 'feature_masks': Feature importance masks
            - 'prediction': Original prediction
            - 'masked_prediction': Prediction with masks applied
        """
        batch_size, channels, num_nodes, seq_length = input_tensor.shape
        
        # Get adp once at the beginning, following net.py lines 98-105 logic
        if self.model.gcn_true:
            if self.model.buildA_true:
                # If buildA_true, get learned adjacency from gc
                if idx is None:
                    self.adp = self.model.gc(self.model.idx)
                else:
                    self.adp = self.model.gc(idx)
            else:
                # Use predefined_A
                self.adp = self.model.predefined_A
        
        # Initialize masks
        adjacency_masks = self._init_adjacency_masks(num_nodes, seq_length)
        feature_masks = self._init_feature_masks(input_tensor.shape)
        
        # Move input tensor to same device as model
        input_tensor = input_tensor.to(self.device)
        
        # Get original prediction
        with torch.no_grad():
            original_pred = self.model(input_tensor, idx=idx)
            if target_node is not None:
                target_pred = original_pred[:, :, target_node, target_timestep].detach()
            else:
                target_pred = original_pred[:, :, :, target_timestep].detach()
        
        # Optimize masks
        adjacency_masks, feature_masks = self._optimize_masks(
            input_tensor, target_pred, adjacency_masks, feature_masks, idx
        )
        
        # Get masked prediction
        masked_input = self._apply_masks(input_tensor, adjacency_masks, feature_masks, idx)
        with torch.no_grad():
            if hasattr(self, 'timestamp_adj_matrices'):
                masked_pred = self.model(masked_input, idx=idx, timestamp_adj_matrices=self.timestamp_adj_matrices)
            else:
                masked_pred = self.model(masked_input, idx=idx)
            if target_node is not None:
                masked_pred = masked_pred[:, :, target_node, target_timestep]
            else:
                masked_pred = masked_pred[:, :, :, target_timestep]
        
        # Restore original adjacency matrix
        self._restore_adjacency()
        
        # Create explanation result
        explanation = {
            'adjacency_masks': adjacency_masks,
            'feature_masks': feature_masks,
            'prediction': original_pred,
            'masked_prediction': masked_pred,
            'target_prediction': target_pred
        }
        
        # Add importance scores as flat keys using update
        explanation.update(self.get_importance_scores(explanation))
        
        return explanation
    
    def _init_adjacency_masks(self, num_nodes: int, seq_length: int) -> torch.Tensor:
        """Initialize adjacency masks for graph edges."""
        if not (0 < self.num_mask_groups < seq_length):
            # Per-timestamp masks (original behavior)
            masks = torch.randn(num_nodes, num_nodes, seq_length, device=self.device)
        else:
            # Shared masks across timestamp groups
            masks = torch.randn(num_nodes, num_nodes, self.num_mask_groups, device=self.device)
        masks.requires_grad_(True)
        return masks
    
    def _init_feature_masks(self, input_shape: Tuple[int, ...]) -> torch.Tensor:
        """Initialize feature masks with optional grouping strategy.

        Note: `num_mask_groups` is always an int; values outside [1, seq_length]
        are treated as invalid and we fall back to per-timestamp masks.
        """
        batch_size, channels, num_nodes, seq_length = input_shape
        
        if not (0 < self.num_mask_groups < seq_length):
            # Per-timestamp feature masks (original behavior)
            masks = torch.randn(input_shape, device=self.device)
        else:
            # Grouped feature masks - create masks for each group
            masks = torch.randn(batch_size, channels, num_nodes, self.num_mask_groups, device=self.device)
        
        masks.requires_grad_(True)
        return masks
    
    def _optimize_masks(self, input_tensor: torch.Tensor, target_pred: torch.Tensor,
                       adjacency_masks: torch.Tensor, feature_masks: torch.Tensor,
                       idx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimize masks to minimize prediction difference with L1 regularization."""
        
        optimizer = torch.optim.Adam([adjacency_masks, feature_masks], lr=self.lr)
        
        # Initialize hard masks to track which elements are actually used
        hard_adjacency_mask = None
        hard_feature_mask = None
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Apply masks to input and adjacency
            masked_input = self._apply_masks(input_tensor, adjacency_masks, feature_masks, idx)
            
            # Get masked prediction with timestamp-specific adjacency
            if hasattr(self, 'timestamp_adj_matrices'):
                # Create a fresh copy to avoid gradient issues
                timestamp_adj_copy = self.timestamp_adj_matrices.clone().detach()
                masked_pred = self.model(masked_input, idx=idx, timestamp_adj_matrices=timestamp_adj_copy)
            else:
                masked_pred = self.model(masked_input, idx=idx)
            
            # Calculate base loss: minimize difference between original and masked prediction
            # Create a fresh copy of target_pred to avoid gradient issues
            target_pred_copy = target_pred.clone().detach()
            pred_diff = torch.mean((masked_pred - target_pred_copy) ** 2)
            loss = pred_diff
            
            # Add L1 regularization for sparsity
            adjacency_l1 = torch.mean(torch.abs(torch.sigmoid(adjacency_masks)))
            loss = loss + self.coeffs['edge_size'] * adjacency_l1
            
            # Add feature regularization only if feature masking is enabled
            if self.mask_features:
                feature_l1 = torch.mean(torch.abs(torch.sigmoid(feature_masks)))
                loss = loss + self.coeffs['node_feat_size'] * feature_l1
            
            loss.backward()
            
            # Collect hard masks from gradients (inspired by PyG GNNExplainer)
            if epoch == 0:
                hard_adjacency_mask = self._get_hard_mask(adjacency_masks)
                if self.mask_features:
                    hard_feature_mask = self._get_hard_mask(feature_masks)
            
            optimizer.step()
            
            # Restore original adjacency matrix after each step
            self._restore_adjacency()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Pred Diff: {pred_diff.item():.4f}", flush=True)
        
        # Apply hard masks: set invalid (unused) elements to -inf for sigmoid
        # hard_mask: True = valid (keep value), False = invalid (set to -inf)
        if hard_adjacency_mask is not None:
            adjacency_masks = torch.where(
                hard_adjacency_mask,
                adjacency_masks,
                torch.tensor(-float('inf'), device=self.device)
            )
        if hard_feature_mask is not None and self.mask_features:
            feature_masks = torch.where(
                hard_feature_mask,
                feature_masks,
                torch.tensor(-float('inf'), device=self.device)
            )
            
        return adjacency_masks.detach(), feature_masks.detach()
    
    def _get_hard_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Get hard mask based on gradients (PyG inspired).
        
        Returns boolean mask where:
        - True (1) = valid (element is used, grad != 0.0)
        - False (0) = invalid (element is not used, grad == 0.0)
        """
        if masks.grad is None:
            raise ValueError("Could not compute gradients for masks. "
                           "Please make sure that masks are used inside the model.")
        
        # Hard mask: elements with non-zero gradients are valid (True), others are invalid (False)
        # True = valid/used element, False = invalid/unused element
        hard_mask = masks.grad != 0.0
        return hard_mask
    
    def _apply_masks(self, input_tensor: torch.Tensor, adjacency_masks: torch.Tensor,
                    feature_masks: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply masks to input tensor and create timestamp-specific masked adjacency matrices."""
        batch_size, channels, num_nodes, seq_length = input_tensor.shape
        
        # Apply feature masks conditionally
        if self.mask_features:
            if not (0 < self.num_mask_groups < seq_length):
                # Per-timestamp feature masks (original behavior)
                masked_input = input_tensor * F.sigmoid(feature_masks)
            else:
                # Grouped feature masks - expand to per-timestamp
                timestamps_per_group = seq_length // self.num_mask_groups
                feature_masks_expanded = torch.zeros_like(input_tensor)
                for t in range(seq_length):
                    mask_group_idx = min(t // timestamps_per_group, self.num_mask_groups - 1)
                    feature_masks_expanded[:, :, :, t] = feature_masks[:, :, :, mask_group_idx]
                masked_input = input_tensor * F.sigmoid(feature_masks_expanded)
        else:
            masked_input = input_tensor  # No feature masking
        
        # Apply timestamp-specific masks to the stored adjacency matrix
        # For original models, adp is a single adjacency matrix
        if self.model.gcn_true and self.adp is not None:
            sigmoid_adjacency_masks = F.sigmoid(adjacency_masks)
            
            # Create timestamp-specific adjacency matrices
            timestamp_adj_matrices = []
            
            if not (0 < self.num_mask_groups < seq_length):
                # Per-timestamp masks (original behavior)
                # adjacency_masks shape: (num_nodes, num_nodes, seq_length)
                for t in range(seq_length):
                    adj_mask_t = sigmoid_adjacency_masks[:, :, t]  # (num_nodes, num_nodes)
                    masked_adp_t = self.adp * adj_mask_t
                    timestamp_adj_matrices.append(masked_adp_t)
            else:
                # Shared masks across timestamp groups
                # adjacency_masks shape: (num_nodes, num_nodes, num_mask_groups)
                timestamps_per_group = seq_length // self.num_mask_groups
                
                for t in range(seq_length):
                    # Determine which mask group this timestamp belongs to
                    mask_group_idx = min(t // timestamps_per_group, self.num_mask_groups - 1)
                    # Get adjacency mask for this timestamp group
                    adj_mask_t = sigmoid_adjacency_masks[:, :, mask_group_idx]  # (num_nodes, num_nodes)
                    # Apply mask to original adjacency matrix
                    masked_adp_t = self.adp * adj_mask_t
                    timestamp_adj_matrices.append(masked_adp_t)
            
            # Stack to create tensor of shape (seq_length, num_nodes, num_nodes)
            timestamp_adj_matrices = torch.stack(timestamp_adj_matrices, dim=0)
            
            # Store for use in model forward pass
            self.timestamp_adj_matrices = timestamp_adj_matrices
        
        return masked_input
    
    def _restore_adjacency(self):
        """Restore the original adjacency matrix."""
        if self.adp is not None:
            self.model.predefined_A = self.adp
    
    def get_importance_scores(self, explanation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract importance scores from explanation results.
        
        Args:
            explanation: Result from explain() method
            
        Returns:
            Dictionary with importance scores as torch tensors
        """
        import torch.nn.functional as F
        
        # Get masks (keep as tensors)
        adjacency_masks = explanation['adjacency_masks']
        feature_masks = explanation['feature_masks']
        
        # Apply sigmoid to get actual importance scores (keep as tensors)
        adjacency_importance = torch.sigmoid(adjacency_masks)  # (num_nodes, num_nodes, 5)
        feature_importance = torch.sigmoid(feature_masks)
        
        # Get seq_length from feature_masks shape (always 4D: batch, channels, nodes, timesteps)
        batch_size, channels, num_nodes, seq_length = feature_masks.shape
        
        # Overall adjacency importance (average across timesteps)
        overall_adjacency_importance = torch.mean(adjacency_importance, dim=2)  # (num_nodes, num_nodes)
        
        # Overall node importance (mean of incoming/outgoing edges)
        incoming_importance = torch.mean(overall_adjacency_importance, dim=0)  # (num_nodes,)
        outgoing_importance = torch.mean(overall_adjacency_importance, dim=1)  # (num_nodes,)
        overall_node_importance = (incoming_importance + outgoing_importance) / 2
        
        # Feature importance per group (unified handling)
        feature_importance_per_group = torch.mean(feature_importance, dim=(0, 2))  # (channels, timesteps)
        overall_feature_importance = torch.mean(feature_importance, dim=(0, 2, 3))  # (channels,)
        
        return {
            'adjacency_importance_per_group': adjacency_importance,
            'overall_adjacency_importance': overall_adjacency_importance,
            'incoming_importance': incoming_importance,
            'outgoing_importance': outgoing_importance,
            'overall_node_importance': overall_node_importance,
            'feature_importance_per_group': feature_importance_per_group,
            'overall_feature_importance': overall_feature_importance,
            'adjacency_importance': adjacency_importance,
            'feature_importance': feature_importance
        }
    
    def get_fidelity_sparsity_curve(self, input_tensor: torch.Tensor, explanation: Dict[str, torch.Tensor],
                                  sparsity_range: tuple = (0.5, 0.9), num_points: int = 10,
                                  idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate fidelity-sparsity curve by varying sparsity levels.
        
        Args:
            input_tensor: Original input tensor used for the explanation
            explanation: Explanation dict returned by explain() (uses its importance and prediction)
            sparsity_range: Tuple of (min_sparsity, max_sparsity) 
            num_points: Number of sparsity points to evaluate
            idx: Optional node indices
            
        Returns:
            Dictionary with sparsity levels and corresponding metrics (mae, rmse, mape)
        """
        import torch.nn.functional as F
        
        # Use importance scores from provided explanation (no new explanation run)
        adjacency_importance = explanation['adjacency_importance']  # (num_nodes, num_nodes, groups/timestamps)
        feature_importance = None  # no feature sparsity
        target_pred = explanation['prediction']
        
        # Generate sparsity levels
        sparsity_levels = torch.linspace(sparsity_range[0], sparsity_range[1], num_points)
        maes: list[torch.Tensor] = []
        rmses: list[torch.Tensor] = []
        mapes: list[torch.Tensor] = []
        
        # Count actual edges for reference
        if self.adp is not None:
            num_actual_edges = (self.adp > 0).sum().item()
            print(f"Total actual edges in adjacency matrix: {num_actual_edges}")
        
        for sparsity in sparsity_levels:
            # Create hard masks based on sparsity (adjacency only)
            masked_input, timestamp_adj_matrices = self._apply_sparsity_mask(
                input_tensor, adjacency_importance, sparsity.item()
            )
            
            # Get prediction with sparsity mask
            with torch.no_grad():
                if timestamp_adj_matrices is not None:
                    masked_pred = self.model(masked_input, idx=idx, timestamp_adj_matrices=timestamp_adj_matrices)
                else:
                    masked_pred = self.model(masked_input, idx=idx)
            
            # Metrics: MAE, RMSE, MAPE
            diff = masked_pred - target_pred
            abs_diff = torch.abs(diff)
            mae = abs_diff.mean()
            rmse = torch.sqrt((diff ** 2).mean())
            # Avoid division by zero using EPS
            mape = (abs_diff / (torch.abs(target_pred) + self.coeffs['EPS'])).mean()
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
        
        return {
            'sparsity_levels': sparsity_levels,
            'mae': torch.stack(maes),
            'rmse': torch.stack(rmses),
            'mape': torch.stack(mapes)
        }
    
    def _apply_sparsity_mask(self, input_tensor: torch.Tensor, adjacency_importance: torch.Tensor,
                            sparsity: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply sparsity mask by keeping only top (1-sparsity) fraction of important adjacency edges.
        Features are not masked - adjacency-only.
        
        Args:
            input_tensor: Input tensor
            adjacency_importance: Adjacency importance scores
            sparsity: Fraction of edges to remove (0.5 = remove 50%)
            
        Returns:
            Tuple of (masked input tensor, timestamp adjacency matrices)
        """
        
        batch_size, channels, num_nodes, seq_length = input_tensor.shape
        masked_input = input_tensor.clone()
        
        # Skip feature sparsity mask - only apply to adjacency matrix
        
        # Apply adjacency sparsity mask
        if self.adp is not None:
            # Create timestamp-specific adjacency matrices with sparsity
            timestamp_adj_matrices = []
            
            if not (0 < self.num_mask_groups < seq_length):
                # Per-timestamp: adjacency_importance shape (num_nodes, num_nodes, timesteps)
                edge_mask = (self.adp > 0)
                num_actual_edges = edge_mask.sum().item()
                edge_indices = torch.where(edge_mask)
                for t in range(seq_length):
                    adj_imp_t = adjacency_importance[:, :, t]  # (num_nodes, num_nodes)
                    num_keep = int((1 - sparsity) * num_actual_edges)
                    if num_keep > 0:
                        # Only consider edges that exist in the original adjacency matrix
                        edge_importance = adj_imp_t[edge_mask]
                        _, top_indices = torch.topk(edge_importance, num_keep)
                        
                        # Create mask for actual edges
                        mask = torch.zeros_like(self.adp, device=self.adp.device)
                        selected_edge_indices = (edge_indices[0][top_indices], edge_indices[1][top_indices])
                        mask[selected_edge_indices] = 1.0
                        masked_adj = self.adp * mask
                    else:
                        masked_adj = torch.zeros_like(self.adp)
                    timestamp_adj_matrices.append(masked_adj)
            else:
                # Grouped: adjacency_importance shape (num_nodes, num_nodes, groups)
                timestamps_per_group = seq_length // self.num_mask_groups
                # Precompute one masked adjacency per group and reuse across timestamps
                masked_adj_per_group = []
                edge_mask = (self.adp > 0)
                num_actual_edges = edge_mask.sum().item()
                edge_indices = torch.where(edge_mask)
                for g in range(self.num_mask_groups):
                    adj_imp_g = adjacency_importance[:, :, g]  # (num_nodes, num_nodes)
                    num_keep = int((1 - sparsity) * num_actual_edges)
                    if num_keep > 0:
                        edge_importance = adj_imp_g[edge_mask]
                        _, top_indices = torch.topk(edge_importance, num_keep)
                        mask = torch.zeros_like(self.adp, device=self.adp.device)
                        selected_edge_indices = (edge_indices[0][top_indices], edge_indices[1][top_indices])
                        mask[selected_edge_indices] = 1.0
                        masked_adj = self.adp * mask
                    else:
                        masked_adj = torch.zeros_like(self.adp)
                    masked_adj_per_group.append(masked_adj)
                for t in range(seq_length):
                    group_idx = min(t // timestamps_per_group, self.num_mask_groups - 1)
                    timestamp_adj_matrices.append(masked_adj_per_group[group_idx])
            
            # Stack to create tensor of shape (seq_length, num_nodes, num_nodes)
            timestamp_adj_matrices = torch.stack(timestamp_adj_matrices, dim=0)
        else:
            timestamp_adj_matrices = None
        
        # Ensure masked input is on the same device as the model
        return masked_input.to(self.device), timestamp_adj_matrices
    
    def visualize_importance(self, importance_scores: Dict[str, Union[torch.Tensor, np.ndarray]], 
                           explanation: Optional[Dict[str, torch.Tensor]] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Visualize importance scores.
        
        Args:
            importance_scores: Result from get_importance_scores() (tensors or numpy arrays)
            explanation: Optional explanation results for prediction comparison
            save_path: Optional path to save plots
        """
        import matplotlib.pyplot as plt
        
        # Convert tensors to numpy for visualization (accepts both)
        importance_scores_np = {}
        for key, value in importance_scores.items():
            if isinstance(value, torch.Tensor):
                importance_scores_np[key] = value.detach().cpu().numpy()
            else:
                importance_scores_np[key] = value
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Overall node importance (derived from adjacency)
        axes[0].bar(range(len(importance_scores_np['overall_node_importance'])), 
                     importance_scores_np['overall_node_importance'])
        axes[0].set_title('Overall Node Importance (from Adjacency)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Node Index')
        axes[0].set_ylabel('Importance Score')
        axes[0].grid(True, alpha=0.3)
        
        # Adjacency importance heatmap (edges)
        im1 = axes[1].imshow(importance_scores_np['overall_adjacency_importance'], 
                             aspect='auto', cmap='viridis')
        axes[1].set_title('Adjacency Importance (Edge Weights)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Target Node')
        axes[1].set_ylabel('Source Node')
        plt.colorbar(im1, ax=axes[1])
        
        # Feature importance
        axes[2].bar(range(len(importance_scores_np['overall_feature_importance'])), 
                     importance_scores_np['overall_feature_importance'])
        axes[2].set_title('Feature Importance', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Feature Index')
        axes[2].set_ylabel('Importance Score')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_timestamp_heatmaps(self, adjacency_importance: Union[torch.Tensor, np.ndarray], 
                                    timesteps_to_show: list = None, 
                                    save_dir: str = None, 
                                    max_nodes: int = 50):
        """
        Create separate adjacency heatmaps for each timestamp.
        
        Args:
            adjacency_importance: Adjacency importance scores (num_nodes, num_nodes, seq_length) 
                                  as tensor or numpy array (already passed through sigmoid)
            timesteps_to_show: List of timesteps to visualize (if None, shows all)
            save_dir: Directory to save individual heatmap files
            max_nodes: Maximum number of nodes to show (for readability)
        """
        import matplotlib.pyplot as plt
        import os
        
        # Convert to numpy if tensor (accepts both)
        if isinstance(adjacency_importance, torch.Tensor):
            adjacency_importance = adjacency_importance.detach().cpu().numpy()
        
        num_nodes, _, seq_length = adjacency_importance.shape
        
        if timesteps_to_show is None:
            timesteps_to_show = list(range(seq_length))
        
        # Limit nodes for readability
        if num_nodes > max_nodes:
            # Select top important nodes
            overall_importance = np.sum(adjacency_importance, axis=(1, 2))
            top_nodes = np.argsort(overall_importance)[-max_nodes:]
            adjacency_importance = adjacency_importance[np.ix_(top_nodes, top_nodes, timesteps_to_show)]
            num_nodes = max_nodes
        else:
            adjacency_importance = adjacency_importance[:, :, timesteps_to_show]
        
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for i, t in enumerate(timesteps_to_show):
            # Get adjacency matrix for this timestep
            adj_matrix = adjacency_importance[:, :, i]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            im = ax.imshow(adj_matrix, cmap='viridis', aspect='equal')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Importance Score', rotation=270, labelpad=20)
            
            # Set labels
            ax.set_title(f'Adjacency Importance Heatmap - Timestep {t}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Target Node')
            ax.set_ylabel('Source Node')
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3)
            
            # Set ticks
            ax.set_xticks(range(num_nodes))
            ax.set_yticks(range(num_nodes))
            
            # Add node labels with smart spacing based on total nodes
            if num_nodes <= 20:
                # Show all labels for small graphs
                ax.set_xticklabels(range(num_nodes), fontsize=8)
                ax.set_yticklabels(range(num_nodes), fontsize=8)
            elif num_nodes <= 50:
                # Show every 2nd label for medium graphs
                step = 2
                tick_positions = list(range(0, num_nodes, step))
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels(tick_positions, fontsize=7)
                ax.set_yticklabels(tick_positions, fontsize=7)
            elif num_nodes <= 100:
                # Show every 5th label for large graphs
                step = 5
                tick_positions = list(range(0, num_nodes, step))
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels(tick_positions, fontsize=6)
                ax.set_yticklabels(tick_positions, fontsize=6)
            else:
                # Show every 10th label for very large graphs
                step = 10
                tick_positions = list(range(0, num_nodes, step))
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels(tick_positions, fontsize=5)
                ax.set_yticklabels(tick_positions, fontsize=5)
            
            plt.tight_layout()
            
            # Save individual file
            if save_dir:
                save_path = os.path.join(save_dir, f'adjacency_heatmap_timestep_{t:03d}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved heatmap for timestep {t} to {save_path}")
            
            plt.show()
        
        print(f"\nGenerated {len(timesteps_to_show)} adjacency heatmaps")
        if save_dir:
            print(f"All heatmaps saved to directory: {save_dir}")


