#!/usr/bin/env python3
"""
MTGNN Explainer Execution Runner

This script demonstrates how to use the MTGNNExplainer to explain predictions
from trained MTGNN models.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mtgnn_explainer import MTGNNExplainer
from net_timestamp import gtnet_timestamp
from util import *


def create_sample_data(num_nodes: int = 50, seq_length: int = 12, 
                      batch_size: int = 1, channels: int = 2) -> torch.Tensor:
    """Create sample time series data for testing."""
    # Generate synthetic time series data
    t = np.linspace(0, 4*np.pi, seq_length)
    data = np.zeros((batch_size, channels, num_nodes, seq_length))
    
    for b in range(batch_size):
        for c in range(channels):
            for n in range(num_nodes):
                # Create different patterns for different nodes
                freq = 0.5 + n * 0.1
                phase = n * 0.2
                amplitude = 1.0 + n * 0.05
                
                data[b, c, n, :] = amplitude * np.sin(freq * t + phase) + np.random.normal(0, 0.1, seq_length)
    
    return torch.FloatTensor(data)


def load_real_data(data_path: str, num_nodes: int, seq_length: int = 12, device: str = 'cpu') -> torch.Tensor:
    """Load real data using DataLoaderS (same as training)."""
    try:
        # Use DataLoaderS to match training data format exactly
        horizon = 1  # Single step prediction
        normalize = 2  # Standard normalization
        
        data_loader = DataLoaderS(data_path, 0.6, 0.2, device, horizon, seq_length, normalize)
        
        # Get test data (same format as training)
        test_X, test_Y = data_loader.test
        
        # Process like in training: unsqueeze and transpose
        # X shape: (batch, seq_length, num_nodes)
        X = torch.unsqueeze(test_X, dim=1)  # (batch, 1, seq_length, num_nodes)
        X = X.transpose(2, 3)  # (batch, 1, num_nodes, seq_length)
        
        return X[:1]  # Return first sample for testing
        
    except Exception as e:
        print(f"Error loading real data with DataLoaderS: {e}")
        print("Using synthetic data instead...")
        return create_sample_data(num_nodes, seq_length, channels=1)  # Use 1 channel for solar data


def run_explainer_example(model_path: str = None, data_path: str = None, 
                          num_nodes: int = 50, seq_length: int = 12,
                          target_node: int = None, epochs: int = 100,
                          device: str = 'cpu', save_results: bool = True,
                          model_config: argparse.Namespace = None,
                          mask_features: bool = True, num_mask_groups: int = 5):
    """
    Run MTGNN Explainer example.
    
    Args:
        model_path: Path to trained model (if None, uses synthetic model)
        data_path: Path to data file (if None, uses synthetic data)
        num_nodes: Number of nodes in the graph
        seq_length: Length of input sequence
        target_node: Specific node to explain (if None, explains all)
        epochs: Number of epochs for mask optimization
        device: Device to run on
        save_results: Whether to save results and plots
        mask_features: Whether to apply feature masks
        num_mask_groups: Number of mask groups (None for per-timestamp masks)
    """
    
    print("=" * 60)
    print("MTGNN Explainer Demo")
    print("=" * 60)
    
    # Load or create data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        input_data = load_real_data(data_path, num_nodes, seq_length, device)
    else:
        print("Creating synthetic data...")
        input_data = create_sample_data(num_nodes, seq_length)
    
    print(f"Input data shape: {input_data.shape}")
    
    # Load or create model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        # Use hyperparameters from arguments
        if model_config is not None:
            config_dict = {
                'gcn_true': model_config.gcn_true,
                'buildA_true': model_config.buildA_true,
                'gcn_depth': model_config.gcn_depth,
                'num_nodes': model_config.num_nodes,
                'device': device,
                'dropout': model_config.dropout,
                'subgraph_size': min(model_config.subgraph_size, model_config.num_nodes),
                'node_dim': model_config.node_dim,
                'dilation_exponential': model_config.dilation_exponential,
                'conv_channels': model_config.conv_channels,
                'residual_channels': model_config.residual_channels,
                'skip_channels': model_config.skip_channels,
                'end_channels': model_config.end_channels,
                'seq_length': model_config.seq_in_len,
                'in_dim': model_config.in_dim,
                'out_dim': model_config.seq_out_len,
                'layers': model_config.layers,
                'propalpha': model_config.propalpha,
                'tanhalpha': model_config.tanhalpha,
                'layer_norm_affline': True
            }
        else:
            # Fallback configuration
            config_dict = {
                'gcn_true': True,
                'buildA_true': True,
                'gcn_depth': 2,
                'num_nodes': num_nodes,
                'device': device,
                'dropout': 0.3,
                'subgraph_size': min(20, num_nodes),
                'node_dim': 40,
                'dilation_exponential': 2,
                'conv_channels': 16,
                'residual_channels': 16,
                'skip_channels': 32,
                'end_channels': 64,
                'seq_length': seq_length,
                'in_dim': input_data.shape[1],
                'out_dim': 1,
                'layers': 5,
                'propalpha': 0.05,
                'tanhalpha': 3,
                'layer_norm_affline': True
            }
        model = load_model(model_path, config_dict, device)
    else:
        print("Creating synthetic model...")
        from net import gtnet
        model_config = {
            'gcn_true': True,
            'buildA_true': True,
            'gcn_depth': 2,
            'num_nodes': num_nodes,
            'device': device,
            'dropout': 0.3,
            'subgraph_size': min(20, num_nodes),  # Ensure subgraph_size <= num_nodes
            'node_dim': 40,
            'dilation_exponential': 1,
            'conv_channels': 32,
            'residual_channels': 32,
            'skip_channels': 64,
            'end_channels': 128,
            'seq_length': seq_length,
            'in_dim': input_data.shape[1],
            'out_dim': 1,
            'layers': 3,
            'propalpha': 0.05,
            'tanhalpha': 3,
            'layer_norm_affline': True
        }
        model = gtnet(**model_config)
        model.to(device)
        model.eval()
    
    print(f"Model loaded successfully")
    
    # Initialize explainer with hyperparameters
    explainer_coeffs = {}
    if model_config is not None:
        explainer_coeffs = {
            'edge_size': model_config.edge_size,
            'node_feat_size': model_config.node_feat_size,
        }
    
    explainer = MTGNNExplainer(
        model, 
        epochs=epochs, 
        lr=model_config.lr if model_config is not None else 0.01,
        device=device,
        mask_features=mask_features,
        num_mask_groups=num_mask_groups,
        **explainer_coeffs
    )
    
    # Run explanation
    print(f"\nRunning explanation...")
    print(f"Target node: {target_node if target_node is not None else 'All nodes'}")
    print(f"Optimization epochs: {epochs}")
    
    explanation = explainer.explain(
        input_data, 
        target_node=target_node,
        target_timestep=0
    )
    
    # Print results (work with tensors/numpy arrays as needed)
    print("\n" + "=" * 40)
    print("EXPLANATION RESULTS")
    print("=" * 40)
    
    print(f"Adjacency masks shape: {explanation['adjacency_masks'].shape}")
    print(f"Feature masks shape: {explanation['feature_masks'].shape}")
    
    # Top important nodes (derived from adjacency)
    node_importance = explanation['overall_node_importance'].cpu()
    top_nodes = torch.argsort(node_importance, descending=True)[:10]
    print(f"\nTop 10 most important nodes (from adjacency):")
    for i, node_idx in enumerate(top_nodes):
        print(f"  {i+1:2d}. Node {node_idx.item():3d}: {node_importance[node_idx].item():.4f}")
    
    # Top important edges
    adjacency_importance = explanation['overall_adjacency_importance'].cpu()
    # Get top edges (excluding diagonal)
    mask = ~torch.eye(adjacency_importance.shape[0], dtype=torch.bool)
    edge_importance = adjacency_importance[mask]
    edge_indices = torch.where(mask)
    top_edges = torch.argsort(edge_importance, descending=True)[:10]
    print(f"\nTop 10 most important edges:")
    for i, edge_idx in enumerate(top_edges):
        src, tgt = edge_indices[0][edge_idx].item(), edge_indices[1][edge_idx].item()
        print(f"  {i+1:2d}. Edge {src:3d} -> {tgt:3d}: {edge_importance[edge_idx].item():.4f}")
    
    # Feature importance
    feature_importance = explanation['overall_feature_importance'].cpu()
    print(f"\nFeature importance:")
    for i, importance in enumerate(feature_importance):
        print(f"  Feature {i}: {importance.item():.4f}")
    
    # Prediction difference (keep as tensor)
    pred_diff = (explanation['prediction'] - explanation['masked_prediction']).cpu()
    print(f"\nPrediction difference statistics:")
    print(f"  Mean: {pred_diff.mean().item():.4f}")
    print(f"  Std:  {pred_diff.std().item():.4f}")
    print(f"  Max:  {pred_diff.max().item():.4f}")
    print(f"  Min:  {pred_diff.min().item():.4f}")
    
    # Calculate sparsity curve with MAE/RMSE/MAPE (always run)
    print("\nEvaluating sparsity curve (MAE/RMSE/MAPE)...")
    try:
        curve_data = explainer.get_fidelity_sparsity_curve(
            input_data, explanation,
            sparsity_range=(0.5, 0.9), 
            num_points=5
        )
        
        print("Sparsity\tMAE\t\tRMSE\t\tMAPE")
        print("-" * 40)
        for i in range(len(curve_data['sparsity_levels'])):
            sparsity = curve_data['sparsity_levels'][i].item()
            mae = curve_data['mae'][i].item()
            rmse = curve_data['rmse'][i].item()
            mape = curve_data['mape'][i].item()
            print(f"{sparsity:.2f}\t\t{mae:.4f}\t\t{rmse:.4f}\t\t{mape:.4f}")
    except Exception as e:
        print(f"Error in sparsity curve: {e}")
        import traceback
        traceback.print_exc()
    
    # Visualize results
    if save_results:
        print(f"\nGenerating visualizations...")
        # Extract importance scores from explanation for visualization
        vis_importance_scores = {
            'overall_node_importance': explanation['overall_node_importance'],
            'overall_adjacency_importance': explanation['overall_adjacency_importance'],
            'overall_feature_importance': explanation['overall_feature_importance']
        }
        explainer.visualize_importance(vis_importance_scores, explanation, save_path='mtgnn_explanation.png')
        
        # Generate timestamp-specific adjacency heatmaps
        print(f"Generating timestamp-specific adjacency heatmaps...")
        adjacency_importance = explanation['adjacency_importance']
        # visualize_timestamp_heatmaps will handle tensor conversion internally
        # Get shape (works for both tensor and numpy)
        seq_length = adjacency_importance.shape[2]
        explainer.visualize_timestamp_heatmaps(
            adjacency_importance, 
            timesteps_to_show=list(range(min(10, seq_length))),  # Show first 10 timesteps
            save_dir='adjacency_heatmaps',
            max_nodes=30
        )
        
        # Save explanation as torch tensor format (all tensors moved to CPU)
        explanation_to_save = {key: value.cpu() for key, value in explanation.items()}
        torch.save(explanation_to_save, 'mtgnn_explanation_results.pt')
        print(f"Results saved to 'mtgnn_explanation_results.pt' ({len(explanation_to_save)} keys)")
    
    return explanation


def main():
    parser = argparse.ArgumentParser(description='MTGNN Explainer Demo')
    
    # Data and model arguments
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model file')
    parser.add_argument('--data', type=str, default='./data/solar_AL.txt',
                       help='Location of the data file')
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='Device to run on (cpu, cuda, cuda:0, cuda:1, etc.)')
    
    # Model architecture arguments (same as training)
    parser.add_argument('--gcn_true', type=bool, default=True, 
                       help='Whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=bool, default=True, 
                       help='Whether to construct adaptive adjacency matrix')
    parser.add_argument('--gcn_depth', type=int, default=2,
                       help='Graph convolution depth')
    parser.add_argument('--num_nodes', type=int, default=137,
                       help='Number of nodes/variables')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--subgraph_size', type=int, default=20,
                       help='Subgraph size k')
    parser.add_argument('--node_dim', type=int, default=40,
                       help='Dimension of nodes')
    parser.add_argument('--dilation_exponential', type=int, default=2,
                       help='Dilation exponential')
    parser.add_argument('--conv_channels', type=int, default=16,
                       help='Convolution channels')
    parser.add_argument('--residual_channels', type=int, default=16,
                       help='Residual channels')
    parser.add_argument('--skip_channels', type=int, default=32,
                       help='Skip channels')
    parser.add_argument('--end_channels', type=int, default=64,
                       help='End channels')
    parser.add_argument('--in_dim', type=int, default=1,
                       help='Input dimension')
    parser.add_argument('--seq_in_len', type=int, default=168,
                       help='Input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=1,
                       help='Output sequence length')
    parser.add_argument('--horizon', type=int, default=3,
                       help='Horizon')
    parser.add_argument('--layers', type=int, default=5,
                       help='Number of layers')
    parser.add_argument('--propalpha', type=float, default=0.05,
                       help='Prop alpha')
    parser.add_argument('--tanhalpha', type=float, default=3,
                       help='Tanh alpha')
    parser.add_argument('--normalize', type=int, default=2,
                       help='Normalization type')
    
    # Explainer-specific arguments
    parser.add_argument('--target_node', type=int, default=None,
                       help='Specific node to explain (if None, explains all)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for mask optimization')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate for mask optimization')
    parser.add_argument('--edge_size', type=float, default=0.005,
                       help='Edge size regularization coefficient')
    parser.add_argument('--node_feat_size', type=float, default=1.0,
                       help='Node feature size regularization coefficient')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results and plots')
    parser.add_argument('--mask_features', action='store_true', default=True,
                       help='Whether to apply feature masks (True) or only adjacency masks (False)')
    parser.add_argument('--num_mask_groups', type=int, default=5,
                       help='Number of mask groups to share across timestamps')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # Run the explainer
    try:
        explanation = run_explainer_example(
            model_path=args.model_path,
            data_path=args.data,
            num_nodes=args.num_nodes,
            seq_length=args.seq_in_len,
            target_node=args.target_node,
            epochs=args.epochs,
            device=args.device,
            save_results=args.save_results,
            model_config=args,
            mask_features=args.mask_features,
            num_mask_groups=args.num_mask_groups
        )
        
        print("\n" + "=" * 60)
        print("EXPLANATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running explainer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
