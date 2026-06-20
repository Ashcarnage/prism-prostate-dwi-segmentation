import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set the style for modern, visually appealing plots with light background
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_training_data(file_path):
    """Load training history from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def create_reward_plot(rewards, save_path):
    """Create a modern reward progression plot with trend analysis."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    episodes = np.arange(1, len(rewards) + 1)
    
    # Raw rewards
    ax.plot(episodes, rewards, alpha=0.3, color='#64b5f6', linewidth=0.8, label='Raw Rewards')
    
    # Smoothed trend
    smoothed = gaussian_filter1d(rewards, sigma=5)
    ax.plot(episodes, smoothed, color='#ffb74d', linewidth=3, label='Smoothed Trend')
    
    # Moving average
    window_size = 20
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    ax.plot(episodes[window_size-1:], moving_avg, color='#e57373', linewidth=2, label=f'{window_size}-Episode Moving Average')
    
    # Add performance phases
    phase1_end = 50
    phase2_end = 150
    
    ax.axvspan(1, phase1_end, alpha=0.1, color='red', label='Learning Phase')
    ax.axvspan(phase1_end, phase2_end, alpha=0.1, color='yellow', label='Improvement Phase')
    ax.axvspan(phase2_end, len(rewards), alpha=0.1, color='green', label='Convergence Phase')
    
    # Styling
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax.set_title('RONASMIS Training: Reward Progression Analysis', fontsize=18, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # Add statistics box
    stats_text = f'Final Reward: {rewards[-1]:.4f}\nMax Reward: {max(rewards):.4f}\nMean Reward: {np.mean(rewards):.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_entropy_plot(entropies, save_path):
    """Create an entropy visualization showing exploration behavior."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    episodes = np.arange(1, len(entropies) + 1)
    
    # Main entropy plot
    ax1.plot(episodes, entropies, color='#ab47bc', linewidth=1.5, alpha=0.7, label='Policy Entropy')
    
    # Smoothed entropy
    smoothed_entropy = gaussian_filter1d(entropies, sigma=3)
    ax1.plot(episodes, smoothed_entropy, color='#26c6da', linewidth=3, label='Smoothed Entropy')
    
    # Add exploration phases
    high_exploration = np.percentile(entropies, 75)
    medium_exploration = np.percentile(entropies, 50)
    
    ax1.axhline(y=high_exploration, color='#4caf50', linestyle='--', alpha=0.7, label='High Exploration Threshold')
    ax1.axhline(y=medium_exploration, color='#ff9800', linestyle='--', alpha=0.7, label='Medium Exploration Threshold')
    
    ax1.set_ylabel('Entropy', fontsize=14, fontweight='bold')
    ax1.set_title('Policy Entropy: Exploration vs Exploitation Balance', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12)
    
    # Entropy distribution histogram
    ax2.hist(entropies, bins=50, color='#ab47bc', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax2.axvline(np.mean(entropies), color='#ffeb3b', linewidth=3, label=f'Mean: {np.mean(entropies):.4f}')
    ax2.axvline(np.median(entropies), color='#e91e63', linewidth=3, label=f'Median: {np.median(entropies):.4f}')
    
    ax2.set_xlabel('Entropy Value', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax2.set_title('Entropy Distribution', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_controller_losses_plot(losses, save_path):
    """Create controller losses plot with moving averages."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    episodes = np.arange(1, len(losses) + 1)
    
    # Raw losses
    ax.plot(episodes, losses, alpha=0.4, color='#f06292', linewidth=0.8, label='Raw Controller Loss')
    
    # Moving averages
    window_10 = np.convolve(losses, np.ones(10)/10, mode='valid')
    window_30 = np.convolve(losses, np.ones(30)/30, mode='valid')
    
    ax.plot(episodes[9:], window_10, color='#42a5f5', linewidth=2, label='10-Episode Moving Average')
    ax.plot(episodes[29:], window_30, color='#66bb6a', linewidth=3, label='30-Episode Moving Average')
    
    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Highlight positive and negative regions
    positive_mask = np.array(losses) > 0
    negative_mask = np.array(losses) < 0
    
    ax.fill_between(episodes, 0, losses, where=positive_mask, color='#4caf50', alpha=0.2, label='Positive Loss')
    ax.fill_between(episodes, 0, losses, where=negative_mask, color='#f44336', alpha=0.2, label='Negative Loss')
    
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Controller Loss', fontsize=14, fontweight='bold')
    ax.set_title('Controller Loss Evolution: Learning Dynamics', fontsize=18, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, framealpha=0.9)
    
    # Add statistics
    stats_text = f'Final Loss: {losses[-1]:.4f}\nMean Loss: {np.mean(losses):.4f}\nStd Loss: {np.std(losses):.4f}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_comprehensive_dashboard(data, save_path):
    """Create a comprehensive dashboard with all metrics."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    rewards = data['rewards']
    entropies = data['entropies']
    losses = data['controller_losses']
    episodes = np.arange(1, len(rewards) + 1)
    
    # Rewards plot (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(episodes, rewards, alpha=0.5, color='#64b5f6', linewidth=1)
    smoothed_rewards = gaussian_filter1d(rewards, sigma=5)
    ax1.plot(episodes, smoothed_rewards, color='#ffb74d', linewidth=3)
    ax1.set_title('Reward Progression', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # Reward distribution (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(rewards, bins=30, color='#64b5f6', alpha=0.7, orientation='horizontal')
    ax2.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Entropy plot (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(episodes, entropies, color='#ab47bc', linewidth=1.5)
    ax3.set_title('Policy Entropy', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Entropy')
    ax3.grid(True, alpha=0.3)
    
    # Controller losses (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(episodes, losses, alpha=0.6, color='#f06292', linewidth=1)
    window_20 = np.convolve(losses, np.ones(20)/20, mode='valid')
    ax4.plot(episodes[19:], window_20, color='#42a5f5', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Controller Loss', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Loss')
    ax4.grid(True, alpha=0.3)
    
    # Correlation heatmap (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    correlation_data = np.array([rewards, entropies, losses])
    correlation_matrix = np.corrcoef(correlation_data)
    im = ax5.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax5.set_xticks([0, 1, 2])
    ax5.set_yticks([0, 1, 2])
    ax5.set_xticklabels(['Rewards', 'Entropy', 'Losses'], rotation=45)
    ax5.set_yticklabels(['Rewards', 'Entropy', 'Losses'])
    ax5.set_title('Metric Correlations', fontsize=14, fontweight='bold')
    
    # Add correlation values
    for i in range(3):
        for j in range(3):
            text = ax5.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Performance metrics summary (bottom row)
    ax6 = fig.add_subplot(gs[2, :])
    
    # Calculate key metrics
    final_reward = rewards[-1]
    max_reward = max(rewards)
    reward_improvement = (final_reward - rewards[0]) / abs(rewards[0]) * 100
    entropy_stability = np.std(entropies[-50:])  # Last 50 episodes
    loss_trend = np.polyfit(episodes[-50:], losses[-50:], 1)[0]  # Slope of last 50 episodes
    
    metrics_text = f"""
    TRAINING SUMMARY (300 Episodes)
    
    🎯 Final Reward: {final_reward:.4f}
    📈 Maximum Reward: {max_reward:.4f}
    📊 Improvement: {reward_improvement:+.1f}%
    
    🎲 Final Entropy: {entropies[-1]:.4f}
    📉 Entropy Stability: {entropy_stability:.4f}
    
    ⚖️ Final Loss: {losses[-1]:.4f}
    📈 Loss Trend: {loss_trend:.6f}
    
    🚀 Training Status: {'Converged' if abs(loss_trend) < 0.001 else 'Still Learning'}
    """
    
    ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace', color='black',
             bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', alpha=0.9))
    
    # Performance timeline
    ax6_right = ax6.twinx()
    performance_score = np.array(rewards) * (1 - np.array(entropies)/max(entropies))  # Reward weighted by exploration
    ax6_right.plot(episodes, performance_score, color='#4caf50', linewidth=2, alpha=0.7)
    ax6_right.set_ylabel('Performance Score', color='#4caf50')
    ax6_right.tick_params(axis='y', labelcolor='#4caf50')
    
    ax6.set_xlim(0, len(episodes))
    ax6.set_xlabel('Episode')
    ax6.set_title('Training Performance Summary', fontsize=16, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add colorbar for correlation heatmap
    cbar = plt.colorbar(im, ax=ax5, shrink=0.6)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    fig.suptitle('RONASMIS Training Dashboard - Complete Analysis', fontsize=24, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    # Load data
    data = load_training_data('/Users/ayushbhakat/Desktop/BioAi_v2/experiments/ronasmis_300ep/training_history.json')
    
    # Create output directory
    import os
    output_dir = '/Users/ayushbhakat/Desktop/BioAi_v2/training_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    print("Creating reward progression plot...")
    create_reward_plot(data['rewards'], f'{output_dir}/reward_progression.png')
    
    print("Creating entropy visualization...")
    create_entropy_plot(data['entropies'], f'{output_dir}/entropy_analysis.png')
    
    print("Creating controller losses plot...")
    create_controller_losses_plot(data['controller_losses'], f'{output_dir}/controller_losses.png')
    
    print("Creating comprehensive dashboard...")
    create_comprehensive_dashboard(data, f'{output_dir}/training_dashboard.png')
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Generated files:")
    print("- reward_progression.png")
    print("- entropy_analysis.png") 
    print("- controller_losses.png")
    print("- training_dashboard.png")

if __name__ == "__main__":
    main()