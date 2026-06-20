#!/usr/bin/env python3
"""
PRISM Training Results Visualization
Creates modern, interactive visualizations of the neural architecture search training process
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

class PRISMVisualizer:
    """Modern visualization tool for PRISM training results"""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'background': '#F5F5F5',
            'text': '#2C3E50'
        }
        
    def load_training_data(self, experiment_name: str = "prism_search"):
        """Load training data from experiment directory"""
        exp_path = self.experiments_dir / experiment_name
        
        # Load training history
        history_file = exp_path / "training_history.json"
        if not history_file.exists():
            raise FileNotFoundError(f"Training history not found: {history_file}")
            
        with open(history_file, 'r') as f:
            history = json.load(f)
            
        # Load search results
        results_file = exp_path / "search_results.json"
        search_results = {}
        if results_file.exists():
            with open(results_file, 'r') as f:
                search_results = json.load(f)
                
        return history, search_results
    
    def create_training_progress_chart(self, history: dict):
        """Create comprehensive training progress visualization"""
        episodes = list(range(1, len(history['rewards']) + 1))
        rewards = history['rewards']
        entropies = history['entropies']
        
        # Calculate moving averages
        window = 10
        rewards_ma = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        entropies_ma = pd.Series(entropies).rolling(window=window, min_periods=1).mean()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Episode Rewards (Dice Scores)', 
                'Controller Entropy', 
                'Training Progress Overview',
                'Performance Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"type": "histogram"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Episode Rewards
        fig.add_trace(
            go.Scatter(
                x=episodes, y=rewards,
                mode='markers+lines',
                name='Episode Reward',
                line=dict(color=self.colors['primary'], width=1),
                marker=dict(size=4, opacity=0.6),
                hovertemplate='Episode: %{x}<br>Dice Score: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=episodes, y=rewards_ma,
                mode='lines',
                name='Moving Average (10)',
                line=dict(color=self.colors['accent'], width=3),
                hovertemplate='Episode: %{x}<br>MA Dice: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Controller Entropy
        fig.add_trace(
            go.Scatter(
                x=episodes, y=entropies,
                mode='lines',
                name='Entropy',
                line=dict(color=self.colors['secondary'], width=2),
                hovertemplate='Episode: %{x}<br>Entropy: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=episodes, y=entropies_ma,
                mode='lines',
                name='Entropy MA',
                line=dict(color=self.colors['success'], width=2, dash='dash'),
                hovertemplate='Episode: %{x}<br>Entropy MA: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Combined Progress Overview
        fig.add_trace(
            go.Scatter(
                x=episodes, y=rewards,
                mode='lines',
                name='Dice Score',
                line=dict(color=self.colors['primary'], width=2),
                yaxis='y3'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=episodes, y=entropies,
                mode='lines',
                name='Entropy',
                line=dict(color=self.colors['secondary'], width=2),
                yaxis='y4'
            ),
            row=2, col=1
        )
        
        # 4. Performance Distribution
        fig.add_trace(
            go.Histogram(
                x=rewards,
                name='Dice Distribution',
                nbinsx=20,
                marker_color=self.colors['primary'],
                opacity=0.7,
                hovertemplate='Dice Range: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🧠 PRISM Neural Architecture Search - Training Progress',
                'x': 0.5,
                'font': {'size': 24, 'color': self.colors['text']}
            },
            showlegend=True,
            height=800,
            plot_bgcolor='white',
            paper_bgcolor=self.colors['background'],
            font=dict(family="Arial, sans-serif", size=12, color=self.colors['text'])
        )
        
        # Update axes
        fig.update_xaxes(title_text="Episode", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Dice Score", row=1, col=1, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Episode", row=1, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Entropy", row=1, col=2, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Episode", row=2, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Dice Score", row=2, col=1, side='left', gridcolor='lightgray')
        
        # Set up secondary y-axis for entropy
        fig.update_layout(yaxis4=dict(title="Entropy", side="right", overlaying="y3"))
        
        fig.update_xaxes(title_text="Dice Score", row=2, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Frequency", row=2, col=2, gridcolor='lightgray')
        
        return fig
    
    def create_architecture_analysis(self, search_results: dict):
        """Create visualization of the best architecture found"""
        if not search_results:
            return None
            
        best_arch = search_results.get('best_architecture', {})
        best_score = search_results.get('best_dice_score', 0)
        patch_sizes = search_results.get('patch_sizes', {})
        
        # Create architecture breakdown
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Best Architecture Components',
                'Patch Size Configuration',
                'Architecture Parameters',
                'Performance Summary'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "indicator"}]]
        )
        
        # 1. Architecture components
        if best_arch:
            components = []
            values = []
            for key, value in best_arch.items():
                if key not in ['patch_hw', 'patch_d']:
                    components.append(key.replace('_', ' ').title())
                    values.append(value)
            
            fig.add_trace(
                go.Bar(
                    x=components,
                    y=values,
                    marker_color=self.colors['primary'],
                    name='Parameters',
                    hovertemplate='%{x}: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Patch sizes
        if patch_sizes:
            hw_sizes = patch_sizes.get('hw_sizes', [])
            d_sizes = patch_sizes.get('d_sizes', [])
            
            if hw_sizes and d_sizes:
                selected_hw = hw_sizes[best_arch.get('patch_hw', 0)] if best_arch.get('patch_hw', 0) < len(hw_sizes) else 64
                selected_d = d_sizes[best_arch.get('patch_d', 0)] if best_arch.get('patch_d', 0) < len(d_sizes) else 32
                
                fig.add_trace(
                    go.Bar(
                        x=['Height/Width', 'Depth', 'Channels'],
                        y=[selected_hw, selected_d, patch_sizes.get('base_channels', 32)],
                        marker_color=[self.colors['accent'], self.colors['secondary'], self.colors['success']],
                        name='Patch Config',
                        hovertemplate='%{x}: %{y}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # 3. Architecture breakdown pie chart
        if best_arch:
            activation_names = ['ReLU', 'LeakyReLU', 'ELU']
            activation = activation_names[best_arch.get('activation_function', 0)]
            
            arch_breakdown = {
                'Skip Connections': best_arch.get('skip_connections', 0),
                'Dilation Stages': sum([
                    best_arch.get('dilation_rate_stage2', 0),
                    best_arch.get('dilation_rate_stage3', 0),
                    best_arch.get('dilation_rate_stage4', 0)
                ]),
                'Pooling Stages': sum([
                    best_arch.get('pooling_stride_stage3', 0),
                    best_arch.get('pooling_stride_stage4', 0)
                ])
            }
            
            fig.add_trace(
                go.Pie(
                    labels=list(arch_breakdown.keys()),
                    values=list(arch_breakdown.values()),
                    marker_colors=[self.colors['primary'], self.colors['accent'], self.colors['secondary']],
                    hovertemplate='%{label}: %{value}<br>%{percent}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Performance indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=best_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Best Dice Score"},
                delta={'reference': 0.5, 'increasing': {'color': self.colors['success']}},
                gauge={
                    'axis': {'range': [None, 1.0]},
                    'bar': {'color': self.colors['primary']},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': self.colors['accent']},
                        {'range': [0.8, 1.0], 'color': self.colors['success']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': '🏗️ Best Architecture Analysis',
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['text']}
            },
            height=700,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor=self.colors['background']
        )
        
        return fig
    
    def create_convergence_analysis(self, history: dict):
        """Create convergence analysis visualization"""
        rewards = history['rewards']
        episodes = list(range(1, len(rewards) + 1))
        
        # Calculate convergence metrics
        window = 20
        std_rolling = pd.Series(rewards).rolling(window=window, min_periods=1).std()
        mean_rolling = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Convergence Analysis', 'Learning Progress'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Convergence plot
        fig.add_trace(
            go.Scatter(
                x=episodes, y=mean_rolling,
                mode='lines',
                name='Mean Performance',
                line=dict(color=self.colors['primary'], width=3),
                hovertemplate='Episode: %{x}<br>Mean: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add confidence bands
        upper_band = mean_rolling + std_rolling
        lower_band = mean_rolling - std_rolling
        
        fig.add_trace(
            go.Scatter(
                x=episodes, y=upper_band,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=episodes, y=lower_band,
                mode='lines',
                fill='tonexty',
                fillcolor=f'rgba(46, 134, 171, 0.2)',
                line=dict(width=0),
                name='Confidence Band',
                hovertemplate='Episode: %{x}<br>Range: ±%{customdata:.4f}<extra></extra>',
                customdata=std_rolling
            ),
            row=1, col=1
        )
        
        # Standard deviation on secondary axis
        fig.add_trace(
            go.Scatter(
                x=episodes, y=std_rolling,
                mode='lines',
                name='Std Deviation',
                line=dict(color=self.colors['accent'], width=2),
                yaxis='y2',
                hovertemplate='Episode: %{x}<br>Std: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Learning progress (cumulative best)
        cumulative_best = pd.Series(rewards).cummax()
        fig.add_trace(
            go.Scatter(
                x=episodes, y=cumulative_best,
                mode='lines+markers',
                name='Best So Far',
                line=dict(color=self.colors['success'], width=3),
                marker=dict(size=4),
                hovertemplate='Episode: %{x}<br>Best: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=episodes, y=rewards,
                mode='markers',
                name='Episode Rewards',
                marker=dict(
                    color=rewards,
                    colorscale='Viridis',
                    size=6,
                    opacity=0.6,
                    colorbar=dict(title="Dice Score")
                ),
                hovertemplate='Episode: %{x}<br>Dice: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title={
                'text': '📈 Training Convergence Analysis',
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['text']}
            },
            height=500,
            plot_bgcolor='white',
            paper_bgcolor=self.colors['background']
        )
        
        # Set y-axis titles
        fig.update_yaxes(title_text="Dice Score", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Standard Deviation", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Dice Score", row=1, col=2)
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=1, col=2)
        
        return fig
    
    def generate_report(self, experiment_name: str = "prism_search", output_dir: str = "visualization_results"):
        """Generate complete visualization report"""
        print("🎨 Generating PRISM Training Visualizations...")
        
        # Load data
        history, search_results = self.load_training_data(experiment_name)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate visualizations
        print("📊 Creating training progress chart...")
        fig1 = self.create_training_progress_chart(history)
        fig1.write_html(output_path / "training_progress.html")
        
        print("🏗️ Creating architecture analysis...")
        fig2 = self.create_architecture_analysis(search_results)
        if fig2:
            fig2.write_html(output_path / "architecture_analysis.html")
        
        print("📈 Creating convergence analysis...")
        fig3 = self.create_convergence_analysis(history)
        fig3.write_html(output_path / "convergence_analysis.html")
        
        # Create summary dashboard
        print("🎯 Creating summary dashboard...")
        self.create_dashboard(history, search_results, output_path)
        
        print(f"✅ Visualizations saved to: {output_path.absolute()}")
        print(f"🌐 Open {output_path.absolute()}/dashboard.html in your browser!")
        
        return output_path
    
    def create_dashboard(self, history: dict, search_results: dict, output_path: Path):
        """Create a comprehensive dashboard HTML file"""
        best_score = search_results.get('best_dice_score', 0)
        total_episodes = len(history['rewards'])
        final_score = history['rewards'][-1] if history['rewards'] else 0
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRISM Training Dashboard</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            border-radius: 10px;
            color: white;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2E86AB;
            margin-bottom: 10px;
        }}
        .stat-label {{
            font-size: 1.1em;
            color: #666;
            font-weight: 500;
        }}
        .visualization-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
        }}
        .viz-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2E86AB;
        }}
        .viz-title {{
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #2E86AB;
            font-weight: bold;
        }}
        .viz-description {{
            color: #666;
            margin-bottom: 20px;
            line-height: 1.6;
        }}
        .btn {{
            display: inline-block;
            padding: 12px 25px;
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
            transition: transform 0.3s ease;
        }}
        .btn:hover {{
            transform: scale(1.05);
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #2E86AB, #F18F01);
            width: {min(best_score * 100, 100)}%;
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 PRISM Neural Architecture Search</h1>
            <p>Training Results Dashboard - {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{best_score:.4f}</div>
                <div class="stat-label">Best Dice Score</div>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_episodes}</div>
                <div class="stat-label">Total Episodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{final_score:.4f}</div>
                <div class="stat-label">Final Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{((best_score - history['rewards'][0]) / history['rewards'][0] * 100):.1f}%</div>
                <div class="stat-label">Improvement</div>
            </div>
        </div>
        
        <div class="visualization-grid">
            <div class="viz-card">
                <div class="viz-title">📊 Training Progress</div>
                <div class="viz-description">
                    Comprehensive view of the training process including episode rewards, moving averages, 
                    controller entropy, and performance distribution across all episodes.
                </div>
                <a href="training_progress.html" class="btn">View Training Progress</a>
            </div>
            
            <div class="viz-card">
                <div class="viz-title">🏗️ Architecture Analysis</div>
                <div class="viz-description">
                    Detailed breakdown of the best architecture found during the search, including 
                    component analysis, patch size configuration, and performance metrics.
                </div>
                <a href="architecture_analysis.html" class="btn">View Architecture Analysis</a>
            </div>
            
            <div class="viz-card">
                <div class="viz-title">📈 Convergence Analysis</div>
                <div class="viz-description">
                    Analysis of training convergence patterns, including confidence bands, 
                    standard deviation trends, and cumulative best performance tracking.
                </div>
                <a href="convergence_analysis.html" class="btn">View Convergence Analysis</a>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h3>🎉 PRISM Training Completed Successfully!</h3>
            <p>The neural architecture search has identified an optimal architecture with a Dice score of <strong>{best_score:.4f}</strong></p>
            <p>Ready for deployment and clinical evaluation! 🏥</p>
        </div>
    </div>
</body>
</html>
        """
        
        with open(output_path / "dashboard.html", 'w') as f:
            f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Visualize PRISM training results')
    parser.add_argument('--experiment', default='prism_search', help='Experiment name')
    parser.add_argument('--output', default='visualization_results', help='Output directory')
    
    args = parser.parse_args()
    
    visualizer = PRISMVisualizer()
    output_path = visualizer.generate_report(args.experiment, args.output)
    
    print(f"\n🎨 Visualization complete!")
    print(f"📁 Files saved to: {output_path}")
    print(f"🌐 Open the dashboard: file://{output_path.absolute()}/dashboard.html")

if __name__ == "__main__":
    main() 