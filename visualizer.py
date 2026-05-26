"""
Visualization Module
Creates visualizations for failure analysis and performance comparison.
"""

import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


class Visualizer:
    """Creates visualizations for experiment results."""
    
    def __init__(self, output_dir: str = "visualizations", dpi: int = 300):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for PNG outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_failure_rates(self, failure_rates: Dict[str, Dict[str, float]],
                          filename: str = "failure_rates"):
        """
        Plot failure rates comparison across models and strategies.
        
        Args:
            failure_rates: Dict[model][strategy] = failure_rate
            filename: Output filename (without extension)
        """
        # Prepare data
        data = []
        for model, strategies in failure_rates.items():
            for strategy, rate in strategies.items():
                data.append({
                    "Model": model,
                    "Strategy": strategy,
                    "Failure Rate": rate
                })
        
        df = pd.DataFrame(data)
        
        # Matplotlib version
        plt.figure(figsize=(12, 6))
        
        models = df["Model"].unique()
        strategies = df["Strategy"].unique()
        x = range(len(models))
        width = 0.8 / len(strategies)
        
        for i, strategy in enumerate(strategies):
            strategy_data = df[df["Strategy"] == strategy]
            values = [strategy_data[strategy_data["Model"] == model]["Failure Rate"].values[0] 
                     if len(strategy_data[strategy_data["Model"] == model]) > 0 else 0
                     for model in models]
            plt.bar([xi + i * width for xi in x], values, width, label=strategy)
        
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Failure Rate", fontsize=12)
        plt.title("Failure Rates by Model and Strategy", fontsize=14, fontweight='bold')
        plt.xticks([xi + width * (len(strategies) - 1) / 2 for xi in x], models, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=self.dpi)
        plt.close()
        
        # Plotly interactive version
        fig = px.bar(df, x="Model", y="Failure Rate", color="Strategy",
                    barmode="group",
                    title="Failure Rates by Model and Strategy",
                    labels={"Failure Rate": "Failure Rate", "Model": "Model"})
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            font=dict(size=12)
        )
        fig.write_html(self.output_dir / f"{filename}.html")
    
    def plot_mutation_distribution(self, mutation_counts: Dict[str, int],
                                   title: str = "Failure Distribution by Mutation Type",
                                   filename: str = None):
        """
        Plot distribution of failures by mutation type.
        
        Args:
            mutation_counts: Dict[mutation_type] = count
            title: Plot title
            filename: Output filename (auto-generated if None)
        """
        if not mutation_counts:
            return
        
        if filename is None:
            filename = f"mutation_dist_{title.replace(' ', '_').replace('-', '_')}"
        
        # Matplotlib version
        plt.figure(figsize=(10, 6))
        
        mutations = list(mutation_counts.keys())
        counts = list(mutation_counts.values())
        
        colors = sns.color_palette("husl", len(mutations))
        plt.bar(mutations, counts, color=colors)
        plt.xlabel("Mutation Type", fontsize=12)
        plt.ylabel("Failure Count", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=self.dpi)
        plt.close()
        
        # Plotly interactive version
        fig = go.Figure(data=[
            go.Bar(x=mutations, y=counts, marker_color=colors)
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Mutation Type",
            yaxis_title="Failure Count",
            xaxis_tickangle=-45,
            height=500,
            font=dict(size=12)
        )
        fig.write_html(self.output_dir / f"{filename}.html")
    
    def plot_category_distribution(self, category_counts: Dict[str, int],
                                   title: str = "Failure Distribution by Category",
                                   filename: str = None):
        """
        Plot distribution of failures by category.
        
        Args:
            category_counts: Dict[category] = count
            title: Plot title
            filename: Output filename (auto-generated if None)
        """
        if not category_counts:
            return
        
        if filename is None:
            filename = f"category_dist_{title.replace(' ', '_').replace('-', '_')}"
        
        # Matplotlib version - Pie chart
        plt.figure(figsize=(10, 8))
        
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        colors = sns.color_palette("husl", len(categories))
        plt.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}_pie.png", dpi=self.dpi)
        plt.close()
        
        # Matplotlib version - Bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(categories, counts, color=colors)
        plt.xlabel("Failure Category", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}_bar.png", dpi=self.dpi)
        plt.close()
        
        # Plotly interactive version
        fig = go.Figure(data=[
            go.Pie(labels=categories, values=counts, hole=0.3)
        ])
        fig.update_layout(
            title=title,
            height=500,
            font=dict(size=12)
        )
        fig.write_html(self.output_dir / f"{filename}.html")
    
    def plot_difficulty_distribution(self, difficulty_counts: Dict[int, int],
                                     title: str = "Failures by Problem Difficulty",
                                     filename: str = "difficulty_distribution"):
        """
        Plot distribution of failures by problem difficulty.
        
        Args:
            difficulty_counts: Dict[difficulty_level] = count
            title: Plot title
            filename: Output filename
        """
        if not difficulty_counts:
            return
        
        # Sort by difficulty
        difficulties = sorted(difficulty_counts.keys())
        counts = [difficulty_counts[d] for d in difficulties]
        
        # Matplotlib version
        plt.figure(figsize=(10, 6))
        plt.plot(difficulties, counts, marker='o', linewidth=2, markersize=8)
        plt.xlabel("Difficulty Level", fontsize=12)
        plt.ylabel("Failure Count", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(difficulties)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=self.dpi)
        plt.close()
        
        # Plotly interactive version
        fig = go.Figure(data=[
            go.Scatter(x=difficulties, y=counts, mode='lines+markers',
                      line=dict(width=2), marker=dict(size=10))
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Difficulty Level",
            yaxis_title="Failure Count",
            height=500,
            font=dict(size=12)
        )
        fig.write_html(self.output_dir / f"{filename}.html")
    
    def plot_pattern_frequency(self, patterns: List[tuple],
                               title: str = "Common Failure Patterns",
                               filename: str = "pattern_frequency",
                               top_n: int = 10):
        """
        Plot frequency of common failure patterns.
        
        Args:
            patterns: List of (pattern_name, count) tuples
            title: Plot title
            filename: Output filename
            top_n: Number of top patterns to show
        """
        if not patterns:
            return
        
        # Take top N patterns
        patterns = patterns[:top_n]
        pattern_names = [p[0] for p in patterns]
        counts = [p[1] for p in patterns]
        
        # Matplotlib version
        plt.figure(figsize=(12, 6))
        colors = sns.color_palette("viridis", len(pattern_names))
        plt.barh(pattern_names, counts, color=colors)
        plt.xlabel("Frequency", fontsize=12)
        plt.ylabel("Pattern", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=self.dpi)
        plt.close()
        
        # Plotly interactive version
        fig = go.Figure(data=[
            go.Bar(y=pattern_names, x=counts, orientation='h',
                  marker=dict(color=counts, colorscale='Viridis'))
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Frequency",
            yaxis_title="Pattern",
            height=max(400, len(pattern_names) * 40),
            font=dict(size=12)
        )
        fig.write_html(self.output_dir / f"{filename}.html")
    
    def plot_strategy_comparison(self, strategy_results: Dict[str, Dict[str, Any]],
                                filename: str = "strategy_comparison"):
        """
        Compare performance of different search strategies.
        
        Args:
            strategy_results: Dict[strategy_name] = {metrics}
            filename: Output filename
        """
        if not strategy_results:
            return
        
        # Prepare data
        strategies = list(strategy_results.keys())
        failure_rates = [strategy_results[s].get("failure_rate", 0) for s in strategies]
        total_evals = [strategy_results[s].get("total_evaluations", 0) for s in strategies]
        failures_found = [strategy_results[s].get("failures", 0) for s in strategies]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Failure rate comparison
        axes[0].bar(strategies, failure_rates, color=sns.color_palette("husl", len(strategies)))
        axes[0].set_xlabel("Strategy", fontsize=12)
        axes[0].set_ylabel("Failure Rate", fontsize=12)
        axes[0].set_title("Failure Rate by Strategy", fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Total evaluations
        axes[1].bar(strategies, total_evals, color=sns.color_palette("muted", len(strategies)))
        axes[1].set_xlabel("Strategy", fontsize=12)
        axes[1].set_ylabel("Total Evaluations", fontsize=12)
        axes[1].set_title("Evaluations per Strategy", fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Failures found
        axes[2].bar(strategies, failures_found, color=sns.color_palette("dark", len(strategies)))
        axes[2].set_xlabel("Strategy", fontsize=12)
        axes[2].set_ylabel("Failures Found", fontsize=12)
        axes[2].set_title("Failures Found per Strategy", fontsize=12, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=self.dpi)
        plt.close()
        
        # Plotly interactive version
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='Failure Rate', x=strategies, y=failure_rates))
        fig.add_trace(go.Bar(name='Failures Found (scaled)', x=strategies, 
                            y=[f/max(failures_found) if max(failures_found) > 0 else 0 
                               for f in failures_found]))
        
        fig.update_layout(
            title="Strategy Performance Comparison",
            xaxis_title="Strategy",
            yaxis_title="Normalized Metrics",
            barmode='group',
            height=500,
            font=dict(size=12)
        )
        fig.write_html(self.output_dir / f"{filename}.html")
    
    def plot_domain_comparison(self, domain_counts: Dict[str, int],
                              title: str = "Failures by Domain",
                              filename: str = "domain_comparison"):
        """
        Compare failures across different problem domains.
        
        Args:
            domain_counts: Dict[domain] = count
            title: Plot title
            filename: Output filename
        """
        if not domain_counts:
            return
        
        domains = list(domain_counts.keys())
        counts = list(domain_counts.values())
        
        # Matplotlib version
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("Set2", len(domains))
        plt.bar(domains, counts, color=colors)
        plt.xlabel("Domain", fontsize=12)
        plt.ylabel("Failure Count", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=self.dpi)
        plt.close()
        
        # Plotly interactive version
        fig = px.bar(x=domains, y=counts, labels={'x': 'Domain', 'y': 'Failure Count'},
                    title=title, color=domains)
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            font=dict(size=12),
            showlegend=False
        )
        fig.write_html(self.output_dir / f"{filename}.html")
    
    def create_summary_dashboard(self, experiment_results: Dict[str, Any],
                                 filename: str = "summary_dashboard"):
        """
        Create a comprehensive summary dashboard.
        
        Args:
            experiment_results: Complete experiment results
            filename: Output filename
        """
        # Extract key metrics
        total_models = len(experiment_results)
        total_strategies = len(list(experiment_results.values())[0]) if experiment_results else 0
        
        # Create figure with subplots
        fig = go.Figure()
        
        # This would be a complex dashboard - simplified version
        fig.add_trace(go.Indicator(
            mode="number",
            value=total_models,
            title="Models Tested",
            domain={'x': [0, 0.5], 'y': [0.5, 1]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=total_strategies,
            title="Strategies Used",
            domain={'x': [0.5, 1], 'y': [0.5, 1]}
        ))
        
        fig.update_layout(
            title="Experiment Summary Dashboard",
            height=600,
            font=dict(size=14)
        )
        
        fig.write_html(self.output_dir / f"{filename}.html")


# if __name__ == "__main__":
#     # Test visualizer
#     print("=== Testing Visualizer ===\n")
    
#     visualizer = Visualizer(output_dir="experiments/test_visualizations")
    
#     # Test data
#     failure_rates = {
#         "openai:gpt-4": {"random": 0.35, "evolutionary": 0.42, "hill_climbing": 0.38},
#         "anthropic:claude-3": {"random": 0.30, "evolutionary": 0.38, "hill_climbing": 0.33}
#     }
    
#     mutation_counts = {
#         "constraint": 15,
#         "instruction": 22,
#         "composition": 18,
#         "adversarial_trap": 25
#     }
    
#     category_counts = {
#         "arithmetic_error": 20,
#         "logical_error": 15,
#         "instruction_following": 12,
#         "trap_susceptibility": 18,
#         "parsing_error": 8
#     }
    
#     difficulty_counts = {1: 5, 2: 12, 3: 20, 4: 15, 5: 8}
    
#     patterns = [
#         ("off_by_one", 15),
#         ("missing_elements", 12),
#         ("fell_for_trap", 20),
#         ("wrong_operation", 8),
#         ("extra_elements", 10)
#     ]
    
#     # Generate visualizations
#     print("Generating failure rates comparison...")
#     visualizer.plot_failure_rates(failure_rates)
    
#     print("Generating mutation distribution...")
#     visualizer.plot_mutation_distribution(mutation_counts)
    
#     print("Generating category distribution...")
#     visualizer.plot_category_distribution(category_counts)
    
#     print("Generating difficulty distribution...")
#     visualizer.plot_difficulty_distribution(difficulty_counts)
    
#     print("Generating pattern frequency...")
#     visualizer.plot_pattern_frequency(patterns)
    
#     print(f"\nVisualizations saved to test_visualizations/")
#     print("Generated files:")
#     for file in Path("test_visualizations").glob("*"):
#         print(f"  - {file.name}")
