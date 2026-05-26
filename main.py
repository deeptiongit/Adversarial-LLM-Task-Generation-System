"""
Main Orchestrator
Coordinates all modules and runs experiments.
"""

from ast import arg
import os
import yaml
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

from problem_generator import ProblemGenerator
from mutation_engine import MutationEngine
from llm_interface import LLMInterface, LLMFactory
from evaluator import Evaluator
from failure_analyzer import FailureAnalyzer
from search_optimizer import SearchOptimizer
from visualizer import Visualizer


class ExperimentConfig:
    """Configuration for an experiment."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load environment variables for API keys
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment variables."""
        api_keys = self.config.get("api_keys", {})
        
        for provider, key_name in api_keys.items():
            if isinstance(key_name, str) and key_name.startswith("${") and key_name.endswith("}"):
                env_var = key_name[2:-1]
                api_keys[provider] = os.getenv(env_var)
        
        self.config["api_keys"] = api_keys
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value if value is not None else default


class Experiment:
    """Represents a single experiment run."""
    
    def __init__(self, name: str, config: ExperimentConfig, 
                 models: List[str], strategies: List[str]):
        """
        Initialize experiment.
        
        Args:
            name: Experiment name
            config: Configuration object
            models: List of models to test
            strategies: List of search strategies to use
        """
        self.name = name
        self.config = config
        self.models = models
        self.strategies = strategies
        self.results = {}
        
        # Create experiment directory
        self.exp_dir = Path(f"experiments/{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        self.logger.info(f"Initialized experiment: {name}")
        self.logger.info(f"Models: {models}")
        self.logger.info(f"Strategies: {strategies}")
    
    def _setup_logging(self):
        """Set up logging for the experiment."""
        log_file = self.exp_dir / "experiment.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"Experiment-{self.name}")
    
    def run(self, max_iterations: int = 100, batch_size: int = 5):
        """
        Run the experiment.
        
        Args:
            max_iterations: Maximum iterations per strategy
            batch_size: Batch size for search
        """
        self.logger.info(f"Starting experiment with {max_iterations} iterations")
        
        # Initialize components
        optimizer_config = {
            "seed": self.config.get("search.random_seed", 42),
            "log_level": self.config.get("logging.level", "INFO"),
            "log_dir": str(self.exp_dir / "logs"),
            "population_size": self.config.get("search.population_size", 20),
            "mutation_rate": self.config.get("search.mutation_rate", 0.3),
            "crossover_rate": self.config.get("search.crossover_rate", 0.7)
        }
        
        optimizer = SearchOptimizer(optimizer_config)
        
        # Run for each model and strategy combination
        for model_spec in self.models:
            # Parse model spec (format: "provider:model")
            if ":" in model_spec:
                provider, model = model_spec.split(":", 1)
            else:
                provider = "openai"
                model = model_spec
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing model: {provider}:{model}")
            self.logger.info(f"{'='*60}\n")
            
            # Create LLM interface
            try:
                llm_interface = LLMInterface(
                    provider=provider,
                    model=model,
                    temperature=self.config.get("llm.temperature", 0.0),
                    max_tokens=self.config.get("llm.max_tokens", 2000),
                    timeout=self.config.get("llm.timeout", 60),
                    max_retries=self.config.get("llm.max_retries", 3),
                    retry_delay=self.config.get("llm.retry_delay", 2),
                    log_dir=str(self.exp_dir / "logs")
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM interface for {provider}:{model}: {e}")
                continue
            
            model_results = {}
            
            for strategy_name in self.strategies:
                self.logger.info(f"\nRunning strategy: {strategy_name}")
                
                try:
                    # Create strategy
                    strategy = optimizer.create_strategy(strategy_name, llm_interface)
                    
                    # Run search
                    results = strategy.search(
                        max_iterations=max_iterations,
                        batch_size=batch_size
                    )
                    
                    # Analyze failures
                    analyzer = FailureAnalyzer()
                    for result in results:
                        if not result.evaluation.is_correct:
                            analyzer.analyze_failure(
                                result.problem,
                                result.mutated_problem,
                                result.llm_response,
                                result.evaluation
                            )
                    
                    analysis = analyzer.generate_analysis()
                    
                    # Store results
                    model_results[strategy_name] = {
                        "total_evaluations": len(results),
                        "failures": len(strategy.get_failures()),
                        "failure_rate": strategy.get_failure_rate(),
                        "analysis": {
                            "by_category": analysis.failure_by_category,
                            "by_mutation": analysis.failure_by_mutation,
                            "by_domain": analysis.failure_by_domain,
                            "by_difficulty": analysis.failure_by_difficulty,
                            "common_patterns": analysis.common_patterns,
                            "severity_distribution": analysis.severity_distribution
                        },
                        "results": results
                    }
                    
                    self.logger.info(f"Strategy {strategy_name} complete:")
                    self.logger.info(f"  Failure rate: {strategy.get_failure_rate():.2%}")
                    self.logger.info(f"  Failures found: {len(strategy.get_failures())}")
                    
                except Exception as e:
                    self.logger.error(f"Strategy {strategy_name} failed: {e}", exc_info=True)
                    model_results[strategy_name] = {"error": str(e)}
            
            self.results[f"{provider}:{model}"] = model_results
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        self.logger.info(f"\nExperiment complete! Results saved to {self.exp_dir}")
    
    def _save_results(self):
        """Save experiment results."""
        results_file = self.exp_dir / "results.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for model, model_results in self.results.items():
            json_results[model] = {}
            for strategy, strategy_results in model_results.items():
                if "results" in strategy_results:
                    # Convert SearchResult objects to dicts
                    json_results[model][strategy] = {
                        "total_evaluations": strategy_results["total_evaluations"],
                        "failures": strategy_results["failures"],
                        "failure_rate": strategy_results["failure_rate"],
                        "analysis": strategy_results["analysis"]
                    }
                else:
                    json_results[model][strategy] = strategy_results
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _generate_visualizations(self):
        """Generate visualizations from results."""
        self.logger.info("Generating visualizations...")
        
        viz_dir = self.exp_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        visualizer = Visualizer(output_dir=str(viz_dir))
        
        try:
            # Prepare data for visualization
            models = list(self.results.keys())
            strategies = self.strategies
            
            # Failure rate comparison
            failure_rates = {}
            for model in models:
                failure_rates[model] = {}
                for strategy in strategies:
                    if strategy in self.results[model]:
                        failure_rates[model][strategy] = self.results[model][strategy].get("failure_rate", 0)
            
            visualizer.plot_failure_rates(failure_rates)
            
            # Mutation type distribution
            for model in models:
                for strategy in strategies:
                    if strategy in self.results[model] and "analysis" in self.results[model][strategy]:
                        analysis = self.results[model][strategy]["analysis"]
                        if analysis.get("by_mutation"):
                            visualizer.plot_mutation_distribution(
                                analysis["by_mutation"],
                                title=f"{model} - {strategy}"
                            )
            
            # Category distribution
            for model in models:
                for strategy in strategies:
                    if strategy in self.results[model] and "analysis" in self.results[model][strategy]:
                        analysis = self.results[model][strategy]["analysis"]
                        if analysis.get("by_category"):
                            visualizer.plot_category_distribution(
                                analysis["by_category"],
                                title=f"{model} - {strategy}"
                            )
            
            self.logger.info(f"Visualizations saved to {viz_dir}")
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}", exc_info=True)


class AdversarialLLMSystem:
    """Main system orchestrator."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the system."""
        self.config = ExperimentConfig(config_path)
        self.logger = logging.getLogger("AdversarialLLMSystem")
    
    def run_experiment(self, name: str, models: List[str], strategies: List[str],
                      max_iterations: int = 100, batch_size: int = 5):
        """
        Run a complete experiment.
        
        Args:
            name: Experiment name
            models: List of models (format: "provider:model")
            strategies: List of search strategies
            max_iterations: Maximum iterations per strategy
            batch_size: Batch size for search
        """
        experiment = Experiment(name, self.config, models, strategies)
        experiment.run(max_iterations=max_iterations, batch_size=batch_size)
        
        return experiment


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adversarial LLM Task Generation System")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--name", type=str, required=True,
                       help="Experiment name")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                       help="Models to test (format: provider:model)")
    parser.add_argument("--strategies", type=str, nargs="+",
                       default=["random", "evolutionary", "hill_climbing"],
                       help="Search strategies to use")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Maximum iterations per strategy")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Batch size for search")
    
    args = parser.parse_args()
    print(args.models)
    
    # Initialize system
    system = AdversarialLLMSystem(config_path=args.config)
    
    # Run experiment
    system.run_experiment(
        name=args.name,
        models=args.models,
        strategies=args.strategies,
        max_iterations=args.iterations,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
