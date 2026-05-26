"""
Example Usage Script
Demonstrates various ways to use the Adversarial LLM Task Generation System.
"""

import os
import logging
from problem_generator import ProblemGenerator
from mutation_engine import MutationEngine
from llm_interface import LLMInterface
from evaluator import Evaluator
from failure_analyzer import FailureAnalyzer
from search_optimizer import SearchOptimizer
from visualizer import Visualizer
from main import AdversarialLLMSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_problem_generation():
    """Example 1: Basic problem generation and mutation."""
    print("\n" + "="*60)
    print("Example 1: Basic Problem Generation and Mutation")
    print("="*60 + "\n")
    
    # Initialize components
    generator = ProblemGenerator(seed=42)
    mutator = MutationEngine(seed=42)
    
    # Generate a problem
    problem = generator.generate_problem(template_id="nt_prime_filtering")
    print(f"Original Problem:")
    print(f"  Template: {problem.template_name}")
    print(f"  Domain: {problem.domain}")
    print(f"  Difficulty: {problem.difficulty}")
    print(f"  Prompt: {problem.prompt}")
    print(f"  Ground Truth: {problem.ground_truth}\n")
    
    # Apply mutations
    print("Applying mutations:")
    mutation_types = ["constraint", "instruction", "adversarial_trap"]
    
    for mut_type in mutation_types:
        mutated = mutator.apply_specific_mutation(problem, mut_type)
        if mutated:
            print(f"\n  {mut_type.upper()} mutation:")
            print(f"    Mutated Prompt: {mutated.mutated_prompt[:100]}...")
            print(f"    Ground Truth Preserved: {mutated.ground_truth == problem.ground_truth}")


def example_2_llm_evaluation():
    """Example 2: Querying LLM and evaluating responses."""
    print("\n" + "="*60)
    print("Example 2: LLM Query and Evaluation")
    print("="*60 + "\n")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping this example.")
        print("   Set your API key: export OPENAI_API_KEY='your-key'\n")
        return
    
    # Initialize components
    generator = ProblemGenerator(seed=42)
    mutator = MutationEngine(seed=42)
    llm = LLMInterface(provider="openai", model="gpt-3.5-turbo", temperature=0.0)
    evaluator = Evaluator()
    
    # Generate and mutate a simple problem
    problem = generator.generate_problem(template_id="nt_sequence_sum")
    mutated = mutator.apply_specific_mutation(problem, "constraint")
    
    print(f"Problem: {mutated.mutated_prompt}\n")
    
    # Query LLM
    print("Querying LLM...")
    response = llm.query(mutated.mutated_prompt)
    
    print(f"\nLLM Response:")
    print(f"  Raw: {response.raw_response[:200]}...")
    print(f"  Extracted Answer: {response.extracted_answer}")
    print(f"  Latency: {response.latency:.2f}s")
    print(f"  Tokens: {response.tokens_used}")
    
    # Evaluate
    evaluation = evaluator.evaluate(response, problem.ground_truth)
    
    print(f"\nEvaluation:")
    print(f"  Correct: {evaluation.is_correct}")
    print(f"  Ground Truth: {evaluation.ground_truth}")
    print(f"  LLM Answer: {evaluation.llm_answer}")
    if not evaluation.is_correct:
        print(f"  Error Type: {evaluation.error_type}")


def example_3_failure_analysis():
    """Example 3: Analyzing failures with the failure analyzer."""
    print("\n" + "="*60)
    print("Example 3: Failure Analysis")
    print("="*60 + "\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping this example.")
        return
    
    # Initialize components
    generator = ProblemGenerator(seed=42)
    mutator = MutationEngine(seed=42)
    llm = LLMInterface(provider="openai", model="gpt-3.5-turbo", temperature=0.0)
    evaluator = Evaluator()
    analyzer = FailureAnalyzer()
    
    print("Testing multiple problems to find failures...\n")
    
    # Test multiple problems
    for i in range(10):
        problem = generator.generate_problem()
        mutated = mutator.apply_random_mutations(problem, max_mutations=2)
        
        if not mutated:
            continue
        
        response = llm.query(mutated.mutated_prompt)
        evaluation = evaluator.evaluate(response, problem.ground_truth)
        
        # Analyze failures
        if not evaluation.is_correct:
            failure_case = analyzer.analyze_failure(
                problem, mutated, response, evaluation
            )
            print(f"Failure {len(analyzer.failure_cases)}:")
            print(f"  Template: {problem.template_name}")
            print(f"  Category: {failure_case.failure_category}")
            print(f"  Patterns: {failure_case.failure_patterns}")
            print(f"  Severity: {failure_case.severity}")
            print()
    
    # Generate analysis
    if analyzer.failure_cases:
        analysis = analyzer.generate_analysis()
        print(f"\n{'='*60}")
        print("Failure Analysis Summary")
        print(f"{'='*60}")
        print(f"\nTotal Failures: {analysis.total_failures}")
        print(f"\nBy Category:")
        for category, count in analysis.failure_by_category.items():
            print(f"  {category}: {count}")
        print(f"\nBy Mutation Type:")
        for mutation, count in analysis.failure_by_mutation.items():
            print(f"  {mutation}: {count}")
        print(f"\nCommon Patterns:")
        for pattern, count in analysis.common_patterns[:5]:
            print(f"  {pattern}: {count}")
    else:
        print("No failures found in this sample!")


def example_4_search_strategy():
    """Example 4: Using search strategies to find failures."""
    print("\n" + "="*60)
    print("Example 4: Search Strategy")
    print("="*60 + "\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping this example.")
        return
    
    # Initialize components
    config = {
        "seed": 42,
        "log_level": "INFO",
        "log_dir": "logs",
        "population_size": 10,
        "mutation_rate": 0.3,
        "crossover_rate": 0.7
    }
    
    optimizer = SearchOptimizer(config)
    llm = LLMInterface(provider="openai", model="gpt-3.5-turbo", temperature=0.0)
    
    print("Running random search strategy (10 iterations)...\n")
    
    # Create and run random search
    strategy = optimizer.create_strategy("random", llm)
    results = strategy.search(max_iterations=10, batch_size=2)
    
    print(f"\n{'='*60}")
    print("Search Results")
    print(f"{'='*60}")
    print(f"\nTotal Evaluations: {len(results)}")
    print(f"Failures Found: {len(strategy.get_failures())}")
    print(f"Failure Rate: {strategy.get_failure_rate():.2%}")
    
    # Show some failure cases
    failures = strategy.get_failures()[:3]
    if failures:
        print(f"\nSample Failure Cases:")
        for i, failure in enumerate(failures, 1):
            print(f"\n{i}. {failure.problem.template_name}")
            print(f"   Prompt: {failure.mutated_problem.mutated_prompt[:80]}...")
            print(f"   Expected: {failure.problem.ground_truth}")
            print(f"   Got: {failure.llm_response.extracted_answer}")
            print(f"   Error: {failure.evaluation.error_type}")


def example_5_visualization():
    """Example 5: Creating visualizations."""
    print("\n" + "="*60)
    print("Example 5: Visualization")
    print("="*60 + "\n")
    
    # Create visualizer
    viz = Visualizer(output_dir="example_visualizations", dpi=300)
    
    # Sample data
    print("Generating sample visualizations...\n")
    
    # 1. Failure rates
    failure_rates = {
        "openai:gpt-4": {"random": 0.35, "evolutionary": 0.42, "hill_climbing": 0.38},
        "openai:gpt-3.5": {"random": 0.45, "evolutionary": 0.52, "hill_climbing": 0.48}
    }
    viz.plot_failure_rates(failure_rates)
    print("✓ Created: failure_rates.png and failure_rates.html")
    
    # 2. Mutation distribution
    mutation_counts = {
        "constraint": 15,
        "instruction": 22,
        "composition": 18,
        "adversarial_trap": 25
    }
    viz.plot_mutation_distribution(mutation_counts)
    print("✓ Created: mutation distribution visualizations")
    
    # 3. Category distribution
    category_counts = {
        "arithmetic_error": 20,
        "logical_error": 15,
        "instruction_following": 12,
        "trap_susceptibility": 18,
        "parsing_error": 8
    }
    viz.plot_category_distribution(category_counts)
    print("✓ Created: category distribution visualizations")
    
    # 4. Pattern frequency
    patterns = [
        ("off_by_one", 15),
        ("missing_elements", 12),
        ("fell_for_trap", 20),
        ("wrong_operation", 8),
        ("extra_elements", 10)
    ]
    viz.plot_pattern_frequency(patterns)
    print("✓ Created: pattern_frequency.png and pattern_frequency.html")
    
    print(f"\n✓ All visualizations saved to 'example_visualizations/' directory")


def example_6_full_experiment():
    """Example 6: Running a complete experiment."""
    print("\n" + "="*60)
    print("Example 6: Complete Experiment")
    print("="*60 + "\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping this example.")
        print("   This example requires API keys to run a full experiment.")
        return
    
    print("Running a small-scale experiment...")
    print("This will test one model with two strategies.\n")
    
    # Initialize system
    system = AdversarialLLMSystem(config_path="config.yaml")
    
    # Run experiment
    experiment = system.run_experiment(
        name="example_experiment",
        models=["openai:gpt-3.5-turbo"],
        strategies=["random"],
        max_iterations=10,
        batch_size=2
    )
    
    print(f"\n✓ Experiment complete!")
    print(f"  Results saved to: {experiment.exp_dir}")
    print(f"  View experiment.log for detailed logs")
    print(f"  View visualizations/ for generated charts")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" "*20 + "ADVERSARIAL LLM TASK GENERATION SYSTEM")
    print(" "*30 + "Example Usage")
    print("="*80)
    
    examples = [
        ("Basic Problem Generation", example_1_basic_problem_generation),
        ("LLM Evaluation", example_2_llm_evaluation),
        ("Failure Analysis", example_3_failure_analysis),
        ("Search Strategy", example_4_search_strategy),
        ("Visualization", example_5_visualization),
        ("Complete Experiment", example_6_full_experiment)
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "-"*80)
    choice = input("\nEnter example number to run (1-6), 'all' for all, or 'q' to quit: ").strip().lower()
    
    if choice == 'q':
        print("Goodbye!")
        return
    elif choice == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                logger.error(f"Example '{name}' failed: {e}", exc_info=True)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                name, func = examples[idx]
                func()
            else:
                print(f"Invalid choice: {choice}")
        except ValueError:
            print(f"Invalid choice: {choice}")
    
    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
