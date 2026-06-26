"""
System Test Script
Tests all modules to ensure the system is working correctly.
"""

import sys
import traceback


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import problem_generator
        import mutation_engine
        import llm_interface
        import evaluator
        import failure_analyzer
        import search_optimizer
        import visualizer
        import main
        print(" All modules imported successfully\n")
        return True
    except ImportError as e:
        print(f" Import failed: {e}\n")
        return False


def test_problem_generator():
    """Test problem generator."""
    print("Testing Problem Generator...")
    try:
        from problem_generator import ProblemGenerator
        
        generator = ProblemGenerator(seed=42)
        
        # Test random generation
        problem = generator.generate_problem()
        assert problem is not None
        assert problem.ground_truth is not None
        
        # Test domain filtering
        nt_problem = generator.generate_problem(domain="number_theory")
        assert "number_theory" in nt_problem.domain
        
        gt_problem = generator.generate_problem(domain="game_theory")
        assert "game_theory" in gt_problem.domain
        
        # Test specific template
        prime_problem = generator.generate_problem(template_id="nt_prime_filtering")
        assert prime_problem.template_id == "nt_prime_filtering"
        
        # Test batch generation
        problems = generator.generate_batch(5)
        assert len(problems) == 5
        
        print(" Problem Generator working correctly\n")
        return True
    except Exception as e:
        print(f"Problem Generator failed: {e}")
        traceback.print_exc()
        print()
        return False


def test_mutation_engine():
    """Test mutation engine."""
    print("Testing Mutation Engine...")
    try:
        from problem_generator import ProblemGenerator
        from mutation_engine import MutationEngine
        
        generator = ProblemGenerator(seed=42)
        mutator = MutationEngine(seed=42)
        
        problem = generator.generate_problem()
        
        # Test each mutation type
        for mut_type in ["constraint", "instruction", "composition", "adversarial_trap"]:
            mutated = mutator.apply_specific_mutation(problem, mut_type)
            assert mutated is not None
            assert mutated.ground_truth == problem.ground_truth
            assert len(mutated.mutations) > 0
            assert mutated.mutations[0].mutation_type == mut_type
        
        # Test random mutations
        mutated = mutator.apply_random_mutations(problem, max_mutations=3)
        assert mutated is not None
        
        print("Mutation Engine working correctly\n")
        return True
    except Exception as e:
        print(f"Mutation Engine failed: {e}")
        traceback.print_exc()
        print()
        return False


def test_evaluator():
    """Test evaluator."""
    print("Testing Evaluator...")
    try:
        from evaluator import Evaluator
        from llm_interface import LLMResponse
        from datetime import datetime
        
        evaluator = Evaluator()
        
        # Test correct answer
        response = LLMResponse(
            raw_response="42",
            extracted_answer=42,
            provider="test",
            model="test",
            prompt="test",
            timestamp=datetime.now().isoformat(),
            latency=0.0
        )
        result = evaluator.evaluate(response, 42)
        assert result.is_correct == True
        
        # Test incorrect answer
        response = LLMResponse(
            raw_response="100",
            extracted_answer=100,
            provider="test",
            model="test",
            prompt="test",
            timestamp=datetime.now().isoformat(),
            latency=0.0
        )
        result = evaluator.evaluate(response, 42)
        assert result.is_correct == False
        assert result.error_type is not None
        
        # Test list comparison
        response = LLMResponse(
            raw_response="[1, 2, 3]",
            extracted_answer=[1, 2, 3],
            provider="test",
            model="test",
            prompt="test",
            timestamp=datetime.now().isoformat(),
            latency=0.0
        )
        result = evaluator.evaluate(response, [3, 2, 1])
        assert result.is_correct == True  # Order doesn't matter
        
        print("Evaluator working correctly\n")
        return True
    except Exception as e:
        print(f"Evaluator failed: {e}")
        traceback.print_exc()
        print()
        return False


def test_failure_analyzer():
    """Test failure analyzer."""
    print("Testing Failure Analyzer...")
    try:
        from problem_generator import ProblemGenerator
        from mutation_engine import MutationEngine
        from llm_interface import LLMResponse
        from evaluator import Evaluator
        from failure_analyzer import FailureAnalyzer
        from datetime import datetime
        
        generator = ProblemGenerator(seed=42)
        mutator = MutationEngine(seed=42)
        evaluator = Evaluator()
        analyzer = FailureAnalyzer()
        
        # Create a failure case
        problem = generator.generate_problem()
        mutated = mutator.apply_random_mutations(problem)
        
        response = LLMResponse(
            raw_response="wrong answer",
            extracted_answer="wrong",
            provider="test",
            model="test",
            prompt=mutated.mutated_prompt,
            timestamp=datetime.now().isoformat(),
            latency=0.0
        )
        
        evaluation = evaluator.evaluate(response, problem.ground_truth)
        
        if not evaluation.is_correct:
            failure_case = analyzer.analyze_failure(problem, mutated, response, evaluation)
            assert failure_case.failure_category is not None
            assert failure_case.severity >= 1
            
            # Test analysis generation
            analysis = analyzer.generate_analysis()
            assert analysis.total_failures > 0
        
        print("Failure Analyzer working correctly\n")
        return True
    except Exception as e:
        print(f"Failure Analyzer failed: {e}")
        traceback.print_exc()
        print()
        return False


def test_visualizer():
    """Test visualizer."""
    print("Testing Visualizer...")
    try:
        from visualizer import Visualizer
        import os
        
        viz = Visualizer(output_dir="test_viz", dpi=100)
        
        # Test data
        failure_rates = {
            "model1": {"strategy1": 0.3, "strategy2": 0.4},
            "model2": {"strategy1": 0.35, "strategy2": 0.45}
        }
        
        mutation_counts = {"constraint": 10, "instruction": 15}
        category_counts = {"arithmetic": 8, "logical": 12}
        
        # Generate visualizations
        viz.plot_failure_rates(failure_rates, filename="test_failure_rates")
        viz.plot_mutation_distribution(mutation_counts, filename="test_mutations")
        viz.plot_category_distribution(category_counts, filename="test_categories")
        
        # Check files were created
        assert os.path.exists("test_viz/test_failure_rates.png")
        assert os.path.exists("test_viz/test_failure_rates.html")
        
        print("Visualizer working correctly\n")
        return True
    except Exception as e:
        print(f" Visualizer failed: {e}")
        traceback.print_exc()
        print()
        return False


def test_configuration():
    """Test configuration loading."""
    print("Testing Configuration...")
    try:
        from main import ExperimentConfig
        
        config = ExperimentConfig(config_path="config.yaml")
        
        # Test configuration access
        assert config.get("llm.temperature") == 0.0
        assert config.get("search.random_seed") == 42
        assert isinstance(config.get("search.strategies"), list)
        
        print(" Configuration loading correctly\n")
        return True
    except Exception as e:
        print(f" Configuration failed: {e}")
        traceback.print_exc()
        print()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" "*25 + "SYSTEM TEST SUITE")
    print("="*80 + "\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Problem Generator", test_problem_generator),
        ("Mutation Engine", test_mutation_engine),
        ("Evaluator", test_evaluator),
        ("Failure Analyzer", test_failure_analyzer),
        ("Visualizer", test_visualizer),
        ("Configuration", test_configuration),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("="*80)
    print(" "*30 + "TEST SUMMARY")
    print("="*80 + "\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"{status:10} {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n All tests passed! System is ready to use.")
        return 0
    else:
        print(f"\n {total - passed} test(s) failed. Please check the errors above.")
        return 1


# if __name__ == "__main__":
#     sys.exit(main())
