"""
Search Optimizer Module
Implements multiple search strategies to find LLM failure cases.
"""

import random
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from problem_generator import ProblemGenerator, Problem
from mutation_engine import MutationEngine, MutatedProblem
from llm_interface import LLMInterface, LLMResponse
from evaluator import Evaluator, EvaluationResult


@dataclass
class SearchResult:
    """Result from a single search evaluation."""
    problem: Problem
    mutated_problem: MutatedProblem
    llm_response: LLMResponse
    evaluation: EvaluationResult
    iteration: int
    strategy: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "iteration": self.iteration,
            "strategy": self.strategy,
            "timestamp": self.timestamp,
            "template_id": self.problem.template_id,
            "domain": self.problem.domain,
            "difficulty": self.problem.difficulty,
            "mutations": [m.mutation_type for m in self.mutated_problem.mutations],
            "mutation_count": self.mutated_problem.mutation_count,
            "prompt": self.mutated_problem.mutated_prompt,
            "ground_truth": str(self.problem.ground_truth),
            "llm_answer": str(self.llm_response.extracted_answer),
            "is_correct": self.evaluation.is_correct,
            "error_type": self.evaluation.error_type,
            "latency": self.llm_response.latency,
            "tokens": self.llm_response.tokens_used
        }


class SearchStrategy:
    """Base class for search strategies."""
    
    def __init__(self, problem_generator: ProblemGenerator,
                 mutation_engine: MutationEngine,
                 llm_interface: LLMInterface,
                 evaluator: Evaluator,
                 seed: int = 42,
                 log_dir: str = "logs"):
        self.problem_generator = problem_generator
        self.mutation_engine = mutation_engine
        self.llm_interface = llm_interface
        self.evaluator = evaluator
        self.seed = seed
        self.rng = random.Random(seed)
        self.log_dir = log_dir
        self.results = []
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def search(self, max_iterations: int, batch_size: int = 1) -> List[SearchResult]:
        """Execute search strategy."""
        raise NotImplementedError
    
    def _evaluate_problem(self, problem: Problem, mutated: MutatedProblem, 
                         iteration: int) -> SearchResult:
        """Evaluate a single problem."""
        # Query LLM
        llm_response = self.llm_interface.query(mutated.mutated_prompt)
        
        # Evaluate response
        evaluation = self.evaluator.evaluate(llm_response, problem.ground_truth)
        
        # Create result
        result = SearchResult(
            problem=problem,
            mutated_problem=mutated,
            llm_response=llm_response,
            evaluation=evaluation,
            iteration=iteration,
            strategy=self.__class__.__name__
        )
        
        self.results.append(result)
        
        # Log result
        self._log_result(result)
        
        return result
    
    def _log_result(self, result: SearchResult):
        """Log search result."""
        log_file = f"{self.log_dir}/{self.llm_interface.provider}_{self.__class__.__name__}_results.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')
    
    def get_failure_rate(self) -> float:
        """Calculate failure rate."""
        if not self.results:
            return 0.0
        failures = sum(1 for r in self.results if not r.evaluation.is_correct)
        return failures / len(self.results)
    
    def get_failures(self) -> List[SearchResult]:
        """Get all failure cases."""
        return [r for r in self.results if not r.evaluation.is_correct]


class RandomSearchStrategy(SearchStrategy):
    """Random search with failure region exploitation."""
    
    def __init__(self, *args, exploitation_rate: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploitation_rate = exploitation_rate
        self.failure_templates = set()
        self.failure_mutations = []
        
    def search(self, max_iterations: int, batch_size: int = 1) -> List[SearchResult]:
        """
        Random search with adaptive exploitation of failure regions.
        
        Args:
            max_iterations: Maximum number of iterations
            batch_size: Number of problems to evaluate per iteration
            
        Returns:
            List of SearchResult objects
        """
        self.logger.info(f"Starting random search for {max_iterations} iterations")
        
        for iteration in range(max_iterations):
            # Decide whether to exploit or explore
            exploit = self.rng.random() < self.exploitation_rate and self.failure_templates
            
            for _ in range(batch_size):
                if exploit and self.failure_templates:
                    # Exploit: generate from failure template
                    template_id = self.rng.choice(list(self.failure_templates))
                    problem = self.problem_generator.generate_problem(template_id=template_id)
                    
                    # Use similar mutations that caused failures
                    if self.failure_mutations:
                        mutation_types = self.rng.choice(self.failure_mutations)
                        mutated = self.mutation_engine.mutate(
                            problem, 
                            mutation_types=mutation_types,
                            num_mutations=len(mutation_types)
                        )
                    else:
                        mutated = self.mutation_engine.apply_random_mutations(problem)
                else:
                    # Explore: random generation
                    problem = self.problem_generator.generate_problem()
                    mutated = self.mutation_engine.apply_random_mutations(problem)
                
                if mutated is None:
                    continue
                
                # Evaluate
                result = self._evaluate_problem(problem, mutated, iteration)
                
                # Update failure tracking
                if not result.evaluation.is_correct:
                    self.failure_templates.add(problem.template_id)
                    mutation_types = [m.mutation_type for m in mutated.mutations]
                    self.failure_mutations.append(mutation_types)
                    
                    self.logger.info(f"Iteration {iteration}: Found failure (template: {problem.template_id})")
                else:
                    self.logger.debug(f"Iteration {iteration}: Success")
        
        self.logger.info(f"Random search complete. Failure rate: {self.get_failure_rate():.2%}")
        return self.results


class EvolutionaryStrategy(SearchStrategy):
    """Evolutionary/genetic algorithm for finding failures."""
    
    def __init__(self, *args, population_size: int = 20,
                 mutation_rate: float = 0.3, crossover_rate: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        
    def search(self, max_iterations: int, batch_size: int = 1) -> List[SearchResult]:
        """
        Evolutionary search using genetic algorithm principles.
        
        Args:
            max_iterations: Number of generations
            batch_size: Number of offspring per generation
            
        Returns:
            List of SearchResult objects
        """
        self.logger.info(f"Starting evolutionary search for {max_iterations} generations")
        
        # Initialize population
        self._initialize_population()
        
        for generation in range(max_iterations):
            # Evaluate population
            generation_results = []
            
            for individual in self.population[:batch_size]:
                problem, mutated = individual
                result = self._evaluate_problem(problem, mutated, generation)
                generation_results.append((individual, result))
            
            # Selection: keep individuals that cause failures
            self.population = self._selection(generation_results)
            
            # Generate offspring through crossover and mutation
            offspring = self._generate_offspring()
            
            # Add offspring to population
            self.population.extend(offspring)
            
            # Limit population size
            self.population = self.population[:self.population_size]
            
            failure_count = sum(1 for _, r in generation_results if not r.evaluation.is_correct)
            self.logger.info(f"Generation {generation}: {failure_count}/{len(generation_results)} failures")
        
        self.logger.info(f"Evolutionary search complete. Failure rate: {self.get_failure_rate():.2%}")
        return self.results
    
    def _initialize_population(self):
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            problem = self.problem_generator.generate_problem()
            mutated = self.mutation_engine.apply_random_mutations(problem)
            if mutated:
                self.population.append((problem, mutated))
    
    def _selection(self, generation_results: List[Tuple[Any, SearchResult]]) -> List[Tuple[Problem, MutatedProblem]]:
        """Select individuals for next generation (favor failures)."""
        # Sort by fitness (failures have higher fitness)
        sorted_results = sorted(
            generation_results,
            key=lambda x: (not x[1].evaluation.is_correct, x[1].problem.difficulty),
            reverse=True
        )
        
        # Keep top 50%
        keep_count = max(len(sorted_results) // 2, 1)
        return [individual for individual, _ in sorted_results[:keep_count]]
    
    def _generate_offspring(self) -> List[Tuple[Problem, MutatedProblem]]:
        """Generate offspring through crossover and mutation."""
        offspring = []
        
        while len(offspring) < self.population_size // 2:
            if self.rng.random() < self.crossover_rate and len(self.population) >= 2:
                # Crossover
                parent1, parent2 = self.rng.sample(self.population, 2)
                child = self._crossover(parent1, parent2)
                if child:
                    offspring.append(child)
            else:
                # Mutation
                if self.population:
                    parent = self.rng.choice(self.population)
                    child = self._mutate_individual(parent)
                    if child:
                        offspring.append(child)
        
        return offspring
    
    def _crossover(self, parent1: Tuple[Problem, MutatedProblem], 
                   parent2: Tuple[Problem, MutatedProblem]) -> Optional[Tuple[Problem, MutatedProblem]]:
        """Crossover two individuals."""
        prob1, mut1 = parent1
        prob2, mut2 = parent2
        
        # Use template from one parent
        template_id = self.rng.choice([prob1.template_id, prob2.template_id])
        
        # Generate new problem from that template
        new_problem = self.problem_generator.generate_problem(template_id=template_id)
        
        # Combine mutation types from both parents
        mutations1 = [m.mutation_type for m in mut1.mutations]
        mutations2 = [m.mutation_type for m in mut2.mutations]
        
        # Take mutations from both parents
        combined_mutations = mutations1[:len(mutations1)//2] + mutations2[len(mutations2)//2:]
        
        if combined_mutations:
            new_mutated = self.mutation_engine.mutate(
                new_problem,
                mutation_types=combined_mutations,
                num_mutations=len(combined_mutations)
            )
            if new_mutated:
                return (new_problem, new_mutated)
        
        return None
    
    def _mutate_individual(self, individual: Tuple[Problem, MutatedProblem]) -> Optional[Tuple[Problem, MutatedProblem]]:
        """Mutate an individual."""
        problem, mutated = individual
        
        # Generate new problem from same template
        new_problem = self.problem_generator.generate_problem(template_id=problem.template_id)
        
        # Apply mutations with some randomness
        if self.rng.random() < self.mutation_rate:
            # High mutation: different mutations
            new_mutated = self.mutation_engine.apply_random_mutations(new_problem)
        else:
            # Low mutation: similar mutations
            mutation_types = [m.mutation_type for m in mutated.mutations]
            new_mutated = self.mutation_engine.mutate(
                new_problem,
                mutation_types=mutation_types,
                num_mutations=len(mutation_types)
            )
        
        if new_mutated:
            return (new_problem, new_mutated)
        
        return None


class HillClimbingStrategy(SearchStrategy):
    """Hill climbing search for finding failures."""
    
    def __init__(self, *args, neighbors_per_step: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.neighbors_per_step = neighbors_per_step
        
    def search(self, max_iterations: int, batch_size: int = 1) -> List[SearchResult]:
        """
        Hill climbing search that moves toward failure regions.
        
        Args:
            max_iterations: Maximum number of iterations
            batch_size: Number of neighbors to explore per iteration
            
        Returns:
            List of SearchResult objects
        """
        self.logger.info(f"Starting hill climbing search for {max_iterations} iterations")
        
        # Start with random problem
        current_problem = self.problem_generator.generate_problem()
        current_mutated = self.mutation_engine.apply_random_mutations(current_problem)
        
        if not current_mutated:
            self.logger.error("Failed to generate initial problem")
            return []
        
        # Evaluate initial problem
        current_result = self._evaluate_problem(current_problem, current_mutated, 0)
        current_score = self._score_result(current_result)
        
        for iteration in range(1, max_iterations):
            # Generate neighbors
            neighbors = self._generate_neighbors(current_problem, current_mutated, batch_size)
            
            # Evaluate neighbors
            best_neighbor = None
            best_score = current_score
            best_result = None
            
            for neighbor_problem, neighbor_mutated in neighbors:
                result = self._evaluate_problem(neighbor_problem, neighbor_mutated, iteration)
                score = self._score_result(result)
                
                if score > best_score:
                    best_score = score
                    best_neighbor = (neighbor_problem, neighbor_mutated)
                    best_result = result
            
            # Move to better neighbor if found
            if best_neighbor:
                current_problem, current_mutated = best_neighbor
                current_score = best_score
                current_result = best_result
                self.logger.info(f"Iteration {iteration}: Moved to better neighbor (score: {current_score:.2f})")
            else:
                # Random restart
                self.logger.info(f"Iteration {iteration}: No improvement, random restart")
                current_problem = self.problem_generator.generate_problem()
                current_mutated = self.mutation_engine.apply_random_mutations(current_problem)
                if current_mutated:
                    current_result = self._evaluate_problem(current_problem, current_mutated, iteration)
                    current_score = self._score_result(current_result)
        
        self.logger.info(f"Hill climbing complete. Failure rate: {self.get_failure_rate():.2%}")
        return self.results
    
    def _score_result(self, result: SearchResult) -> float:
        """
        Score a result (higher is better for finding failures).
        
        We want to maximize failures and difficulty.
        """
        score = 0.0
        
        # Failure is good
        if not result.evaluation.is_correct:
            score += 10.0
        
        # Higher difficulty is good
        score += result.problem.difficulty
        
        # More mutations is good
        score += result.mutated_problem.mutation_count * 0.5
        
        return score
    
    def _generate_neighbors(self, problem: Problem, mutated: MutatedProblem, 
                           n: int) -> List[Tuple[Problem, MutatedProblem]]:
        """Generate neighbor solutions."""
        neighbors = []
        
        for _ in range(n):
            # Strategy 1: Same template, different parameters
            neighbor_problem = self.problem_generator.generate_problem(template_id=problem.template_id)
            
            # Strategy 2: Similar mutations
            if self.rng.random() < 0.5:
                # Same mutation types
                mutation_types = [m.mutation_type for m in mutated.mutations]
                neighbor_mutated = self.mutation_engine.mutate(
                    neighbor_problem,
                    mutation_types=mutation_types,
                    num_mutations=len(mutation_types)
                )
            else:
                # Add one more mutation
                mutation_types = [m.mutation_type for m in mutated.mutations]
                mutation_types.append(self.rng.choice(["constraint", "instruction", "adversarial_trap"]))
                neighbor_mutated = self.mutation_engine.mutate(
                    neighbor_problem,
                    mutation_types=mutation_types,
                    num_mutations=len(mutation_types)
                )
            
            if neighbor_mutated:
                neighbors.append((neighbor_problem, neighbor_mutated))
        
        return neighbors


class SearchOptimizer:
    """Main search optimizer that coordinates different strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.problem_generator = ProblemGenerator(seed=config.get("seed", 42))
        self.mutation_engine = MutationEngine(seed=config.get("seed", 42))
        self.evaluator = Evaluator()
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, config.get("log_level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def create_strategy(self, strategy_name: str, llm_interface: LLMInterface) -> SearchStrategy:
        """Create a search strategy instance."""
        if strategy_name == "random":
            return RandomSearchStrategy(
                self.problem_generator,
                self.mutation_engine,
                llm_interface,
                self.evaluator,
                seed=self.config.get("seed", 42),
                log_dir=self.config.get("log_dir", "logs")
            )
        elif strategy_name == "evolutionary":
            return EvolutionaryStrategy(
                self.problem_generator,
                self.mutation_engine,
                llm_interface,
                self.evaluator,
                seed=self.config.get("seed", 42),
                population_size=self.config.get("population_size", 20),
                mutation_rate=self.config.get("mutation_rate", 0.3),
                crossover_rate=self.config.get("crossover_rate", 0.7),
                log_dir=self.config.get("log_dir", "logs")
            )
        elif strategy_name == "hill_climbing":
            return HillClimbingStrategy(
                self.problem_generator,
                self.mutation_engine,
                llm_interface,
                self.evaluator,
                seed=self.config.get("seed", 42),
                log_dir=self.config.get("log_dir", "logs")
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")


# if __name__ == "__main__":
#     # Test search optimizer
#     import os
    
#     print("=== Testing Search Optimizer ===\n")
    
#     # Mock configuration
#     config = {
#         "seed": 42,
#         "log_level": "INFO",
#         "log_dir": "logs",
#         "population_size": 10,
#         "mutation_rate": 0.3,
#         "crossover_rate": 0.7
#     }
    
#     optimizer = SearchOptimizer(config)
    
#     # Test with mock LLM (just for structure testing)
#     if os.getenv("OPENAI_API_KEY"):
#         print("Testing with OpenAI (limited iterations)...")
#         llm = LLMInterface(provider="openai", model="gpt-3.5-turbo", log_dir="logs")
        
#         # Test random strategy
#         strategy = optimizer.create_strategy("random", llm)
#         results = strategy.search(max_iterations=2, batch_size=1)
        
#         print(f"\nRandom Strategy Results:")
#         print(f"  Total evaluations: {len(results)}")
#         print(f"  Failure rate: {strategy.get_failure_rate():.2%}")
#         print(f"  Failures found: {len(strategy.get_failures())}")
#     else:
#         print("OPENAI_API_KEY not set. Skipping live tests.")
#         print("Strategy classes initialized successfully.")
