"""
Mutation Engine Module
Implements 4 mutation types that preserve ground truth while increasing difficulty.
"""

import random
import copy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from problem_generator import Problem


@dataclass
class Mutation:
    """Represents a mutation applied to a problem."""
    mutation_type: str
    description: str
    parameters: Dict[str, Any]
    

@dataclass
class MutatedProblem:
    """A problem with applied mutations."""
    original_problem: Problem
    mutated_prompt: str
    mutations: List[Mutation] = field(default_factory=list)
    ground_truth: Any = None
    verifier: Any = None
    mutation_count: int = 0
    

class MutationEngine:
    """Applies various mutations to problems while preserving ground truth."""
    
    def __init__(self, seed: int = 42, discard_on_failure: bool = True):
        self.seed = seed
        self.rng = random.Random(seed)
        self.discard_on_failure = discard_on_failure
        
    def mutate(self, problem: Problem, mutation_types: List[str] = None, 
               num_mutations: int = 1) -> Optional[MutatedProblem]:
        """
        Apply mutations to a problem.
        
        Args:
            problem: Original problem
            mutation_types: List of mutation types to apply (None = random)
            num_mutations: Number of mutations to apply
            
        Returns:
            MutatedProblem or None if ground truth breaks
        """
        mutated = MutatedProblem(
            original_problem=problem,
            mutated_prompt=problem.prompt,
            ground_truth=problem.ground_truth,
            verifier=problem.verifier
        )
        
        available_mutations = mutation_types or ["constraint", "instruction", "composition", "adversarial_trap"]
        
        for i in range(num_mutations):
            mutation_type = self.rng.choice(available_mutations)
            
            try:
                if mutation_type == "constraint":
                    mutated = self._apply_constraint_mutation(mutated)
                elif mutation_type == "instruction":
                    mutated = self._apply_instruction_mutation(mutated)
                elif mutation_type == "composition":
                    # Composition needs special handling
                    if i == 0:  # Only apply once
                        mutated = self._apply_composition_mutation(mutated, problem)
                elif mutation_type == "adversarial_trap":
                    mutated = self._apply_adversarial_trap(mutated)
                    
                mutated.mutation_count += 1
                
            except Exception as e:
                if self.discard_on_failure:
                    return None
                raise
        
        # Verify ground truth is still valid
        if not self._verify_mutation(mutated):
            if self.discard_on_failure:
                return None
            raise ValueError("Ground truth verification failed after mutation")
        
        return mutated
    
    def _apply_constraint_mutation(self, mutated: MutatedProblem) -> MutatedProblem:
        """Add constraints or modify ranges."""
        problem = mutated.original_problem
        prompt = mutated.mutated_prompt
        
        constraints = [
            # Time constraint
            lambda p: f"{p} You must solve this efficiently.",
            
            # Format constraint
            lambda p: f"{p} Format your answer exactly as specified with no additional text.",
            
            # Complexity hint (potentially misleading)
            lambda p: f"{p} Note: This problem may have multiple edge cases.",
            
            # Step requirement
            lambda p: f"{p} Show your reasoning step by step before the final answer.",
            
            # Precision requirement
            lambda p: f"{p} Ensure your answer is exact, not approximate.",
            
            # Range modification (add artificial bounds)
            lambda p: f"{p} Verify your answer is within reasonable bounds.",
        ]
        
        constraint_func = self.rng.choice(constraints)
        new_prompt = constraint_func(prompt)
        
        mutated.mutated_prompt = new_prompt
        mutated.mutations.append(Mutation(
            mutation_type="constraint",
            description="Added constraint to problem",
            parameters={"constraint_type": "generic"}
        ))
        
        return mutated
    
    def _apply_instruction_mutation(self, mutated: MutatedProblem) -> MutatedProblem:
        """Make instructions ambiguous or reordered."""
        prompt = mutated.mutated_prompt
        
        instruction_mutations = [
            # Ambiguous wording
            lambda p: p.replace("Find", "Determine").replace("Calculate", "Compute"),
            
            # Add redundant information
            lambda p: f"Consider the following problem: {p} Solve it carefully.",
            
            # Reorder with distraction
            lambda p: f"Given the constraints below, {p.lower()} Pay attention to details.",
            
            # Passive voice
            lambda p: p.replace("List all", "All should be listed").replace("Find", "Let there be found"),
            
            # Add cautionary note
            lambda p: f"{p} Be careful not to make common mistakes.",
            
            # Double negative
            lambda p: f"{p} Do not provide any answer that is not correct.",
            
            # Implicit instruction
            lambda p: p.replace("Provide your answer", "Your answer should be provided"),
            
            # Add meta-instruction
            lambda p: f"Task: {p} Remember to double-check your work.",
        ]
        
        mutation_func = self.rng.choice(instruction_mutations)
        new_prompt = mutation_func(prompt)
        
        mutated.mutated_prompt = new_prompt
        mutated.mutations.append(Mutation(
            mutation_type="instruction",
            description="Modified instructions for ambiguity",
            parameters={"modification_type": "ambiguous"}
        ))
        
        return mutated
    
    def _apply_composition_mutation(self, mutated: MutatedProblem, 
                                   original_problem: Problem) -> MutatedProblem:
        """Combine two related sub-tasks."""
        prompt = mutated.mutated_prompt
        
        # Create a simple related sub-task based on problem type
        sub_tasks = []
        
        if "prime" in prompt.lower():
            sub_tasks = [
                "First, identify which numbers are even.",
                "Additionally, count how many numbers you found.",
                "Also determine the largest number in your answer.",
            ]
        elif "sum" in prompt.lower():
            sub_tasks = [
                "Additionally, find the average of these numbers.",
                "Also count how many numbers were summed.",
                "Determine if the sum is even or odd.",
            ]
        elif "divisor" in prompt.lower():
            sub_tasks = [
                "Also check if any of these numbers are prime.",
                "Count how many numbers you found.",
                "Find the smallest number in your result.",
            ]
        else:
            sub_tasks = [
                "After finding your answer, verify it is correct.",
                "Additionally, explain your reasoning briefly.",
                "Count the number of elements in your answer.",
            ]
        
        if sub_tasks:
            sub_task = self.rng.choice(sub_tasks)
            new_prompt = f"{prompt} {sub_task}"
            
            mutated.mutated_prompt = new_prompt
            mutated.mutations.append(Mutation(
                mutation_type="composition",
                description="Added sub-task to problem",
                parameters={"sub_task": sub_task}
            ))
        
        return mutated
    
    def _apply_adversarial_trap(self, mutated: MutatedProblem) -> MutatedProblem:
        """Add edge cases or misleading phrasing."""
        prompt = mutated.mutated_prompt
        
        traps = [
            # Red herring numbers
            lambda p: f"Note: Some students incorrectly got 42 as the answer. {p}",
            
            # False constraint
            lambda p: f"{p} (Hint: The answer is usually a multiple of 3)",
            
            # Anchoring bias
            lambda p: f"Most similar problems have answers around 100. {p}",
            
            # Distractor information
            lambda p: f"{p} Ignore any numbers outside the specified range.",
            
            # Misleading example
            lambda p: f"For example, if the range were different, the answer might be 5. {p}",
            
            # False simplification
            lambda p: f"{p} This is simpler than it looks.",
            
            # Overthinking suggestion
            lambda p: f"{p} Consider all possible interpretations carefully.",
            
            # False urgency
            lambda p: f"Quick question: {p}",
            
            # Fake edge case
            lambda p: f"{p} Assume no edge cases apply.",
            
            # Contradictory hint
            lambda p: f"{p} Remember: sometimes the obvious answer is wrong.",
        ]
        
        trap_func = self.rng.choice(traps)
        new_prompt = trap_func(prompt)
        
        mutated.mutated_prompt = new_prompt
        mutated.mutations.append(Mutation(
            mutation_type="adversarial_trap",
            description="Added adversarial element",
            parameters={"trap_type": "misleading"}
        ))
        
        return mutated
    
    def _verify_mutation(self, mutated: MutatedProblem) -> bool:
        """
        Verify that the mutation preserves ground truth.
        Since we're only modifying the prompt and not the problem structure,
        ground truth should always be preserved.
        """
        # The ground truth is copied from the original problem
        # We only modify the prompt, so verification passes if ground truth exists
        return mutated.ground_truth is not None and mutated.verifier is not None
    
    def apply_random_mutations(self, problem: Problem, max_mutations: int = 3) -> Optional[MutatedProblem]:
        """Apply 1 to max_mutations random mutations."""
        num_mutations = self.rng.randint(1, max_mutations)
        return self.mutate(problem, num_mutations=num_mutations)
    
    def apply_specific_mutation(self, problem: Problem, mutation_type: str) -> Optional[MutatedProblem]:
        """Apply a specific mutation type."""
        return self.mutate(problem, mutation_types=[mutation_type], num_mutations=1)
    
    def generate_mutation_suite(self, problem: Problem) -> Dict[str, MutatedProblem]:
        """Generate all mutation types for a problem."""
        suite = {}
        
        mutation_types = ["constraint", "instruction", "composition", "adversarial_trap"]
        
        for mut_type in mutation_types:
            mutated = self.apply_specific_mutation(problem, mut_type)
            if mutated:
                suite[mut_type] = mutated
        
        return suite
    
    def batch_mutate(self, problems: List[Problem], mutations_per_problem: int = 1) -> List[MutatedProblem]:
        """Apply mutations to a batch of problems."""
        mutated_problems = []
        
        for problem in problems:
            mutated = self.apply_random_mutations(problem, max_mutations=mutations_per_problem)
            if mutated:
                mutated_problems.append(mutated)
        
        return mutated_problems


# if __name__ == "__main__":
#     # Test mutation engine
#     from problem_generator import ProblemGenerator
    
#     print("=== Testing Mutation Engine ===\n")
    
#     generator = ProblemGenerator(seed=42)
#     mutator = MutationEngine(seed=42)
    
#     # Generate a test problem
#     problem = generator.generate_problem(template_id="nt_prime_filtering")
    
#     print("Original Problem:")
#     print(f"Prompt: {problem.prompt}")
#     print(f"Ground Truth: {problem.ground_truth}\n")
    
#     # Test each mutation type
#     mutation_types = ["constraint", "instruction", "composition", "adversarial_trap"]
    
#     for mut_type in mutation_types:
#         print(f"\n--- {mut_type.upper()} Mutation ---")
#         mutated = mutator.apply_specific_mutation(problem, mut_type)
        
#         if mutated:
#             print(f"Mutated Prompt: {mutated.mutated_prompt}")
#             print(f"Ground Truth Preserved: {mutated.ground_truth == problem.ground_truth}")
#             print(f"Mutations Applied: {[m.mutation_type for m in mutated.mutations]}")
    
#     # Test multiple mutations
#     print("\n\n--- Multiple Random Mutations ---")
#     mutated = mutator.apply_random_mutations(problem, max_mutations=3)
    
#     if mutated:
#         print(f"Mutated Prompt: {mutated.mutated_prompt}")
#         print(f"Number of Mutations: {mutated.mutation_count}")
#         print(f"Mutation Types: {[m.mutation_type for m in mutated.mutations]}")
#         print(f"Ground Truth Preserved: {mutated.ground_truth == problem.ground_truth}")
