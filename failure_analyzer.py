"""
Failure Analyzer Module
Analyzes LLM failures and categorizes them by type and pattern.
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from evaluator import EvaluationResult
from llm_interface import LLMResponse
from problem_generator import Problem
from mutation_engine import MutatedProblem


@dataclass
class FailureCase:
    """Represents a single failure case."""
    problem: Problem
    mutated_problem: MutatedProblem
    llm_response: LLMResponse
    evaluation: EvaluationResult
    failure_category: str
    failure_patterns: List[str] = field(default_factory=list)
    severity: int = 1  # 1-5 scale


@dataclass
class FailureAnalysis:
    """Analysis results for a set of failures."""
    total_failures: int
    failure_by_category: Dict[str, int]
    failure_by_mutation: Dict[str, int]
    failure_by_domain: Dict[str, int]
    failure_by_difficulty: Dict[int, int]
    common_patterns: List[Tuple[str, int]]
    severity_distribution: Dict[int, int]
    

class FailureAnalyzer:
    """Analyzes LLM failures to identify patterns and weaknesses."""
    
    def __init__(self):
        self.failure_cases = []
        self.categories = [
            "arithmetic_error",
            "logical_error",
            "parsing_error",
            "instruction_following",
            "edge_case_handling",
            "trap_susceptibility",
            "constraint_violation",
            "incomplete_solution"
        ]
    
    def analyze_failure(self, problem: Problem, mutated_problem: MutatedProblem,
                       llm_response: LLMResponse, evaluation: EvaluationResult) -> FailureCase:
        """
        Analyze a single failure case.
        
        Args:
            problem: Original problem
            mutated_problem: Mutated problem
            llm_response: LLM response
            evaluation: Evaluation result
            
        Returns:
            FailureCase object
        """
        # Categorize the failure
        category = self._categorize_failure(
            problem, mutated_problem, llm_response, evaluation
        )
        
        # Identify patterns
        patterns = self._identify_patterns(
            problem, mutated_problem, llm_response, evaluation
        )
        
        # Determine severity
        severity = self._assess_severity(evaluation, mutated_problem)
        
        failure_case = FailureCase(
            problem=problem,
            mutated_problem=mutated_problem,
            llm_response=llm_response,
            evaluation=evaluation,
            failure_category=category,
            failure_patterns=patterns,
            severity=severity
        )
        
        self.failure_cases.append(failure_case)
        
        return failure_case
    
    def _categorize_failure(self, problem: Problem, mutated_problem: MutatedProblem,
                           llm_response: LLMResponse, evaluation: EvaluationResult) -> str:
        """Categorize the type of failure."""
        error_type = evaluation.error_type
        
        # Arithmetic errors
        if error_type and "arithmetic" in error_type:
            return "arithmetic_error"
        
        # Parsing errors
        if error_type in ["type_mismatch"]:
            return "parsing_error"
        
        # List handling errors
        if error_type and "list" in error_type:
            # Check if it's due to mutation
            mutation_types = [m.mutation_type for m in mutated_problem.mutations]
            if "adversarial_trap" in mutation_types:
                return "trap_susceptibility"
            return "logical_error"
        
        # Check for instruction following issues
        if self._is_instruction_error(llm_response, mutated_problem):
            return "instruction_following"
        
        # Check for constraint violations
        if self._is_constraint_violation(evaluation, mutated_problem):
            return "constraint_violation"
        
        # Check for edge case handling
        if self._is_edge_case_error(problem, evaluation):
            return "edge_case_handling"
        
        # Check for trap susceptibility
        mutation_types = [m.mutation_type for m in mutated_problem.mutations]
        if "adversarial_trap" in mutation_types:
            return "trap_susceptibility"
        
        # Default to logical error
        return "logical_error"
    
    def _is_instruction_error(self, llm_response: LLMResponse, 
                             mutated_problem: MutatedProblem) -> bool:
        """Check if failure is due to not following instructions."""
        raw_response = llm_response.raw_response.lower()
        
        # Check if response contains excessive explanation when only answer was requested
        if "provide only" in mutated_problem.mutated_prompt.lower():
            # Response should be concise
            if len(raw_response.split()) > 20:
                return True
        
        # Check if response format doesn't match requirements
        if "comma-separated" in mutated_problem.mutated_prompt.lower():
            if "," not in str(llm_response.extracted_answer):
                return True
        
        return False
    
    def _is_constraint_violation(self, evaluation: EvaluationResult,
                                mutated_problem: MutatedProblem) -> bool:
        """Check if failure violates explicit constraints."""
        mutation_types = [m.mutation_type for m in mutated_problem.mutations]
        
        if "constraint" in mutation_types:
            # Check if the error is related to constraint handling
            if evaluation.error_type in ["list_extra_elements", "list_incomplete"]:
                return True
        
        return False
    
    def _is_edge_case_error(self, problem: Problem, evaluation: EvaluationResult) -> bool:
        """Check if failure is due to edge case mishandling."""
        # Check for boundary errors
        if isinstance(evaluation.ground_truth, list):
            if len(evaluation.ground_truth) == 0:
                return True  # Empty list edge case
        
        # Check for small/large number edge cases
        if isinstance(evaluation.ground_truth, int):
            if evaluation.ground_truth == 0 or evaluation.ground_truth == 1:
                return True
        
        return False
    
    def _identify_patterns(self, problem: Problem, mutated_problem: MutatedProblem,
                          llm_response: LLMResponse, evaluation: EvaluationResult) -> List[str]:
        """Identify specific failure patterns."""
        patterns = []
        
        # Off-by-one errors
        if isinstance(evaluation.normalized_llm_answer, int) and \
           isinstance(evaluation.normalized_ground_truth, int):
            diff = abs(evaluation.normalized_llm_answer - evaluation.normalized_ground_truth)
            if diff == 1:
                patterns.append("off_by_one")
        
        # Missing elements pattern
        if evaluation.error_type == "list_incomplete":
            patterns.append("missing_elements")
        
        # Extra elements pattern
        if evaluation.error_type == "list_extra_elements":
            patterns.append("extra_elements")
        
        # Wrong operation pattern (e.g., addition instead of multiplication)
        if isinstance(evaluation.normalized_llm_answer, (int, float)) and \
           isinstance(evaluation.normalized_ground_truth, (int, float)):
            ratio = evaluation.normalized_llm_answer / evaluation.normalized_ground_truth \
                    if evaluation.normalized_ground_truth != 0 else 0
            if 0.4 < ratio < 0.6:
                patterns.append("possible_wrong_operation")
        
        # Case sensitivity pattern
        if isinstance(evaluation.llm_answer, str) and \
           isinstance(evaluation.ground_truth, str):
            if evaluation.llm_answer.lower() == evaluation.ground_truth.lower() and \
               evaluation.llm_answer != evaluation.ground_truth:
                patterns.append("case_sensitivity")
        
        # Trap fall pattern
        mutation_types = [m.mutation_type for m in mutated_problem.mutations]
        if "adversarial_trap" in mutation_types:
            patterns.append("fell_for_trap")
        
        # Mutation-specific patterns
        if "instruction" in mutation_types and \
           evaluation.error_type == "parsing_error":
            patterns.append("ambiguous_instruction_confusion")
        
        return patterns
    
    def _assess_severity(self, evaluation: EvaluationResult, 
                        mutated_problem: MutatedProblem) -> int:
        """Assess severity of failure (1-5 scale)."""
        severity = 1
        
        # Increase severity based on error magnitude
        if evaluation.error_type and "large" in evaluation.error_type:
            severity += 2
        elif evaluation.error_type and "medium" in evaluation.error_type:
            severity += 1
        
        # Increase severity for multiple mutations
        severity += min(len(mutated_problem.mutations) - 1, 2)
        
        # Increase severity for high difficulty problems
        if mutated_problem.original_problem.difficulty >= 4:
            severity += 1
        
        return min(severity, 5)
    
    def generate_analysis(self) -> FailureAnalysis:
        """Generate comprehensive failure analysis."""
        if not self.failure_cases:
            return FailureAnalysis(
                total_failures=0,
                failure_by_category={},
                failure_by_mutation={},
                failure_by_domain={},
                failure_by_difficulty={},
                common_patterns=[],
                severity_distribution={}
            )
        
        # Count failures by category
        failure_by_category = defaultdict(int)
        for case in self.failure_cases:
            failure_by_category[case.failure_category] += 1
        
        # Count failures by mutation type
        failure_by_mutation = defaultdict(int)
        for case in self.failure_cases:
            for mutation in case.mutated_problem.mutations:
                failure_by_mutation[mutation.mutation_type] += 1
        
        # Count failures by domain
        failure_by_domain = defaultdict(int)
        for case in self.failure_cases:
            failure_by_domain[case.problem.domain] += 1
        
        # Count failures by difficulty
        failure_by_difficulty = defaultdict(int)
        for case in self.failure_cases:
            failure_by_difficulty[case.problem.difficulty] += 1
        
        # Count pattern frequencies
        pattern_counts = defaultdict(int)
        for case in self.failure_cases:
            for pattern in case.failure_patterns:
                pattern_counts[pattern] += 1
        
        common_patterns = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Count severity distribution
        severity_distribution = defaultdict(int)
        for case in self.failure_cases:
            severity_distribution[case.severity] += 1
        
        return FailureAnalysis(
            total_failures=len(self.failure_cases),
            failure_by_category=dict(failure_by_category),
            failure_by_mutation=dict(failure_by_mutation),
            failure_by_domain=dict(failure_by_domain),
            failure_by_difficulty=dict(failure_by_difficulty),
            common_patterns=common_patterns,
            severity_distribution=dict(severity_distribution)
        )
    
    def get_failure_cases_by_category(self, category: str) -> List[FailureCase]:
        """Get all failure cases of a specific category."""
        return [case for case in self.failure_cases if case.failure_category == category]
    
    def get_most_severe_failures(self, n: int = 10) -> List[FailureCase]:
        """Get the n most severe failures."""
        return sorted(
            self.failure_cases,
            key=lambda x: x.severity,
            reverse=True
        )[:n]
    
    def compare_reasoning(self, llm_response: LLMResponse, ground_truth_reasoning: str = None) -> Dict[str, Any]:
        """
        Compare LLM reasoning with expected reasoning.
        
        Args:
            llm_response: LLM response
            ground_truth_reasoning: Expected reasoning (optional)
            
        Returns:
            Comparison analysis
        """
        response_text = llm_response.raw_response.lower()
        
        # Extract reasoning indicators
        has_step_by_step = bool(re.search(r'(step \d|first|second|then|finally)', response_text))
        has_calculation = bool(re.search(r'(\d+\s*[+\-*/]\s*\d+|calculate|compute)', response_text))
        has_explanation = len(response_text.split()) > 20
        
        return {
            "has_step_by_step": has_step_by_step,
            "has_calculation": has_calculation,
            "has_explanation": has_explanation,
            "response_length": len(response_text.split())
        }
    
    def reset(self):
        """Reset the analyzer."""
        self.failure_cases = []


# if __name__ == "__main__":
#     # Test failure analyzer
#     from problem_generator import ProblemGenerator
#     from mutation_engine import MutationEngine
#     from llm_interface import LLMResponse
#     from evaluator import Evaluator, EvaluationResult
#     from datetime import datetime
    
#     print("=== Testing Failure Analyzer ===\n")
    
#     analyzer = FailureAnalyzer()
#     generator = ProblemGenerator(seed=42)
#     mutator = MutationEngine(seed=42)
#     evaluator = Evaluator()
    
#     # Generate some test failures
#     for i in range(5):
#         problem = generator.generate_problem()
#         mutated = mutator.apply_random_mutations(problem, max_mutations=2)
        
#         if mutated:
#             # Simulate wrong answer
#             wrong_answer = 999 if isinstance(problem.ground_truth, int) else "wrong"
            
#             llm_response = LLMResponse(
#                 raw_response=f"The answer is {wrong_answer}",
#                 extracted_answer=wrong_answer,
#                 provider="test",
#                 model="test",
#                 prompt=mutated.mutated_prompt,
#                 timestamp=datetime.now().isoformat(),
#                 latency=0.5
#             )
            
#             evaluation = evaluator.evaluate(llm_response, problem.ground_truth)
            
#             if not evaluation.is_correct:
#                 failure_case = analyzer.analyze_failure(
#                     problem, mutated, llm_response, evaluation
#                 )
#                 print(f"Failure {i+1}:")
#                 print(f"  Category: {failure_case.failure_category}")
#                 print(f"  Patterns: {failure_case.failure_patterns}")
#                 print(f"  Severity: {failure_case.severity}")
#                 print(f"  Mutations: {[m.mutation_type for m in mutated.mutations]}\n")
    
#     # Generate analysis
#     print("\n=== Failure Analysis ===")
#     analysis = analyzer.generate_analysis()
    
#     print(f"Total Failures: {analysis.total_failures}")
#     print(f"\nBy Category: {analysis.failure_by_category}")
#     print(f"\nBy Mutation: {analysis.failure_by_mutation}")
#     print(f"\nBy Domain: {analysis.failure_by_domain}")
#     print(f"\nCommon Patterns: {analysis.common_patterns}")
#     print(f"\nSeverity Distribution: {analysis.severity_distribution}")
