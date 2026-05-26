"""
Evaluator Module
Implements hard correctness checking with output normalization.
"""

import re
from typing import Any, Tuple, Dict
from dataclasses import dataclass
from llm_interface import LLMResponse


@dataclass
class EvaluationResult:
    """Result of evaluating an LLM response."""
    is_correct: bool
    llm_answer: Any
    ground_truth: Any
    normalized_llm_answer: Any
    normalized_ground_truth: Any
    error_type: str = None
    confidence: float = 1.0  # Confidence in the evaluation


class Evaluator:
    """Evaluates LLM responses against ground truth."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize evaluator.
        
        Args:
            strict_mode: If True, requires exact match without normalization
        """
        self.strict_mode = strict_mode
    
    def evaluate(self, llm_response: LLMResponse, ground_truth: Any) -> EvaluationResult:
        """
        Evaluate LLM response against ground truth.
        
        Args:
            llm_response: Response from LLM
            ground_truth: Expected answer
            
        Returns:
            EvaluationResult
        """
        llm_answer = llm_response.extracted_answer
        
        if self.strict_mode:
            # Strict comparison - no normalization
            is_correct = llm_answer == ground_truth
            return EvaluationResult(
                is_correct=is_correct,
                llm_answer=llm_answer,
                ground_truth=ground_truth,
                normalized_llm_answer=llm_answer,
                normalized_ground_truth=ground_truth,
                error_type=None if is_correct else "mismatch"
            )
        
        # Normalize both answers
        norm_llm = self._normalize_answer(llm_answer)
        norm_truth = self._normalize_answer(ground_truth)
        
        # Compare normalized answers
        is_correct = self._compare_normalized(norm_llm, norm_truth)
        
        # Determine error type if incorrect
        error_type = None
        if not is_correct:
            error_type = self._classify_error(llm_answer, ground_truth, norm_llm, norm_truth)
        
        return EvaluationResult(
            is_correct=is_correct,
            llm_answer=llm_answer,
            ground_truth=ground_truth,
            normalized_llm_answer=norm_llm,
            normalized_ground_truth=norm_truth,
            error_type=error_type
        )
    
    def _normalize_answer(self, answer: Any) -> Any:
        """
        Normalize answer for comparison.
        
        Handles:
        - Whitespace
        - Case sensitivity
        - List ordering (for unordered sets)
        - Number formatting
        - String formatting
        """
        if answer is None:
            return None
        
        # Handle numbers
        if isinstance(answer, (int, float)):
            return answer
        
        # Handle lists
        if isinstance(answer, (list, tuple)):
            # Sort lists for unordered comparison
            try:
                return sorted([self._normalize_answer(item) for item in answer])
            except TypeError:
                # Can't sort - return as is
                return [self._normalize_answer(item) for item in answer]
        
        # Handle strings
        if isinstance(answer, str):
            # Remove extra whitespace
            answer = ' '.join(answer.split())
            
            # Try to parse as number
            number_match = re.search(r'^-?\d+(?:\.\d+)?$', answer.strip())
            if number_match:
                try:
                    if '.' in answer:
                        return float(answer)
                    else:
                        return int(answer)
                except ValueError:
                    pass
            
            # Try to parse as list
            list_match = re.match(r'^\[(.*)\]$', answer.strip())
            if list_match:
                try:
                    items = [x.strip() for x in list_match.group(1).split(',')]
                    numbers = [int(x) if '.' not in x else float(x) for x in items if x]
                    return sorted(numbers)
                except ValueError:
                    pass
            
            # Comma-separated numbers
            if re.match(r'^[\d,\s]+$', answer):
                try:
                    numbers = [int(x.strip()) for x in answer.split(',') if x.strip()]
                    return sorted(numbers)
                except ValueError:
                    pass
            
            # Return lowercase string for case-insensitive comparison
            return answer.lower().strip()
        
        return answer
    
    def _compare_normalized(self, norm_llm: Any, norm_truth: Any) -> bool:
        """Compare normalized answers."""
        # Direct equality check
        if norm_llm == norm_truth:
            return True
        
        # Handle lists with tolerance for floating point
        if isinstance(norm_llm, list) and isinstance(norm_truth, list):
            if len(norm_llm) != len(norm_truth):
                return False
            
            for llm_item, truth_item in zip(norm_llm, norm_truth):
                if isinstance(llm_item, float) and isinstance(truth_item, float):
                    if not self._float_equal(llm_item, truth_item):
                        return False
                elif llm_item != truth_item:
                    return False
            return True
        
        # Handle floating point comparison
        if isinstance(norm_llm, float) and isinstance(norm_truth, float):
            return self._float_equal(norm_llm, norm_truth)
        
        # Handle int/float comparison
        if isinstance(norm_llm, (int, float)) and isinstance(norm_truth, (int, float)):
            return self._float_equal(float(norm_llm), float(norm_truth))
        
        return False
    
    def _float_equal(self, a: float, b: float, tolerance: float = 1e-6) -> bool:
        """Compare floats with tolerance."""
        return abs(a - b) < tolerance
    
    def _classify_error(self, llm_answer: Any, ground_truth: Any,
                       norm_llm: Any, norm_truth: Any) -> str:
        """
        Classify the type of error.
        
        Returns:
            Error type string
        """
        # Type mismatch
        if type(norm_llm) != type(norm_truth):
            return "type_mismatch"
        
        # For numbers, check if it's close
        if isinstance(norm_llm, (int, float)) and isinstance(norm_truth, (int, float)):
            diff = abs(norm_llm - norm_truth)
            if diff <= 10:
                return "arithmetic_error_small"
            elif diff <= 100:
                return "arithmetic_error_medium"
            else:
                return "arithmetic_error_large"
        
        # For lists, check difference
        if isinstance(norm_llm, list) and isinstance(norm_truth, list):
            llm_set = set(norm_llm) if norm_llm else set()
            truth_set = set(norm_truth) if norm_truth else set()
            
            missing = truth_set - llm_set
            extra = llm_set - truth_set
            
            if missing and extra:
                return "list_partial_overlap"
            elif missing:
                return "list_incomplete"
            elif extra:
                return "list_extra_elements"
            else:
                return "list_order_error"
        
        # For strings
        if isinstance(norm_llm, str) and isinstance(norm_truth, str):
            if norm_llm in norm_truth or norm_truth in norm_llm:
                return "string_partial_match"
            else:
                return "string_mismatch"
        
        return "unknown_error"
    
    def batch_evaluate(self, llm_responses: list, ground_truths: list) -> list:
        """
        Evaluate a batch of responses.
        
        Args:
            llm_responses: List of LLMResponse objects
            ground_truths: List of ground truth answers
            
        Returns:
            List of EvaluationResult objects
        """
        if len(llm_responses) != len(ground_truths):
            raise ValueError("Number of responses and ground truths must match")
        
        return [
            self.evaluate(response, truth)
            for response, truth in zip(llm_responses, ground_truths)
        ]
    
    def calculate_accuracy(self, evaluation_results: list) -> Dict[str, float]:
        """
        Calculate accuracy metrics from evaluation results.
        
        Args:
            evaluation_results: List of EvaluationResult objects
            
        Returns:
            Dictionary with accuracy metrics
        """
        if not evaluation_results:
            return {"accuracy": 0.0, "total": 0, "correct": 0, "incorrect": 0}
        
        correct = sum(1 for result in evaluation_results if result.is_correct)
        total = len(evaluation_results)
        
        # Count error types
        error_counts = {}
        for result in evaluation_results:
            if result.error_type:
                error_counts[result.error_type] = error_counts.get(result.error_type, 0) + 1
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "error_distribution": error_counts
        }


# if __name__ == "__main__":
#     # Test evaluator
#     from llm_interface import LLMResponse
#     from datetime import datetime
    
#     print("=== Testing Evaluator ===\n")
    
#     evaluator = Evaluator(strict_mode=False)
    
#     # Test cases
#     test_cases = [
#         # (llm_answer, ground_truth, expected_result)
#         (42, 42, True, "exact_match"),
#         ("42", 42, True, "string_to_int"),
#         ([1, 2, 3], [3, 2, 1], True, "list_order"),
#         ([2, 3, 5, 7], [2, 3, 5, 7, 11], False, "list_incomplete"),
#         (100, 42, False, "arithmetic_error"),
#         ("winning", "winning", True, "string_match"),
#         ("Winning", "winning", True, "case_insensitive"),
#         ("2, 3, 5, 7", [2, 3, 5, 7], True, "comma_separated"),
#         ("[1, 2, 3]", [1, 2, 3], True, "string_list"),
#     ]
    
#     for llm_ans, truth, expected, description in test_cases:
#         # Create mock LLMResponse
#         response = LLMResponse(
#             raw_response=str(llm_ans),
#             extracted_answer=llm_ans,
#             provider="test",
#             model="test",
#             prompt="test",
#             timestamp=datetime.now().isoformat(),
#             latency=0.0
#         )
        
#         result = evaluator.evaluate(response, truth)
        
#         status = "✓" if result.is_correct == expected else "✗"
#         print(f"{status} {description}:")
#         print(f"  LLM: {llm_ans} -> Ground Truth: {truth}")
#         print(f"  Normalized LLM: {result.normalized_llm_answer}")
#         print(f"  Normalized Truth: {result.normalized_ground_truth}")
#         print(f"  Correct: {result.is_correct}, Error Type: {result.error_type}\n")
    
#     # Test batch evaluation
#     print("\n=== Batch Evaluation Test ===")
#     responses = [
#         LLMResponse("42", 42, "test", "test", "test", datetime.now().isoformat(), 0.0),
#         LLMResponse("100", 42, "test", "test", "test", datetime.now().isoformat(), 0.0),
#         LLMResponse("[2, 3, 5]", [2, 3, 5], "test", "test", "test", datetime.now().isoformat(), 0.0),
#     ]
#     truths = [42, 42, [2, 3, 5]]
    
#     results = evaluator.batch_evaluate(responses, truths)
#     metrics = evaluator.calculate_accuracy(results)
    
#     print(f"Accuracy: {metrics['accuracy']:.2%}")
#     print(f"Correct: {metrics['correct']}/{metrics['total']}")
#     print(f"Error Distribution: {metrics['error_distribution']}")
