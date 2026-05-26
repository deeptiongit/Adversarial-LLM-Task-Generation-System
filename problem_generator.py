"""
Problem Generator Module
Generates structured problems in Number Theory and Game Theory domains
with programmatic ground truth verification.
"""

import random
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
import math


@dataclass
class Problem:
    """Represents a generated problem with ground truth."""
    domain: str # e.g., "number_theory" or "game_theory"
    template_id: str # e.g., "nt_prime_filtering"
    template_name: str # e.g., "Prime Number Filtering"
    prompt: str # The problem statement to present to the user
    ground_truth: Any # The correct answer(s) for verification
    verifier: Callable[[Any], bool] # Function to verify user answer against ground truth
    parameters: Dict[str, Any] # Parameters used to generate the problem (for reproducibility)
    difficulty: int  # 1-5 scale
    

class ProblemGenerator:
    """Generates structured problems with programmatic verification."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, Dict]:
        """Initialize all problem templates."""
        return {
            **self._number_theory_templates(),
            **self._game_theory_templates()
        }
    
    def _number_theory_templates(self) -> Dict[str, Dict]:
        """Define 10 number theory problem templates."""
        return {
            "nt_prime_filtering": {
                "name": "Prime Number Filtering",
                "difficulty": 2,
                "generator": self._gen_prime_filtering
            },
            "nt_divisor_count": {
                "name": "Divisor Count Problem",
                "difficulty": 3,
                "generator": self._gen_divisor_count
            },
            "nt_sequence_sum": {
                "name": "Sequence Sum with Condition",
                "difficulty": 2,
                "generator": self._gen_sequence_sum
            },
            "nt_modular_arithmetic": {
                "name": "Modular Arithmetic",
                "difficulty": 3,
                "generator": self._gen_modular_arithmetic
            },
            "nt_set_intersection": {
                "name": "Set Intersection with Properties",
                "difficulty": 2,
                "generator": self._gen_set_intersection
            },
            "nt_fibonacci_property": {
                "name": "Fibonacci Sequence Property",
                "difficulty": 3,
                "generator": self._gen_fibonacci_property
            },
            "nt_perfect_numbers": {
                "name": "Perfect Number Search",
                "difficulty": 4,
                "generator": self._gen_perfect_numbers
            },
            "nt_digit_manipulation": {
                "name": "Digit Sum and Product",
                "difficulty": 2,
                "generator": self._gen_digit_manipulation
            },
            "nt_gcd_lcm": {
                "name": "GCD and LCM Problems",
                "difficulty": 3,
                "generator": self._gen_gcd_lcm
            },
            "nt_arithmetic_progression": {
                "name": "Arithmetic Progression",
                "difficulty": 2,
                "generator": self._gen_arithmetic_progression
            }
        }
    
    def _game_theory_templates(self) -> Dict[str, Dict]:
        """Define 10 game theory problem templates."""
        return {
            "gt_nim_game": {
                "name": "Nim Game Strategy",
                "difficulty": 3,
                "generator": self._gen_nim_game
            },
            "gt_coin_game": {
                "name": "Coin Selection Game",
                "difficulty": 3,
                "generator": self._gen_coin_game
            },
            "gt_stone_removal": {
                "name": "Stone Removal Game",
                "difficulty": 3,
                "generator": self._gen_stone_removal
            },
            "gt_prisoner_dilemma": {
                "name": "Prisoner's Dilemma Variant",
                "difficulty": 4,
                "generator": self._gen_prisoner_dilemma
            },
            "gt_auction_strategy": {
                "name": "Auction Bidding Strategy",
                "difficulty": 4,
                "generator": self._gen_auction_strategy
            },
            "gt_resource_allocation": {
                "name": "Resource Allocation Game",
                "difficulty": 3,
                "generator": self._gen_resource_allocation
            },
            "gt_matching_pennies": {
                "name": "Matching Pennies Game",
                "difficulty": 2,
                "generator": self._gen_matching_pennies
            },
            "gt_sequential_choice": {
                "name": "Sequential Choice Game",
                "difficulty": 3,
                "generator": self._gen_sequential_choice
            },
            "gt_voting_strategy": {
                "name": "Strategic Voting Problem",
                "difficulty": 4,
                "generator": self._gen_voting_strategy
            },
            "gt_fair_division": {
                "name": "Fair Division Problem",
                "difficulty": 3,
                "generator": self._gen_fair_division
            }
        }
    
    # Number Theory Problem Generators
    
    def _gen_prime_filtering(self) -> Problem:
        """Generate prime filtering problem."""
        start = self.rng.randint(10, 50)
        end = start + self.rng.randint(30, 60)
        
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True
        
        primes = [n for n in range(start, end + 1) if is_prime(n)]
        
        prompt = f"List all prime numbers between {start} and {end} (inclusive). Provide your answer as a comma-separated list of numbers."
        
        return Problem(
            domain="number_theory",
            template_id="nt_prime_filtering",
            template_name="Prime Number Filtering",
            prompt=prompt,
            ground_truth=primes,
            verifier=lambda x: x == primes,
            parameters={"start": start, "end": end},
            difficulty=2
        )
    
    def _gen_divisor_count(self) -> Problem:
        """Generate divisor count problem."""
        target_count = self.rng.randint(4, 8)
        start = self.rng.randint(1, 30)
        end = start + self.rng.randint(40, 80)
        
        def count_divisors(n):
            count = 0
            for i in range(1, n + 1):
                if n % i == 0:
                    count += 1
            return count
        
        numbers = [n for n in range(start, end + 1) if count_divisors(n) == target_count]
        
        prompt = f"Find all numbers between {start} and {end} (inclusive) that have exactly {target_count} divisors. Provide your answer as a comma-separated list."
        
        return Problem(
            domain="number_theory",
            template_id="nt_divisor_count",
            template_name="Divisor Count Problem",
            prompt=prompt,
            ground_truth=numbers,
            verifier=lambda x: x == numbers,
            parameters={"start": start, "end": end, "target_count": target_count},
            difficulty=3
        )
    
    def _gen_sequence_sum(self) -> Problem:
        """Generate sequence sum problem."""
        start = self.rng.randint(1, 10)
        end = start + self.rng.randint(15, 30)
        divisor = self.rng.choice([3, 4, 5, 7])
        
        numbers = [n for n in range(start, end + 1) if n % divisor == 0]
        total = sum(numbers)
        
        prompt = f"What is the sum of all numbers from {start} to {end} (inclusive) that are divisible by {divisor}? Provide only the numerical answer."
        
        return Problem(
            domain="number_theory",
            template_id="nt_sequence_sum",
            template_name="Sequence Sum with Condition",
            prompt=prompt,
            ground_truth=total,
            verifier=lambda x: x == total,
            parameters={"start": start, "end": end, "divisor": divisor},
            difficulty=2
        )
    
    def _gen_modular_arithmetic(self) -> Problem:
        """Generate modular arithmetic problem."""
        base = self.rng.randint(5, 20)
        exponent = self.rng.randint(3, 8)
        modulus = self.rng.randint(7, 15)
        
        result = pow(base, exponent, modulus)
        
        prompt = f"Calculate {base}^{exponent} mod {modulus}. Provide only the numerical answer."
        
        return Problem(
            domain="number_theory",
            template_id="nt_modular_arithmetic",
            template_name="Modular Arithmetic",
            prompt=prompt,
            ground_truth=result,
            verifier=lambda x: x == result,
            parameters={"base": base, "exponent": exponent, "modulus": modulus},
            difficulty=3
        )
    
    def _gen_set_intersection(self) -> Problem:
        """Generate set intersection problem."""
        range_a = (self.rng.randint(1, 20), self.rng.randint(40, 60))
        range_b = (self.rng.randint(30, 50), self.rng.randint(70, 90))
        divisor_a = self.rng.choice([3, 4, 5])
        divisor_b = self.rng.choice([2, 6, 7])
        
        set_a = set(range(range_a[0], range_a[1] + 1, divisor_a))
        set_b = set(range(range_b[0], range_b[1] + 1, divisor_b))
        intersection = sorted(list(set_a & set_b))
        
        prompt = f"Set A contains multiples of {divisor_a} from {range_a[0]} to {range_a[1]}. Set B contains multiples of {divisor_b} from {range_b[0]} to {range_b[1]}. Find the intersection of sets A and B. Provide as a comma-separated list."
        
        return Problem(
            domain="number_theory",
            template_id="nt_set_intersection",
            template_name="Set Intersection with Properties",
            prompt=prompt,
            ground_truth=intersection,
            verifier=lambda x: x == intersection,
            parameters={"range_a": range_a, "range_b": range_b, "divisor_a": divisor_a, "divisor_b": divisor_b},
            difficulty=2
        )
    
    def _gen_fibonacci_property(self) -> Problem:
        """Generate Fibonacci sequence property problem."""
        n = self.rng.randint(8, 15)
        
        def fib(k):
            if k <= 1:
                return k
            a, b = 0, 1
            for _ in range(2, k + 1):
                a, b = b, a + b
            return b
        
        fib_numbers = [fib(i) for i in range(1, n + 1)]
        even_fibs = [f for f in fib_numbers if f % 2 == 0]
        result = sum(even_fibs)
        
        prompt = f"Generate the first {n} Fibonacci numbers (starting with 1, 1, 2, 3, 5...). Sum only the even Fibonacci numbers. Provide the numerical sum."
        
        return Problem(
            domain="number_theory",
            template_id="nt_fibonacci_property",
            template_name="Fibonacci Sequence Property",
            prompt=prompt,
            ground_truth=result,
            verifier=lambda x: x == result,
            parameters={"n": n},
            difficulty=3
        )
    
    def _gen_perfect_numbers(self) -> Problem:
        """Generate perfect number search problem."""
        limit = self.rng.randint(50, 100)
        
        def is_perfect(n):
            if n < 2:
                return False
            divisor_sum = sum(i for i in range(1, n) if n % i == 0)
            return divisor_sum == n
        
        perfect_nums = [n for n in range(1, limit + 1) if is_perfect(n)]
        
        prompt = f"Find all perfect numbers up to {limit}. A perfect number equals the sum of its proper divisors. Provide as a comma-separated list."
        
        return Problem(
            domain="number_theory",
            template_id="nt_perfect_numbers",
            template_name="Perfect Number Search",
            prompt=prompt,
            ground_truth=perfect_nums,
            verifier=lambda x: x == perfect_nums,
            parameters={"limit": limit},
            difficulty=4
        )
    
    def _gen_digit_manipulation(self) -> Problem:
        """Generate digit sum and product problem."""
        start = self.rng.randint(10, 30)
        end = start + self.rng.randint(20, 40)
        
        def digit_sum(n):
            return sum(int(d) for d in str(n))
        
        def digit_product(n):
            product = 1
            for d in str(n):
                product *= int(d)
            return product
        
        numbers = [n for n in range(start, end + 1) if digit_sum(n) == digit_product(n)]
        
        prompt = f"Find all numbers from {start} to {end} where the sum of digits equals the product of digits. Provide as a comma-separated list."
        
        return Problem(
            domain="number_theory",
            template_id="nt_digit_manipulation",
            template_name="Digit Sum and Product",
            prompt=prompt,
            ground_truth=numbers,
            verifier=lambda x: x == numbers,
            parameters={"start": start, "end": end},
            difficulty=2
        )
    
    def _gen_gcd_lcm(self) -> Problem:
        """Generate GCD and LCM problem."""
        a = self.rng.randint(12, 48)
        b = self.rng.randint(12, 48)
        
        def gcd(x, y):
            while y:
                x, y = y, x % y
            return x
        
        def lcm(x, y):
            return abs(x * y) // gcd(x, y)
        
        result = gcd(a, b) + lcm(a, b)
        
        prompt = f"For numbers {a} and {b}, calculate GCD + LCM. Provide only the numerical answer."
        
        return Problem(
            domain="number_theory",
            template_id="nt_gcd_lcm",
            template_name="GCD and LCM Problems",
            prompt=prompt,
            ground_truth=result,
            verifier=lambda x: x == result,
            parameters={"a": a, "b": b},
            difficulty=3
        )
    
    def _gen_arithmetic_progression(self) -> Problem:
        """Generate arithmetic progression problem."""
        first = self.rng.randint(2, 10)
        diff = self.rng.randint(2, 7)
        n_terms = self.rng.randint(8, 15)
        
        # Sum of arithmetic progression: n/2 * (2a + (n-1)d)
        total = (n_terms * (2 * first + (n_terms - 1) * diff)) // 2
        
        prompt = f"An arithmetic sequence starts with {first} and has a common difference of {diff}. What is the sum of the first {n_terms} terms? Provide only the numerical answer."
        
        return Problem(
            domain="number_theory",
            template_id="nt_arithmetic_progression",
            template_name="Arithmetic Progression",
            prompt=prompt,
            ground_truth=total,
            verifier=lambda x: x == total,
            parameters={"first": first, "diff": diff, "n_terms": n_terms},
            difficulty=2
        )
    
    # Game Theory Problem Generators
    
    def _gen_nim_game(self) -> Problem:
        """Generate Nim game problem."""
        piles = [self.rng.randint(3, 8) for _ in range(3)]
        
        # Nim game: XOR of all piles. If 0, losing position; otherwise winning
        xor_sum = piles[0] ^ piles[1] ^ piles[2]
        winning = "winning" if xor_sum != 0 else "losing"
        
        prompt = f"In a Nim game, there are three piles with {piles[0]}, {piles[1]}, and {piles[2]} stones. Players alternate removing any number of stones from a single pile. The player who takes the last stone wins. Is the current position winning or losing for the player to move? Answer with 'winning' or 'losing'."
        
        return Problem(
            domain="game_theory",
            template_id="gt_nim_game",
            template_name="Nim Game Strategy",
            prompt=prompt,
            ground_truth=winning,
            verifier=lambda x: x.lower().strip() == winning,
            parameters={"piles": piles},
            difficulty=3
        )
    
    def _gen_coin_game(self) -> Problem:
        """Generate coin selection game."""
        coins = sorted([self.rng.randint(1, 20) for _ in range(6)], reverse=True)
        
        # Optimal strategy: take from ends greedily
        def optimal_play(coins_list):
            if not coins_list:
                return 0, 0
            
            # Dynamic programming approach
            n = len(coins_list)
            dp = [[0] * n for _ in range(n)]
            
            for i in range(n):
                dp[i][i] = coins_list[i]
            
            for length in range(2, n + 1):
                for i in range(n - length + 1):
                    j = i + length - 1
                    left = coins_list[i] + min(dp[i+2][j] if i+2 <= j else 0, dp[i+1][j-1] if i+1 <= j-1 else 0)
                    right = coins_list[j] + min(dp[i+1][j-1] if i+1 <= j-1 else 0, dp[i][j-2] if i <= j-2 else 0)
                    dp[i][j] = max(left, right)
            
            player1_score = dp[0][n-1]
            total = sum(coins_list)
            return player1_score, total - player1_score
        
        p1_score, p2_score = optimal_play(coins)
        
        prompt = f"In a coin game, coins with values {coins} are arranged in a line. Two players alternate picking a coin from either end. Both play optimally. What is the maximum score the first player can guarantee? Provide only the numerical answer."
        
        return Problem(
            domain="game_theory",
            template_id="gt_coin_game",
            template_name="Coin Selection Game",
            prompt=prompt,
            ground_truth=p1_score,
            verifier=lambda x: x == p1_score,
            parameters={"coins": coins},
            difficulty=3
        )
    
    def _gen_stone_removal(self) -> Problem:
        """Generate stone removal game."""
        total_stones = self.rng.randint(15, 30)
        max_remove = self.rng.choice([2, 3, 4])
        
        # If total_stones % (max_remove + 1) == 0, it's a losing position
        is_losing = (total_stones % (max_remove + 1)) == 0
        result = "losing" if is_losing else "winning"
        
        prompt = f"In a game with {total_stones} stones, players alternate removing 1 to {max_remove} stones per turn. The player who takes the last stone wins. Is the starting position winning or losing for the first player? Answer with 'winning' or 'losing'."
        
        return Problem(
            domain="game_theory",
            template_id="gt_stone_removal",
            template_name="Stone Removal Game",
            prompt=prompt,
            ground_truth=result,
            verifier=lambda x: x.lower().strip() == result,
            parameters={"total_stones": total_stones, "max_remove": max_remove},
            difficulty=3
        )
    
    def _gen_prisoner_dilemma(self) -> Problem:
        """Generate prisoner's dilemma variant."""
        cooperate_both = self.rng.randint(5, 8)
        defect_both = self.rng.randint(1, 3)
        defect_vs_coop = self.rng.randint(9, 12)
        coop_vs_defect = self.rng.randint(0, 2)
        
        # Nash equilibrium is (Defect, Defect)
        answer = f"({defect_both}, {defect_both})"
        
        prompt = f"Two players play a prisoner's dilemma. Payoffs: (Cooperate, Cooperate) = ({cooperate_both}, {cooperate_both}), (Defect, Defect) = ({defect_both}, {defect_both}), (Defect, Cooperate) = ({defect_vs_coop}, {coop_vs_defect}), (Cooperate, Defect) = ({coop_vs_defect}, {defect_vs_coop}). What are the payoffs at the Nash equilibrium? Format: (payoff1, payoff2)"
        
        return Problem(
            domain="game_theory",
            template_id="gt_prisoner_dilemma",
            template_name="Prisoner's Dilemma Variant",
            prompt=prompt,
            ground_truth=answer,
            verifier=lambda x: x.strip() == answer,
            parameters={"cooperate_both": cooperate_both, "defect_both": defect_both,
                       "defect_vs_coop": defect_vs_coop, "coop_vs_defect": coop_vs_defect},
            difficulty=4
        )
    
    def _gen_auction_strategy(self) -> Problem:
        """Generate auction bidding strategy problem."""
        valuations = sorted([self.rng.randint(10, 50) for _ in range(4)])
        
        # In second-price auction, optimal bid equals valuation
        # Winner pays second-highest bid
        highest = valuations[-1]
        second_highest = valuations[-2]
        
        prompt = f"In a second-price sealed-bid auction, four bidders have valuations {valuations[0]}, {valuations[1]}, {valuations[2]}, and {valuations[3]}. If all bid their true valuations, what price does the winner pay? Provide only the numerical answer."
        
        return Problem(
            domain="game_theory",
            template_id="gt_auction_strategy",
            template_name="Auction Bidding Strategy",
            prompt=prompt,
            ground_truth=second_highest,
            verifier=lambda x: x == second_highest,
            parameters={"valuations": valuations},
            difficulty=4
        )
    
    def _gen_resource_allocation(self) -> Problem:
        """Generate resource allocation game."""
        resources = self.rng.randint(10, 20)
        players = 3
        equal_share = resources // players
        
        # Equal division is fair allocation
        result = equal_share * players
        
        prompt = f"{players} players must divide {resources} identical resources. Using equal division (each gets ⌊{resources}/{players}⌋), how many resources are allocated in total? Provide only the numerical answer."
        
        return Problem(
            domain="game_theory",
            template_id="gt_resource_allocation",
            template_name="Resource Allocation Game",
            prompt=prompt,
            ground_truth=result,
            verifier=lambda x: x == result,
            parameters={"resources": resources, "players": players},
            difficulty=3
        )
    
    def _gen_matching_pennies(self) -> Problem:
        """Generate matching pennies game."""
        # Zero-sum game, mixed strategy equilibrium is (0.5, 0.5) for both
        # Expected payoff is 0
        expected_value = 0
        
        prompt = "In matching pennies, Player 1 wins if coins match (both heads or tails), Player 2 wins if they differ. Payoffs are +1/-1. What is Player 1's expected payoff in Nash equilibrium? Provide only the numerical answer."
        
        return Problem(
            domain="game_theory",
            template_id="gt_matching_pennies",
            template_name="Matching Pennies Game",
            prompt=prompt,
            ground_truth=expected_value,
            verifier=lambda x: x == expected_value,
            parameters={},
            difficulty=2
        )
    
    def _gen_sequential_choice(self) -> Problem:
        """Generate sequential choice game."""
        options = sorted([self.rng.randint(5, 30) for _ in range(4)])
        
        # Backward induction: last player takes highest, second-to-last anticipates this
        # Work backwards
        result = options[-2] if len(options) >= 2 else options[0]
        
        prompt = f"In a sequential game, Player 1 chooses first from options {options}. After Player 1 chooses, Player 2 picks from the remaining options. Both want to maximize their own choice. Using backward induction, what does Player 1 choose? Provide only the numerical answer."
        
        # Actually, with backward induction in this setup, P1 should choose second-highest
        # because P2 will take the highest remaining
        
        return Problem(
            domain="game_theory",
            template_id="gt_sequential_choice",
            template_name="Sequential Choice Game",
            prompt=prompt,
            ground_truth=result,
            verifier=lambda x: x == result,
            parameters={"options": options},
            difficulty=3
        )
    
    def _gen_voting_strategy(self) -> Problem:
        """Generate strategic voting problem."""
        candidates = 3
        voters = self.rng.randint(5, 9)
        threshold = (voters // 2) + 1
        
        prompt = f"In a {voters}-voter election with {candidates} candidates, plurality voting is used. What is the minimum number of votes needed to guarantee winning? Provide only the numerical answer."
        
        return Problem(
            domain="game_theory",
            template_id="gt_voting_strategy",
            template_name="Strategic Voting Problem",
            prompt=prompt,
            ground_truth=threshold,
            verifier=lambda x: x == threshold,
            parameters={"candidates": candidates, "voters": voters},
            difficulty=4
        )
    
    def _gen_fair_division(self) -> Problem:
        """Generate fair division problem."""
        cake_value = 100
        people = self.rng.choice([3, 4, 5])
        fair_share = cake_value // people
        
        prompt = f"Using the divide-and-choose method for {people} people dividing a cake worth {cake_value} units, what is the minimum value each person can guarantee (using floor division)? Provide only the numerical answer."
        
        return Problem(
            domain="game_theory",
            template_id="gt_fair_division",
            template_name="Fair Division Problem",
            prompt=prompt,
            ground_truth=fair_share,
            verifier=lambda x: x == fair_share,
            parameters={"cake_value": cake_value, "people": people},
            difficulty=3
        )
    
    def generate_problem(self, template_id: str = None, domain: str = None) -> Problem:
        """
        Generate a problem from a specific template or domain.
        
        Args:
            template_id: Specific template to use (optional)
            domain: Domain to sample from (optional)
            
        Returns:
            Problem instance
        """
        if template_id:
            if template_id not in self.templates:
                raise ValueError(f"Unknown template: {template_id}")
            template = self.templates[template_id]
            return template["generator"]()
        
        # Filter by domain if specified
        if domain:
            # Map domain names to template prefixes
            domain_map = {
                "number_theory": "nt",
                "game_theory": "gt"
            }
            prefix = domain_map.get(domain.lower(), domain[:2].lower())
            available = {k: v for k, v in self.templates.items() if k.startswith(prefix + "_")}
        else:
            available = self.templates
        
        # Randomly select a template
        template_id = self.rng.choice(list(available.keys()))
        template = available[template_id]
        
        return template["generator"]()
    
    def generate_batch(self, n: int, domain: str = None) -> List[Problem]:
        """Generate a batch of problems."""
        return [self.generate_problem(domain=domain) for _ in range(n)]
    
    def get_template_info(self) -> Dict[str, Dict]:
        """Get information about all templates."""
        info = {}
        for tid, template in self.templates.items():
            info[tid] = {
                "name": template["name"],
                "difficulty": template["difficulty"]
            }
        return info


# if __name__ == "__main__":
#     # Test problem generation
#     generator = ProblemGenerator(seed=42)
    
#     print("=== Testing Problem Generation ===\n")
    
#     # Test number theory
#     print("Number Theory Problems:")
#     for _ in range(3):
#         problem = generator.generate_problem(domain="number_theory")
#         print(f"\nTemplate: {problem.template_name}")
#         print(f"Prompt: {problem.prompt}")
#         print(f"Ground Truth: {problem.ground_truth}")
#         print(f"Difficulty: {problem.difficulty}")
    
#     # Test game theory
#     print("\n\nGame Theory Problems:")
#     for _ in range(3):
#         problem = generator.generate_problem(domain="game_theory")
#         print(f"\nTemplate: {problem.template_name}")
#         print(f"Prompt: {problem.prompt}")
#         print(f"Ground Truth: {problem.ground_truth}")
#         print(f"Difficulty: {problem.difficulty}")
    
#     print("\n\n=== Available Templates ===")
#     for tid, info in generator.get_template_info().items():
#         print(f"{tid}: {info['name']} (Difficulty: {info['difficulty']})")
