# Adversarial LLM Task Generation System (ALTG)

A comprehensive system for discovering structured failure modes in Large Language Models through evolutionary search and systematic mutation of mathematical and game-theoretic problems.

## Overview

This system analyzes LLM robustness by:
- **Generating** structured problems with verifiable ground truth
- **Mutating** problems to increase reasoning difficulty while preserving correctness
- **Searching** for failure cases using multiple optimization strategies
- **Analyzing** failure patterns and distributions
- **Visualizing** results for model comparison

## Features

###  Problem Generation
- **20 problem templates** across two domains:
  - **Number Theory** (10 templates): Prime filtering, divisor counting, modular arithmetic, sequences, etc.
  - **Game Theory** (10 templates): Nim game, strategic voting, auctions, fair division, etc.
- Programmatic ground truth verification for all problems
- Configurable difficulty levels (1-5 scale)

###  Mutation Engine
Four mutation types that preserve ground truth:
1. **Constraint Mutation**: Add limits, format requirements, precision constraints
2. **Instruction Mutation**: Ambiguous wording, passive voice, reordering
3. **Composition**: Combine multiple sub-tasks
4. **Adversarial Traps**: Misleading hints, anchoring bias, red herrings

###  Search Strategies
Three optimization strategies to find failure cases:
1. **Random Search**: Exploration with failure region exploitation
2. **Evolutionary Algorithm**: Genetic approach with crossover and mutation
3. **Hill Climbing**: Gradient-based local search with random restarts

###  LLM Interface
- **Multi-provider support**: OpenAI (GPT-4, GPT-3.5), Anthropic (Claude 3)
- **Deterministic mode**: Temperature 0, fixed seeds for reproducibility
- **Robust parsing**: Multiple strategies to extract answers from messy outputs
- **Retry mechanism**: Automatic retry with exponential backoff

###  Analysis & Visualization
- **Failure categorization**: Arithmetic errors, logical errors, instruction following, etc.
- **Pattern detection**: Off-by-one errors, missing elements, trap susceptibility
- **Comparative analysis**: Model vs. model, strategy vs. strategy
- **Interactive visualizations**: Matplotlib (PNG) and Plotly (HTML) outputs

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the repository**:
```bash
cd /home/ubuntu/altg
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure API keys**:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Or set environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

4. **Verify configuration**:
```bash
# Edit config.yaml to adjust settings
cat config.yaml
```

## Quick Start

### Basic Usage

```bash
# Run a simple experiment with GPT-3.5
python main.py \
  --name "quick_test" \
  --models "openai:gpt-3.5-turbo" \
  --strategies "random" \
  --iterations 20 \
  --batch-size 3
```

### Multi-Model Comparison

```bash
# Compare multiple models and strategies
python main.py \
  --name "model_comparison" \
  --models "openai:gpt-4" "openai:gpt-3.5-turbo" "anthropic:claude-3-sonnet-20240229" \
  --strategies "random" "evolutionary" "hill_climbing" \
  --iterations 100 \
  --batch-size 5
```

### Using the Python API

```python
from main import AdversarialLLMSystem

# Initialize system
system = AdversarialLLMSystem(config_path="config.yaml")

# Run experiment
experiment = system.run_experiment(
    name="my_experiment",
    models=["openai:gpt-4", "anthropic:claude-3-opus-20240229"],
    strategies=["random", "evolutionary"],
    max_iterations=50,
    batch_size=5
)

# Results are automatically saved to experiments/ directory
```

## Module Documentation

### 1. Problem Generator (`problem_generator.py`)

Generate structured problems with ground truth:

```python
from problem_generator import ProblemGenerator

generator = ProblemGenerator(seed=42)

# Generate random problem
problem = generator.generate_problem()

# Generate from specific template
problem = generator.generate_problem(template_id="nt_prime_filtering")

# Generate from domain
problem = generator.generate_problem(domain="number_theory")

# Batch generation
problems = generator.generate_batch(n=10, domain="game_theory")
```

### 2. Mutation Engine (`mutation_engine.py`)

Apply mutations to problems:

```python
from mutation_engine import MutationEngine

mutator = MutationEngine(seed=42)

# Apply random mutations
mutated = mutator.apply_random_mutations(problem, max_mutations=3)

# Apply specific mutation type
mutated = mutator.apply_specific_mutation(problem, "adversarial_trap")

# Generate mutation suite (all types)
suite = mutator.generate_mutation_suite(problem)
```

### 3. LLM Interface (`llm_interface.py`)

Query LLMs with robust error handling:

```python
from llm_interface import LLMInterface

# OpenAI
llm = LLMInterface(provider="openai", model="gpt-4", temperature=0.0)

# Anthropic Claude
llm = LLMInterface(provider="anthropic", model="claude-3-opus-20240229")

# Query
response = llm.query("What is 2 + 2?")
print(response.extracted_answer)  # Parsed answer
print(response.raw_response)       # Full response text
```

### 4. Evaluator (`evaluator.py`)

Evaluate correctness with normalization:

```python
from evaluator import Evaluator

evaluator = Evaluator(strict_mode=False)

# Evaluate single response
result = evaluator.evaluate(llm_response, ground_truth=42)
print(result.is_correct)
print(result.error_type)

# Batch evaluation
results = evaluator.batch_evaluate(responses, ground_truths)
metrics = evaluator.calculate_accuracy(results)
```

### 5. Failure Analyzer (`failure_analyzer.py`)

Analyze failure patterns:

```python
from failure_analyzer import FailureAnalyzer

analyzer = FailureAnalyzer()

# Analyze individual failure
failure_case = analyzer.analyze_failure(problem, mutated, llm_response, evaluation)

# Generate comprehensive analysis
analysis = analyzer.generate_analysis()
print(analysis.failure_by_category)
print(analysis.common_patterns)
```

### 6. Search Optimizer (`search_optimizer.py`)

Run search strategies:

```python
from search_optimizer import SearchOptimizer

optimizer = SearchOptimizer(config={
    "seed": 42,
    "log_dir": "logs",
    "population_size": 20
})

# Create strategy
strategy = optimizer.create_strategy("evolutionary", llm_interface)

# Run search
results = strategy.search(max_iterations=100, batch_size=5)

# Get failures
failures = strategy.get_failures()
failure_rate = strategy.get_failure_rate()
```

### 7. Visualizer (`visualizer.py`)

Generate visualizations:

```python
from visualizer import Visualizer

viz = Visualizer(output_dir="visualizations", dpi=300)

# Plot failure rates
viz.plot_failure_rates(failure_rates)

# Plot mutation distribution
viz.plot_mutation_distribution(mutation_counts)

# Plot category distribution
viz.plot_category_distribution(category_counts)
```

## Project Structure

```
altg/
├── config.yaml                 # Main configuration file
├── .env.example               # Environment variables template
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── problem_generator.py       # Problem generation module
├── mutation_engine.py         # Mutation engine module
├── llm_interface.py          # LLM interface module
├── evaluator.py              # Evaluation module
├── failure_analyzer.py       # Failure analysis module
├── search_optimizer.py       # Search optimization module
├── visualizer.py             # Visualization module
├── main.py                   # Main orchestrator
│
├── example_usage.py          # Example usage script
├── logs/                     # Log files
├── visualizations/           # Generated visualizations
└── experiments/              # Experiment results
    └── experiment_name_timestamp/
        ├── results.json
        ├── experiment.log
        ├── logs/
        └── visualizations/
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# API Keys
api_keys:
  openai: ${OPENAI_API_KEY}
  anthropic: ${ANTHROPIC_API_KEY}

# LLM Settings
llm:
  temperature: 0.0           # Deterministic generation
  max_tokens: 2000
  timeout: 60

# Search Settings
search:
  random_seed: 42            # For reproducibility
  max_iterations: 100
  population_size: 20        # For evolutionary strategy
  mutation_rate: 0.3
  crossover_rate: 0.7
  batch_size: 5

# Logging
logging:
  level: "INFO"
  log_dir: "logs"
  save_raw_responses: true
  save_prompts: true
```

## Output

### Experiment Results

Results are saved in `experiments/experiment_name_timestamp/`:

- **results.json**: Comprehensive metrics and analysis
- **experiment.log**: Detailed execution log
- **logs/**: Raw LLM responses and prompts
- **visualizations/**: Generated charts and graphs

### Visualizations

Generated visualizations include:

1. **Failure Rates**: Bar charts comparing models and strategies
2. **Mutation Distribution**: Distribution of failures by mutation type
3. **Category Distribution**: Pie charts and bar charts by failure category
4. **Difficulty Analysis**: Failure rates vs. problem difficulty
5. **Pattern Frequency**: Common failure patterns
6. **Strategy Comparison**: Performance metrics across strategies

Both static (PNG) and interactive (HTML) versions are generated.

## Examples

See `example_usage.py` for complete examples:

```bash
python example_usage.py
```

## Troubleshooting

### API Key Issues

```bash
# Verify API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Test API connectivity
python -c "from llm_interface import LLMInterface; llm = LLMInterface('openai'); print('OK')"
```

### Module Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Verify Python version
python --version  # Should be 3.8+
```

### Rate Limiting

If you encounter rate limits:
- Reduce `batch_size` in config
- Increase `retry_delay` in config
- Use smaller `max_iterations` for testing

## Performance Tips

1. **Start small**: Test with 10-20 iterations before full runs
2. **Use batch_size wisely**: Balance speed vs. API costs
3. **Enable logging**: Set log_level to INFO for monitoring
4. **Cache results**: Previous experiment results are saved
5. **Parallel experiments**: Run different models in separate processes



## Contributing

Contributions welcome! Areas for improvement:
- Additional problem domains (logic puzzles, coding problems)
- More mutation types
- Additional search strategies
- Enhanced visualization options
- Performance optimizations

## Contact

For questions or issues:
- Open an issue on GitHub

## Acknowledgments

Built with:
- OpenAI GPT models
- Anthropic Claude models
- Matplotlib, Seaborn, Plotly for visualizations
- PyYAML for configuration

---

