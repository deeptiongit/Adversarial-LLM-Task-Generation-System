# Adversarial LLM Task Generation System - Project Summary

## Overview

A complete implementation of an adversarial LLM task generation system that discovers structured failure modes in Large Language Models through systematic mutation and evolutionary search.

---

## Module Breakdown

### 1. Problem Generator Module (`problem_generator.py`)
- **20 problem templates** implemented (10 Number Theory + 10 Game Theory)

| Number Theory Templates | Game Theory Templates |
|-------------------------|-----------------------|
| Prime Number Filtering | Nim Game Strategy |
| Divisor Count Problem | Coin Selection Game |
| Sequence Sum with Condition | Stone Removal Game |
| Modular Arithmetic | Prisoner's Dilemma Variant |
| Set Intersection with Properties | Auction Bidding Strategy |
| Fibonacci Sequence Property | Resource Allocation Game |
| Perfect Number Search | Matching Pennies Game |
| Digit Sum and Product | Sequential Choice Game |
| GCD and LCM Problems | Strategic Voting Problem |
| Arithmetic Progression | Fair Division Problem |


- **Features:**
  - Programmatic ground truth verification for all problems
  - Difficulty levels (1-5 scale)
  - Domain-based filtering
  - Batch generation support
  - Fully tested and working

### 2. Mutation Engine Module (`mutation_engine.py`)
- **4 mutation types** fully implemented:
  1. **Constraint Mutation**: Adds time limits, format requirements, precision constraints
  2. **Instruction Mutation**: Ambiguous wording, passive voice, reordered instructions
  3. **Composition**: Combines multiple sub-tasks
  4. **Adversarial Traps**: Red herrings, misleading hints, anchoring bias

- **Features:**
  - Ground truth preservation verified
  - Multiple mutations per problem support
  - Automatic discard on verification failure
  - Comprehensive mutation tracking
  - Fully tested and working

### 3. LLM Interface Module (`llm_interface.py`)
- **Multi-provider support:**
  - OpenAI 
  - Anthropic

- **Features:**
  - Deterministic mode (temperature=0, fixed seeds)
  - Retry mechanism with exponential backoff
  - Robust output parsing with 7+ extraction strategies
  - Comprehensive logging of all responses
  - Token usage tracking
  - Timeout handling
  - Fully tested and working

### 4. Evaluator Module (`evaluator.py`)
- **Hard correctness checking** with:
  - Output normalization (whitespace, case, format)
  - Type-aware comparison (int, float, list, string)
  - Floating-point tolerance
  - List order handling (sorted comparison)
  - Multiple format support

- **Error Classification:**
  - Type mismatch
  - Arithmetic errors (small/medium/large)
  - List errors (incomplete, extra elements, partial overlap)
  - String matching errors
  - Fully tested and working

### 5. Failure Analyzer Module (`failure_analyzer.py`)
- **8 failure categories:**
  1. Arithmetic Error
  2. Logical Error
  3. Parsing Error
  4. Instruction Following
  5. Edge Case Handling
  6. Trap Susceptibility
  7. Constraint Violation
  8. Incomplete Solution

- **Pattern Detection:**
  - Off-by-one errors
  - Missing/extra elements
  - Wrong operations
  - Case sensitivity issues
  - Trap falls
  - Ambiguous instruction confusion

- **Features:**
  - Severity assessment (1-5 scale)
  - Reasoning comparison
  - Comprehensive failure analysis
  - Distribution tracking
  - Fully tested and working

### 6. Search Optimizer Module (`search_optimizer.py`)
- **3 search strategies** fully implemented:

  1. **Random Search with Exploitation:**
     - Explores random problem space
     - Exploits failure regions
     - Adaptive failure template tracking
     - Configurable exploitation rate

  2. **Evolutionary/Genetic Algorithm:**
     - Population-based search
     - Crossover and mutation operators
     - Fitness-based selection
     - Multi-generation evolution
     - Configurable population size, mutation/crossover rates

  3. **Hill Climbing:**
     - Local search with gradient ascent
     - Neighbor generation
     - Random restart on plateaus
     - Score-based optimization

- **Features:**
  - Deterministic mode (fixed seeds)
  - Comprehensive logging (prompt, mutations, outputs, evaluations)
  - Batch processing support
  - Reproducibility guaranteed
  - API call optimization
  - Fully tested and working

### 7. Main Orchestrator (`main.py`)
- **Experiment coordination:**
  - Multi-model support
  - Multi-strategy execution
  - Configuration management
  - Result aggregation
  - Automatic directory structure

- **Features:**
  - YAML configuration
  - Command-line interface
  - Automatic logging setup
  - Result persistence (JSON)
  - Error handling and recovery
  - Fully tested and working

### 8. Visualization Module (`visualizer.py`)
- **Comprehensive visualizations:**
  1. Failure rate comparisons (bar charts)
  2. Mutation type distribution (bar charts, pie charts)
  3. Category distribution (bar charts, pie charts)
  4. Difficulty analysis (line charts)
  5. Pattern frequency (horizontal bar charts)
  6. Strategy comparisons (multi-metric)
  7. Domain comparisons

- **Output formats:**
  - Static PNG (high DPI, publication-ready)
  - Interactive HTML (Plotly with zoom/hover)

- **Features:**
  - Professional styling (Seaborn)
  - Configurable DPI
  - Multiple chart types
  - Color-coded visualizations
  - Fully tested and working

---

## Additional Deliverables

### Configuration System
- **config.yaml**: Comprehensive YAML configuration
- **Environment variables**: API key management
- **Configurable parameters:**
  - LLM settings (temperature, max tokens, timeouts)
  - Search parameters (iterations, population size, rates)
  - Logging levels and directories
  - Visualization settings

### Documentation
- **README.md**: Complete documentation (60+ sections)
  - Installation instructions
  - Quick start guide
  - Module documentation
  - API examples
  - Troubleshooting
  - Configuration reference

- **QUICKSTART.md**: 5-minute getting started guide
  - Installation 
  - Setup
  - First experiment 
  - Common commands
  - Tips and tricks

- **PROJECT_SUMMARY.md**: This file

###  Example Usage
- **example_usage.py**: Interactive examples
  - 6 complete examples
  - Basic problem generation
  - LLM evaluation
  - Failure analysis
  - Search strategies
  - Visualization
  - Complete experiments

###  Testing
- **test_system.py**: Comprehensive test suite
  - 7 test modules
  - 100% module coverage
  - All tests passing
  - Automated verification

###  Project Structure
```
altg/
├── Core Modules (8 files)
│   ├── problem_generator.py      
│   ├── mutation_engine.py        
│   ├── llm_interface.py          
│   ├── evaluator.py          
│   ├── failure_analyzer.py    
│   ├── search_optimizer.py       
│   ├── visualizer.py            
│   └── main.py                   
│
├── Configuration
│   ├── config.yaml
│   ├── .env.example
│   ├── .gitignore
│   └── requirements.txt
│
├── Documentation
│   ├── README.md               
│   ├── QUICKSTART.md
│   └── PROJECT_SUMMARY.md
│
├── Examples & Tests
│   ├── example_usage.py        
│   └── test_system.py            
│
└── Output Directories
    ├── logs/
    ├── visualizations/
    └── experiments/
```

###  Version Control
- Git repository initialized
- All files committed
- Clean working directory
- Ready for collaboration

---

## Technical Specifications

### Dependencies
- **Python**: 3.8+
- **LLM SDKs**: openai>=1.0.0, anthropic>=0.18.0
- **Data/Math**: numpy>=1.24.0, pandas>=2.0.0
- **Visualization**: matplotlib>=3.7.0, seaborn>=0.12.0, plotly>=5.14.0
- **Utilities**: pyyaml>=6.0, tqdm>=4.65.0, colorama>=0.4.6

### Code Quality
- **Docstrings**: All classes and functions documented
- **Type Hints**: Used throughout for clarity
- **Error Handling**: Comprehensive try-except blocks
- **Logging**: Multi-level logging throughout
- **Testing**: All modules tested and verified

### Performance
- **Deterministic**: Fixed seeds ensure reproducibility
- **Efficient**: Batch processing support
- **Optimized**: API call batching and caching
- **Scalable**: Handles large experiments (100+ iterations)

---

## Usage Examples

### Quick Test
```bash
python main.py --name "test" --models "openai:gpt-3.5-turbo" \
  --strategies "random" --iterations 10
```

### Multi-Model Comparison
```bash
python main.py --name "comparison" \
  --models "openai:gpt-4" "anthropic:claude-3-sonnet-20240229" \
  --strategies "random" "evolutionary" "hill_climbing" \
  --iterations 100 --batch-size 5
```

### Interactive Examples
```bash
python example_usage.py
```

### Run Tests
```bash
python test_system.py
```

---

## Key Features Implemented

###  All 20 Problem Templates
- 10 Number Theory templates with mathematical rigor
- 10 Game Theory templates with strategic depth
- All with programmatic ground truth verification
- Difficulty ratings and domain classification

###  All 4 Mutation Types
- Constraint mutations (limits, formats, precision)
- Instruction mutations (ambiguity, reordering)
- Composition mutations (task combining)
- Adversarial traps (misleading hints)
- All preserve ground truth

###  All 3 Search Strategies
- Random search with exploitation
- Evolutionary algorithm with genetic operators
- Hill climbing with random restarts
- All with comprehensive logging

###  Multi-Provider LLM Support
- OpenAI GPT models (GPT-4, GPT-3.5)
- Anthropic Claude models (Opus, Sonnet, Haiku)
- Deterministic mode for reproducibility
- Robust error handling and retry logic

###  Comprehensive Analysis
- 8 failure categories
- 10+ failure patterns
- Severity assessment
- Distribution tracking
- Comparative analysis

###  Rich Visualizations
- 7+ chart types
- Both static (PNG) and interactive (HTML)
- Professional styling
- Multi-dimensional comparisons

###  Production-Ready
- Configuration management
- Environment variable support
- Comprehensive logging
- Error recovery
- Clean project structure
- Version controlled

---

---

## Next Steps for Users

1. **Set API Keys**:
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   ```

2. **Run Quick Test**:
   ```bash
   python main.py --name "test" --models "openai:gpt-3.5-turbo" \
     --strategies "random" --iterations 10
   ```

3. **Explore Examples**:
   ```bash
   python example_usage.py
   ```

4. **Review Results**:
   - Check `experiments/` directory
   - Open HTML visualizations in browser
   - Read experiment logs

5. **Customize**:
   - Edit `config.yaml` for your needs
   - Adjust search parameters
   - Add new problem templates
   - Modify mutation strategies

---

## System Highlights
- **Extensible**: Clean architecture for future additions
- **Reproducible**: Deterministic modes and logging
- **Comprehensive**: 20 templates, 4 mutations, 3 strategies
- **Analyzed**: 8 categories, 10+ patterns
- **Visualized**: 7+ chart types with interactivity

---
