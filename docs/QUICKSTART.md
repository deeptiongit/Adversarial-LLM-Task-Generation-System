# Quick Start Guide

## Installation 

```bash
cd /path /to/repository
pip install -r requirements.txt
```

## Setup API Keys 

```bash
# Option 1: Environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Option 2: Create .env file
cp .env.example .env
# Edit .env and add your keys
```

## Run Your First Experiment

```bash
# Quick test with GPT-3.5 (10 iterations)
python main.py \
  --name "quick_test" \
  --models "openai:gpt-3.5-turbo" \
  --strategies "random" \
  --iterations 10 \
  --batch-size 2
```

Results will be saved in `experiments/quick_test_<timestamp>/`

## Run Examples (Interactive)

```bash
python example_usage.py
```

Choose from 6 interactive examples:
1. Basic Problem Generation
2. LLM Evaluation
3. Failure Analysis
4. Search Strategy
5. Visualization
6. Complete Experiment

## Test Individual Modules

```bash
# Test problem generator (20 templates)
python problem_generator.py

# Test mutation engine (4 mutation types)
python mutation_engine.py

# Test evaluator
python evaluator.py

# Test visualizer
python visualizer.py
```

## Common Commands

### Single Model Test
```bash
python main.py --name "gpt4_test" --models "openai:gpt-4" --strategies "random" --iterations 50
```

### Multi-Model Comparison
```bash
python main.py \
  --name "model_comparison" \
  --models "openai:gpt-4" "openai:gpt-3.5-turbo" "anthropic:claude-3-sonnet-20240229" \
  --strategies "random" "evolutionary" \
  --iterations 100
```

### Full Experiment (All Strategies)
```bash
python main.py \
  --name "full_experiment" \
  --models "openai:gpt-4" \
  --strategies "random" "evolutionary" "hill_climbing" \
  --iterations 100 \
  --batch-size 5
```

## Output Structure

```
experiments/
└── experiment_name_timestamp/
    ├── results.json              # Metrics and analysis
    ├── experiment.log            # Detailed logs
    ├── logs/                     # Raw LLM responses
    │   └── provider_model_responses.jsonl
    └── visualizations/           # Charts and graphs
        ├── failure_rates.png
        ├── failure_rates.html
        ├── mutation_dist_*.png
        └── category_dist_*.png
```

## Troubleshooting

### API Key Not Found
```bash
# Verify keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Rate Limits
- Reduce `--batch-size` (try 1-2)
- Increase retry delay in `config.yaml`
- Use fewer iterations for testing

## Next Steps

1. **Review Results**: Check `experiments/` for your experiment results
2. **View Visualizations**: Open HTML files in browser for interactive charts
3. **Analyze Failures**: Review `results.json` for detailed failure analysis
4. **Customize**: Edit `config.yaml` to adjust search parameters
5. **Read Full Docs**: See `README.md` for complete documentation

## Tips

- Start with 10-20 iterations for quick tests
- Use `gpt-3.5-turbo` for rapid prototyping (cheaper/faster)
- Use `random` strategy first (simplest/fastest)
- Check logs/ for debugging LLM responses
- Interactive visualizations (HTML) allow zooming and hovering

## Support

- Full documentation: `README.md`
- Example code: `example_usage.py`
- Module tests: Run any `.py` file directly
- Issues: Check logs in `experiments/*/experiment.log`

---


