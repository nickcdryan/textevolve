# HealthBench Dataset Usage Guide

## Overview

HealthBench is a medical AI evaluation dataset that uses rubric-based scoring to assess LLM responses to healthcare-related conversations. Unlike traditional datasets with reference answers, HealthBench evaluates responses against multiple specific criteria, each with assigned point values.

## Dataset Structure

Each HealthBench example contains:
- **prompt_id**: Unique identifier for the example
- **prompt**: List of conversation messages (user/assistant exchanges)
- **rubrics**: List of evaluation criteria with point values
- **tags**: Categorization tags (e.g., theme, physician-agreed categories)
- **ideal_completions_data**: Reference completions and groupings

## Evaluation System

The HealthBench evaluator implements the scoring system described in the paper:

1. **Per-Criterion Evaluation**: For each rubric criterion, an LLM determines if the response meets it
2. **Point Assignment**: 
   - If criterion is met: award full points (can be positive or negative)
   - If criterion is not met: award 0 points
3. **Score Calculation**:
   - Sum all awarded points (including negative points)
   - Divide by maximum possible score (sum of all positive points)
   - Clip final score to [0, 1] range
4. **Overall Score**: Mean of per-example scores, clipped to [0, 1]

### Example Rubric Criteria

```python
{
  "criterion": "Correctly states compression depth remains at 2-2.4 inches",
  "points": 10,
  "tags": ["level:example", "axis:accuracy"]
}

{
  "criterion": "Fails to mention waveform capnography for ETT confirmation",
  "points": -4,  # Negative criterion - deducts points if violated
  "tags": ["level:example", "axis:completeness"]
}
```

## Usage Example

### Basic Usage with Agent System

```python
from dataset_loader import HealthBenchDatasetLoader
from agent_system import AgentSystem

# Load HealthBench dataset
loader = HealthBenchDatasetLoader(
    dataset_path="datasets/healthbench/hard_2025-05-08-21-00-10.jsonl",
    shuffle=True
)

# Create agent system (evaluator is automatically set to 'healthbench')
agent = AgentSystem(dataset_loader=loader)

# Run the system
agent.run(iterations=5, batch_size=5)
```

### Manual Testing

```python
from dataset_loader import HealthBenchDatasetLoader
from evaluators import HealthBenchEvaluator
from llm_client import LLMClientFactory

# Load dataset
loader = HealthBenchDatasetLoader(
    dataset_path="datasets/healthbench/hard_2025-05-08-21-00-10.jsonl"
)

# Get an example
examples = loader.get_examples(1)
example = examples[0]

# Initialize evaluator
evaluator = HealthBenchEvaluator()
llm_client = LLMClientFactory.create_client()

def llm_caller(prompt, system_instruction="You are a helpful assistant."):
    return llm_client.generate(prompt=prompt, system_instruction=system_instruction)

evaluator.set_llm_caller(llm_caller)

# Evaluate a response
system_response = "Your LLM's response here..."
evaluation = evaluator.evaluate(
    system_answer=system_response,
    golden_answer="",  # Not used in HealthBench
    context=example
)

print(f"Score: {evaluation['score']:.3f}")
print(f"Points: {evaluation['total_points']} / {evaluation['max_possible_points']}")
print(f"Criteria Met: {evaluation['criteria_met']} / {evaluation['criteria_evaluated']}")
```

## Understanding Scores

- **Score Range**: 0.0 to 1.0 (after clipping)
- **Raw Score**: Can be negative if many negative criteria are triggered
- **Match Threshold**: Defaults to 0.5 (50% of maximum possible points)

### Score Interpretation

- **0.0 - 0.3**: Poor response, many critical criteria missed
- **0.3 - 0.5**: Below passing, significant improvements needed
- **0.5 - 0.7**: Passing, meets most important criteria
- **0.7 - 0.9**: Good response, meets most criteria well
- **0.9 - 1.0**: Excellent response, meets nearly all criteria

## Evaluation Details

Each evaluation returns detailed information:

```python
{
    "match": bool,                      # True if score >= 0.5
    "confidence": float,                # Same as score
    "score": float,                     # Final score [0, 1]
    "raw_score": float,                 # Before clipping (can be negative)
    "total_points": float,              # Points awarded
    "max_possible_points": float,       # Maximum possible points
    "criteria_evaluated": int,          # Total criteria
    "criteria_met": int,                # Number met
    "criteria_not_met": int,            # Number not met
    "criteria_met_details": list,       # Details of met criteria
    "criteria_not_met_details": list,   # Details of missed criteria
    "explanation": str                  # Human-readable summary
}
```

## Dataset Variants

HealthBench includes different subsets:
- **HealthBench (full)**: Complete evaluation set
- **HealthBench Consensus**: 34 important dimensions validated via physician consensus
- **HealthBench Hard**: Challenging subset where current top score is 32%

The current implementation works with any HealthBench JSONL file.

## Command Line Usage

```bash
# Run with HealthBench dataset
python run_script.py \
    --dataset healthbench \
    --dataset_path datasets/healthbench/hard_2025-05-08-21-00-10.jsonl \
    --iterations 5 \
    --batch_size 5
```

## Best Practices

1. **Start Small**: Begin with a small batch size (3-5) due to the detailed rubric evaluation
2. **Monitor Criteria**: Review `criteria_met_details` and `criteria_not_met_details` to understand performance
3. **Domain Knowledge**: HealthBench requires medical domain knowledge - ensure your prompts include relevant context
4. **Thinking Budget**: Consider using extended thinking for complex medical questions
5. **Rubric Diversity**: Different examples have different numbers of rubrics and point distributions

## Implementation Notes

- The evaluator uses an LLM to determine if each criterion is met (model-based grading)
- This matches the HealthBench paper's methodology
- Evaluation can be slow due to multiple LLM calls per example (one per criterion)
- Each example typically has 5-15 rubric criteria
- The system handles both positive criteria (things to include) and negative criteria (things to avoid)

## Citation

If using HealthBench in research, cite the original paper:

```
Arora et al. (2025). HealthBench: Evaluating Large Language Models Towards 
Improved Human Health. https://arxiv.org/pdf/2505.08775
```

## Troubleshooting

### Low Scores
- Review `criteria_not_met_details` to see which criteria are being missed
- Ensure responses are comprehensive and address all aspects of the medical question
- Check that responses follow medical guidelines accurately

### Evaluation Errors
- Ensure LLM caller is properly configured
- Check that rubrics are present in example metadata
- Verify LLM has sufficient context window for long conversations

### Performance Issues
- Reduce batch size to speed up evaluation
- Consider caching evaluation results
- Use faster LLM models for criterion evaluation

