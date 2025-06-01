# TextEvolve

An advanced AI system that uses LLM-driven reasoning to iteratively improve its approach to solving problems from datasets. The system employs dynamic exploration/exploitation strategies and adapts its approach based on performance feedback.

## 🚀 Quick Start

1. **Set your Gemini API key:**
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

2. **Run the system:**
   ```bash
   # Basic usage with 5 iterations
   python run_script.py --dataset your_dataset.jsonl --loader jsonl --iterations 5

   # Example with MATH benchmark
   python run_script.py --dataset hendrycks_math/math_test.jsonl --loader math --iterations 5
   ```

3. **Validate results:**
   ```bash
   # Test the best script on examples 100-199
   python validate_script.py --script scripts/script_iteration_4.py --dataset hendrycks_math/math_test.jsonl --loader math --start 100 --end 199
   ```

## 📊 Supported Dataset Formats

The system supports multiple dataset formats through modular loaders:

### Built-in Loaders

| Loader | Dataset Type | Example Usage |
|--------|--------------|---------------|
| `arc` | ARC (Abstraction and Reasoning Corpus) | `--loader arc` |
| `jsonl` | JSONL files (one JSON per line) | `--loader jsonl` |
| `json` | JSON files with configurable fields | `--loader json` |
| `simpleqa` | SimpleQA dataset | `--loader simpleqa` |
| `math` | MATH dataset | `--loader math` |
| `natural_plan` | Natural Plan dataset | `--loader natural_plan` |
| `gpqa` | GPQA dataset | `--gpqa` |
| `hotpotqa` | HotpotQA dataset | `--gpqa` |
| `custom` | Your own custom format | `--loader custom` |

### Usage Examples

```bash
# ARC dataset (directory of JSON files)
python run_script.py --dataset ARC_2024_Training/ --loader arc --iterations 10

# JSONL dataset (like MATH benchmark)
python run_script.py --dataset math_test.jsonl --loader math --iterations 5

# Custom JSON with specific fields
python run_script.py --dataset custom.json --loader json --input-field question --output-field answer --iterations 5

# JSONL with custom fields (like DROP dataset)
python run_script.py --dataset drop_dataset.jsonl --loader jsonl --input-field question --output-field answers_spans --iterations 5

# Disable shuffling for consistent ordering
python run_script.py --dataset dataset.jsonl --loader jsonl --no-shuffle --iterations 5
```

## 🔧 Command Line Options

### run_script.py

| Option | Description | Default |
|--------|-------------|---------|
| `--iterations` | Number of iterations to run | 5 |
| `--dataset` | Path to dataset file/directory | required |
| `--loader` | Type of dataset loader | required |
| `--input-field` | Input field name (JSON/JSONL) | "input" |
| `--output-field` | Output field name (JSON/JSONL) | "output" |
| `--passage-field` | Passage field (JSONL) | "passage" |
| `--no-shuffle` | Disable dataset shuffling | False |
| `--seed` | Random seed | 42 |

### validate_script.py

| Option | Description | Default |
|--------|-------------|---------|
| `--script` | Path to script to validate | required |
| `--dataset` | Path to dataset | required |
| `--loader` | Dataset loader type | required |
| `--start` | Start index for validation | 0 |
| `--end` | End index for validation | 99 |
| `--detailed` | Show detailed results | False |

## 🏗️ Creating Custom Dataset Loaders

### Method 1: Simple Custom Loader

For basic custom formats, extend the `DatasetLoader` class:

```python
from dataset_loader import DatasetLoader
import json

class MyDatasetLoader(DatasetLoader):
    def _load_examples(self):
        """Load examples from your custom format"""
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)

        for key, example in data.items():
            # Convert to standard format
            self.examples.append({
                "id": key,
                "question": example["my_input_field"],  # Standard field: "question"
                "answer": example["my_output_field"],   # Standard field: "answer"
                "meta": {
                    "source": "my_dataset",
                    "original_data": example
                }
            })

        print(f"Loaded {len(self.examples)} examples from custom dataset")

# Register and use your loader
from dataset_loader import create_dataset_loader

def create_my_loader(**kwargs):
    return MyDatasetLoader(**kwargs)

# Add to the create_dataset_loader function or use directly
loader = MyDatasetLoader(dataset_path="my_data.json", shuffle=True)
```

### Method 2: Using the Custom Loader Framework

For more complex formats, use the built-in custom loader:

```python
from dataset_loader import create_dataset_loader

def load_my_examples(dataset_path):
    """Load examples from your dataset"""
    # Your custom loading logic
    with open(dataset_path, 'r') as f:
        raw_data = f.read()

    # Process and return list of examples
    examples = []
    # ... your processing logic ...
    return examples

def get_my_input(example):
    """Extract input from example"""
    return example["my_question_field"]

def get_my_output(example):
    """Extract output from example"""
    return example["my_answer_field"]

# Create the custom loader
loader = create_dataset_loader(
    "custom",
    dataset_path="my_dataset.xyz",
    load_examples_fn=load_my_examples,
    get_input_fn=get_my_input,
    get_output_fn=get_my_output,
    shuffle=True
)

# Use with agent system
from agent_system import AgentSystem
agent = AgentSystem(dataset_loader=loader)
```

## 📈 How It Works

### 1. Adaptive Strategy

The system uses three main strategies:
- **Explore** (60% initially): Try completely new approaches
- **Exploit** (20% initially): Combine successful techniques  
- **Refine** (20% initially): Make targeted improvements to the best script

The balance between these strategies adapts based on performance.

### 2. Progressive Testing

- Starts with small batches (3-5 examples)
- For promising scripts (>60% accuracy), runs progressive testing on all previously seen examples
- Adjusts batch size based on performance stability

### 3. LLM-Driven Improvements

- Uses LLM reasoning for strategy decisions, error analysis, and script generation
- Employs advanced agentic patterns like ReAct, chain-of-thought, and verification loops
- Automatically repairs and debugs generated scripts

### 4. Example Workflow

```
Iteration 0: Baseline script (simple LLM call) → 45% accuracy
Iteration 1: Explore new approach → 62% accuracy → Progressive testing → 58% overall
Iteration 2: Exploit successful techniques → 71% accuracy → Progressive testing → 65% overall  
Iteration 3: Refine best approach → 73% accuracy → Progressive testing → 68% overall
...
```

## 📁 Output Structure

The system creates several directories:

```
├── archive/                 # Iteration data and summaries
│   ├── iteration_0.json    # Detailed data for each iteration
│   ├── iteration_1.json
│   └── summaries.json      # Performance summaries
├── scripts/                # Generated scripts
│   ├── script_iteration_0.py
│   ├── script_iteration_1.py
│   └── ...
├── learnings.txt           # Accumulated insights and patterns
└── README.md
```

## 🎯 Performance Tracking

The system tracks multiple metrics:

- **Batch Accuracy**: Performance on current test batch
- **Progressive Accuracy**: Performance on all previously seen examples  
- **Combined Accuracy**: Weighted average across all tested examples
- **Capability Assessment**: Strengths, weaknesses, and improvement areas

Example output:
```
Iteration  Strategy     Batch Acc.   Prog. Acc.      Combined    Batch Size  Prog. Size
8          exploit      75.00%       68.33% (60)     69.23%      4           60
```

## 🔍 Validation and Testing

Test your best script on specific example ranges:

```bash
# Test on examples 0-99
python validate_script.py --script scripts/script_iteration_5.py --dataset data.jsonl --loader jsonl --start 0 --end 99

# Test on examples 500-599 with detailed output
python validate_script.py --script scripts/script_iteration_5.py --dataset data.jsonl --loader jsonl --start 500 --end 599 --detailed

# Test the current best script (auto-detected)
python validate_script.py --dataset data.jsonl --loader jsonl --start 100 --end 199
```

## 🛠️ Advanced Usage

### Programmatic Usage

```python
from dataset_loader import create_dataset_loader
from agent_system import AgentSystem

# Create dataset loader
loader = create_dataset_loader(
    "jsonl",
    dataset_path="your_dataset.jsonl",
    shuffle=True,
    random_seed=42
)

# Initialize agent system
agent = AgentSystem(dataset_loader=loader)

# Run iterations
for i in range(10):
    result = agent.run_iteration()
    print(f"Iteration {i}: {result.get('performance', {}).get('accuracy', 0):.2f} accuracy")

# Get best script info
best_script = agent.get_best_script_info()
print(f"Best script: {best_script.get('path')} with {best_script.get('combined_accuracy', 0):.2f} accuracy")
```

### Custom Field Mapping

For datasets with non-standard field names:

```bash
# JSON dataset with custom fields
python run_script.py --dataset custom.json --loader json --input-field "problem_statement" --output-field "solution"

# JSONL dataset with passage and question
python run_script.py --dataset reading_comprehension.jsonl --loader jsonl --input-field "question" --passage-field "context" --output-field "answer"
```

## 🤝 Contributing

To add support for a new dataset format:

1. Create a new loader class inheriting from `DatasetLoader`
2. Implement the `_load_examples()` method
3. Ensure examples use standard field names: `"question"`, `"answer"`, `"id"`
4. Add your loader to the `create_dataset_loader()` function
5. Test with both `run_script.py` and `validate_script.py`

## 📜 License

MIT License - see LICENSE file for details.