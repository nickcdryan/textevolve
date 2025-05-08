# Improved Agentic Learning System

This system uses LLM-driven reasoning to iteratively improve its approach to solving problems from a dataset. It employs a dynamic exploration/exploitation strategy and adapts its testing approach based on performance.

## Key Features

### 1. Example Rotation and Progressive Testing

- The system works through dataset examples sequentially, encountering new examples with each iteration
- It starts with small batch sizes (5 examples) and gradually increases as performance improves
- For promising scripts, it runs "progressive testing" on all previously seen examples to verify robustness

### 2. LLM-Driven Decision Making

- Batch size adjustment is determined by LLM reasoning about performance trends
- Explore/exploit balance is adapted based on LLM analysis of accuracy and coverage
- Best script selection considers both accuracy and testing coverage
- Answer evaluation uses semantic matching via LLM, not just exact string matching

### 3. Validation Interface

A separate validation script allows testing the best solution on any range of dataset examples:

```bash
python validate_script.py --start 900 --end 999
```

## How to Run

1. Set your Gemini API key:
   ```
   export GEMINI_API_KEY=your_api_key_here
   ```

2. Run the system for a specified number of iterations:
   ```
   python run_script.py --iterations 10
   ```

## System Workflow

1. **Initialization**: 
   - Starting explore rate: 70%, exploit rate: 30%
   - Starting batch size: 5 examples

2. **Each Iteration**:
   - Retrieves the next N examples from the dataset (where N is the current batch size)
   - Decides whether to explore (try a new approach) or exploit (refine existing approach)
   - Generates a script using LLM guidance
   - Tests the script on the batch
   - Evaluates results using LLM-based semantic matching
   - For promising scripts, runs progressive testing on all previously seen examples
   - Adjusts explore/exploit balance and batch size based on performance

3. **Performance Tracking**:
   - Tracks both batch performance and progressive testing performance
   - Maintains detailed logs of all iterations and performance metrics
   - Identifies the best script based on both accuracy and coverage

## Output and Results

The system provides:
- Detailed iteration logs in the archive directory
- Performance trend summaries
- Information about the best script
- A convenient interface for validating scripts on held-out examples

## Dataset Format

The system expects a JSON dataset with example keys following a pattern (e.g., "calendar_scheduling_example_0"):

```json
{
  "calendar_scheduling_example_0": {
    "prompt_0shot": "...",  // The input question
    "golden_plan": "..."    // The expected answer
  },
  "calendar_scheduling_example_1": {
    ...
  }
}
```




# Custom Dataset Loaders for Agentic Learning System

This system now supports flexible dataset loading through a custom loader interface. Instead of hardcoding field names or making assumptions about dataset structure, the system now uses a modular loader approach that can handle various dataset formats.

## Key Features

- **Modular Dataset Loaders**: Separate the dataset loading logic from the core learning system
- **ARC Dataset Support**: Built-in support for the ARC (Abstraction and Reasoning Corpus) format
- **Optional Shuffling**: Control whether examples are shuffled or used in their original order
- **Extensible Design**: Create custom loaders for your specific dataset formats

## Available Dataset Loaders

### 1. ARC Dataset Loader

Designed for the ARC dataset format, handling both directory-based and single-file formats:

```python
from dataset_loader import ARCDatasetLoader

# For a directory of ARC files
loader = ARCDatasetLoader(
    dataset_path="ARC_2024_Training/",
    shuffle=True,
    random_seed=42
)

# For a single ARC file
loader = ARCDatasetLoader(
    dataset_path="arc_problem.json",
    shuffle=False
)
```

### 2. Generic JSON Loader

For JSON datasets with configurable field names:

```python
from dataset_loader import JSONDatasetLoader

loader = JSONDatasetLoader(
    dataset_path="dataset.json",
    input_field="question",     # Name of input field
    output_field="answer",      # Name of output field
    example_prefix="example_",  # Optional prefix for example keys
    shuffle=True
)
```

### 3. Custom Loader

For completely custom formats, you can provide your own loading functions:

```python
from dataset_loader import CustomDatasetLoader

# Define your custom functions
def load_my_examples(dataset_path):
    # Your logic to load examples from the dataset
    # Return a list of examples in any format
    return examples

def get_my_input(example):
    # Extract input from your example format
    return example["my_input_field"]

def get_my_output(example):
    # Extract output from your example format
    return example["my_output_field"]

# Create the custom loader
loader = CustomDatasetLoader(
    dataset_path="my_custom_dataset.xyz",
    load_examples_fn=load_my_examples,
    get_input_fn=get_my_input,
    get_output_fn=get_my_output,
    shuffle=True
)
```

## Running the System

### Using run_script.py

The main script has been updated to support different loader types:

```bash
# For ARC dataset
python run_script.py --iterations 5 --dataset ARC_2024_Training/ --loader arc

# For a JSON dataset with custom fields
python run_script.py --iterations 5 --dataset custom_data.json --loader json --input-field question --output-field answer

# Disable shuffling
python run_script.py --iterations 5 --dataset ARC_2024_Training/ --loader arc --no-shuffle
```

### Using a Custom Script

For more control, you can create your own script and initialize the system directly:

```python
from dataset_loader import create_dataset_loader
from agent_system import AgentSystem

# Create your desired loader
loader = create_dataset_loader(
    "arc",
    dataset_path="ARC_2024_Training/",
    shuffle=True
)

# Initialize the agent system with the loader
agent = AgentSystem(dataset_loader=loader)

# Run iterations
for i in range(5):
    agent.run_iteration()
```

## Creating Your Own Loader

To create a custom loader for a new dataset format, subclass `DatasetLoader` and implement the required methods:

```python
from dataset_loader import DatasetLoader

class MySpecialDatasetLoader(DatasetLoader):
    def _load_examples(self):
        # Load examples from your dataset format
        # Populate self.examples with your data

    def get_example_input(self, example):
        # Extract input from your example format
        return example["my_input_field"]

    def get_example_output(self, example):
        # Extract output from your example format
        return example["my_output_field"]
```

## Example for ARC Dataset

A complete example for running with the ARC dataset is provided in `run_arc_example.py`:

```bash
python run_arc_example.py
```

This will:
1. Load the ARC dataset from the "ARC_2024_Training/" directory
2. Initialize the agent system
3. Run 3 iterations to demonstrate the system
4. Print information about the examples and results

## Benefits of This Approach

- **Modularity**: Dataset logic is separate from the core learning system
- **Flexibility**: Support for various dataset formats without changing core code
- **Extensibility**: Easy to add support for new dataset formats
- **Control**: Fine-grained control over dataset loading and processing
- **Simplicity**: No need to convert datasets to a specific format