#!/usr/bin/env python
"""
Modified dataset_loader.py - Ensuring a universal interface for all dataset loaders
"""

import os
import json
import glob
import random
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple, Optional, Union

class DatasetLoader:
    """Base interface for dataset loaders with standard field names"""

    def __init__(self, 
                 dataset_path: str,
                 shuffle: bool = True, 
                 random_seed: int = 42):
        """
        Initialize the dataset loader

        Args:
            dataset_path: Path to the dataset file or directory
            shuffle: Whether to shuffle examples
            random_seed: Random seed for shuffling
        """
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.examples = []
        self.current_index = 0

        # Validate dataset path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Load examples
        self._load_examples()

        # Shuffle if requested
        if self.shuffle:
            random.seed(self.random_seed)
            random.shuffle(self.examples)

    def _load_examples(self):
        """Load examples from dataset (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _load_examples method")

    def get_examples(self, count: int) -> List[Any]:
        """
        Get a specified number of examples

        Args:
            count: Number of examples to retrieve

        Returns:
            List of examples - Each example must have "question", "answer", and "id" fields
        """
        examples = []
        for _ in range(count):
            if self.current_index >= len(self.examples):
                # Wrap around if we're at the end
                self.current_index = 0

            examples.append(self.examples[self.current_index])
            self.current_index += 1

        return examples

    def get_example_input(self, example: Any) -> str:
        """
        Extract input from an example (standard field is "question")

        Args:
            example: The example to extract input from

        Returns:
            The input portion of the example as a string
        """
        return example.get("question", "")

    def get_example_output(self, example: Any) -> str:
        """
        Extract output from an example (standard field is "answer")

        Args:
            example: The example to extract output from

        Returns:
            The output portion of the example as a string
        """
        return example.get("answer", "")

    def get_total_count(self) -> int:
        """
        Get total number of examples

        Returns:
            Total number of examples
        """
        return len(self.examples)


class ARCDatasetLoader(DatasetLoader):
    """Loader for ARC datasets, ensuring standard field names with improved formatting"""

    def _format_grid(self, grid):
        """
        Format a grid in a more visually readable way

        Args:
            grid: A 2D array (list of lists)

        Returns:
            A formatted string representation of the grid
        """
        formatted = []
        for row in grid:
            formatted.append("[" + ", ".join(str(cell) for cell in row) + "]")
        return "[\n  " + "\n  ".join(formatted) + "\n]"

    def _load_examples(self):
        """Load examples from ARC dataset directory or file with universal field names and improved formatting"""
        if os.path.isdir(self.dataset_path):
            # Directory of JSON files
            json_files = glob.glob(os.path.join(self.dataset_path, "*.json"))
            if not json_files:
                raise ValueError(f"No JSON files found in directory: {self.dataset_path}")

            for file_path in json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        problem_data = json.load(f)

                    problem_id = os.path.basename(file_path).replace(".json", "")

                    # Process each task as a separate example
                    if "train" in problem_data and "test" in problem_data:
                        train_examples = problem_data.get("train", [])
                        test_cases = problem_data.get("test", [])

                        # Only process if we have both training examples and test case
                        if train_examples and test_cases:
                            test_case = test_cases[0]  # Usually there's just one test case

                            # Format the main training example
                            main_example = ""
                            if train_examples:
                                main_example = f"""
                                The following examples demonstrate a transformation rule. The numbers in the grids represent different colors, and your job is to identify the underlying abstract pattern or rule used across all training example pairs and apply it to new test input.

                                0: Black (sometimes interpreted as background/empty)
                                1: Blue
                                2: Red
                                3: Green
                                4: Yellow
                                5: Gray
                                6: Pink
                                7: Orange
                                8: Cyan
                                9: Purple


                                These tasks involve abstract reasoning and visual pattern recognition operations such as:
                                - Pattern recognition and completion
                                - Object identification and manipulation
                                - Spatial transformations (rotation, reflection, translation, movement, groupings, sizes, etc.)
                                - Color/shape transformations
                                - Counting and arithmetic operations
                                - Boolean operations (AND, OR, XOR)

                                First, analyze each example pair carefully and examine the similarities across different example pairs. Look for consistent abstract rules across example pairs that transform each input into its corresponding output. There is a single meta rule or meta transformation sequence that applies to all the example training pairs. Then apply the inferred rule to solve the test case.

                                Explain your reasoning and describe the transformation rule you've identified. Lastly, you MUST provide the output grid as the test output.


                                
                                Example 1:
Input Grid:
{self._format_grid(train_examples[0]['input'])}

Output Grid:
{self._format_grid(train_examples[0]['output'])}"""

                            # Format additional training examples
                            additional_examples = ""
                            for i, example in enumerate(train_examples[1:], 2):
                                additional_examples += f"""
Example {i}:
Input Grid:
{self._format_grid(example['input'])}

Output Grid:
{self._format_grid(example['output'])}"""

                            # Format as a visually structured question
                            question_str = f"""Grid Transformation Task

=== TRAINING EXAMPLES ===

{main_example}{additional_examples}

=== TEST INPUT ===
{self._format_grid(test_case.get('input'))}

Transform the test input according to the pattern shown in the training examples. State the transformation rule explicitly and then provide the output grid.
"""
                            # For the answer, we'll keep the same format for consistency
                            test_output_json = json.dumps(test_case.get("output"), separators=(',', ':'))

                            # Create the example with STANDARD field names
                            task_data = {
                                "id": f"arc_{problem_id}",
                                "question": question_str.strip(),  # Standard field: "question"
                                "answer": test_output_json,        # Standard field: "answer"
                                "meta": {
                                    "source": "ARC",
                                    "filename": os.path.basename(file_path)
                                }
                            }

                            self.examples.append(task_data)
                except Exception as e:
                    print(f"Warning: Error processing {file_path}: {e}")
        else:
            # Single JSON file case - similar logic but for a single file
            try:
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    problem_data = json.load(f)

                problem_id = os.path.basename(self.dataset_path).replace(".json", "")

                # Process the single file as a task
                if "train" in problem_data and "test" in problem_data:
                    train_examples = problem_data.get("train", [])
                    test_cases = problem_data.get("test", [])

                    # Only process if we have both training examples and test case
                    if train_examples and test_cases:
                        test_case = test_cases[0]  # Usually there's just one test case

                        # Format the main training example
                        main_example = ""
                        if train_examples:
                            main_example = f"""Example 1:
Input Grid:
{self._format_grid(train_examples[0]['input'])}

Output Grid:
{self._format_grid(train_examples[0]['output'])}"""

                        # Format additional training examples
                        additional_examples = ""
                        for i, example in enumerate(train_examples[1:], 2):
                            additional_examples += f"""
Example {i}:
Input Grid:
{self._format_grid(example['input'])}

Output Grid:
{self._format_grid(example['output'])}"""

                        # Format as a visually structured question
                        question_str = f"""Grid Transformation Task

=== TRAINING EXAMPLES ===

{main_example}{additional_examples}

=== TEST INPUT ===
{self._format_grid(test_case.get('input'))}

Transform the test input according to the pattern shown in the training examples.
"""
                        # For the answer, we'll keep the same format for consistency
                        test_output_json = json.dumps(test_case.get("output"), separators=(',', ':'))

                        # Create the example with STANDARD field names
                        task_data = {
                            "id": f"arc_{problem_id}",
                            "question": question_str.strip(),  # Standard field: "question"
                            "answer": test_output_json,        # Standard field: "answer"
                            "meta": {
                                "source": "ARC",
                                "filename": os.path.basename(self.dataset_path)
                            }
                        }

                        self.examples.append(task_data)
            except Exception as e:
                raise ValueError(f"Error loading dataset: {e}")

        if not self.examples:
            raise ValueError("No valid examples found in dataset")

        print(f"Loaded {len(self.examples)} examples from ARC dataset")

    # The get_example_input and get_example_output methods are inherited from the base class
    # and already use the standard field names "question" and "answer"


class JSONDatasetLoader(DatasetLoader):
    """Loader for generic JSON datasets with configurable field names using universal interface"""

    def __init__(self, 
                 dataset_path: str,
                 input_field: str = "input",
                 output_field: str = "output",
                 example_prefix: str = None,
                 shuffle: bool = True, 
                 random_seed: int = 42):
        """
        Initialize the JSON dataset loader

        Args:
            dataset_path: Path to the dataset JSON file
            input_field: Name of the field containing input data in the source JSON
            output_field: Name of the field containing output data in the source JSON
            example_prefix: Optional prefix for example keys (e.g., "example_")
            shuffle: Whether to shuffle examples
            random_seed: Random seed for shuffling
        """
        self.input_field = input_field
        self.output_field = output_field
        self.example_prefix = example_prefix
        super().__init__(dataset_path, shuffle, random_seed)

    def _load_examples(self):
        """Load examples from JSON dataset file and convert to universal format"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Dataset JSON must be an object/dictionary")

            # Process all examples
            for key, example in data.items():
                # Skip keys that don't match the prefix if specified
                if self.example_prefix and not key.startswith(self.example_prefix):
                    continue

                # Check if the example has the required fields
                if self.input_field in example and self.output_field in example:
                    # Convert input and output to strings if they're not already
                    input_str = str(example[self.input_field])
                    output_str = str(example[self.output_field])

                    # Store with STANDARD field names
                    self.examples.append({
                        "id": key,
                        "question": input_str,      # Standard field: "question"
                        "answer": output_str,       # Standard field: "answer"
                        "meta": {
                            "source": "json_dataset",
                            "filename": os.path.basename(self.dataset_path),
                            "original_fields": list(example.keys())
                        }
                    })

            if not self.examples:
                raise ValueError(f"No valid examples found with fields '{self.input_field}' and '{self.output_field}'")

            print(f"Loaded {len(self.examples)} examples from JSON dataset")

        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")

    # The get_example_input and get_example_output methods are inherited from the base class
    # and already use the standard field names "question" and "answer"


class CustomDatasetLoader(DatasetLoader):
    """Loader for custom datasets with user-provided extraction functions"""

    def __init__(self, 
                 dataset_path: str,
                 load_examples_fn: Callable[[str], List[Any]],
                 get_input_fn: Callable[[Any], Any],
                 get_output_fn: Callable[[Any], Any],
                 shuffle: bool = True, 
                 random_seed: int = 42):
        """
        Initialize a custom dataset loader with user-provided functions

        Args:
            dataset_path: Path to the dataset
            load_examples_fn: Function to load examples from the dataset
            get_input_fn: Function to extract input from an example
            get_output_fn: Function to extract output from an example
            shuffle: Whether to shuffle examples
            random_seed: Random seed for shuffling
        """
        self.load_examples_fn = load_examples_fn
        self.get_input_fn = get_input_fn
        self.get_output_fn = get_output_fn
        super().__init__(dataset_path, shuffle, random_seed)

    def _load_examples(self):
        """Load examples using the provided function"""
        try:
            raw_examples = self.load_examples_fn(self.dataset_path)
            if not raw_examples:
                raise ValueError("No examples returned by load_examples_fn")

            # Convert raw examples to the universal format
            for i, raw_example in enumerate(raw_examples):
                # Extract using the provided functions
                question = self.get_input_fn(raw_example)
                answer = self.get_output_fn(raw_example)

                # Convert to standard format 
                self.examples.append({
                    "id": f"custom_{i}",
                    "question": str(question),  # Ensure string type for question
                    "answer": str(answer),      # Ensure string type for answer
                    "meta": {
                        "source": "custom_dataset",
                        "original_data": raw_example  # Store original data for reference
                    }
                })

            print(f"Loaded {len(self.examples)} examples using custom loader")

        except Exception as e:
            raise ValueError(f"Error loading examples with custom function: {e}")

    def get_example_input(self, example: Any) -> str:
        """Extract input using the standardized field"""
        return example.get("question", "")

    def get_example_output(self, example: Any) -> str:
        """Extract output using the standardized field"""
        return example.get("answer", "")



# Factory function to create the appropriate loader
def create_dataset_loader(loader_type: str, **kwargs) -> DatasetLoader:
    """
    Create a dataset loader of the specified type

    Args:
        loader_type: Type of loader to create ("arc", "json", or "custom")
        **kwargs: Additional arguments to pass to the loader constructor

    Returns:
        DatasetLoader: An instance of the requested loader type
    """
    if loader_type.lower() == "arc":
        return ARCDatasetLoader(**kwargs)
    elif loader_type.lower() == "json":
        return JSONDatasetLoader(**kwargs)
    elif loader_type.lower() == "custom":
        return CustomDatasetLoader(**kwargs)
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")