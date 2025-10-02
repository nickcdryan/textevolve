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
import csv
import random


class DatasetLoader:
    """Base interface for dataset loaders with standard field names and tool specification"""
    
    # Default evaluator for this dataset type (can be overridden by subclasses)
    default_evaluator = "llm"
    
    # Default tools required for this dataset type (can be overridden by subclasses)
    required_tools = ["call_llm"]  # All datasets need at least call_llm
    
    # Tool-specific configuration (can be overridden by subclasses)
    tool_config = {}

    def __init__(self,
                 dataset_path: str,
                 shuffle: bool = True,
                 random_seed: int = 42,
                 evaluator: str = None):
        """
        Initialize the dataset loader

        Args:
            dataset_path: Path to the dataset file or directory
            shuffle: Whether to shuffle examples
            random_seed: Random seed for shuffling
            evaluator: Evaluator type to use (overrides class default)
        """
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.examples = []
        self.current_index = 0

        # Set evaluator with precedence: param > class default > base default
        if evaluator is not None:
            self.evaluator = evaluator
        elif hasattr(self.__class__, 'default_evaluator') and self.__class__.default_evaluator:
            self.evaluator = self.__class__.default_evaluator
        else:
            self.evaluator = "llm"
            print(f"Warning: No evaluator specified for {self.__class__.__name__}, using 'llm' as default.")
            print(f"  For different evaluation methods, specify --evaluator [llm|f1|exact_match|ticketworld] or see evaluators.py")

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
        raise NotImplementedError(
            "Subclasses must implement _load_examples method")

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

    def get_evaluator(self) -> str:
        """
        Get the evaluator type for this dataset
        
        Returns:
            The evaluator type name (e.g., "llm", "f1", "exact_match")
        """
        return self.evaluator

    def get_total_count(self) -> int:
        """
        Get total number of examples

        Returns:
            Total number of examples
        """
        return len(self.examples)
    
    def get_required_tools(self) -> List[str]:
        """
        Get the list of tools required for this dataset
        
        Returns:
            List of tool names required for this dataset
        """
        return self.required_tools
    
    def get_tool_config(self) -> Dict[str, Any]:
        """
        Get tool-specific configuration for this dataset
        
        Returns:
            Dictionary of tool configuration parameters
        """
        return self.tool_config
    
    def get_tool_categories(self) -> List[str]:
        """
        Get the categories of tools needed for this dataset
        
        Returns:
            List of tool categories (e.g., ["llm"], ["llm", "database", "file"])
        """
        from system_tools import TOOL_REGISTRY
        
        categories = set()
        for tool_name in self.required_tools:
            if tool_name in TOOL_REGISTRY:
                categories.add(TOOL_REGISTRY[tool_name]["category"])
        
        return list(categories)


class ARCDatasetLoader(DatasetLoader):
    """Loader for ARC datasets, ensuring standard field names with improved formatting"""
    
    default_evaluator = "llm"
    required_tools = ["call_llm", "execute_code"]  # ARC only needs LLM calls
    tool_config = {}

    def _format_grid(self, grid):
        """Format a grid in a more visually readable way"""
        formatted = []
        for row in grid:
            formatted.append("[" + ", ".join(str(cell) for cell in row) + "]")
        return "[\n  " + "\n  ".join(formatted) + "\n]"

    def _process_arc_file(self, file_path):
        """Process a single ARC JSON file"""
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
                    test_case = test_cases[
                        0]  # Usually there's just one test case

                    # Format training examples
                    examples_text = ""
                    for i, example in enumerate(train_examples, 1):
                        examples_text += f"""Example {i}:
Input Grid:
{self._format_grid(example['input'])}

Output Grid:
{self._format_grid(example['output'])}

"""

                    # Format as a visually structured question
                    question_str = f"""Grid Transformation Task

=== TRAINING EXAMPLES ===

{examples_text}=== TEST INPUT ===
{self._format_grid(test_case.get('input'))}

Transform the test input according to the pattern shown in the training examples.
"""

                    # Add learnings if available
                    # try:
                    #     with open('learnings.txt', 'r') as l:
                    #         learnings = l.read()
                    #     question_str = "\n\n Here are the learnings from previous iterations: \n\n" + learnings + "\n\n" + question_str
                    # except:
                    #     pass

                    # For the answer, keep the same format for consistency
                    test_output_json = json.dumps(test_case.get("output"),
                                                  separators=(',', ':'))

                    # Create the example with STANDARD field names
                    task_data = {
                        "id": f"arc_{problem_id}",
                        "question": question_str.strip(),
                        "answer": test_output_json,
                        "meta": {
                            "source": "ARC",
                            "filename": os.path.basename(file_path)
                        }
                    }

                    return task_data
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")

        return None

    def _load_examples(self):
        """Load examples from ARC dataset directory or file"""
        examples = []

        if os.path.isdir(self.dataset_path):
            # Directory of JSON files
            json_files = glob.glob(os.path.join(self.dataset_path, "*.json"))
            if not json_files:
                raise ValueError(
                    f"No JSON files found in directory: {self.dataset_path}")

            for file_path in json_files:
                task_data = self._process_arc_file(file_path)
                if task_data:
                    examples.append(task_data)
        else:
            # Single JSON file
            task_data = self._process_arc_file(self.dataset_path)
            if task_data:
                examples.append(task_data)

        if not examples:
            raise ValueError("No valid examples found in dataset")

        self.examples = examples
        print(f"Loaded {len(self.examples)} examples from ARC dataset")

    # The get_example_input and get_example_output methods are inherited from the base class
    # and already use the standard field names "question" and "answer"


class HotpotQADatasetLoader(DatasetLoader):
    """Loader specifically for HotpotQA multi-hop reasoning datasets"""
    
    default_evaluator = "llm"
    required_tools = ["call_llm"]  # HotpotQA only needs LLM calls
    tool_config = {}

    def _load_examples(self):
        """Load examples from HotpotQA JSON dataset file and convert to universal format"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(
                    "HotpotQA dataset JSON must be a list of objects")

            examples = []
            for item in data:
                # Extract basic fields
                original_question = item.get("question", "")
                answer = item.get("answer", "")
                example_id = item.get("id", f"hotpotqa_{len(examples)}")

                # Skip examples that don't have required fields
                if not original_question or not answer:
                    print(
                        f"Warning: Skipping {example_id} - missing question or answer"
                    )
                    continue

                # Extract and format context
                context_data = item.get("context", {})
                titles = context_data.get("title", [])
                sentences_lists = context_data.get("sentences", [])

                if not titles or not sentences_lists:
                    print(f"Warning: Skipping {example_id} - missing context")
                    continue

                # Format the supporting documents
                formatted_context = ""
                for i, (title,
                        sentences) in enumerate(zip(titles, sentences_lists)):
                    formatted_context += f"\n=== Document {i+1}: {title} ===\n"
                    for j, sentence in enumerate(sentences):
                        formatted_context += f"{sentence.strip()} "
                    formatted_context += "\n"

                # Create the structured question format
                structured_question = f"""Multi-hop reasoning task:

Question: {original_question}

Supporting Documents:{formatted_context}

Provide your answer based on the information in the supporting documents."""

                # Create standardized example with universal field names
                standardized_example = {
                    "id": example_id,
                    "question":
                    structured_question.strip(),  # Standard field: "question"
                    "answer": str(answer).strip(),  # Standard field: "answer"
                    "meta": {
                        "source": "hotpotqa",
                        "filename": self.dataset_path,
                        "type": item.get("type", "unknown"),
                        "level": item.get("level", "unknown"),
                        "original_question":
                        original_question,  # Keep original for reference
                        "num_documents": len(titles)
                    }
                }

                examples.append(standardized_example)

            self.examples = examples
            print(f"Loaded {len(examples)} examples from HotpotQA dataset")

            if not self.examples:
                raise ValueError("No valid examples found in HotpotQA dataset")

        except Exception as e:
            raise ValueError(f"Error loading HotpotQA dataset: {e}")


class JSONDatasetLoader(DatasetLoader):
    """Loader for generic JSON datasets with configurable field names using universal interface"""
    
    default_evaluator = "llm"
    required_tools = ["call_llm"]  # Generic JSON datasets typically only need LLM
    tool_config = {}

    def __init__(self,
                 dataset_path: str,
                 input_field: str = "input",
                 output_field: str = "output",
                 example_prefix: str = None,
                 shuffle: bool = True,
                 random_seed: int = 42,
                 evaluator: str = None):
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
        super().__init__(dataset_path, shuffle, random_seed, evaluator)

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
                if self.example_prefix and not key.startswith(
                        self.example_prefix):
                    continue

                # Check if the example has the required fields
                if self.input_field in example and self.output_field in example:
                    # Convert input and output to strings if they're not already
                    input_str = str(example[self.input_field])
                    output_str = str(example[self.output_field])

                    # Store with STANDARD field names
                    self.examples.append({
                        "id": key,
                        "question": input_str,  # Standard field: "question"
                        "answer": output_str,  # Standard field: "answer"
                        "meta": {
                            "source": "json_dataset",
                            "filename": os.path.basename(self.dataset_path),
                            "original_fields": list(example.keys())
                        }
                    })

            if not self.examples:
                raise ValueError(
                    f"No valid examples found with fields '{self.input_field}' and '{self.output_field}'"
                )

            print(f"Loaded {len(self.examples)} examples from JSON dataset")

        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")

    # The get_example_input and get_example_output methods are inherited from the base class
    # and already use the standard field names "question" and "answer"


class CustomDatasetLoader(DatasetLoader):
    """Loader for custom datasets with user-provided extraction functions"""
    
    default_evaluator = "llm"

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
                    "question":
                    str(question),  # Ensure string type for question
                    "answer": str(answer),  # Ensure string type for answer
                    "meta": {
                        "source": "custom_dataset",
                        "original_data":
                        raw_example  # Store original data for reference
                    }
                })

            print(f"Loaded {len(self.examples)} examples using custom loader")

        except Exception as e:
            raise ValueError(
                f"Error loading examples with custom function: {e}")

    def get_example_input(self, example: Any) -> str:
        """Extract input using the standardized field"""
        return example.get("question", "")

    def get_example_output(self, example: Any) -> str:
        """Extract output using the standardized field"""
        return example.get("answer", "")


class JSONLDatasetLoader(DatasetLoader):
    """Loader for JSONL datasets with configurable field mapping
    Used for DROP"""
    
    default_evaluator = "llm"

    def __init__(
            self,
            dataset_path: str,
            input_field: str = "question",
            output_field: str = "answers_spans",
            passage_field: str = "passage",
            answer_extraction:
        str = "spans",  # Field within answers_spans to extract
            shuffle: bool = True,
            random_seed: int = 42,
            **kwargs):  # Added **kwargs to accept any additional parameters
        """
        Initialize the JSONL dataset loader

        Args:
            dataset_path: Path to the dataset JSONL file
            input_field: Field name containing the question
            output_field: Field name containing the answer data
            passage_field: Field name containing the context passage
            answer_extraction: Key for extracting the answer from answer_field (e.g., 'spans')
            shuffle: Whether to shuffle examples
            random_seed: Random seed for shuffling
            **kwargs: Additional arguments that might be passed
        """
        self.input_field = input_field
        self.output_field = output_field
        self.passage_field = passage_field
        self.answer_extraction = answer_extraction
        # Extract evaluator from kwargs if present
        evaluator = kwargs.get('evaluator', None)
        super().__init__(dataset_path, shuffle, random_seed, evaluator)

    def _load_examples(self):
        """Load examples from JSONL dataset file and convert to universal format"""
        import json

        try:
            # JSONL format has one JSON object per line
            examples = []
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse the JSON object from this line
                        example = json.loads(line)

                        # Extract passage and question
                        passage = example.get(self.passage_field, "")
                        question = example.get(self.input_field, "")

                        # For the DROP dataset, answers are in a nested structure
                        # Extract the answer based on the specified extraction method
                        answer_data = example.get(self.output_field, {})

                        # Get all answer spans (instead of just the first one)
                        answer = ""
                        if isinstance(
                                answer_data, dict
                        ) and self.answer_extraction in answer_data:
                            spans = answer_data.get(self.answer_extraction, [])
                            if spans and isinstance(spans, list):
                                # Join all spans with a comma and space instead of taking just the first item
                                answer = ", ".join(spans)
                            else:
                                answer = str(spans)

                        # Combine passage and question for the standard "question" field
                        formatted_question = f"PASSAGE: {passage}\n\nQUESTION: {question}"

                        # Create standardized example with universal field names
                        examples.append({
                            "id":
                            example.get("query_id", f"example_{line_num}"),
                            "question":
                            formatted_question,  # Standard field: "question"
                            "answer":
                            answer,  # Standard field: "answer"
                            "meta": {
                                "source": "jsonl_dataset",
                                "original_passage": passage,
                                "original_question": question,
                                "original_answer_data": answer_data,
                                "line_number": line_num
                            }
                        })

                    except json.JSONDecodeError:
                        print(
                            f"Warning: Invalid JSON on line {line_num+1}, skipping"
                        )
                    except Exception as e:
                        print(
                            f"Warning: Error processing line {line_num+1}: {e}"
                        )

            self.examples = examples
            print(f"Loaded {len(examples)} examples from JSONL dataset")

            if not self.examples:
                raise ValueError("No valid examples found in dataset")

        except Exception as e:
            raise ValueError(f"Error loading JSONL dataset: {e}")


"""
simpleqa_loader.py - Custom dataset loader for SimpleQA dataset
"""


class SimpleQADatasetLoader(DatasetLoader):
    """Loader specifically for SimpleQA datasets with 'problem', 'answer', and 'id' fields"""
    
    default_evaluator = "llm"

    def _load_examples(self):
        """Load examples from SimpleQA JSONL dataset file"""
        try:
            examples = []
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse the JSON object from this line
                        data = json.loads(line)

                        # Extract the required fields
                        problem = data.get("problem", "")
                        answer = data.get("answer", "")
                        example_id = data.get("id", f"simpleqa_{line_num}")

                        # Create standardized example with universal field names
                        examples.append({
                            "id": example_id,
                            "question": problem,  # Standard field: "question"
                            "answer":
                            str(answer
                                ),  # Standard field: "answer" (ensure string)
                            "meta": {
                                "source": "SimpleQA",
                                "line_number": line_num,
                                "original_data": data
                            }
                        })

                    except json.JSONDecodeError:
                        print(
                            f"Warning: Invalid JSON on line {line_num+1}, skipping"
                        )
                    except Exception as e:
                        print(
                            f"Warning: Error processing line {line_num+1}: {e}"
                        )

            self.examples = examples
            print(f"Loaded {len(examples)} examples from SimpleQA dataset")

            if not self.examples:
                raise ValueError("No valid examples found in SimpleQA dataset")

        except Exception as e:
            raise ValueError(f"Error loading SimpleQA dataset: {e}")


class MathDatasetLoader(DatasetLoader):
    """Loader specifically for Hendrycks Math datasets with 'problem', 'answer', and 'id' fields"""
    
    default_evaluator = "llm"
    required_tools = ["call_llm", "execute_code"]  # Math problems may benefit from code execution
    tool_config = {}

    def _load_examples(self):
        """Load examples from Math JSONL dataset file"""
        try:
            examples = []
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse the JSON object from this line
                        data = json.loads(line)

                        # Extract the required fields
                        problem = data.get("problem", "")
                        answer = data.get("solution", "")
                        example_id = data.get("id", f"math_{line_num}")
                        problem_type = data.get("type", "")

                        # Create standardized example with universal field names
                        examples.append({
                            "id": example_id,
                            "question": problem,  # Standard field: "question"
                            "answer":
                            str(answer
                                ),  # Standard field: "answer" (ensure string)
                            "meta": {
                                "source": "Math",
                                "line_number": line_num,
                                "original_data": data,
                                "problem_type": problem_type,
                            }
                        })

                    except json.JSONDecodeError:
                        print(
                            f"Warning: Invalid JSON on line {line_num+1}, skipping"
                        )
                    except Exception as e:
                        print(
                            f"Warning: Error processing line {line_num+1}: {e}"
                        )

            self.examples = examples
            print(f"Loaded {len(examples)} examples from Math dataset")

            if not self.examples:
                raise ValueError("No valid examples found in Math dataset")

        except Exception as e:
            raise ValueError(f"Error loading Math dataset: {e}")


class NaturalPlanDatasetLoader(DatasetLoader):
    """Loader specifically for Natural Plan trip planning datasets"""
    
    default_evaluator = "llm"
    required_tools = ["call_llm"] 

    def _load_examples(self):
        """Load examples from Natural Plan dataset file and convert to universal format"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError(
                    "Natural Plan dataset JSON must be an object/dictionary")

            examples = []
            for example_key, example_data in data.items():
                # Skip if this doesn't look like a trip planning example
                if not isinstance(example_data, dict):
                    continue

                # Extract the question (problem statement) and answer (golden plan)
                question = example_data.get("prompt_0shot", "")
                # try:
                #     with open('learnings.txt', 'r') as l:
                #         learnings = l.read()
                #     question += "\n\n Here are the learnings from previous iterations: \n\n" + learnings
                # except:
                #     pass
                answer = example_data.get("golden_plan", "")

                # with open('learnings.txt', 'r') as l:
                #     learnings = l.read()
                # question += "\n\n Here are the learnings from previous iterations: \n\n" + learnings

                # Skip examples that don't have both required fields
                if not question or not answer:
                    print(
                        f"Warning: Skipping {example_key} - missing prompt_0shot or golden_plan"
                    )
                    continue

                # Create standardized example with universal field names
                standardized_example = {
                    "id": example_key,
                    "question": question.strip(),  # Standard field: "question"
                    "answer":
                    answer,  #.strip(),      # Standard field: "answer"
                    "meta": {
                        "source": "natural_plan",
                        "filename": self.dataset_path,
                        #"num_cities": example_data.get("num_cities", ""),
                        #"cities": example_data.get("cities", ""),
                        #"durations": example_data.get("durations", ""),
                        #"has_5shot_prompt": "prompt_5shot" in example_data,
                        #"has_prediction": "pred_5shot_pro" in example_data
                    }
                }

                examples.append(standardized_example)

            self.examples = examples
            print(f"Loaded {len(examples)} examples from Natural Plan dataset")

            if not self.examples:
                raise ValueError(
                    "No valid examples found in Natural Plan dataset")

        except Exception as e:
            raise ValueError(f"Error loading Natural Plan dataset: {e}")


class GPQADatasetLoader(DatasetLoader):
    """Loader specifically for GPQA datasets with multiple choice questions"""
    
    default_evaluator = "llm"
    required_tools = ["call_llm"]  # GPQA only needs LLM calls
    tool_config = {}

    def __init__(self,
                 dataset_path: str,
                 shuffle_choices: bool = True,
                 **kwargs):
        """
        Initialize GPQA dataset loader

        Args:
            dataset_path: Path to GPQA CSV file
            shuffle_choices: Whether to shuffle answer choices to prevent bias
            **kwargs: Other arguments passed to parent class (shuffle, random_seed, etc.)
        """
        self.shuffle_choices = shuffle_choices
        super().__init__(dataset_path, **kwargs)

    def _load_examples(self):
        """Load examples from GPQA CSV file with shuffled answer choices"""
        # Set random seed for reproducible choice shuffling
        # Use a separate random instance to avoid interfering with dataset shuffling
        choice_random = random.Random(self.random_seed) if hasattr(
            self, 'random_seed') else random.Random(42)

        try:
            with open(self.dataset_path, 'r', encoding='utf-8',
                      newline='') as csvfile:
                reader = csv.DictReader(csvfile)

                for row_num, row in enumerate(reader):
                    question = row.get('Question', '').strip()
                    # try:
                    #     with open('learnings.txt', 'r') as l:
                    #         learnings = l.read()
                    #     question = "\n\n Here are the learnings from previous iterations: \n\n" + learnings + question
                    # except:
                    #     pass
                    correct_answer = row.get('Correct Answer', '').strip()
                    incorrect_1 = row.get('Incorrect Answer 1', '').strip()
                    incorrect_2 = row.get('Incorrect Answer 2', '').strip()
                    incorrect_3 = row.get('Incorrect Answer 3', '').strip()

                    # Skip empty rows or rows with missing answers
                    if not question or not correct_answer or not all(
                        [incorrect_1, incorrect_2, incorrect_3]):
                        print(
                            f"Warning: Skipping row {row_num + 1} due to missing data"
                        )
                        continue

                    # Create list of all answers with their types
                    all_answers = [(correct_answer, 'correct'),
                                   (incorrect_1, 'incorrect'),
                                   (incorrect_2, 'incorrect'),
                                   (incorrect_3, 'incorrect')]

                    # Shuffle the answers to prevent bias (if enabled)
                    if self.shuffle_choices:
                        choice_random.shuffle(all_answers)

                    # Find which position the correct answer ended up in
                    correct_position = None
                    for i, (answer_text,
                            answer_type) in enumerate(all_answers):
                        if answer_type == 'correct':
                            correct_position = ['A', 'B', 'C', 'D'][i]
                            break

                    # Format the question with answer choices (prompt template from openai simple eval)
                    formatted_question = f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {all_answers[0][0]}
B) {all_answers[1][0]}
C) {all_answers[2][0]}
D) {all_answers[3][0]}
"""

                    # Create standardized example
                    example = {
                        "id": f"gpqa_{row_num}",
                        "question":
                        formatted_question,  # Standard field: "question"
                        "answer":
                        correct_position,  # Standard field: "answer" - now the letter (A/B/C/D)
                        "meta": {
                            "source":
                            "GPQA",
                            "filename":
                            os.path.basename(self.dataset_path),
                            "row_number":
                            row_num + 1,
                            "correct_answer_text":
                            correct_answer,  # Store the original text
                            "incorrect_answers":
                            [incorrect_1, incorrect_2,
                             incorrect_3],  # Store incorrect answers
                            "all_answer_choices":
                            [answer[0] for answer in all_answers
                             ],  # Store all choices in order
                            "correct_position":
                            correct_position,  # Store which letter is correct
                            "shuffled":
                            self.
                            shuffle_choices  # Track if choices were shuffled
                        }
                    }

                    self.examples.append(example)

            if not self.examples:
                raise ValueError("No valid examples found in GPQA dataset")

            print(f"Loaded {len(self.examples)} examples from GPQA dataset")
            if self.shuffle_choices:
                print("Answer choices are shuffled to prevent bias")

        except Exception as e:
            raise ValueError(f"Error loading GPQA dataset: {e}")


class MedMCQADatasetLoader(DatasetLoader):
    """Loader specifically for MedMCQA medical multiple choice datasets"""
    
    default_evaluator = "llm"
    required_tools = ["call_llm"]  # MedMCQA only needs LLM calls
    tool_config = {}

    def __init__(self,
                 dataset_path: str,
                 shuffle_choices: bool = True,
                 **kwargs):
        """
        Initialize MedMCQA dataset loader

        Args:
            dataset_path: Path to MedMCQA JSON/JSONL file
            shuffle_choices: Whether to shuffle answer choices to prevent bias
            **kwargs: Other arguments passed to parent class (shuffle, random_seed, etc.)
        """
        self.shuffle_choices = shuffle_choices
        super().__init__(dataset_path, **kwargs)

    def _load_examples(self):
        """Load examples from MedMCQA dataset file with shuffled answer choices"""
        # Set random seed for reproducible choice shuffling
        choice_random = random.Random(self.random_seed) if hasattr(
            self, 'random_seed') else random.Random(42)

        try:
            examples = []
            
            # Handle both JSON and JSONL formats
            if self.dataset_path.endswith('.jsonl'):
                # JSONL format - one JSON object per line
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            example = self._process_medmcqa_item(data, line_num, choice_random)
                            if example:
                                examples.append(example)
                        except json.JSONDecodeError:
                            print(f"Warning: Invalid JSON on line {line_num+1}, skipping")
                        except Exception as e:
                            print(f"Warning: Error processing line {line_num+1}: {e}")
            else:
                # JSON format - list of objects or single object
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        example = self._process_medmcqa_item(item, idx, choice_random)
                        if example:
                            examples.append(example)
                elif isinstance(data, dict):
                    example = self._process_medmcqa_item(data, 0, choice_random)
                    if example:
                        examples.append(example)
                else:
                    raise ValueError("MedMCQA dataset must be a JSON object or array")

            self.examples = examples
            
            if not self.examples:
                raise ValueError("No valid examples found in MedMCQA dataset")

            print(f"Loaded {len(self.examples)} examples from MedMCQA dataset")
            if self.shuffle_choices:
                print("Answer choices are shuffled to prevent bias")

        except Exception as e:
            raise ValueError(f"Error loading MedMCQA dataset: {e}")

    def _process_medmcqa_item(self, item, item_num, choice_random):
        """Process a single MedMCQA item"""
        question = item.get('question', '').strip()
        option_a = item.get('opa', '').strip()
        option_b = item.get('opb', '').strip()
        option_c = item.get('opc', '').strip()
        option_d = item.get('opd', '').strip()
        correct_option_index = item.get('cop', -1)  # 0=A, 1=B, 2=C, 3=D
        
        # Skip items with missing data
        if not question or not all([option_a, option_b, option_c, option_d]) or correct_option_index < 0 or correct_option_index > 3:
            print(f"Warning: Skipping item {item_num} due to missing or invalid data")
            return None

        # Create list of all answers with their correctness
        all_answers = [
            (option_a, 0 == correct_option_index),
            (option_b, 1 == correct_option_index),
            (option_c, 2 == correct_option_index),
            (option_d, 3 == correct_option_index)
        ]

        # Shuffle the answers to prevent bias (if enabled)
        if self.shuffle_choices:
            choice_random.shuffle(all_answers)

        # Find which position the correct answer ended up in
        correct_position = None
        for i, (answer_text, is_correct) in enumerate(all_answers):
            if is_correct:
                correct_position = ['A', 'B', 'C', 'D'][i]
                break

        # Format the question with answer choices
        formatted_question = f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {all_answers[0][0]}
B) {all_answers[1][0]}
C) {all_answers[2][0]}
D) {all_answers[3][0]}
"""

        # Create standardized example
        example = {
            "id": item.get("id", f"medmcqa_{item_num}"),
            "question": formatted_question,  # Standard field: "question"
            "answer": correct_position,  # Standard field: "answer" - the letter (A/B/C/D)
            "meta": {
                "source": "MedMCQA",
                "filename": os.path.basename(self.dataset_path),
                "item_number": item_num,
                "original_question": question,
                "original_options": {
                    "A": option_a,
                    "B": option_b, 
                    "C": option_c,
                    "D": option_d
                },
                "original_correct_index": correct_option_index,
                "correct_position": correct_position,
                "subject_name": item.get("subject_name", ""),
                "topic_name": item.get("topic_name", ""),
                "explanation": item.get("exp", ""),
                "choice_type": item.get("choice_type", "single"),
                "shuffled": self.shuffle_choices
            }
        }

        return example


class TicketWorldDatasetLoader(DatasetLoader):
    """Loader specifically for TicketWorld customer service datasets"""
    
    default_evaluator = "ticketworld"
    required_tools = [
        "call_llm",
        "read_query", 
        "write_query",
        "list_tables",
        "describe_table",
        "read_file",
        "search_file"
    ]
    tool_config = {
        "database_path": "datasets/ticketworld/customer_database.db",
        "policy_files": ["datasets/ticketworld/company_policy.txt"],
        "database_schema": {
            "customers": "Customer information with IDs, emails, addresses",
            "orders": "Order details with items, status, tracking",
            "products": "Product catalog with pricing, warranties"
        }
    }

    def _load_examples(self):
        """Load examples from TicketWorld JSON dataset file"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("TicketWorld dataset JSON must be a list of objects")

            examples = []
            for item in data:
                # Extract customer email fields for the question
                customer_email = item.get("customer_email", "")
                subject = item.get("subject", "")
                body = item.get("body", "")
                timestamp = item.get("timestamp", "")
                ticket_id = item.get("ticket_id", f"ticket_{len(examples)}")
                
                # Skip examples that don't have required fields
                if not customer_email or not subject or not body:
                    print(f"Warning: Skipping {ticket_id} - missing customer email, subject, or body")
                    continue

                # Extract resolution plan for the answer
                resolution_plan = item.get("resolution_plan", {})
                if not resolution_plan:
                    print(f"Warning: Skipping {ticket_id} - missing resolution_plan")
                    continue

                # Format the question from customer email components
                formatted_question = f"""# Customer Service Resolution Plan Instructions

## Overview
When you receive a customer email, create a comprehensive resolution plan following our standardized schema. Analyze the customer's issue, look up relevant information in our databases, and determine the appropriate response based on company policies.
Note: it is impossible to accurately resolve the customer's issue without multiple steps of reasoning and retrieving the relevant information from the provided database and policy document.
Therefore you must use these tools.

🔥 CRITICAL: YOU MUST OUTPUT THE RESOLUTION PLAN IN DEMONSTRATED FORMAT. DO NOT OUTPUT ANYTHING EXCEPT FOR THE RESOLUTION PLAN.

🔥 CRITICAL: FILE ACCESS AND DATABASE ACCESS CAPABILITIES AVAILABLE 🔥

You have access to powerful file reading and searching functions for accessing local files safely.


## Required Resources
You have access to these files located at datasets/ticketworld/:
- **customer_database.db** - SQLite database with customer, order, and product information
- **company_policy.txt** - Complete company policies with policy IDs and rules


You have advanced database and file access functions - YOU MUST USE THESE! 

## Enhanced Database Functions (Recommended):
- **read_query(sql_query)** - Execute SELECT queries with automatic connection to TicketWorld database
- **write_query(sql_query)** - Execute INSERT/UPDATE/DELETE queries 
- **list_tables()** - Show all available database tables
- **describe_table(table_name)** - Get detailed table schema information

## File Access Functions:
- **read_file(path_to_file)** - Read file contents with optional line ranges
- **search_file(path_to_file, pattern)** - Search for patterns within files

## Legacy Database Function (still supported):
- **call_database(path_to_database, SQL_query)** - Original database access method

EXAMPLE: HOW TO USE read_query() (RECOMMENDED):
```python
# Find customer by email (auto-connects to TicketWorld database)
customer = read_query("SELECT customer_id, name FROM customers WHERE primary_email = 'customer@email.com'")

# Get customer's recent orders
orders = read_query("SELECT order_id, order_date, total_amount FROM orders WHERE customer_id = 'CUST-0001' ORDER BY order_date DESC LIMIT 5")

# Get complete order details
order_details = read_query("SELECT * FROM orders WHERE order_id = 'ORD-20250224-1008'")

# Explore database if needed
tables = list_tables()                           # See all tables
customer_schema = describe_table("customers")    # Get table structure
```

EXAMPLE: HOW TO USE call_database() (LEGACY):
```python
# Legacy method - still works but requires full database path
customers = call_database("datasets/ticketworld/customer_database.db", "SELECT name, email FROM customers WHERE customer_id = 'CUST-0001'")
```

HOW TO USE read_file():
```python
# Read entire file
content = read_file("config.txt")

# Read specific line range (1-based line numbers)
content = read_file("large_file.log", start_line=100, end_line=200)

# Read from start to specific line
content = read_file("script.py", end_line=50)

# Read from specific line to end
content = read_file("data.txt", start_line=25)
```

DO NOT TRY TO CALL_LLM TO READ THESE FILES. YOU MUST USE THE PROVIDED REAL ASSETS. CALLING THE LLM TO READ FILES OR WITH DATABASE QUERIES IS FORBIDDEN.

🔥 CRITICAL: YOU MUST USE THESE TOOLS TO ACCESS THE DATABASES AND FILES: THEY ARE REQUIRED TO SOLVE THE TICKET 🔥

## Database Schema (customer_database.db)

### Table: customers
- customer_id (TEXT, PRIMARY KEY): Format CUST-XXXX
- name (TEXT): Customer full name
- primary_email (TEXT): Primary email address
- alternate_email (TEXT): Secondary email address
- phone (TEXT): Phone number
- shipping_street, shipping_city, shipping_state, shipping_zip (TEXT): Shipping address
- billing_street, billing_city, billing_state, billing_zip (TEXT): Billing address
- created_date (DATE): Account creation date

### Table: products
- product_id (TEXT, PRIMARY KEY): Format PROD-XXXX
- name (TEXT): Product name
- category, brand (TEXT): Product classification
- base_price (DECIMAL): Product price
- warranty_period (INTEGER): Warranty days
- weight (DECIMAL): Product weight
- requires_signature (BOOLEAN): Delivery requirement
- in_stock (BOOLEAN): Availability status
- description (TEXT): Product description

### Table: orders
- order_id (TEXT, PRIMARY KEY): Format ORD-YYYYMMDD-XXXX
- customer_id (TEXT): Links to customers table
- order_date (DATE): Order placement date
- items (TEXT): JSON array of order items
- shipping_method (TEXT): Shipping method used
- tracking_number (TEXT): Package tracking
- total_amount (DECIMAL): Total order amount
- payment_method (TEXT): Payment method
- order_status (TEXT): Current status

### Order Items JSON Structure
```
[
  {{
    "product_id": "PROD-XXXX",
    "quantity": 1,
    "price_paid": 99.99,
    "item_status": "delivered"
  }}
]
```

## Resolution Plan Schema

### Main Structure
```
{{
  "order_id": "string",              // Order ID or "N/A"
  "order_date": "string",            // "YYYY-MM-DD" or "N/A"
  "customer_lookup": {{object}},       // Customer identification
  "policy_references": [string],     // Array of policy IDs
  "policy_reasoning": "string",      // Detailed explanation
  "actions": [object],               // Array of actions
  "escalation_required": boolean,    // true/false
  "escalation_reason": "string",     // null if no escalation
  "priority": "enum",                // Priority level
  "total_resolution_value": number   // Sum of action values
}}
```

### Customer Lookup Object
```
{{
  "status": "found|not_found",
  "customer_id": "string",
  "lookup_method": "email_match",
  "notes": "string"
}}
```

### Action Object
```
{{
  "type": "enum",         // See action types below
  "reason": "string",     // Policy citation
  "value": number,        // Dollar amount (0 for denials)
  "details": "string"     // Implementation details
}}
```

### Action Types (Complete List)
- process_return
- send_replacement
- provide_tracking
- honor_warranty
- request_photo
- update_shipping_address
- cancel_order
- deny_return
- deny_refund
- deny_warranty_claim
- deny_price_match
- deny_order_modification
- deny_cancellation
- provide_product_information
- initiate_investigation
- process_exchange
- deny_exchange
- honor_price_match
- accept_order_modification


### Priority Levels
- low
- medium
- high
- urgent

## Process Steps
1. **Customer Lookup**: Search database by email, record customer_id and status
2. **Order Information**: Extract order_id and order_date if applicable, use "N/A" if none
3. **Policy Research**: Find all relevant policies from company_policy.txt
4. **Policy Reasoning**: Explain which policies apply and how they interact
5. **Determine Actions**: Choose appropriate actions with monetary values and detailed instructions
6. **Escalation Assessment**: Check for high-value items (>$500), complex interactions, or unusual circumstances
7. **Priority Assignment**: Set based on urgency and issue type
8. **Calculate Total**: Sum all monetary values from actions

## Key Requirements
- Reference specific policy IDs (e.g., "POL-WARRANTY-001")
- Use exact dollar amounts from product prices
- Provide detailed reasoning connecting issue to policies
- When in doubt, escalate rather than assume 

---
## HERE IS THE CURRENT CUSTOMER EMAIL TO PROCESS
## Customer Email 

Customer Email: {customer_email}
Subject: {subject}
Timestamp: {timestamp}

Message Body:
{body}

Please provide a resolution plan for this customer service ticket following the schema and instructions above."""

                # Convert resolution plan to JSON string for the answer
                answer = json.dumps(resolution_plan, indent=2)

                # Create standardized example with universal field names
                standardized_example = {
                    "id": ticket_id,
                    "question": formatted_question.strip(),  # Standard field: "question"
                    "answer": answer,  # Standard field: "answer"
                    "meta": {
                        "source": "ticketworld",
                        "filename": os.path.basename(self.dataset_path),
                        "customer_id": item.get("customer_id", ""),
                        "order_id": item.get("order_id", ""),
                        "original_customer_email": customer_email,
                        "original_subject": subject,
                        "original_body": body,
                        "original_timestamp": timestamp
                    }
                }

                examples.append(standardized_example)

            self.examples = examples
            print(f"Loaded {len(examples)} examples from TicketWorld dataset")

            if not self.examples:
                raise ValueError("No valid examples found in TicketWorld dataset")

        except Exception as e:
            raise ValueError(f"Error loading TicketWorld dataset: {e}")


class TicketWorldSimpleDatasetLoader(DatasetLoader):
    """TicketWorld customer service resolution dataset loader with comprehensive task instructions"""
    
    default_evaluator = "ticketworld"
    required_tools = [
        "call_llm",
        "read_query", 
        "write_query",
        "list_tables",
        "describe_table",
        "read_file",
        "search_file"
    ]
    tool_config = {
        "database_path": "datasets/ticketworld/customer_database.db",
        "policy_files": ["datasets/ticketworld/company_policy.txt"],
        "database_schema": {
            "customers": "Customer information with IDs, emails, addresses",
            "orders": "Order details with items, status, tracking",
            "products": "Product catalog with pricing, warranties"
        }
    }

    def _load_examples(self):
        """Load examples from TicketWorld JSON dataset file with comprehensive resolution task"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("TicketWorld dataset JSON must be a list of objects")

            examples = []
            for item in data:
                # Extract customer email fields for the question
                customer_email = item.get("customer_email", "")
                subject = item.get("subject", "")
                body = item.get("body", "")
                timestamp = item.get("timestamp", "")
                ticket_id = item.get("ticket_id", f"ticket_{len(examples)}")
                
                # Skip examples that don't have required fields
                if not customer_email or not subject or not body:
                    print(f"Warning: Skipping {ticket_id} - missing customer email, subject, or body")
                    continue

                # Get the expected resolution from the original data
                resolution_plan = item.get("resolution_plan", {})
                if not resolution_plan:
                    print(f"Warning: Skipping {ticket_id} - missing resolution_plan")
                    continue

                # Format the comprehensive question with all instructions
                formatted_question = f"""# Customer Service Resolution System

## TASK OVERVIEW
You are a customer service resolution system with advanced database access capabilities. Your job is to analyze incoming support tickets and create a resolution plan. This process requires:

1. **Analyze Customer Issue**: Read the customer email and subject to understand their problem
2. **Find Customer**: Use the email address to look up customer information in the database
3. **Locate Orders**: Find relevant orders for the customer
4. **Get Order Details**: Retrieve complete order information
5. **Research Policies**: Read company policies to understand applicable rules
6. **Create Resolution**: Determine appropriate actions based on data and policies
7. **Assess Escalation**: Decide if manager escalation is required

## REQUIRED OUTPUT FIELDS
Your primary objective is to correctly determine these core fields:
- **order_id**: The relevant order identifier  
- **customer_id**: The customer identifier from database lookup
- **actions[type]**: List of action types to resolve the issue
- **escalation_required**: Boolean decision on whether escalation is needed
- **policy_references**: List of policy IDs that apply to this case

🚨 **CRITICAL**: This task is IMPOSSIBLE without using the provided database and file tools.
🚨 **CRITICAL**: Customer emails contain limited information - you MUST use the database and policy documents to gather complete context before making decisions.
All customers and orders exist in the database - there are no missing records.

## ENHANCED DATABASE TOOLS
You have access to reliable, simplified database functions:

### Primary Database Functions (Recommended)
# Find customer by email (auto-connects to TicketWorld database)
customer_data = read_query("SELECT customer_id, name FROM customers WHERE primary_email = 'customer@email.com' OR alternate_email = 'customer@email.com'")

# Get customer's orders
orders = read_query("SELECT order_id, order_date, total_amount FROM orders WHERE customer_id = 'CUST-0001' ORDER BY order_date DESC")

# Get specific order details
order_details = read_query("SELECT * FROM orders WHERE order_id = 'ORD-20250224-1008'")

# Explore database structure (if needed)
tables = list_tables()                    # See available tables
schema = describe_table("customers")      # Get table structure

### Policy Document Access  
policy_text = read_file("datasets/ticketworld/company_policy.txt")

### Key Benefits of New Database Tools:
✅ **Automatic Connection**: No need to specify database path
✅ **Error Handling**: Clear error messages and validation
✅ **Consistent Results**: Reliable data formatting
✅ **Query Safety**: Built-in SQL validation
✅ **Type Safety**: Proper handling of different data types

🔥 **YOU MUST USE THESE TOOLS** - Accurate resolution requires database lookup and policy research.

## DATABASE SCHEMA

### Table: customers
- customer_id (TEXT, PRIMARY KEY): Format CUST-XXXX
- name (TEXT): Customer full name  
- primary_email (TEXT): Primary email address
- alternate_email (TEXT): Secondary email address
- phone (TEXT): Phone number
- shipping_street, shipping_city, shipping_state, shipping_zip (TEXT): Shipping address
- billing_street, billing_city, billing_state, billing_zip (TEXT): Billing address
- created_date (DATE): Account creation date

### Table: orders
- order_id (TEXT, PRIMARY KEY): Format ORD-YYYYMMDD-XXXX
- customer_id (TEXT): Links to customers table
- order_date (DATE): Order placement date
- items (TEXT): JSON array of order items
- shipping_method (TEXT): Shipping method used
- tracking_number (TEXT): Package tracking
- total_amount (DECIMAL): Total order amount
- payment_method (TEXT): Payment method
- order_status (TEXT): Current status

### Table: products
- product_id (TEXT, PRIMARY KEY): Format PROD-XXXX
- name (TEXT): Product name
- category, brand (TEXT): Product classification
- base_price (DECIMAL): Product price
- warranty_period (INTEGER): Warranty days
- weight (DECIMAL): Product weight
- requires_signature (BOOLEAN): Delivery requirement
- in_stock (BOOLEAN): Availability status
- description (TEXT): Product description

## VALID ACTION TYPES
Choose from these action types only:
- process_return
- send_replacement
- provide_tracking
- honor_warranty
- request_photo
- update_shipping_address
- cancel_order
- deny_return
- deny_refund
- deny_warranty_claim
- deny_price_match
- deny_order_modification
- deny_cancellation
- provide_product_information
- initiate_investigation
- process_exchange
- deny_exchange
- honor_price_match
- accept_order_modification


## WORKING EXAMPLE APPROACH
Here's a proven workflow that successfully extracts all required information:

### Step 1: Extract Customer Email
```python
# Extract email from customer ticket using LLM
email_prompt = f"Extract the customer email address from this text: {{question}}. Return only the email address, nothing else."
customer_email = call_llm(email_prompt).strip()
```

### Step 2: Find Customer in Database
```python
# Look up customer using the extracted email
customer_data = read_query(f"SELECT customer_id, name, primary_email FROM customers WHERE primary_email = '{{customer_email}}' OR alternate_email = '{{customer_email}}'")

# Handle errors and empty results
if isinstance(customer_data, dict) and 'error' in customer_data:
    return customer_data
if not customer_data:
    return {{"error": "Customer not found"}}

customer = customer_data[0]
customer_id = customer['customer_id']
customer_name = customer['name']
```

### Step 3: Get Most Recent Order
```python
# Find customer's most recent order (realistic approach - customers rarely mention order IDs)
order_data = read_query(f"SELECT order_id, customer_id, order_date, total_amount, order_status FROM orders WHERE customer_id = '{{customer_id}}' ORDER BY order_date DESC LIMIT 1")

if isinstance(order_data, dict) and 'error' in order_data:
    return order_data
if not order_data:
    return {{"error": "No orders found"}}

order = order_data[0]
order_id = order['order_id']
```

### NOTE: the customer may have made multiple orders, so you need to find the relevant order discussed in the email.

### Step 4: Read Company Policies
```python
# Access policy document for decision making
policy_content = read_file("datasets/ticketworld/company_policy.txt")
if "Error:" in policy_content:
    policy_summary = "Unable to read policy file"
else:
    policy_summary = "Return policy available"
```

### Step 5: Generate Final Resolution
```python
# Create comprehensive prompt with all gathered information
resolution_prompt = f'''
Reason carefully over the following information to create a customer service resolution plan:

Customer: {{customer_name}} ({{customer_id}})
Email: {{customer_email}}
Order: {{order_id}}
Order Date: {{order['order_date']}}
Order Status: {{order['order_status']}}
Order Amount: ${{order['total_amount']}}

Customer Issue: {{question}}

Policy Information: {{policy_summary}}

Reference the policy document, order information, and customer issue to determine:
- Appropriate actions to take
- Whether escalation is required
- Which policies apply to this situation

Create a JSON response with these fields:
- order_id
- customer_id  
- actions (list of action types)
- escalation_required (boolean)
- policy_references (list of policy IDs)

Format as valid JSON.
'''

resolution = call_llm(resolution_prompt)
```

### Key Success Patterns:
✅ **Always extract email first** - Most reliable starting point
✅ **Use most recent order** - More realistic than expecting order IDs in emails  
✅ **Handle all error cases** - Check for database errors and empty results
✅ **Gather complete context** - Get customer, order, and policy information before deciding
✅ **Use structured prompting** - Provide all context to LLM for final reasoning

 ## OUTPUT FORMAT
 Provide ONLY the essential fields as a JSON object:
 
 ```json
 {{
   "order_id": "ORD-YYYYMMDD-XXXX",
   "customer_id": "CUST-XXXX", 
   "actions": ["action_type1", "action_type2"],
   "escalation_required": true/false,
   "policy_references": ["POL-XXX-XXX", "POL-YYY-YYY"]
 }}
 ```
 
 These are the ONLY fields required. Do not include additional fields - focus on getting these core decisions correct.

---

## CUSTOMER TICKET TO RESOLVE

**From:** {customer_email}
**Subject:** {subject}  
**Date:** {timestamp}

**Message:**
{body}

---

**YOUR TASK:** Write a comprehensive program that uses the database and policy tools to accurately resolve this customer service ticket. Remember: this requires multiple steps of investigation, policy research, reasoning, and decision-making."""

                # Extract only the essential fields for the simplified answer
                essential_answer = {
                    "order_id": resolution_plan.get("order_id", "N/A"),
                    "customer_id": resolution_plan.get("customer_lookup", {}).get("customer_id", "N/A"),
                    "actions": [action.get("type") for action in resolution_plan.get("actions", []) if action.get("type")],
                    "escalation_required": resolution_plan.get("escalation_required", False),
                    "policy_references": resolution_plan.get("policy_references", [])
                }
                answer = json.dumps(essential_answer, indent=2)

                # Create standardized example with universal field names
                standardized_example = {
                    "id": ticket_id,
                    "question": formatted_question.strip(),  # Standard field: "question"
                    "answer": answer,  # Standard field: "answer"
                    "meta": {
                        "source": "ticketworld_comprehensive",
                        "filename": os.path.basename(self.dataset_path),
                        "original_customer_email": customer_email,
                        "original_subject": subject,
                        "original_body": body,
                        "original_timestamp": timestamp,
                        "resolution_plan": resolution_plan,
                        "evaluation_note": "Evaluation focuses on: order_id, customer_id, actions[type], escalation_required, policy_references"
                    }
                }

                examples.append(standardized_example)

            self.examples = examples
            print(f"Loaded {len(examples)} examples from TicketWorld Simple dataset")

            if not self.examples:
                raise ValueError("No valid examples found in TicketWorld Simple dataset")

        except Exception as e:
            raise ValueError(f"Error loading TicketWorld Simple dataset: {e}")



class TicketWorldSimpleDatasetLoaderEMAILORDER(DatasetLoader):
    """TicketWorld customer service resolution dataset loader with comprehensive task instructions"""
    
    default_evaluator = "ticketworld"
    required_tools = [
        "call_llm",
        "read_query", 
        "write_query",
        "list_tables",
        "describe_table",
        "read_file",
        "search_file"
    ]
    tool_config = {
        "database_path": "datasets/ticketworld/customer_database.db",
        "policy_files": ["datasets/ticketworld/company_policy.txt"],
        "database_schema": {
            "customers": "Customer information with IDs, emails, addresses",
            "orders": "Order details with items, status, tracking",
            "products": "Product catalog with pricing, warranties"
        }
    }

    def _load_examples(self):
        """Load examples from TicketWorld JSON dataset file with comprehensive resolution task"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("TicketWorld dataset JSON must be a list of objects")

            examples = []
            for item in data:
                # Extract customer email fields for the question
                customer_email = item.get("customer_email", "")
                subject = item.get("subject", "")
                body = item.get("body", "")
                timestamp = item.get("timestamp", "")
                ticket_id = item.get("ticket_id", f"ticket_{len(examples)}")
                
                # Skip examples that don't have required fields
                if not customer_email or not subject or not body:
                    print(f"Warning: Skipping {ticket_id} - missing customer email, subject, or body")
                    continue

                # Get the expected resolution from the original data
                resolution_plan = item.get("resolution_plan", {})
                if not resolution_plan:
                    print(f"Warning: Skipping {ticket_id} - missing resolution_plan")
                    continue

                # Format the comprehensive question with all instructions
                formatted_question = f"""# Customer Service Resolution System

## TASK OVERVIEW
You are a customer service resolution system with advanced database access capabilities. 
Your job is to analyze incoming support tickets and simply look up the customer_id and order_id using the database functions.

1. **Analyze Customer Issue**: Read the customer email and subject
2. **Find Customer**: Use the email address to look up customer information in the database
3. **Locate Orders**: Find relevant orders for the customer
4. **Get Order Details**: Retrieve order_id by looking up the order in the database using the customer_id.


## REQUIRED OUTPUT FIELDS
Your primary objective is to correctly determine these core fields:
- **order_id**: The relevant order identifier  
- **customer_id**: The customer identifier from database lookup


🚨 **CRITICAL**: This task is IMPOSSIBLE without using the provided database and file tools.
🚨 **CRITICAL**: Customer emails contain limited information - you MUST use the database and policy documents to gather complete context before making decisions.
All customers and orders exist in the database - there are no missing records.

## ENHANCED DATABASE TOOLS
You have access to reliable, simplified database functions:

### Primary Database Functions (Recommended)
# Find customer by email (auto-connects to TicketWorld database)
customer_data = read_query("SELECT customer_id, name FROM customers WHERE primary_email = 'customer@email.com' OR alternate_email = 'customer@email.com'")

# Get customer's orders
orders = read_query("SELECT order_id, order_date, total_amount FROM orders WHERE customer_id = 'CUST-0001' ORDER BY order_date DESC")

# Get specific order details
order_details = read_query("SELECT * FROM orders WHERE order_id = 'ORD-20250224-1008'")

# Explore database structure (if needed)
tables = list_tables()                    # See available tables
schema = describe_table("customers")      # Get table structure

### Policy Document Access  
policy_text = read_file("datasets/ticketworld/company_policy.txt")

### Key Benefits of New Database Tools:
✅ **Automatic Connection**: No need to specify database path
✅ **Error Handling**: Clear error messages and validation
✅ **Consistent Results**: Reliable data formatting
✅ **Query Safety**: Built-in SQL validation
✅ **Type Safety**: Proper handling of different data types

🔥 **YOU MUST USE THESE TOOLS** - Accurate resolution requires database lookup and policy research.

## DATABASE SCHEMA

### Table: customers
- customer_id (TEXT, PRIMARY KEY): Format CUST-XXXX
- name (TEXT): Customer full name  
- primary_email (TEXT): Primary email address
- alternate_email (TEXT): Secondary email address
- phone (TEXT): Phone number
- shipping_street, shipping_city, shipping_state, shipping_zip (TEXT): Shipping address
- billing_street, billing_city, billing_state, billing_zip (TEXT): Billing address
- created_date (DATE): Account creation date

### Table: orders
- order_id (TEXT, PRIMARY KEY): Format ORD-YYYYMMDD-XXXX
- customer_id (TEXT): Links to customers table
- order_date (DATE): Order placement date
- items (TEXT): JSON array of order items
- shipping_method (TEXT): Shipping method used
- tracking_number (TEXT): Package tracking
- total_amount (DECIMAL): Total order amount
- payment_method (TEXT): Payment method
- order_status (TEXT): Current status

### Table: products
- product_id (TEXT, PRIMARY KEY): Format PROD-XXXX
- name (TEXT): Product name
- category, brand (TEXT): Product classification
- base_price (DECIMAL): Product price
- warranty_period (INTEGER): Warranty days
- weight (DECIMAL): Product weight
- requires_signature (BOOLEAN): Delivery requirement
- in_stock (BOOLEAN): Availability status
- description (TEXT): Product description




## WORKING EXAMPLE APPROACH
Here's a proven workflow that successfully extracts all required information:

### Step 1: Extract Customer Email
```python
# Extract email from customer ticket using LLM
email_prompt = f"Extract the customer email address from this text: {{question}}. Return only the email address, nothing else."
customer_email = call_llm(email_prompt).strip()
```

### Step 2: Find Customer in Database
```python
# Look up customer using the extracted email
customer_data = read_query(f"SELECT customer_id, name, primary_email FROM customers WHERE primary_email = '{{customer_email}}' OR alternate_email = '{{customer_email}}'")

# Handle errors and empty results
if isinstance(customer_data, dict) and 'error' in customer_data:
    return customer_data
if not customer_data:
    return {{"error": "Customer not found"}}

customer = customer_data[0]
customer_id = customer['customer_id']
customer_name = customer['name']
```

### Step 3: Get Most Recent Order
```python
# Find customer's most recent order (realistic approach - customers rarely mention order IDs)
order_data = read_query(f"SELECT order_id, customer_id, order_date, total_amount, order_status FROM orders WHERE customer_id = '{{customer_id}}' ORDER BY order_date DESC LIMIT 1")

if isinstance(order_data, dict) and 'error' in order_data:
    return order_data
if not order_data:
    return {{"error": "No orders found"}}

order = order_data[0]
order_id = order['order_id']
```

Create a JSON response with these fields:
- order_id
- customer_id  

Format as valid JSON.
'''

```

### Key Success Patterns:
✅ **Always extract email first** - Most reliable starting point
✅ **Use most recent order** - More realistic than expecting order IDs in emails  
✅ **Handle all error cases** - Check for database errors and empty results
✅ **Use structured prompting** - Provide all context to LLM for final reasoning

 ## OUTPUT FORMAT
 Provide ONLY the essential fields as a JSON object:
 
 ```json
 {{
   "order_id": "ORD-YYYYMMDD-XXXX",
   "customer_id": "CUST-XXXX", 
 }}
 ```
 
 These are the ONLY fields required. Do not include additional fields - focus on getting these core decisions correct.

---

## CUSTOMER TICKET TO RESOLVE

**From:** {customer_email}
**Subject:** {subject}  
**Date:** {timestamp}

**Message:**
{body}

---

**YOUR TASK:** Write a comprehensive program that uses the database and policy tools to accurately resolve this customer service ticket. Remember: this requires multiple steps of investigation, policy research, reasoning, and decision-making."""

                # Extract only the essential fields for the simplified answer
                essential_answer = {
                    "order_id": resolution_plan.get("order_id", "N/A"),
                    "customer_id": resolution_plan.get("customer_lookup", {}).get("customer_id", "N/A"),
                }
                answer = json.dumps(essential_answer, indent=2)

                # Create standardized example with universal field names
                standardized_example = {
                    "id": ticket_id,
                    "question": formatted_question.strip(),  # Standard field: "question"
                    "answer": answer,  # Standard field: "answer"
                    "meta": {
                        "source": "ticketworld_comprehensive",
                        "filename": os.path.basename(self.dataset_path),
                        "original_customer_email": customer_email,
                        "original_subject": subject,
                        "original_body": body,
                        "original_timestamp": timestamp,
                        "resolution_plan": resolution_plan,
                        "evaluation_note": "Evaluation focuses on: order_id, customer_id, actions[type], escalation_required, policy_references"
                    }
                }

                examples.append(standardized_example)

            self.examples = examples
            print(f"Loaded {len(examples)} examples from TicketWorld Simple dataset")

            if not self.examples:
                raise ValueError("No valid examples found in TicketWorld Simple dataset")

        except Exception as e:
            raise ValueError(f"Error loading TicketWorld Simple dataset: {e}")





def create_dataset_loader(loader_type: str, **kwargs) -> DatasetLoader:
    """
    Create a dataset loader of the specified type

    Args:
        loader_type: Type of loader to create ("arc", "json", "jsonl", "simpleqa", "natural_plan", "ticketworld", or "custom")
        **kwargs: Additional arguments to pass to the loader constructor

    Returns:
        DatasetLoader: An instance of the requested loader type
    """
    if loader_type.lower() == "arc":
        from dataset_loader import ARCDatasetLoader
        return ARCDatasetLoader(**kwargs)
    elif loader_type.lower() == "json":
        from dataset_loader import JSONDatasetLoader
        return JSONDatasetLoader(**kwargs)
    elif loader_type.lower() == "jsonl":
        from dataset_loader import JSONLDatasetLoader
        return JSONLDatasetLoader(**kwargs)
    elif loader_type.lower() == "simpleqa":
        from dataset_loader import SimpleQADatasetLoader
        return SimpleQADatasetLoader(**kwargs)
    elif loader_type.lower() == "natural_plan":
        return NaturalPlanDatasetLoader(**kwargs)
    elif loader_type.lower() == "hotpotqa":
        return HotpotQADatasetLoader(**kwargs)
    elif loader_type.lower() == "math":
        return MathDatasetLoader(**kwargs)
    elif loader_type.lower() == "gpqa":
        return GPQADatasetLoader(**kwargs)
    elif loader_type.lower() == "medmcqa":
        return MedMCQADatasetLoader(**kwargs)
    elif loader_type.lower() == "ticketworld":
        return TicketWorldDatasetLoader(**kwargs)
    elif loader_type.lower() == "ticketworld_simple":
        return TicketWorldSimpleDatasetLoader(**kwargs)
    elif loader_type.lower() == "custom":
        from dataset_loader import CustomDatasetLoader
        return CustomDatasetLoader(**kwargs)
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
