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
        super().__init__(dataset_path, shuffle, random_seed)

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


def create_dataset_loader(loader_type: str, **kwargs) -> DatasetLoader:
    """
    Create a dataset loader of the specified type

    Args:
        loader_type: Type of loader to create ("arc", "json", "jsonl", "simpleqa", "natural_plan", or "custom")
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
    elif loader_type.lower() == "custom":
        from dataset_loader import CustomDatasetLoader
        return CustomDatasetLoader(**kwargs)
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
