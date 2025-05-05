#!/usr/bin/env python
"""
agent_system.py - Main class for the Agentic Learning System
"""

import os
import json
import time
import datetime
import traceback
import random
import sys
import ast  # Added for script validation
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from google import genai
from google.genai import types  # Added import for GenerateContentConfig
from llm_example_library import ExampleSets, APIExamples, FallbackScripts
from prompt_templates import PromptTemplates


class AgentSystem:
    """
    Agentic Learning System that uses LLM reasoning to continuously improve its approach
    to solving dataset problems through iterative exploration and exploitation.
    Now supports custom dataset loaders.
    """

    def __init__(self, dataset_loader=None):
        """
        Initialize the agent system with a dataset loader

        Args:
            dataset_loader: A DatasetLoader instance for loading and processing examples
        """
        # Initialize configuration
        self.explore_rate = 60
        self.exploit_rate = 40
        self.force_exploitation_next = False

        # Store the dataset loader
        self.dataset_loader = dataset_loader
        if not self.dataset_loader:
            raise ValueError("A dataset loader must be provided")

        # Initialize batch size and tracking for seen examples
        self.current_batch_size = 3  # Start with a small batch
        self.seen_examples = set()
        self.examples_processed = 0

        # Ensure directories exist
        self.archive_dir = Path("archive")
        self.archive_dir.mkdir(exist_ok=True)
        self.scripts_dir = Path("scripts")
        self.scripts_dir.mkdir(exist_ok=True)

        self.capability_tracker = CapabilityTracker()

        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        print(f"System prompt loaded: {len(self.system_prompt)} characters")

        # Initialize Gemini API client
        try:
            self.client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"))
            print("Gemini API client initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini API client: {e}")
            print("Make sure to set the GEMINI_API_KEY environment variable")
            raise

        # Initialize learnings mechanism
        print("Initializing learnings mechanism...")
        learnings = self._load_learnings()
        if learnings:
            print(f"Loaded existing learnings: {len(learnings)} characters")
        else:
            print("No existing learnings found. Will start accumulating learnings.")

        # ADDED: Reserve training examples to prevent data leakage
        self.mark_training_examples()



        # Initialize current iteration
        self.current_iteration = 0

        # Load previous iterations if available
        self._load_previous_state()

        # analyze training examples and add to learnings if cold start
        if self.current_iteration == 0:
            self.analyze_dataset_with_llm()



    def get_training_examples(self, count: int = 5) -> List[Dict]:
        """
        Get a fixed set of examples for initial training (cold start).
        These examples are reserved and never used for testing.

        Args:
            count: Number of training examples to return

        Returns:
            List of example dictionaries with input and output fields
        """
        # Get first 'count' examples from dataset and mark them as seen
        if not self.dataset_loader:
            return []

        # Save original index (though we won't restore it in most cases)
        original_index = self.dataset_loader.current_index 

        # Start from beginning for training examples
        self.dataset_loader.current_index = 0

        # Get training examples
        training_examples = self.dataset_loader.get_examples(count)

        # Mark these examples as seen
        for i in range(count):
            self.seen_examples.add(i)

        # Update dataset position to continue after the training examples
        # Instead of going back to the original index
        self.dataset_loader.current_index = count

        # Update the next example index to continue after training examples
        if hasattr(self, 'next_example_index'):
            self.next_example_index = max(self.next_example_index, count)
        else:
            self.next_example_index = count

        print(f"Reserved {len(training_examples)} examples for training, next example will be {self.dataset_loader.current_index}")

        return training_examples

    def mark_training_examples(self):
        """
        Mark the initial training examples as seen to ensure they're not used for testing.
        This should be called during initialization to establish the training set.
        """
        # Only run this if we don't have any seen examples yet
        if not self.seen_examples:
            # The training examples will be added to seen_examples
            training_examples = self.get_training_examples(5)  # Get 5 training examples

            # Update examples processed to include training examples
            self.examples_processed = len(training_examples)

            print(f"Reserved {len(training_examples)} examples for initial training")
            print(f"Total examples seen: {len(self.seen_examples)}")


    # Add this to the AgentSystem class
    def analyze_dataset_with_llm(self):
        """
        Perform an initial analysis of the dataset to understand patterns,
        structures, and potential approaches before any problem-solving.
        Adds insights to the learnings.txt file.
        """
        print("Performing initial dataset analysis with LLM...")

        # Get training examples - these are already set aside for learning
        # We're not using examples that would be used for testing later

        # Don't mess with the dataset loader's current index

        try: 
            original_index = self.dataset_loader.current_index
            self.dataset_loader.current_index = 0
            
            training_examples = self.get_training_examples(5)
    
            if not training_examples or len(training_examples) == 0:
                print("Warning: No training examples available for analysis.")
                return
    
            # Format examples for LLM analysis
            formatted_examples = []
            for i, example in enumerate(training_examples):
                formatted_examples.append({
                    "id": f"example_{i}",
                    "question": self.dataset_loader.get_example_input(example),
                    "answer": self.dataset_loader.get_example_output(example)
                })

        finally:
            self.dataset_loader.current_index = original_index

        # Create a system instruction for the data analyzer
        data_analyzer_system_instruction = "You are a Data Pattern Analyst specialized in identifying patterns, structures, and challenges in datasets. Your goal is to provide deep insights and creative strategies for approaching problems."

        # Build the prompt for dataset analysis
        prompt = f"""
        You're analyzing a new dataset to identify patterns, structures, and potential problem-solving approaches.

        Here are some representative examples from the dataset:

        {json.dumps(formatted_examples, indent=2)}

        Think deeply about these examples and provide a comprehensive analysis with the following sections:

        ## DATASET CHARACTERISTICS
        - What patterns do you observe in the questions?
        - What patterns do you observe in the answers?
        - What is the structure and format of both inputs and outputs?
        - What domain knowledge might be required?
        - Are all questions of the same type, or are there multiple types?
        - Do all questions require the same type of reasoning?

        ## DATA CHALLENGES
        - What makes these problems difficult?
        - What potential edge cases or complexities should be considered?
        - What types of reasoning are required to solve these problems?

        ## POTENTIAL APPROACHES
        - What solution strategies might work well for this type of problem?
        - What decomposition of the problem would be most effective?
        - What validation techniques would help ensure correct solutions?
        - How would you handle unusual or edge cases?

        ## CREATIVE INSIGHTS
        - What non-obvious patterns or shortcuts might exist?
        - What unique perspectives might help solve these problems?
        - What analogies to other problem domains might be useful?

        ## IMPLEMENTATION RECOMMENDATIONS
        - What verification steps would be crucial?
        - What intermediate steps or representations would be helpful?
        - Assuming you will work mostly with text as inputs and outputs, what specific techniques would be most effective? Remember, overreliance on JSON parsing and complex code generation often leads to errors and low performance. Instead, focus on leveraging the LLM's natural reasoning abilities.

        Be specific and concrete in your analysis. Focus on actionable insights that would help develop effective solutions.
        """

        # Call LLM to analyze the dataset
        try:
            dataset_analysis = self.call_llm(prompt, system_instruction=data_analyzer_system_instruction)
            print("Dataset analysis complete, adding to learnings.txt")

            # Format the analysis for learnings.txt
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_analysis = f"""
    === INITIAL DATASET ANALYSIS [{timestamp}] ===

    {dataset_analysis}

    === END INITIAL DATASET ANALYSIS ===

    """
            print (formatted_analysis)
            # Load existing learnings (if any)
            current_learnings = self._load_learnings()

            # Add analysis at the beginning of learnings.txt
            updated_learnings = formatted_analysis + current_learnings

            # Save updated learnings
            self._save_learnings(updated_learnings)

            print(f"Added {len(dataset_analysis)} characters of dataset analysis to learnings.txt")
            return dataset_analysis

        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            traceback.print_exc()
            return None

    
    def _load_system_prompt(self) -> str:
        """Load the system prompt from the system_prompt.md file"""
        system_prompt_path = Path("system_prompt.md")
        if not system_prompt_path.exists():
            print(
                "Warning: system_prompt.md file not found. Using empty system prompt."
            )
            return ""

        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading system prompt: {e}")
            return ""

    def _load_previous_state(self):
        """Load previous state from archive if available"""
        summaries = self.get_summaries()
        iterations = self.get_all_iterations()

        # Add debugging to verify what's being loaded
        print(f"Loading previous state... Found {len(summaries)} summaries and {len(iterations)} iteration files")

        # First check if we have summaries
        if summaries:
            # Verify summaries have the expected content for debugging
            iteration_nums = [s.get("iteration") for s in summaries]
            print(f"Summary iteration numbers: {iteration_nums}")

            # Sort by iteration number to find the highest
            sorted_summaries = sorted(summaries,
                                      key=lambda x: x.get("iteration", 0),
                                      reverse=True)
            last_iteration = sorted_summaries[0].get("iteration", 0)
            self.current_iteration = last_iteration + 1

            # Use the explore/exploit balance from the last iteration
            self.explore_rate = sorted_summaries[0].get(
                "new_explore_rate", self.explore_rate)
            self.exploit_rate = sorted_summaries[0].get(
                "new_exploit_rate", self.exploit_rate)

            # Use the batch size from the last iteration
            self.current_batch_size = sorted_summaries[0].get(
                "new_batch_size", self.current_batch_size)

            print(
                f"Loaded previous state: iteration {self.current_iteration}, "
                +
                f"explore/exploit: {self.explore_rate}/{self.exploit_rate}, " +
                f"batch size: {self.current_batch_size}")

            # CRITICAL FIX: Always include the 5 training examples
            for i in range(5):
                self.seen_examples.add(i)

            # Track total examples seen so far (starting after training examples)
            total_examples_seen = 5  # Start with the training examples

            # Reconstruct set of seen examples
            for iteration in iterations:
                if iteration and "sample_count" in iteration:
                    # Each iteration represents sample_count examples
                    sample_count = iteration.get("sample_count", 0)

                    # Add the examples this iteration would have seen
                    # Starting from the current total (which includes training examples)
                    for i in range(total_examples_seen, total_examples_seen + sample_count):
                        self.seen_examples.add(i)

                    # Update the total examples seen
                    total_examples_seen += sample_count

                    # Also update our internal examples_processed counter
                    self.examples_processed = total_examples_seen

            # Set next example index to after the last seen example
            self.next_example_index = total_examples_seen

            print(
                f"Loaded {len(self.seen_examples)} seen examples, next example index: {self.next_example_index}"
            )

            # After calculating next_example_index
            if hasattr(self, 'next_example_index') and self.dataset_loader:
                self.dataset_loader.current_index = self.next_example_index
                print(f"Updated dataset loader to start at example index {self.next_example_index}")
        else:
            # Double-check iterations as a backup in case summaries.json is missing
            if iterations:
                # Get the highest iteration number from iteration files
                highest_iter = max([it.get("iteration", 0) for it in iterations if it])
                self.current_iteration = highest_iter + 1
                print(f"No summaries found, but found iteration files. Setting next iteration to {self.current_iteration}")
            else:
                self.current_iteration = 0
                print("No previous state found. Starting from iteration 0.")
    
    def call_llm(self, prompt: str, system_instruction: str = None) -> str:
        """Call the Gemini LLM with a prompt and return the response"""
        try:
            # Use provided system instruction or default to the loaded system prompt
            sys_instruction = system_instruction if system_instruction is not None else ""

            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction),
                contents=prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error: {str(e)}"

    def load_dataset(self) -> Dict:
        """Load the entire dataset from the specified file"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return {}

    def get_samples(self) -> Dict:
        """
        Get samples from the dataset loader for the current batch.
        Uses universal field names (question/answer) for consistency.

        Returns:
            Dict containing:
            - samples: List of sample dictionaries with question/answer/id fields
            - new_examples_added: Count of new examples added
            - total_seen_examples: Total count of seen examples
        """
        if not self.dataset_loader:
            return {
                "samples": [],
                "new_examples_added": 0,
                "total_seen_examples": 0
            }

        # Get current_batch_size examples
        examples = self.dataset_loader.get_examples(self.current_batch_size)

        # Convert to standardized format for system
        samples = []
        new_examples_added = 0

        for i, example in enumerate(examples):
            # Get current example index
            example_index = self.examples_processed + i

            # Track that we've seen this example
            if example_index not in self.seen_examples:
                self.seen_examples.add(example_index)
                new_examples_added += 1

            # Extract input and output
            try:
                example_input = self.dataset_loader.get_example_input(example)
                example_output = self.dataset_loader.get_example_output(example)

                # Create standardized sample using universal field names
                standardized_sample = {
                    "question": example_input,  # Universal field name
                    "answer": example_output,   # Universal field name
                    "id": f"example_{example_index}",
                    "meta": example.get("meta", {})

                }

                samples.append(standardized_sample)
            except Exception as e:
                print(f"Error processing example: {e}")

        # Update the number of examples processed
        self.examples_processed += len(samples)

        return {
            "samples": samples,
            "new_examples_added": new_examples_added,
            "total_seen_examples": len(self.seen_examples)
        }
    
    def save_to_archive(self, data: Dict, filename: str) -> None:
        """Save data to the archive directory"""
        filepath = self.archive_dir / filename
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)

    def read_from_archive(self, filename: str) -> Dict:
        """Read data from the archive directory"""
        filepath = self.archive_dir / filename
        if not filepath.exists():
            return {}

        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)

    def get_all_iterations(self) -> List[Dict]:
        """Get data from all past iterations"""
        iterations = []
        for file in self.archive_dir.glob("iteration_*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                iterations.append(json.load(f))
        return sorted(iterations, key=lambda x: x.get('iteration', 0))

    def get_summaries(self) -> List[Dict]:
        """Get all iteration summaries"""
        summary_file = self.archive_dir / "summaries.json"
        if not summary_file.exists():
            print(f"Warning: Summaries file {summary_file} does not exist")
            return []

        try:
            with open(summary_file, 'r', encoding='utf-8') as file:
                summaries = json.load(file)
                print(f"Successfully loaded {len(summaries)} summaries from {summary_file}")
                return summaries
        except Exception as e:
            print(f"Error loading summaries file: {e}")
            return []

    def update_summaries(self, new_summary: Dict) -> None:
        """Add a new summary to the summaries file"""
        # Add capability data to summary
        if hasattr(self, 'capability_tracker'):
            new_summary[
                "capability_report"] = self.capability_tracker.generate_report()

        summaries = self.get_summaries()
        summaries.append(new_summary)

        summary_file = self.archive_dir / "summaries.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as file:
                json.dump(summaries, file, indent=2)
            print(f"Successfully updated summaries file with iteration {new_summary.get('iteration')}")
        except Exception as e:
            print(f"Error updating summaries file: {e}")

    def adjust_batch_size_with_llm(self, performance: Dict) -> Tuple[int, str]:
        """Use LLM to determine appropriate batch size based on performance"""
        # Get performance history
        iterations = self.get_all_iterations()

        # Extract relevant information for LLM
        performance_history = []
        for iteration in iterations[-5:]:  # Last 5 iterations
            if iteration is None:  # Skip None entries
                continue

            perf = iteration.get("performance", {})
            accuracy = perf.get("accuracy", 0) if perf else 0

            performance_history.append({
                "iteration":
                iteration.get("iteration"),
                "batch_size":
                iteration.get("batch_size", 5),
                "accuracy":
                accuracy,
                "error_patterns":
                iteration.get("performance",
                              {}).get("error_analysis",
                                      {}).get("error_patterns", [])
            })

        # Default response if no LLM available
        default_response = (
            self.current_batch_size,
            "Maintaining current batch size due to insufficient performance data"
        )

        # If no performance history, just keep current batch size
        if not performance_history:
            return default_response

        # Role-specific system instruction for batch size optimizer
        batch_optimizer_system_instruction = "You are a Batch Size Optimizer. Your task is to analyze performance trends and recommend the optimal batch size for testing, balancing between stability and throughput."

        prompt = f"""
        As an AI optimization system, you need to determine the appropriate batch size for testing.

        Current batch size: {self.current_batch_size}
        Current accuracy: {performance.get("accuracy", 0):.2f}
        Total examples seen so far: {len(self.seen_examples)}

        Recent performance history:
        {json.dumps(performance_history, indent=2)}

        Based on this information, determine if the batch size should be adjusted.
        Consider:

        1. Recent performance trend
        2. Stability of results
        3. Need for more diverse examples

        Rules:
        - Batch size should be between 3 and 10
        - Increase batch size when performance is stable and good. For example, if every script tested on batches of 5 has been 100% accurate, then we cannot differentiate between them and tell how good they are relative to one another. In this case, you would increase the batch size to 10. 
        - Decrease batch size if performance is consistently poor. For example, if every script tested on batches of 10 has been 0% accurate, then we are simply wasting compute and data examples, and it would be sufficient to just test on the minimum batch size. In this case you would decrease the batch size to 5.
        - Keep batch size stable when exploring new approaches
        - Remember: batch size should ONLY be increased when the current batch size is performing well and we want to test more diverse examples

        Return only a JSON object with:
        {{"new_batch_size": <integer>, "rationale": "<brief explanation>"}}
        """

        try:
            response = self.call_llm(
                prompt, system_instruction=batch_optimizer_system_instruction)

            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.endswith("```"):
                response = response.split("```")[0]

            result = json.loads(response)

            # Validate and extract new batch size
            new_batch_size = int(
                result.get("new_batch_size", self.current_batch_size))

            # Ensure batch size is within reasonable limits
            new_batch_size = max(3, min(10, new_batch_size))

            return new_batch_size, result.get("rationale",
                                              "No rationale provided")
        except Exception as e:
            print(f"Error adjusting batch size: {e}")
            return default_response

    def _load_learnings(self) -> str:
        """Load accumulated learnings from the learnings.txt file"""
        learnings_path = Path("learnings.txt")
        if not learnings_path.exists():
            print("No existing learnings file found. Starting with fresh learnings.")
            return ""

        try:
            with open(learnings_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                print(f"Successfully loaded learnings.txt ({len(content)} characters)")
                return content
        except Exception as e:
            print(f"Error loading learnings: {e}")
            traceback.print_exc()  # Print full traceback for better debugging
            return ""

    def _save_learnings(self, learnings: str) -> None:
        """Save updated learnings to the learnings.txt file"""
        learnings_path = Path("learnings.txt")
        try:
            with open(learnings_path, 'w', encoding='utf-8') as f:
                f.write(learnings)
            print(f"Learnings successfully saved to {learnings_path} ({len(learnings)} characters)")
        except Exception as e:
            print(f"Error saving learnings: {e}")
            traceback.print_exc()  # Print full traceback for better debugging


    def generate_batch_learnings(self, iteration_data: Dict) -> str:
        """Generate learnings from the current batch results with focus on dataset-specific insights"""
        learnings_generator_system_instruction = "You are a Knowledge Synthesizer. Your role is to extract concrete, dataset-specific insights from experiment results, focusing on patterns in the data, effective strategies for this specific task, and precise failure modes."

        # Get full original samples from iteration_data
        samples = []
        if "samples" in iteration_data:
            samples = iteration_data.get("samples", [])

        # Get example questions - prefer direct samples if available
        sample_questions = []
        for i in range(min(3, len(samples))):
            if i < len(samples):
                # Use the universal "question" field 
                sample_questions.append(samples[i].get("question", "N/A"))
            else:
                sample_questions.append("N/A")

        # If we couldn't get samples directly, try to infer from results
        if not sample_questions and "results" in iteration_data:
            results = iteration_data.get("results", [])
            for i in range(min(3, len(results))):
                if "question" in results[i]:
                    sample_questions.append(results[i].get("question", "N/A"))

        # Get script source code (truncated for prompts)
        script_code = iteration_data.get("script", "")[:500] if iteration_data.get("script") else "No script available"

        # Get performance metrics
        accuracy = iteration_data.get("performance", {}).get("accuracy", 0)

        # Get error examples
        error_examples = []
        for i, result in enumerate(iteration_data.get("results", [])):
            if not result.get("match", True) and result.get("success", False):
                # Find the corresponding sample
                sample_question = "N/A"
                golden_answer = "N/A"

                if i < len(samples):
                    # Use the universal "question" and "answer" fields
                    sample_question = samples[i].get("question", "N/A")
                    golden_answer = samples[i].get("answer", "N/A")

                error_examples.append({
                    "question": sample_question,
                    "expected": golden_answer,
                    "actual": result.get("answer", "N/A"),
                    "explanation": result.get("evaluation", {}).get("explanation", "No explanation")
                })

        # Get capability assessment if available
        capability_insights = ""
        if "capability_report" in iteration_data and iteration_data["capability_report"]:
            report = iteration_data["capability_report"]
            capability_insights = f"""
            Key Capabilities:
            - Strengths: {', '.join(report.get('strengths', [])[:2])}
            - Weaknesses: {', '.join(report.get('weaknesses', [])[:2])}
            - Focus Area: {report.get('improvement_focus', 'None identified')}
            """

        # Add trace insights to the prompt
        trace_insights = iteration_data.get("trace_insights", "")
        
        prompt = f"""
        Extract specific, concrete learnings from this iteration's results, focusing on dataset-specific insights:

        Iteration: {iteration_data.get("iteration")}
        Strategy: {iteration_data.get("strategy", "Unknown")}
        Accuracy: {accuracy:.2f}
        Approach summary: {iteration_data.get("approach_summary", "No summary available")}

        EXECUTION TRACE INSIGHTS:
        {trace_insights if trace_insights else "No trace analysis available"}

        Sample questions from dataset:
        {json.dumps(sample_questions, indent=2)}

        Script approach (excerpt):
        ```python
        {script_code}
        ```

        Primary issue identified: {iteration_data.get("performance", {}).get("error_analysis", {}).get("primary_issue", "None identified")}

        Error patterns:
        {json.dumps(iteration_data.get("performance", {}).get("error_analysis", {}).get("error_patterns", []), indent=2)}

        Error examples (first {len(error_examples)} failures):
        {json.dumps(error_examples[:3], indent=2)}

        {capability_insights}

        Based on this information, provide specific learnings in the following format:

        1. DATASET PATTERNS: Identify 2-3 specific patterns or characteristics in this dataset. What format do questions take? What structures appear repeatedly? What's unique about this task?

        2. WORKING STRATEGIES: What specific techniques worked well for this particular dataset and why?

        3. FAILURE MODES: What specific aspects of the dataset or task caused failures? Describe exactly how and why the approach failed on specific examples.

        4. EXPERIMENT RESULTS: What did we learn from this specific experimental approach? What hypotheses were confirmed or rejected?

        5. NEXT STEPS: What specific adaptations should be made for this particular dataset and task?

        6. TRACE INSIGHTS SUMMARY: What key insights did we learn from the execution traces?

        Focus on concrete, specific insights that are directly tied to the dataset and task at hand, not general principles of system design.
        Keep your summary focused on what we've learned about solving THIS specific dataset problem.
        """

        try:
            response = self.call_llm(
                prompt,
                system_instruction=learnings_generator_system_instruction)
            return f"--- LEARNINGS FROM ITERATION {iteration_data.get('iteration')} ---\n{response.strip()}\n\n"
        except Exception as e:
            error_message = f"Error generating batch learnings: {str(e)}"
            print(error_message)
            return f"--- LEARNINGS FROM ITERATION {iteration_data.get('iteration')} ---\n{error_message}\n\n"


    def analyze_iteration_traces(self, iteration_number: int) -> Dict:
        """Analyze traces for a specific iteration (useful for retroactive analysis)"""
        if not hasattr(self, 'archive_dir'):
            return {"error": "No archive directory configured"}

        trace_analyzer = TraceAnalyzer(self.archive_dir)
        insights = trace_analyzer.analyze_traces_with_llm(iteration_number)

        # Update the iteration file with trace insights if not already present
        iteration_file = self.archive_dir / f"iteration_{iteration_number}.json"
        if iteration_file.exists():
            with open(iteration_file, 'r', encoding='utf-8') as f:
                iteration_data = json.load(f)

            if not iteration_data.get("trace_insights"):
                iteration_data["trace_insights"] = insights
                iteration_data["trace_analysis"] = {
                    "analyzed_at": datetime.datetime.now().isoformat(),
                    "insights": insights,
                    "trace_file": f"trace_iteration_{iteration_number}.jsonl"
                }

                with open(iteration_file, 'w', encoding='utf-8') as f:
                    json.dump(iteration_data, f, indent=2)

        return {"iteration": iteration_number, "insights": insights}
        
    
    def synthesize_learnings(self, current_learnings: str, new_batch_learnings: str) -> str:
        """Synthesize existing learnings with new batch learnings using a section-by-section approach"""
        # Define the standard sections
        sections = [
            "1. DATASET PATTERNS & CHARACTERISTICS",
            "2. EFFECTIVE TASK-SPECIFIC STRATEGIES",
            "3. COMMON FAILURE MODES ON THIS DATASET",
            "4. EXPERIMENT LOG & FINDINGS", 
            "5. NEXT RESEARCH DIRECTIONS"
        ]

        # Parse the current learnings into sections
        current_sections = {}

        # Handle header/intro text
        header = current_learnings
        for i, section in enumerate(sections):
            if section in current_learnings:
                if i == 0:  # First section - extract header
                    header = current_learnings.split(section)[0].strip()

                # Extract section content
                section_text = current_learnings.split(section)[1]
                # If there's another section after this one, only grab until that section
                if i < len(sections) - 1 and sections[i+1] in section_text:
                    section_text = section_text.split(sections[i+1])[0].strip()
                current_sections[section] = section_text.strip()

        # Parse the new batch learnings to identify which sections have new content
        new_sections = {}
        for section in sections:
            if section in new_batch_learnings:
                section_text = new_batch_learnings.split(section)[1]
                # If there's another section after this one, only grab until that section
                next_sections = [s for s in sections if s in section_text]
                if next_sections:
                    section_text = section_text.split(next_sections[0])[0].strip()
                new_sections[section] = section_text.strip()

        # Get current iteration number
        current_iteration = self.current_iteration - 1  # Since we increment at the end of run_iteration

        # For each section, update if there's new content
        updated_sections = {}
        for section in sections:
            if section not in new_sections:
                # No new content for this section
                updated_sections[section] = current_sections.get(section, "")
                continue

            # Special handling for EXPERIMENT LOG & FINDINGS section
            if section == "4. EXPERIMENT LOG & FINDINGS":
                current_section_text = current_sections.get(section, "")
                new_section_text = new_sections[section]

                # Check if the previous iteration is in the current content
                prev_iteration_marker = f"**Iteration {current_iteration-1}:**"
                iteration_missing = current_iteration > 1 and prev_iteration_marker not in current_section_text

                # Check combined length
                combined_length = len(current_section_text) + len(new_section_text)

                # If too large or missing iterations, ask for condensed version
                if combined_length > 30000 or iteration_missing:
                    print(f"EXPERIMENT LOG section too large ({combined_length} chars) or missing iterations - condensing")

                    prompt = f"""
                    You are managing the EXPERIMENT LOG & FINDINGS section of our research document.
                    This section has grown too large and needs condensing while preserving key information.

                    CURRENT EXPERIMENT LOG:
                    {current_section_text}

                    NEW EXPERIMENT FINDINGS (FOR ITERATION {current_iteration}):
                    {new_section_text}

                    Create a condensed version of the EXPERIMENT LOG that:
                    1. PRESERVES ALL ITERATIONS - it's critical that ALL iterations have some representation
                    2. Maintains MORE DETAIL for recent iterations ({max(0, current_iteration-5)}-{current_iteration})
                    3. CONDENSES older iterations (0-{max(0, current_iteration-6)}) to just key findings and essential details
                    4. Ensures each iteration is clearly marked with "**Iteration X:**" format
                    5. Maintains the hierarchical bullet point structure

                    You must include information about ALL iterations from 0 to {current_iteration}, with no gaps.
                    Focus on reducing verbose descriptions of older iterations while preserving their key findings.

                    Return ONLY the condensed experiment log section with no explanation.
                    """

                    try:
                        updated_text = self.call_llm(
                            prompt, 
                            system_instruction="You are a Research Documentation Specialist who excels at preserving essential information while reducing verbosity."
                        )

                        # Verify all iterations are present by checking for markers
                        all_iterations_present = True
                        for i in range(current_iteration + 1):
                            if f"**Iteration {i}:**" not in updated_text and f"* **Iteration {i}:**" not in updated_text:
                                all_iterations_present = False
                                print(f"Warning: Iteration {i} missing from condensed log")

                        if all_iterations_present:
                            updated_sections[section] = updated_text.strip()
                        else:
                            # If iterations are missing, append new content with a warning
                            print("Some iterations missing - falling back to concatenation with warning")
                            updated_sections[section] = current_section_text + "\n\n=== WARNING: EXPERIMENT LOG REACHED SIZE LIMIT ===\n" + \
                                                     "Earlier iterations have been abbreviated to save space.\n\n" + \
                                                     new_section_text
                    except Exception as e:
                        print(f"Error condensing experiment log: {e}")
                        # Fallback to concatenation with warning
                        updated_sections[section] = current_section_text + "\n\n=== WARNING: EXPERIMENT LOG REACHED SIZE LIMIT ===\n" + \
                                                 "Consider manually condensing earlier iterations.\n\n" + \
                                                 new_section_text
                else:
                    # Standard concatenation for experiment log if not too large
                    updated_sections[section] = current_section_text + "\n\n" + new_section_text
            else:
                # For all other sections, use the original logic
                if section in current_sections:
                    current_section_text = current_sections[section]
                    new_section_text = new_sections[section]

                    # Check combined length
                    combined_length = len(current_section_text) + len(new_section_text)
                    if combined_length > 30000:  # Conservative character limit
                        print(f"Section {section} too large ({combined_length} chars) - using simple append")
                        updated_sections[section] = current_section_text + "\n\n=== NEW ADDITIONS ===\n\n" + new_section_text
                        continue

                    # Otherwise, synthesize this section
                    try:
                        print(f"Synthesizing section: {section}")
                        prompt = f"""
                        You are updating a specific section of our research learnings document about a Grid Transformation Task dataset.

                        CURRENT CONTENT FOR SECTION "{section}":
                        {current_section_text}

                        NEW INSIGHTS FOR THIS SECTION:
                        {new_section_text}

                        Create an updated version of ONLY THIS SECTION that:
                        1. Preserves all important information from both sources
                        2. Eliminates redundancies
                        3. Organizes related insights together
                        4. Maintains the existing formatting style with bullet points

                        Return ONLY the updated content for this section with no preamble or explanation.
                        """

                        updated_text = self.call_llm(
                            prompt, 
                            system_instruction="You are a Knowledge Integrator specializing in maintaining research documentation."
                        )

                        # Safety check
                        if len(updated_text) < len(current_section_text) * 0.7:
                            print(f"Warning: Updated section {section} suspiciously short - using concatenation")
                            updated_sections[section] = current_section_text + "\n\n=== NEW ADDITIONS ===\n\n" + new_section_text
                        else:
                            updated_sections[section] = updated_text.strip()

                    except Exception as e:
                        print(f"Error synthesizing section {section}: {e}")
                        # Fallback to concatenation
                        updated_sections[section] = current_section_text + "\n\n=== NEW ADDITIONS ===\n\n" + new_section_text
                else:
                    # Section doesn't exist in current learnings, just use the new content
                    updated_sections[section] = new_sections[section]

        # Reassemble the document
        updated_learnings = header + "\n\n"
        for section in sections:
            if section in updated_sections and updated_sections[section]:
                updated_learnings += f"## {section}\n\n{updated_sections[section]}\n\n"

        return updated_learnings.strip()
    
    def update_learnings(self, iteration_data: Dict) -> None:
        """Update the learnings file with insights from the current iteration"""
        try:
            # Load existing learnings
            current_learnings = self._load_learnings()
            print(f"Loaded existing learnings: {len(current_learnings)} characters")

            # Generate learnings from current batch
            print("Generating new batch learnings...")
            batch_learnings = self.generate_batch_learnings(iteration_data)
            print(f"Generated batch learnings: {len(batch_learnings)} characters")

            # If this is the first iteration, just use the batch learnings
            if not current_learnings:
                print("No existing learnings found. Using batch learnings as initial content.")
                updated_learnings = batch_learnings
            else:
                # Synthesize existing learnings with new batch learnings
                print("Synthesizing existing learnings with new batch learnings...")
                updated_learnings = self.synthesize_learnings(current_learnings, batch_learnings)
                print(f"Synthesized learnings: {len(updated_learnings)} characters")

            # Save updated learnings
            self._save_learnings(updated_learnings)
            print(f"Learnings saved successfully ({len(updated_learnings)} characters)")

        except Exception as e:
            print(f"Error updating learnings: {e}")
            traceback.print_exc()  # Print full traceback for better debugging
            
    def adjust_explore_exploit_with_llm(self) -> Tuple[int, int]:
        """
        Use LLM reasoning to adjust the explore/exploit balance based on 
        performance history and current capability insights. Adapts dynamically
        to any dataset's difficulty level rather than using fixed thresholds.
        """
        iterations = self.get_all_iterations()
        summaries = self.get_summaries()

        # If there aren't enough iterations yet, continue with current balance
        if len(iterations) < 2:
            return self.explore_rate, self.exploit_rate

        # Get the full performance history
        performance_history = []
        for summary in summaries:
            performance_history.append({
                "iteration":
                summary.get("iteration"),
                "accuracy":
                summary.get("performance", {}).get("accuracy", 0),
                "batch_size":
                summary.get("batch_size", 5),
                "explore_rate":
                summary.get("explore_rate"),
                "exploit_rate":
                summary.get("exploit_rate"),
                "strategy":
                summary.get("strategy"),
                "primary_issue":
                summary.get("primary_issue", "None identified")
            })

        # Try to get information about the best script so far
        best_script_info = None
        try:
            best_script_info = self.get_best_script_info()
        except Exception as e:
            print(f"Error getting best script info: {e}")

        # Prepare additional context for the LLM
        context = {
            "iterations_completed":
            len(summaries),
            "best_accuracy":
            best_script_info.get("accuracy", 0) if best_script_info else 0,
            "best_iteration":
            best_script_info.get("iteration", -1) if best_script_info else -1,
            "current_balance":
            f"{self.explore_rate}/{self.exploit_rate}",
            "total_examples_seen":
            len(self.seen_examples)
        }

        # Check for capability insights if available
        capability_context = {}
        if hasattr(self, 'capability_tracker'):
            capability_report = self.capability_tracker.generate_report()
            capability_context = {
                "capability_scores":
                capability_report.get("capability_scores", {}),
                "weakest_capability":
                capability_report.get("weakest_capabilities",
                                      [{}])[0].get("name", None),
                "strongest_capability":
                capability_report.get("strongest_capabilities",
                                      [{}])[0].get("name", None),
                "improvement_focus":
                capability_report.get("improvement_focus"),
                "capability_trend":
                capability_report.get("trend", {})
            }

        # Role-specific system instruction for strategy optimizer
        strategy_optimizer_system_instruction = "You are a Strategy Optimizer. Your role is to analyze performance patterns and determine the optimal balance between exploration (trying new approaches) and exploitation (refining successful approaches)."

        # Create prompt for LLM to reason about explore/exploit adjustment
        prompt = f"""
        You're optimizing the explore/exploit balance for an iterative learning system.

        Current system status:
        - Explore rate: {self.explore_rate}%
        - Exploit rate: {self.exploit_rate}%
        - Iterations completed: {context["iterations_completed"]}
        - Best accuracy so far: {context["best_accuracy"]:.2f} (from iteration {context["best_iteration"]})
        - Total examples seen: {context["total_examples_seen"]}

        Performance history (from newest to oldest):
        {json.dumps(performance_history[-5:] if len(performance_history) > 5 else performance_history, indent=2)}

        {"Capability insights:" if capability_context else ""}
        {json.dumps(capability_context, indent=2) if capability_context else ""}

        IMPORTANT GUIDELINES:
        1. ADAPTIVITY: Don't rely on fixed accuracy thresholds. Assess what constitutes "good" performance in the context of this specific dataset and task.

        2. EXPLOITATION TRIGGERS:
           - When a script performs significantly better than previous iterations
           - When a promising approach emerges that could benefit from refinement
           - When several iterations of the same approach show consistent improvement
           - When a good solution emerges after several exploration attempts

        3. EXPLORATION TRIGGERS:
           - When performance has plateaued despite exploitation attempts
           - When there's little difference between approaches tried so far
           - When the system seems stuck in a local optimum
           - When there have been several consecutive exploitation iterations with minimal improvement

        4. BALANCE CONSIDERATIONS:
           - More aggressive shifts (±20-30%) when clear patterns emerge
           - Moderate shifts (±10-20%) when trends are present but less definitive
           - Small adjustments (±5-10%) when optimizing a working approach
           - The explore/exploit rates must sum to 100
           - Consider capability insights if available

        5. DATASET CALIBRATION:
           - Do not use absolute accuracy thresholds
           - Instead, compare relative performance across iterations
           - Treat the highest achieved accuracy as a temporary ceiling
           - Consider trends rather than absolute values

        Determine the optimal explore/exploit balance based on these patterns rather than fixed thresholds.

        Provide a JSON object with:
        {{"explore_rate": <new_explore_rate_as_integer>, "exploit_rate": <new_exploit_rate_as_integer>, "rationale": "<explanation>"}}
        """

        # Call LLM to reason about adjustment
        try:
            response = self.call_llm(
                prompt,
                system_instruction=strategy_optimizer_system_instruction)
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.endswith("```"):
                response = response.split("```")[0]

            result = json.loads(response)

            # Extract values and rationale
            new_explore = int(result.get("explore_rate", self.explore_rate))
            new_exploit = int(result.get("exploit_rate", self.exploit_rate))
            rationale = result.get("rationale", "No rationale provided")

            # Ensure values are reasonable
            new_explore = max(10, min(90, new_explore))
            new_exploit = max(10, min(90, new_exploit))

            # Ensure they sum to 100
            if new_explore + new_exploit != 100:
                total = new_explore + new_exploit
                new_explore = int(round(new_explore * 100 / total))
                new_exploit = 100 - new_explore

            # Determine if this is a significant change in strategy
            strategy_shift = abs(new_explore - self.explore_rate)
            if strategy_shift >= 20:
                print(
                    f"Major strategy shift: {self.explore_rate}/{self.exploit_rate} → {new_explore}/{new_exploit}"
                )
            elif strategy_shift >= 10:
                print(
                    f"Moderate strategy shift: {self.explore_rate}/{self.exploit_rate} → {new_explore}/{new_exploit}"
                )
            else:
                print(
                    f"Minor strategy adjustment: {self.explore_rate}/{self.exploit_rate} → {new_explore}/{new_exploit}"
                )

            print(f"Rationale: {rationale}")

            return new_explore, new_exploit

        except Exception as e:
            print(f"Error adjusting explore/exploit: {e}")

            # In case of error, use a simple adaptive heuristic based on iteration count and recent performance
            if len(summaries) >= 3:
                # Get the last 3 summaries to check for trends
                recent_summaries = sorted(summaries,
                                          key=lambda x: x.get("iteration", 0),
                                          reverse=True)[:3]
                accuracies = [
                    s.get("performance", {}).get("accuracy", 0)
                    for s in recent_summaries
                ]

                # Calculate mean and standard deviation to determine what's "good" for this dataset
                all_accuracies = [
                    s.get("performance", {}).get("accuracy", 0)
                    for s in summaries
                ]
                mean_accuracy = sum(all_accuracies) / len(
                    all_accuracies) if all_accuracies else 0

                # If recent performance is improving relative to the mean, favor exploitation
                if accuracies[0] > mean_accuracy * 1.1:  # 10% better than mean
                    print(
                        "Fallback: Recent improvement detected, slightly favoring exploitation."
                    )
                    return max(self.explore_rate - 10,
                               40), min(self.exploit_rate + 10, 60)
                # If recent performance is declining relative to mean, favor exploration
                elif accuracies[
                        0] < mean_accuracy * 0.9:  # 10% worse than mean
                    print(
                        "Fallback: Recent decline detected, slightly favoring exploration."
                    )
                    return min(self.explore_rate + 10,
                               60), max(self.exploit_rate - 10, 40)

            # Default fallback: maintain current balance with a slight exploration bias for the first few iterations
            if len(summaries) < 5:
                print("Fallback: Early stages, maintaining exploration bias.")
                return min(self.explore_rate + 5,
                           70), max(self.exploit_rate - 5, 30)
            else:
                print("Fallback: Maintaining current explore/exploit balance.")
                return self.explore_rate, self.exploit_rate





    def _get_historical_data(self):
        """Gather historical data for script generation."""
        iterations = self.get_all_iterations()
        summaries = self.get_summaries()

        # Get best scripts
        best_scripts = []
        if iterations:
            for iteration in sorted(
                    iterations,
                    key=lambda x: x.get('performance', {}).get('accuracy', 0),
                    reverse=True)[:3]:
                if iteration.get('script') and iteration.get(
                        'performance', {}).get('accuracy', 0) > 0:
                    best_scripts.append({
                        "iteration": iteration.get('iteration'),
                        "accuracy": iteration.get('performance', {}).get('accuracy', 0),
                        "approach_summary": iteration.get('approach_summary', 'No summary available'),
                        "performance": iteration.get('performance', {})
                    })

        # Get approach history
        approach_history = []
        for summary in summaries:
            approach_history.append({
                "iteration": summary.get("iteration"),
                "strategy": summary.get("strategy"),
                "accuracy": summary.get("performance", {}).get("accuracy", 0),
                "approach": summary.get("approach_summary", "No summary available")
            })

        # Aggregate error analyses
        error_patterns = []
        primary_issues = []
        targeted_improvements = []

        for iteration in iterations:
            if not iteration:
                continue

            error_analysis = iteration.get("performance", {}).get("error_analysis", {})
            if error_analysis.get("error_patterns"):
                error_patterns.extend(error_analysis.get("error_patterns", []))
            if error_analysis.get("primary_issue"):
                primary_issues.append({
                    "iteration": iteration.get("iteration"),
                    "issue": error_analysis.get("primary_issue")
                })
            if error_analysis.get("targeted_improvements"):
                targeted_improvements.extend(
                    error_analysis.get("targeted_improvements", []))
            elif error_analysis.get("improvement_suggestions"):
                targeted_improvements.extend(
                    error_analysis.get("improvement_suggestions", []))
            elif error_analysis.get("recommendations"):
                targeted_improvements.extend(
                    error_analysis.get("recommendations", []))

        # Format trace insights
        trace_insights_context = ""
        for iteration in iterations[-3:]:  # Last 3 iterations
            if iteration and iteration.get("trace_insights"):
                trace_insights_context += f"\n--- Iteration {iteration.get('iteration')} Trace Insights ---\n"
                trace_insights_context += iteration.get("trace_insights", "")
                trace_insights_context += "\n"

        return {
            "iterations": iterations,
            "summaries": summaries,
            "best_scripts": best_scripts, 
            "approach_history": approach_history,
            "error_patterns": error_patterns,
            "primary_issues": primary_issues,
            "targeted_improvements": targeted_improvements,
            "trace_insights_context": trace_insights_context,
            "summaries_count": len(summaries)
        }

    def _get_approach_history(self, max_approaches=5):
        """
        Get a comprehensive history of past approaches with the approach summary,
        full script, trace insights, and error analysis.

        Args:
            max_approaches: Maximum number of past approaches to include

        Returns:
            Formatted string with comprehensive approach history
        """
        iterations = self.get_all_iterations()
        if not iterations:
            return "No previous approaches available."

        # Sort iterations by most recent first
        sorted_iterations = sorted(
            iterations, 
            key=lambda x: x.get('iteration', 0) if x and 'iteration' in x else 0, 
            reverse=True
        )

        # Take only the requested number of approaches
        approaches_to_show = sorted_iterations[:max_approaches]

        # Format each approach
        formatted_approaches = []
        for approach in approaches_to_show:
            if not approach:
                continue

            # Basic approach information
            iteration_num = approach.get('iteration', 'unknown')
            strategy = approach.get('strategy', 'unknown')
            accuracy = approach.get('performance', {}).get('accuracy', 0)

            # Get the approach summary directly
            approach_summary = approach.get('approach_summary', 'No approach summary available.')

            # Get the full script
            script = approach.get('script', 'No script available.')

            # Get trace insights directly from the iteration data
            trace_insights = approach.get('trace_insights', 'No trace insights available.')

            # Get the error analysis text report directly
            error_analysis = approach.get('performance', {}).get('error_analysis', {}).get('text_report', 'No error analysis available.')

            # Format the approach information
            formatted_approach = f"""
    === APPROACH #{iteration_num} ({strategy}, ACCURACY: {accuracy:.2f}) ===

    APPROACH SUMMARY:
    {approach_summary}

    IMPLEMENTATION:
    ```python
    {script}
    ```

    TRACE INSIGHTS:
    {trace_insights}

    ERROR ANALYSIS:
    {error_analysis}

    ===
    """
            formatted_approaches.append(formatted_approach)

        # Join all formatted approaches
        return "\n".join(formatted_approaches)

    def _format_historical_context(self, historical_data, few_shot_examples):
        """
        Format historical data into a context string using the comprehensive approach.

        Args:
            historical_data: Dictionary with historical data
            few_shot_examples: Few-shot examples string

        Returns:
            Formatted historical context string
        """
        # Get basic history information
        summaries_count = historical_data["summaries_count"]
        best_scripts = historical_data["best_scripts"]

        # Format best accuracy
        best_accuracy_str = f"{best_scripts[0].get('accuracy', 0):.2f} (iteration {best_scripts[0].get('iteration')})" if best_scripts else "None"

        # Create history summary header
        header = f"""
        ITERATION HISTORY SUMMARY:
        - Total iterations completed: {summaries_count}
        - Current explore/exploit balance: {self.explore_rate}/{self.exploit_rate}
        - Best accuracy achieved: {best_accuracy_str}
        """

        # Get comprehensive approach history
        approach_history = self._get_approach_history(max_approaches=5)

        # Combine header, approach history, and examples
        historical_context = f"{header}\n\nPREVIOUS APPROACHES:\n{approach_history}\n\n{few_shot_examples}"

        # Add multi-example prompting guidance
        historical_context += "\n\n" + PromptTemplates.get_multi_example_prompting_guidance()

        return historical_context
    
    def _get_last_scripts_context(self, iterations):
        """Get context of the last 5 scripts for exploration."""
        last_scripts_context = ""
        if not iterations:
            return last_scripts_context

        # Get the last 5 iterations, sorted by recency
        sorted_iterations = sorted(iterations, key=lambda x: x.get('iteration', 0) if x and 'iteration' in x else 0, reverse=True)
        last_scripts = []

        for i, iteration in enumerate(sorted_iterations[:5]):
            if iteration and 'script' in iteration:
                iteration_num = iteration.get('iteration', 'unknown')
                accuracy = iteration.get('performance', {}).get('accuracy', 0)
                strategy = iteration.get('strategy', 'unknown')
                approach = iteration.get('approach_summary', 'No summary available')

                # For space management, limit script size if necessary
                script = iteration.get('script', '')

                last_scripts.append({
                    "iteration": iteration_num,
                    "accuracy": accuracy,
                    "strategy": strategy,
                    "approach": approach,
                    "script": script
                })

        # Generate the context with the scripts
        if last_scripts:
            last_scripts_context = "\nPREVIOUSLY TRIED APPROACHES (LAST 5 SCRIPTS):\n"
            for script_info in last_scripts:
                last_scripts_context += f"\n=== SCRIPT FROM ITERATION {script_info['iteration']} ({script_info['strategy']}, ACCURACY: {script_info['accuracy']:.2f}) ===\n"
                last_scripts_context += f"Approach: {script_info['approach']}\n\n"
                last_scripts_context += f"```python\n{script_info['script']}\n```\n"

        return last_scripts_context

    def _get_exploitation_content(self, iterations, best_scripts):
        """Get content specific to exploitation mode."""
        # Handle best script and top scripts for exploitation
        best_script_to_exploit = None
        top_scripts_to_exploit = []
        best_script_code = ""
        top_scripts_content = ""

        if best_scripts:
            best_script_to_exploit = best_scripts[0]
            for iteration in iterations:
                if iteration.get("iteration") == best_script_to_exploit.get("iteration"):
                    best_script_to_exploit["script"] = iteration.get("script", "")
                    break

            # Get top performing scripts for exploitation instead of just the best one
            # Get the top 2-3 performing scripts (depending on how many are available)
            top_count = min(3, len(best_scripts))
            for i in range(top_count):
                script_info = best_scripts[i]
                # Find the full script content
                for iteration in iterations:
                    if iteration.get("iteration") == script_info.get("iteration"):
                        script_info["script"] = iteration.get("script", "")
                        break
                top_scripts_to_exploit.append(script_info)

        # Format best script code
        if best_script_to_exploit and 'script' in best_script_to_exploit:
            best_script_code = f"\nFULL SCRIPT TO REFINE:\n```python\n{best_script_to_exploit.get('script', '')}\n```"

        # Generate content for multiple top scripts
        if top_scripts_to_exploit:
            for i, script_info in enumerate(top_scripts_to_exploit):
                # Format the accuracy string
                accuracy_str = f"{script_info.get('accuracy', 0):.2f}"

                script_content = ""
                if 'script' in script_info:
                    script_content = f"\n```python\n{script_info.get('script', '')}\n```"

                top_scripts_content += f"\nTOP PERFORMING APPROACH #{i+1}:\n"
                top_scripts_content += f"Iteration: {script_info.get('iteration', 'Unknown')}\n"
                top_scripts_content += f"Accuracy: {accuracy_str}\n"
                top_scripts_content += f"Approach Summary: {script_info.get('approach_summary', 'No summary available')}\n"

                # Only include full code for the best script to avoid making prompt too long
                if i == 0:  # Only for the top script
                    top_scripts_content += f"\nFULL SCRIPT TO REFINE:{script_content}\n"
                else:  # For other scripts, mention they're available for reference
                    top_scripts_content += f"\nKey approach aspects (full code available for reference)\n"

        return {
            "best_script_to_exploit": best_script_to_exploit,
            "top_scripts_to_exploit": top_scripts_to_exploit,
            "best_script_code": best_script_code,
            "top_scripts_content": top_scripts_content
        }

    def _generate_validated_script(self, prompt, system_instruction):
        """Generate a script with LLM and validate it."""
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Call LLM to generate script
            response = self.call_llm(prompt, system_instruction=system_instruction)

            # Extract code block from response
            if "```python" in response:
                script = response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                script = response.split("```")[1].split("```")[0].strip()
            else:
                script = response.strip()

            # Validate script syntax
            try:
                # Test if the script can be parsed
                import ast
                ast.parse(script)

                # Script is syntactically valid
                print(f"Generated valid script on attempt {attempts}")

                # Save the script to a file
                script_path = self.scripts_dir / f"script_iteration_{self.current_iteration}.py"
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script)

                return script

            except SyntaxError as e:
                print(f"Syntax error in generated script (attempt {attempts}/{max_attempts}): {e}")

                if attempts >= max_attempts:
                    print("\n***\nFALLBACK: Maximum attempts reached. Using fallback script from library.\n***\n")
                    # Get fallback script from the library
                    fallback_script = FallbackScripts.basic_fallback()
                    script_path = self.scripts_dir / f"script_iteration_{self.current_iteration}.py"
                    with open(script_path, 'w', encoding='utf-8') as f:
                        f.write(fallback_script)
                    return fallback_script

                # Try again with a more explicit instruction about the error
                prompt = PromptTemplates.build_error_correction_prompt(str(e))


    def _get_example_problems(self, num_examples=3):
        """
        Get example problems from previous iterations or training set.
        Uses cold start examples if it's the first iteration, 
        otherwise randomly selects from past seen examples.

        Args:
            num_examples: Number of examples to return

        Returns:
            List of example problems with question, answer, and id fields.
        """
        example_problems = []

        # Cold start - use training examples for first iteration
        if self.current_iteration == 0:
            print("First iteration - using training examples")
            # For initial training, get examples from a fixed training set
            training_examples = self.get_training_examples(5)
            for i, example in enumerate(training_examples):
                example_problems.append({
                    "id": i,
                    "question": self.dataset_loader.get_example_input(example),
                    "answer": self.dataset_loader.get_example_output(example)
                })
            return example_problems[:num_examples]  # Return requested number of examples

        # For subsequent iterations, collect all samples from previous iterations
        iterations = self.get_all_iterations()
        all_samples = []

        # Gather all samples from all previous iterations
        for iteration in iterations:
            if iteration and 'samples' in iteration:
                all_samples.extend(iteration.get('samples', []))

        # If we have samples, randomly select from them
        if all_samples:
            # Shuffle the samples
            import random
            random.shuffle(all_samples)

            # Select up to num_examples samples
            selected_samples = all_samples[:num_examples] if len(all_samples) > num_examples else all_samples

            # Format the selected samples
            for i, sample in enumerate(selected_samples):
                example_problems.append({
                    "id": i,
                    "question": sample.get("question", ""),  # Use universal "question" field
                    "answer": sample.get("answer", "")       # Use universal "answer" field
                })

            print(f"Randomly selected {len(example_problems)} examples from {len(all_samples)} past samples")

        # If we somehow have no previous samples (unlikely but possible), fall back to training examples
        if not example_problems:
            print(f"No previous samples found - falling back to training examples")
            training_examples = self.get_training_examples(5)
            for i, example in enumerate(training_examples):
                example_problems.append({
                    "id": i,
                    "question": self.dataset_loader.get_example_input(example),
                    "answer": self.dataset_loader.get_example_output(example)
                })

        return example_problems[:num_examples]  # Return requested number of examples
    
    def generate_script_with_llm(self, is_exploration: bool) -> str:
        """
        Use the LLM to generate a script to solve dataset problems.
        Refactored to use helper methods for better organization and maintainability.
        """
        # Get example problems from previous iterations or training
        example_problems = self._get_example_problems(num_examples=3)

        # Load accumulated learnings
        accumulated_learnings = self._load_learnings()

        # Get examples from the library
        example_library = ExampleSets.get_standard_examples()
        gemini_api_example = example_library["gemini_api_example"]
        few_shot_examples = ExampleSets.get_few_shot_examples_text()

        # Get historical analysis data
        iterations = self.get_all_iterations()
        summaries = self.get_summaries()

        # Get best scripts
        best_scripts = []
        if iterations:
            for iteration in sorted(
                    iterations,
                    key=lambda x: x.get('performance', {}).get('accuracy', 0),
                    reverse=True)[:3]:
                if iteration.get('script') and iteration.get(
                        'performance', {}).get('accuracy', 0) > 0:
                    best_scripts.append({
                        "iteration": iteration.get('iteration'),
                        "accuracy": iteration.get('performance', {}).get('accuracy', 0),
                        "approach_summary": iteration.get('approach_summary', 'No summary available'),
                        "performance": iteration.get('performance', {})
                    })

        # Format best accuracy
        best_accuracy_str = f"{best_scripts[0].get('accuracy', 0):.2f} (iteration {best_scripts[0].get('iteration')})" if best_scripts else "None"

        # Create comprehensive historical context with approaches
        historical_context = f"""ITERATION HISTORY SUMMARY:
    - Total iterations completed: {len(summaries)}
    - Current explore/exploit balance: {self.explore_rate}/{self.exploit_rate}
    - Best accuracy achieved: {best_accuracy_str}

    PREVIOUS APPROACHES:
    {self._get_approach_history(max_approaches=5)}
    """

        # Add few-shot examples to historical context
        historical_context += f"\n\n{few_shot_examples}"
        historical_context += "\n\n" + PromptTemplates.get_multi_example_prompting_guidance()

        # Get capability insights
        capability_report = None
        capability_guidance = ""
        if hasattr(self, 'capability_tracker'):
            capability_report = self.capability_tracker.generate_report()
            if capability_report:
                capability_guidance = self._generate_capability_guidance(capability_report)

        # Format contexts for templates
        learning_context = ""
        if accumulated_learnings:
            learning_context = f"""
            ACCUMULATED LEARNINGS FROM PREVIOUS ITERATIONS:
            {accumulated_learnings}
            """

        capability_context = ""
        if capability_guidance:
            capability_context = f"""
            CAPABILITY ASSESSMENT & IMPROVEMENT GUIDANCE:
            {capability_guidance}
            """

        # Set approach type and system instruction
        approach_type = "exploration" if is_exploration else "exploitation"
        script_generator_system_instruction = f"{self.system_prompt}\n\nYou are now acting as a Script Generator for an {approach_type} task. Your goal is to create a Python script that uses LLM-driven agentic approaches with chain-of-thought reasoning, agentic LLM patterns, and python to solve the problem examples provided."

        # Build appropriate prompt based on strategy
        if is_exploration:
            # Get last scripts context for exploration
            last_scripts_context = ""
            if iterations:
                # Get the last 5 iterations, sorted by recency
                sorted_iterations = sorted(iterations, key=lambda x: x.get('iteration', 0) if x and 'iteration' in x else 0, reverse=True)
                last_scripts = []

                for i, iteration in enumerate(sorted_iterations[:5]):
                    if iteration and 'script' in iteration:
                        iteration_num = iteration.get('iteration', 'unknown')
                        accuracy = iteration.get('performance', {}).get('accuracy', 0)
                        strategy = iteration.get('strategy', 'unknown')
                        approach = iteration.get('approach_summary', 'No summary available')

                        # For space management, limit script size if necessary
                        script = iteration.get('script', '')

                        last_scripts.append({
                            "iteration": iteration_num,
                            "accuracy": accuracy,
                            "strategy": strategy,
                            "approach": approach,
                            "script": script
                        })

                # Generate the context with the scripts
                if last_scripts:
                    last_scripts_context = "\nPREVIOUSLY TRIED APPROACHES (LAST 5 SCRIPTS):\n"
                    for script_info in last_scripts:
                        last_scripts_context += f"\n=== SCRIPT FROM ITERATION {script_info['iteration']} ({script_info['strategy']}, ACCURACY: {script_info['accuracy']:.2f}) ===\n"
                        last_scripts_context += f"Approach: {script_info['approach']}\n\n"
                        last_scripts_context += f"```python\n{script_info['script']}\n```\n"

            # Build exploration prompt
            prompt = PromptTemplates.build_exploration_prompt(
                example_problems, 
                historical_context, 
                last_scripts_context,
                learning_context, 
                capability_context, 
                gemini_api_example
            )
            print (prompt)
        else:
            # Get exploitation-specific content
            best_script_to_exploit = None
            top_scripts_to_exploit = []

            if best_scripts:
                best_script_to_exploit = best_scripts[0]
                for iteration in iterations:
                    if iteration.get("iteration") == best_script_to_exploit.get("iteration"):
                        best_script_to_exploit["script"] = iteration.get("script", "")
                        break

                # Get top performing scripts for exploitation instead of just the best one
                # Get the top 2-3 performing scripts (depending on how many are available)
                top_count = min(3, len(best_scripts))
                for i in range(top_count):
                    script_info = best_scripts[i]
                    # Find the full script content
                    for iteration in iterations:
                        if iteration.get("iteration") == script_info.get("iteration"):
                            script_info["script"] = iteration.get("script", "")
                            break
                    top_scripts_to_exploit.append(script_info)

            # Format best script code
            best_script_code = ""
            if best_script_to_exploit and 'script' in best_script_to_exploit:
                best_script_code = f"\nFULL SCRIPT TO REFINE:\n```python\n{best_script_to_exploit.get('script', '')}\n```"

            # Generate content for multiple top scripts
            top_scripts_content = ""
            if top_scripts_to_exploit:
                for i, script_info in enumerate(top_scripts_to_exploit):
                    # Format the accuracy string
                    accuracy_str = f"{script_info.get('accuracy', 0):.2f}"

                    script_content = ""
                    if 'script' in script_info:
                        script_content = f"\n```python\n{script_info.get('script', '')}\n```"

                    top_scripts_content += f"\nTOP PERFORMING APPROACH #{i+1}:\n"
                    top_scripts_content += f"Iteration: {script_info.get('iteration', 'Unknown')}\n"
                    top_scripts_content += f"Accuracy: {accuracy_str}\n"
                    top_scripts_content += f"Approach Summary: {script_info.get('approach_summary', 'No summary available')}\n"

                    # Only include full code for the best script to avoid making prompt too long
                    if i == 0:  # Only for the top script
                        top_scripts_content += f"\nFULL SCRIPT TO REFINE:{script_content}\n"
                    else:  # For other scripts, mention they're available for reference
                        top_scripts_content += f"\nKey approach aspects (full code available for reference)\n"

            # Build exploitation prompt
            prompt = PromptTemplates.build_exploitation_prompt(
                example_problems, 
                historical_context, 
                best_script_code, 
                top_scripts_content,
                learning_context, 
                capability_context, 
                gemini_api_example
            )

        # Generate and validate script
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Call LLM to generate script with the specific system instruction
            response = self.call_llm(
                prompt, system_instruction=script_generator_system_instruction)

            # Extract code block from response
            if "```python" in response:
                script = response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                script = response.split("```")[1].split("```")[0].strip()
            else:
                script = response.strip()

            # Validate script syntax
            try:
                # Test if the script can be parsed
                import ast
                ast.parse(script)

                # Script is syntactically valid
                print(f"Generated valid script on attempt {attempts}")

                # Save the script to a file
                script_path = self.scripts_dir / f"script_iteration_{self.current_iteration}.py"
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script)

                return script

            except SyntaxError as e:
                print(
                    f"Syntax error in generated script (attempt {attempts}/{max_attempts}): {e}"
                )

                if attempts >= max_attempts:
                    print(
                        "\n***\nFALLBACK: Maximum attempts reached. Using fallback script from library.\n***\n"
                    )
                    # Get fallback script from the library
                    fallback_script = FallbackScripts.basic_fallback()
                    script_path = self.scripts_dir / f"script_iteration_{self.current_iteration}.py"
                    with open(script_path, 'w', encoding='utf-8') as f:
                        f.write(fallback_script)

                    return fallback_script

                # Try again with a more explicit instruction about the error
                prompt = PromptTemplates.build_error_correction_prompt(str(e))
    
    def log_script_error_to_learnings(self, error_info: str, script_snippet):
        """
        Add script errors to learnings.txt in a minimal way.

        Args:
            error_info: Description of the error
            script_snippet: Optional short snippet from the script (if available) # DISABLED
        """
        try:
            # Load existing learnings
            current_learnings = self._load_learnings()

            # Create a timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Format the entry for learnings.txt
            new_learning = f"""

    === SCRIPT ERROR ENCOUNTERED [{timestamp}] ===
    {error_info}
    """



            new_learning += """
    === END SCRIPT ERROR ===

    """

            # Add to learnings file
            updated_learnings = current_learnings + new_learning
            self._save_learnings(updated_learnings)

            print("Added script error to learnings.txt")

        except Exception as e:
            print(f"Error logging script error to learnings: {e}")
            # Don't block the main process if logging fails
    
    def repair_script_with_llm(self, script: str, question: str, error_output: str) -> str:
        """
        Use the LLM to repair a script that had execution errors.

        Args:
            script: The original script with errors
            question: The input question that caused the error
            error_output: The error output from executing the script

        Returns:
            A repaired script
        """
        # Role-specific system instruction for the script repairer
        repairer_system_instruction = "You are a Script Repair Specialist. Your task is to analyze error outputs and fix scripts to make them execute correctly. You ARE NOT to check if the answer is correct, only if the script ran successfully."


        prompt = f"""
        I need you to repair a Python script that's encountering errors.

        Here's the original script:
        ```python
        {script}
        ```

        When executing this script with the following input:
        ```
        {question}
        ```

        It produces this error:
        ```
        {error_output}
        ```

        Please analyze the error and provide a corrected version of the script that will fix the issue.
        Focus specifically on the error shown above.

        Return ONLY the complete fixed script without explanations.

        If there are JSON errors, you should attempt to remove calls like json.loads() to explicitly read the output of one LLM call as strict json data. JSON formatting is good to use to structure information as inputs and outputs, but attempting to have functions process JSON data explicitly with strict built-in functionality is error prone due to formatting issues and additional text that appears as documentation, reasoning, or comments when generated by an LLM. When passing data from one LLM call into another another LLM call, it best to process the inputs as plain text rather than trying to load it in strict json format.

        Break it down, think step by step about how to analyze the error and fix the script. When you have an approach that works, return ONLY the complete fixed script without explanations.
        """
        
        try:
            response = self.call_llm(prompt, system_instruction=repairer_system_instruction)

            # Extract code block from response
            if "```python" in response:
                fixed_script = response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                fixed_script = response.split("```")[1].split("```")[0].strip()
            else:
                fixed_script = response.strip()

            return fixed_script
        except Exception as e:
            print(f"Error repairing script with LLM: {e}")
            return script  # Return original script if repair fails


    def check_output_for_errors(self, output: str, answer: str) -> tuple:
        """
        Use LLM to check if the script output contains error messages.

        Args:
            output: Raw output from script execution
            answer: The answer returned by the script

        Returns:
            Tuple of (has_error, error_description)
        """
        # Role-specific system instruction for error checker
        error_checker_system_instruction = "You are an Error Detection Specialist. Your task is to determine if output contains error messages."

        prompt = f"""
        Analyze this output from a script execution and determine if it contains error messages, execution errors, or could not run as intended. You are not to check if the answer is correct, only if the script ran successfully.

        Raw output:
        ```
        {output}
        ```

        Script's answer:
        ```
        {answer}
        ```

        Does this output contain error messages or indicate the script failed to run properly?

        Respond with ONLY "ERROR: <brief error description>" if you detect an error message or execution problem.

        If the output seems normal and indicates successful execution, respond with ONLY "SUCCESS: Script executed normally."

        Your response:
        """

        try:
            response = self.call_llm(prompt, system_instruction=error_checker_system_instruction)

            # Check if the response indicates an error
            response = response.strip()
            if response.startswith("ERROR:"):
                return True, response
            else:
                return False, response
        except Exception as e:
            # If there's an error calling the LLM, assume there's an issue with the script to be safe
            print(f"Error checking for errors with LLM: {e}")
            return True, f"Error checking output with LLM: {str(e)}"

    def attempt_script_repair(self, script: str, max_attempts: int = 3) -> str:
        """
        Attempt to repair a script by testing it with a fixed training example.
        Uses an LLM to determine if the output contains errors.

        Args:
            script: The script to repair
            max_attempts: Maximum number of repair attempts

        Returns:
            The best script (original or repaired)
        """
        print("Attempting script repair and verification...")

        # Save current dataset position
        original_index = self.dataset_loader.current_index

        try:
            # Temporarily set index to 0 to get the first training example
            self.dataset_loader.current_index = 0
            test_example = self.dataset_loader.get_examples(1)[0]
            test_question_str = self.dataset_loader.get_example_input(test_example)
            print(f"Using test question for repair: {test_question_str[:50]}...")

            # Create a properly formatted sample dictionary
            test_sample = {
                "question": test_question_str,
                "answer": self.dataset_loader.get_example_output(test_example),
                "id": "test_sample"
            }

            current_script = script
            best_script = script
            best_result = None

            for attempt in range(max_attempts):
                # Try to execute the current script
                result = self.execute_script(current_script, test_sample)
    
                # Get the output and answer
                output = result.get("output", "")
                answer = result.get("answer", "")
    
                # Check if the script execution succeeded
                if result.get("success", False):
                    # Use LLM to check if the output contains error messages
                    has_error, error_description = self.check_output_for_errors(output, answer)
    
                    # If no errors detected, return this script
                    if not has_error:
                        print(f"Script passed verification on attempt {attempt + 1}!")
                        print(f"LLM verdict: {error_description}")
                        return current_script
    
                    print(f"LLM detected error in output: {error_description}")
    
                print(f"Repair attempt {attempt + 1}/{max_attempts} needed - script has errors.")
    
                # Save the best result so far (prioritize scripts that at least executed partially)
                if best_result is None or (
                    result.get("success", False) and 
                    (not best_result.get("success", False) or
                     ("error" not in answer.lower() and "error" in best_result.get("answer", "").lower()))
                ):
                    best_script = current_script
                    best_result = result
    
                # Extract error information
                error = result.get("error", "Unknown error")
    
                # If LLM detected an error, use that description
                if result.get("success", False) and locals().get('error_description'):
                    error = error_description
    
                # Log the error to learnings.txt
                # Get a small snippet from the script (focus on main function if possible)
                script_snippet = ""
                if "def main" in current_script:
                    lines = current_script.split('\n')
                    main_line = next((i for i, line in enumerate(lines) if "def main" in line), -1)
                    if main_line >= 0:
                        start_line = max(0, main_line - 2)
                        end_line = min(len(lines), main_line + 8)
                        script_snippet = '\n'.join(lines[start_line:end_line])
    
                # Log the error
                self.log_script_error_to_learnings(
                    f"Error detected during script repair (attempt {attempt+1}): {error}",
                    script_snippet
                )
    
                # Don't try to repair if this is the last attempt
                if attempt >= max_attempts - 1:
                    print("Maximum repair attempts reached, using best version.")
                    break
    
                # Repair the script using LLM
                print(f"Repairing script with LLM based on error: {error[:100]}...")
                current_script = self.repair_script_with_llm(current_script, test_question_str, output)
    
                # Validate the script syntax
                try:
                    import ast
                    ast.parse(current_script)
                    print("Repaired script passed syntax check.")
                except SyntaxError as e:
                    print(f"Repaired script has syntax error: {e}")
                    # Keep the previous best script if the repair introduced syntax errors
                    current_script = best_script
    
            print(f"Script repair completed with best available version after {max_attempts} attempts.")
            return best_script

        finally:
            # Always restore original position, even if an error occurs
            self.dataset_loader.current_index = original_index
            print(f"Restored dataset position to {original_index}")
    

    def execute_script(self, script: str, sample: Dict) -> Dict:
        """
        Execute the generated script on a sample and return the result.
        Uses automatic debugging if the script fails with specific errors.
        """
        # Create a temporary script file
        script_path = self.scripts_dir / f"current_script_{self.current_iteration}.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

        # Get the question string from the sample
        question = sample.get("question", "")
        sample_id = sample.get("id", f"example_{self.current_iteration}")

        # Set up trace file in the archive directory
        trace_file = self.archive_dir / f"trace_iteration_{self.current_iteration}.jsonl"

        # Create a test harness for the script with enhanced tracing
        test_script = f"""import sys
import traceback
import os
import json
import datetime
import inspect
import functools
import importlib.util

# Add the scripts directory to the path
sys.path.append("{self.scripts_dir}")

# Ensure the Gemini API key is available to the script
os.environ["GEMINI_API_KEY"] = "{os.environ.get('GEMINI_API_KEY')}"

# Configure tracing
trace_file = "{trace_file}"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {{
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": {self.current_iteration},
        "sample_id": "{sample_id}",
        "question": {repr(question)}
    }}
    f.write(json.dumps(start_entry) + "\\n")

# More reliable method for getting caller information
def get_real_caller():
    #Get information about the caller, skipping intermediate functions like wrappers and decorators.
    frames = inspect.stack()
    # Skip first 2 frames (this function and immediate caller)
    for frame_info in frames[2:]:
        # Get the frame's module
        frame_module = frame_info.frame.f_globals.get('__name__', '')
        # If this frame is from our module (not from system libraries)
        if frame_module == 'current_script_{self.current_iteration}':
            # Check if it's not the call_llm function itself
            if frame_info.function != 'call_llm' and 'wrapper' not in frame_info.function:
                return {{
                    "function": frame_info.function,
                    "filename": frame_info.filename,
                    "lineno": frame_info.lineno
                }}
    # Fallback if we can't find a suitable caller
    return {{"function": "unknown", "filename": "unknown", "lineno": 0}}

# Create a tracing decorator for call_llm
def trace_call_llm(func):
    @functools.wraps(func)
    def wrapper(prompt, system_instruction=None):
        # Get caller information using our improved method
        caller_info = get_real_caller()

        # Create trace entry with caller information
        trace_entry = {{
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "llm_call",
            "iteration": {self.current_iteration},
            "sample_id": "{sample_id}",
            "function": "call_llm",
            "caller": caller_info,
            "input": {{
                "prompt": prompt,
                "system_instruction": system_instruction
            }}
        }}

        # Call the original function
        try:
            result = func(prompt, system_instruction)

            # Log successful response
            trace_entry["output"] = result
            trace_entry["status"] = "success"

            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trace_entry) + "\\n")

            return result

        except Exception as e:
            # Log error
            trace_entry["error"] = str(e)
            trace_entry["status"] = "error"
            trace_entry["traceback"] = traceback.format_exc()

            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trace_entry) + "\\n")

            raise

    return wrapper

try:
    # Import the script as a module
    spec = importlib.util.spec_from_file_location(
        "current_script_{self.current_iteration}", 
        "{script_path}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Patch call_llm function if it exists
    if hasattr(module, 'call_llm'):
        original_call_llm = module.call_llm
        module.call_llm = trace_call_llm(original_call_llm)

    # Also patch any other functions that might call LLM directly
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            try:
                source = inspect.getsource(obj)
                if 'generate_content' in source and obj is not getattr(module, 'call_llm', None):
                    setattr(module, name, trace_call_llm(obj))
            except:
                pass

    # Execute the main function with the question string
    question = {repr(question)}

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {{
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": {self.current_iteration},
            "sample_id": "{sample_id}",
            "answer": str(answer)
        }}
        f.write(json.dumps(end_entry) + "\\n")

    # Print the answer for capture
    print("ANSWER_START")
    print(answer)
    print("ANSWER_END")

except Exception as e:
    # Log the error
    with open(trace_file, 'a', encoding='utf-8') as f:
        error_entry = {{
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_error",
            "iteration": {self.current_iteration},
            "sample_id": "{sample_id}",
            "error": str(e),
            "traceback": traceback.format_exc()
        }}
        f.write(json.dumps(error_entry) + "\\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
"""

        test_path = self.scripts_dir / f"test_script_{self.current_iteration}.py"
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_script)

        # The rest of your function remains unchanged...
        debug_attempts = 0
        max_debug_attempts = 3

        # Rest of the function as before...

        while debug_attempts <= max_debug_attempts:
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, str(test_path)],
                    capture_output=True,
                    text=True,
                    timeout=60  # 60 second timeout - increased for LLM API calls
                )

                # Parse the output
                output = result.stdout + result.stderr

                if "ANSWER_START" in output and "ANSWER_END" in output:
                    answer = output.split("ANSWER_START")[1].split(
                        "ANSWER_END")[0].strip()
                    return {
                        "success": True,
                        "answer": answer,
                        "output": output,
                        "trace_file": str(trace_file)  # Return trace file path
                    }
                elif "ERROR_START" in output and "ERROR_END" in output:
                    error = output.split("ERROR_START")[1].split(
                        "ERROR_END")[0].strip()

                    # If we've reached max debug attempts or this isn't a "missing main" error, return the error
                    if debug_attempts >= max_debug_attempts or "cannot import name 'main'" not in error:
                        return {
                            "success": False,
                            "error": error,
                            "output": output,
                            "trace_file": str(trace_file)  # Return trace file path
                        }

                    # Try to debug the script
                    debug_attempts += 1
                    print(
                        f"  Debugging attempt {debug_attempts}/{max_debug_attempts}..."
                    )

                    # Apply debugging fixes
                    self._debug_script(script_path)

                    # Continue to next attempt
                    continue
                else:
                    return {
                        "success": False,
                        "error": "Unknown execution error",
                        "output": output,
                        "trace_file": str(trace_file)  # Return trace file path
                    }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": "Script execution timed out (60 seconds)",
                    "output": "Timeout",
                    "trace_file": str(trace_file)  # Return trace file path
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "output": traceback.format_exc(),
                    "trace_file": str(trace_file)  # Return trace file path
                }

        # If we get here, we've exhausted our debug attempts
        return {
            "success": False,
            "error": "Maximum debug attempts reached. Could not fix script.",
            "output": "Debug failure",
            "trace_file": str(trace_file)  # Return trace file path
        }
    
    def _debug_script(self, script_path: Path) -> bool:
        """
        Debug a script by checking for common issues and fixing them.

        Args:
            script_path: Path to the script file

        Returns:
            bool: True if debugging was successful, False otherwise
        """
        print(f"  Analyzing script: {script_path}")

        try:
            # Read the script content
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()

            # Check if the script has a 'main' function
            has_main_function = "def main(" in script_content
            if not has_main_function:
                print(
                    "  Issue detected: Script does not have a 'main' function")

                # Look for possible main function alternatives
                possible_main_functions = []
                for line in script_content.split('\n'):
                    if line.strip().startswith("def ") and "(" in line:
                        function_name = line.strip().split("def ")[1].split(
                            "(")[0].strip()
                        if function_name != "main" and (
                                "solve" in function_name
                                or "process" in function_name
                                or "answer" in function_name or
                                function_name.lower() == "process_question"):
                            possible_main_functions.append(function_name)

                if possible_main_functions:
                    primary_function = possible_main_functions[0]
                    print(
                        f"  Found potential main function: {primary_function}")

                    # Add a main function that calls the primary function
                    new_content = script_content + f"\n\ndef main(question):\n    return {primary_function}(question)\n"

                    # Save the modified script
                    with open(script_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                    print(
                        f"  Added 'main' function wrapper for {primary_function}"
                    )
                    return True
                else:
                    # If no suitable function found, try to identify the primary function
                    # This is a more aggressive fix for when function names don't match patterns
                    function_names = []
                    for line in script_content.split('\n'):
                        if line.strip().startswith("def ") and "(" in line:
                            function_name = line.strip().split(
                                "def ")[1].split("(")[0].strip()
                            if function_name != "main":
                                function_names.append(function_name)

                    if function_names:
                        # Choose the first defined function as the main function
                        primary_function = function_names[0]
                        print(
                            f"  Using first defined function as main: {primary_function}"
                        )

                        # Add a main function that calls this function
                        new_content = script_content + f"\n\ndef main(question):\n    return {primary_function}(question)\n"

                        # Save the modified script
                        with open(script_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)

                        print(
                            f"  Added 'main' function wrapper for {primary_function}"
                        )
                        return True
                    else:
                        print("  No functions found to use as main")
                        return False
            else:
                print("  Script already has a 'main' function - no fix needed")
                return True

        except Exception as e:
            print(f"  Error debugging script: {e}")
            return False

    def evaluate_with_llm(self, samples: List[Dict], results: List[Dict]) -> Dict:
        """
        Use the LLM to evaluate results and perform detailed error analysis.
        Modified to work with the standardized sample format and use universal field names.
        """
        evaluations = []

        # First, perform semantic evaluation with LLM
        correct_count = 0
        for i, (sample, result) in enumerate(zip(samples, results)):
            if not result.get("success"):
                evaluations.append({
                    "sample_id": i,
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "output": result.get("output", "No output captured"),
                    "match": False,
                    "capability_failures": ["execution"]
                })
                continue

            # Compare with golden answer using LLM
            if not result.get("evaluation"):
                golden_answer = sample.get("answer", "").strip()  # Use universal "answer" field
                system_answer = result.get("answer", "").strip()
                evaluation = self.evaluate_answer_with_llm(system_answer, golden_answer)
                result["evaluation"] = evaluation
                result["match"] = evaluation.get("match", False)

            if result.get("match", False):
                correct_count += 1
                # For successful cases, still include the output
                evaluations.append({
                    "sample_id": i,
                    "success": True,
                    "system_answer": result.get("answer", "").strip(),
                    "golden_answer": sample.get("answer", "").strip(),  # Use universal "answer" field
                    "output": result.get("output", "No output captured"),
                    "match": True,
                    "evaluation": result.get("evaluation", {})
                })
            else:
                # For failed cases, include output for analysis
                evaluations.append({
                    "sample_id": i,
                    "success": True,
                    "system_answer": result.get("answer", "").strip(),
                    "golden_answer": sample.get("answer", "").strip(),  # Use universal "answer" field
                    "output": result.get("output", "No output captured"),
                    "match": False,
                    "evaluation": result.get("evaluation", {}),
                    "capability_failures": []
                })

        # Calculate accuracy
        accuracy = correct_count / len(samples) if samples else 0

        error_samples = []
        success_samples = []  # NEW: Collect successful examples

        for i, eval_data in enumerate(evaluations):
            sample = samples[i]
            if not eval_data.get("match"):
                # [Existing error collection code]
                error_samples.append({
                    "sample_id": i,
                    "question": sample.get("question", ""),
                    "system_answer": eval_data.get("system_answer", ""),
                    "golden_answer": eval_data.get("golden_answer", ""),
                    "error_message": eval_data.get("error", ""),
                    "output": eval_data.get("output", "No output captured"),
                    "explanation": eval_data.get("evaluation", {}).get("explanation", "")
                })
            else:  # NEW: Collection of successful examples
                success_samples.append({
                    "sample_id": i,
                    "question": sample.get("question", ""),
                    "system_answer": eval_data.get("system_answer", ""),
                    "golden_answer": eval_data.get("golden_answer", ""),
                    "output": eval_data.get("output", "No output captured"),
                    "explanation": eval_data.get("evaluation", {}).get("explanation", "")
                })
    
        print(f"Found {len(error_samples)} error samples and {len(success_samples)} success samples for analysis")
    
        # NEW APPROACH: Generate a text-based capability report instead of trying to parse complex JSON
        capability_report_text = ""
        error_analysis_text = ""
    
        if error_samples or success_samples: # do it regardless
            # Modified system instruction for error analyzer that produces text instead of JSON
            error_analyzer_system_instruction = "You are a Forensic Error Analyzer specializing in debugging complex reasoning systems. Your task is to perform deep, deliberate analysis of errors and successes in past performance to identify specific failure points, successful strategies, and propose targeted improvements."
    
            # Modified prompt for LLM to generate a natural language analysis, now including raw outputs
            prompt = f"""
            Perform a thorough forensic analysis of these error cases in our AI problem-solving system.
    
            For each error case, think step-by-step through what happened:
    
            ERROR CASES:
            {json.dumps(error_samples, indent=2)}

            SUCCESS CASES:
            {json.dumps(success_samples, indent=2)} 
    
            ANALYSIS INSTRUCTIONS:

            SUCCESS PATTERNS: Analyze what led to correct answers. What reasoning steps, approaches, or techniques were effective? What can we learn from these successes and apply to future iterations or refine for future iterations?
            
            1. TRACE THE REASONING PATH: For each error case, reconstruct the likely reasoning path the system took. Where precisely did the reasoning go wrong? In the future can you add print statements and intermediate outputs such that you can see them later to determine why things are going wrong?
    
            2. IDENTIFY SPECIFIC FAILURE POINTS: What exact component, reasoning step, or assumption failed? Pay special attention to the 'output' field which contains the raw execution output with detailed information about errors and execution flow.
    
            3. ANALYZE RAW OUTPUTS: Look for specific error patterns (like JSONDecodeError, TypeError, etc.) in the output field that may reveal implementation issues or runtime errors.
    
            4. COMPARE WITH CORRECT SOLUTION: Analyze how the golden answer's reasoning differs from the system's approach.
    
            5. FIND PATTERNS ACROSS ERRORS: Are there common failure modes or recurring issues?
    
            6. MAP FAILURES TO SYSTEM CAPABILITIES: For each error, identify which of these capabilities failed:
               - information_extraction: Extracting relevant information from the problem statement
               - constraint_handling: Identifying and applying constraints correctly
               - solution_generation: Generating valid potential solutions
               - solution_verification: Verifying solutions against constraints
               - decision_making: Making a final decision on the best solution
    
            7. EVALUATE HYPOTHESIS: Was the hypothesis for this iteration correct? If not, what went wrong?
    
            FORMAT YOUR RESPONSE AS A STRUCTURED TEXT REPORT with the following sections:
    
            ## RUNTIME ERRORS
            (Identify and categorize any error messages or exceptions found in the 'output' fields)
    
            ## STRENGTHS
            (List 2-3 specific strengths of the current approach)
    
            ## WEAKNESSES
            (List 2-3 specific weaknesses identified from error cases)
    
            ## CRITICAL BOTTLENECKS
            (The 1-2 critical bottlenecks limiting performance)
    
            ## ERROR PATTERNS
            (Recurring patterns across multiple errors)
    
            ## PRIMARY ISSUE
            (The single most critical problem to fix - be very specific)
    
            ## IMPROVEMENT AREAS
            (Specific capabilities that need the most improvement)
    
            ## IMPROVEMENT SUGGESTIONS
            (Specific, actionable changes to fix the identified issues)
    
            ## CAPABILITY MAPPING
            (For each sample with errors, list which capabilities failed)
    
            BE EXTREMELY SPECIFIC IN YOUR ANALYSIS. If you see technical errors in the outputs (like JSONDecodeError or TypeError), highlight these explicitly and explain their implications.
            """
    
            # Call LLM for detailed error analysis as text
            try:
                error_analysis_text = self.call_llm(prompt, system_instruction=error_analyzer_system_instruction)
                print("Generated error analysis text report")
    
                # Extract useful information for the dictionary return
                error_analysis = {
                    "text_report": error_analysis_text
                }
    
                # Try to extract basic structured information from the text report
                if "## STRENGTHS" in error_analysis_text and "## WEAKNESSES" in error_analysis_text:
                    strengths = []
                    weaknesses = []
                    primary_issue = "See full text report"
                    improvement_suggestions = []
                    runtime_errors = []
    
                    # Extract runtime errors section if it exists
                    if "## RUNTIME ERRORS" in error_analysis_text:
                        runtime_errors_section = error_analysis_text.split("## RUNTIME ERRORS")[1].split("##")[0].strip()
                        for line in runtime_errors_section.split("\n"):
                            if line.strip() and line.strip()[0] in ["-", "*", "•"]:
                                runtime_errors.append(line.strip().lstrip("-*• "))
    
                    # Very simple extraction - this won't be comprehensive but will give us something
                    strength_section = error_analysis_text.split("## STRENGTHS")[1].split("##")[0].strip()
                    for line in strength_section.split("\n"):
                        if line.strip() and line.strip()[0] in ["-", "*", "•"]:
                            strengths.append(line.strip().lstrip("-*• "))
    
                    weakness_section = error_analysis_text.split("## WEAKNESSES")[1].split("##")[0].strip()
                    for line in weakness_section.split("\n"):
                        if line.strip() and line.strip()[0] in ["-", "*", "•"]:
                            weaknesses.append(line.strip().lstrip("-*• "))
    
                    if "## PRIMARY ISSUE" in error_analysis_text:
                        primary_issue = error_analysis_text.split("## PRIMARY ISSUE")[1].split("##")[0].strip()
    
                    if "## IMPROVEMENT SUGGESTIONS" in error_analysis_text:
                        improvement_section = error_analysis_text.split("## IMPROVEMENT SUGGESTIONS")[1].split("##")[0].strip()
                        for line in improvement_section.split("\n"):
                            if line.strip() and line.strip()[0] in ["-", "*", "•"]:
                                improvement_suggestions.append(line.strip().lstrip("-*• "))
    
                    error_analysis["strengths"] = strengths
                    error_analysis["weaknesses"] = weaknesses
                    error_analysis["primary_issue"] = primary_issue
                    error_analysis["improvement_suggestions"] = improvement_suggestions
                    error_analysis["runtime_errors"] = runtime_errors  # Add runtime errors to the analysis
    
                print("Successfully extracted information from text report")
    
            except Exception as e:
                print(f"Error generating error analysis text: {str(e)}")
                import traceback
                traceback.print_exc()
    
                error_analysis_text = f"Error analysis failed: {str(e)}"
                error_analysis = {
                    "text_report": error_analysis_text,
                    "strengths": ["Analysis failed"],
                    "weaknesses": ["Unable to analyze errors with LLM"],
                    "primary_issue": "Unable to analyze errors with LLM",
                    "improvement_suggestions": ["Retry analysis in next iteration"]
                }
        else:
            # No error samples
            error_analysis_text = "No errors to analyze in this batch."
            error_analysis = {
                "text_report": error_analysis_text,
                "strengths": ["All examples processed correctly"],
                "weaknesses": [],
                "primary_issue": "No issues identified",
                "improvement_suggestions": []
            }
    
        # Generate a capability report text, now including raw outputs
        try:
            # Define a system instruction for the capability reporter
            capability_reporter_system_instruction = "You are a System Capability Analyst who specializes in providing actionable insights for improving AI systems."
    
            # Collect all outputs for analysis
            all_outputs = [eval_data.get("output", "No output captured") for eval_data in evaluations]
            sample_outputs = all_outputs[:3]  # Take a sample of outputs for the prompt
    
            capability_prompt = f"""
            Generate a comprehensive capability report for our AI system based on its performance.
    
            PERFORMANCE SUMMARY:
            - Accuracy: {accuracy:.2f} ({correct_count}/{len(samples)})
            - Error samples: {len(error_samples)}/{len(samples)}
    
            ERROR ANALYSIS REPORT:
            {error_analysis_text}
    
            SAMPLE EXECUTION OUTPUTS:
            {json.dumps(sample_outputs, indent=2)}
    
            Please provide a thorough capability assessment that includes:
    
            ## EXECUTION ANALYSIS
            (Analysis of the raw execution outputs, including any errors or issues observed)
    
            ## CAPABILITY ASSESSMENT
            (Overall assessment of the system's capabilities)
    
            ## KEY STRENGTHS
            (The most important strengths to maintain)
    
            ## KEY WEAKNESSES
            (The most critical weaknesses to address)
    
            ## IMPROVEMENT FOCUS
            (The single most important capability to focus on improving)
    
            ## ACTIONABLE RECOMMENDATIONS
            (Specific changes to implement in the next iteration)
    
            ## CAPABILITY TREND
            (Assessment of whether capabilities are improving, declining, or stable)
    
            Your assessment should be specific, actionable, and focused on concrete improvements.
            Pay particular attention to any patterns in the execution outputs that might reveal issues with the implementation.
            """
    
            capability_report_text = self.call_llm(capability_prompt, system_instruction=capability_reporter_system_instruction)
            print("Generated capability report text successfully")
    
            # Extract the improvement focus for the dictionary return
            improvement_suggestions = []
            if "## IMPROVEMENT SUGGESTIONS" in capability_report_text:
                improvement_section = capability_report_text.split("## IMPROVEMENT SUGGESTIONS")[1].split("##")[0].strip()
                for line in improvement_section.split("\n"):
                    if line.strip() and line.strip()[0] in ["-", "*", "•"]:
                        improvement_suggestions.append(line.strip().lstrip("-*• "))
    
            # Create a structured capability report for the dictionary return
            capability_report = {
                "text_report": capability_report_text,
                "strengths": error_analysis.get("strengths", []),
                "weaknesses": error_analysis.get("weaknesses", []),
                "improvement_suggestions": error_analysis.get("improvement_suggestions", []),
                "runtime_errors": error_analysis.get("runtime_errors", [])
            }
    
        except Exception as e:
            print(f"Error generating capability report: {str(e)}")
            import traceback
            traceback.print_exc()
    
            capability_report_text = f"Capability report generation failed: {str(e)}"
            capability_report = {
                "text_report": capability_report_text,
                "improvement_focus": "information_extraction",  # Default
                "strengths": error_analysis.get("strengths", []),
                "weaknesses": error_analysis.get("weaknesses", []),
                "improvement_suggestions": error_analysis.get("improvement_suggestions", [])
            }

        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(samples),
            "evaluations": evaluations,
            "error_analysis": error_analysis,
            "capability_report": capability_report,
            "error_analysis_text": error_analysis_text,
            "capability_report_text": capability_report_text
        }
    
    def _generate_capability_guidance(self, capability_report):
        """
        Generate specific guidance based on qualitative assessment.

        Args:
            capability_report: The report generated by capability_tracker.generate_report()

        Returns:
            str: Detailed guidance text for script generation
        """
        if not capability_report:
            return ""

        # Extract key components from assessment
        strengths = capability_report.get("strengths", [])
        weaknesses = capability_report.get("weaknesses", [])
        bottlenecks = capability_report.get("bottlenecks", [])
        improvement_suggestions = capability_report.get("improvement_suggestions", [])

        # Build guidance
        guidance = "SYSTEM ANALYSIS & GUIDANCE\n\n"


        if weaknesses:
            guidance += "WEAKNESSES TO ADDRESS:\n"
            for weakness in weaknesses:
                guidance += f"- {weakness}\n"
            guidance += "\n"

        if bottlenecks:
            guidance += "CRITICAL BOTTLENECKS:\n"
            for bottleneck in bottlenecks:
                guidance += f"- {bottleneck}\n"
            guidance += "\n"

        if improvement_suggestions:
            guidance += "SPECIFIC IMPROVEMENTS TO IMPLEMENT:\n"
            for suggestion in improvement_suggestions:
                guidance += f"- {suggestion}\n"
            guidance += "\n"

        if strengths:
            guidance += "STRENGTHS TO MAINTAIN:\n"
            for strength in strengths:
                guidance += f"- {strength}\n"
            guidance += "\n"

        # Add trend analysis if available
        trends = capability_report.get("trend", "insufficient_data")
        if trends != "insufficient_data" and isinstance(trends, dict):
            improving = [cap.replace("_", " ") for cap, trend in trends.items() if trend == "improving"]
            declining = [cap.replace("_", " ") for cap, trend in trends.items() if trend == "declining"]

            if improving:
                guidance += "IMPROVING CAPABILITIES (CONTINUE THIS DIRECTION):\n"
                for cap in improving:
                    guidance += f"- {cap}\n"
                guidance += "\n"

            if declining:
                guidance += "DECLINING CAPABILITIES (NEED IMMEDIATE ATTENTION):\n"
                for cap in declining:
                    guidance += f"- {cap}\n"
                guidance += "\n"

        return guidance
    
    def evaluate_answer_with_llm(self, system_answer: str,
                                 golden_answer: str) -> Dict:
        """Use LLM to determine if answers are semantically equivalent"""

        # Role-specific system instruction for the evaluator
        evaluator_system_instruction = "You are now acting as an Answer Evaluator. Your task is to determine if two answers convey the same meaning, even if they are worded differently."

        prompt = f"""
        You're evaluating two answers to determine if they convey the same information.

        System answer: {system_answer}
        Golden answer: {golden_answer}

        Do these answers effectively communicate the same information, even if worded differently?
        Return only a JSON object with: {{"match": true/false, "confidence": 0-1, "explanation": "reason"}}
        """
        print ("SYSTEM ANSWER: ....", system_answer[-300:])
        print ("GOLDEN ANSWER: ", golden_answer)
        try:
            response = self.call_llm(
                prompt, system_instruction=evaluator_system_instruction)

            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.endswith("```"):
                response = response.split("```")[0]

            result = json.loads(response)

            # Extract match information
            match = result.get("match", False)
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "No explanation provided")

            return {
                "match": match,
                "confidence": confidence,
                "explanation": explanation
            }
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            # Fallback to exact match
            exact_match = system_answer.strip() == golden_answer.strip()
            return {
                "match":
                exact_match,
                "confidence":
                1.0 if exact_match else 0.0,
                "explanation":
                f"Fallback to exact match comparison due to error: {str(e)}"
            }

    def generate_approach_summary(self, script: str) -> str:
        """
        Use the LLM to generate a brief summary of the approach used in the script.
        """
        # Role-specific system instruction for approach summarizer
        summarizer_system_instruction = "You are an Approach Summarizer. Your task is to analyze code and provide concise explanations of the techniques and methods used."

        prompt = f"""
        You're given a Python script that processes input and generates output using LLM-driven techniques.
        Provide a brief summary of the approach used in this script in 2-3 sentences.

        Focus on:
        1. What LLM-based techniques are used (chain-of-thought, verification, etc.)
        2. How the problem is decomposed
        3. What agent roles are involved
        4. What other functions are used
        5. A brief list of the function names used and how they are used with one another
        6. The overall workflow

        Script:
        ```python
        {script}
        ```

        Return only the summary with no introduction or additional comments.
        """

        try:
            summary = self.call_llm(
                prompt, system_instruction=summarizer_system_instruction)
            return summary.strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    def get_best_script_info(self) -> Dict:
        """
        Get information about the best performing script, considering both 
        accuracy and testing coverage, with special emphasis on progressive testing results.
        """
        iterations = self.get_all_iterations()
        if not iterations:
            return None

        # Prepare data for LLM evaluation
        iteration_data = []
        for it in iterations:
            if it is None:
                continue  # Skip None entries

            # Safely access nested data
            progressive_accuracy = None
            progressive_samples = 0
            if "progressive_testing" in it and it["progressive_testing"]:
                progressive_accuracy = it["progressive_testing"].get("accuracy", None)
                progressive_samples = it["progressive_testing"].get("total_examples", 0)

            # Calculate combined accuracy across both batch and progressive tests
            combined_acc = None
            if progressive_accuracy is not None:
                batch_acc = it.get("performance", {}).get("accuracy", 0)
                batch_size = it.get("batch_size", 5)
                # Calculate weighted average based on sample counts
                total_correct = (batch_acc * batch_size) + (progressive_accuracy * progressive_samples)
                total_samples = batch_size + progressive_samples
                combined_acc = total_correct / total_samples if total_samples > 0 else 0

            iteration_data.append({
                "iteration": it.get("iteration"),
                "accuracy": it.get("performance", {}).get("accuracy", 0),
                "batch_size": it.get("batch_size", 5),
                "progressive_accuracy": progressive_accuracy,
                "progressive_samples": progressive_samples,
                "combined_accuracy": combined_acc,
                "approach": it.get("approach_summary", "Unknown approach"),
                "strategy": it.get("strategy", "Unknown")
            })

        if not iteration_data:
            # Fallback if no valid iteration data
            return {
                "iteration": 0,
                "accuracy": 0,
                "batch_size": 5,
                "progressive_accuracy": None,
                "progressive_samples": 0,
                "combined_accuracy": 0,
                "path": "No valid scripts available",
                "approach": "No approaches tried yet",
                "rationale": "No valid iterations completed"
            }

        # Role-specific system instruction for script evaluator
        script_evaluator_system_instruction = "You are a Script Evaluator. Your task is to analyze performance metrics of different script iterations and determine which one represents the best overall approach."

        # Handle API rate limit issues - don't try to use LLM if we've hit limits
        try:
            # Use LLM to determine best script with enhanced guidance about progressive testing
            prompt = f"""
            As an AI system, determine which iteration produced the best script.

            Here is data about all iterations:
            {json.dumps(iteration_data, indent=2)}

            Consider multiple factors with the following priorities:
            1. HIGHEST PRIORITY: Combined accuracy across all tested examples
               - This is pre-calculated in "combined_accuracy" when available
               - It represents performance across both batch and progressive tests
               - This is the most reliable indicator of general performance

            2. HIGH PRIORITY: Progressive testing performance
               - "progressive_accuracy" is performance on previously seen examples
               - Progressive testing covers more examples and is a better measure of generalization
               - Scripts with high progressive accuracy have proven robustness

            3. MEDIUM PRIORITY: Batch accuracy and testing coverage
               - "accuracy" is performance on the current batch being tested
               - "batch_size" indicates how many examples were in the current batch

            4. LOW PRIORITY: Recency and approach diversity
               - Recent iterations may reflect learned improvements
               - Different strategies (exploration vs exploitation) have different goals

            For your analysis:
              * If multiple scripts have similar combined accuracy, prefer those with more total examples tested
              * If only some scripts have progressive testing data, give them preference if their combined accuracy is reasonable
              * Consider trends over multiple iterations - consistent improvement is valuable

            Return only a JSON object with:
            {{"best_iteration": <integer>, "rationale": "<brief explanation>"}}
            """

            response = self.call_llm(
                prompt, system_instruction=script_evaluator_system_instruction)

            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.endswith("```"):
                response = response.split("```")[0]

            result = json.loads(response)

            # Get detailed info about the best iteration
            best_iteration_number = int(result.get("best_iteration", -1))

            best_iteration = next(
                (it for it in iterations
                 if it.get("iteration") == best_iteration_number), None)

            if not best_iteration:
                # Fallback: use combined accuracy method
                raise Exception("Best iteration not found, using fallback method")

            # Find progressive testing data in the best iteration
            progressive_accuracy = None
            progressive_samples = 0
            if "progressive_testing" in best_iteration and best_iteration["progressive_testing"]:
                progressive_accuracy = best_iteration["progressive_testing"].get("accuracy", None)
                progressive_samples = best_iteration["progressive_testing"].get("total_examples", 0)

            # Calculate combined accuracy
            combined_acc = None
            if progressive_accuracy is not None:
                batch_acc = best_iteration.get("performance", {}).get("accuracy", 0)
                batch_size = best_iteration.get("batch_size", 5)
                # Calculate weighted average based on sample counts
                total_correct = (batch_acc * batch_size) + (progressive_accuracy * progressive_samples)
                total_samples = batch_size + progressive_samples
                combined_acc = total_correct / total_samples if total_samples > 0 else 0

            return {
                "iteration": best_iteration.get("iteration"),
                "accuracy": best_iteration.get("performance", {}).get("accuracy", 0),
                "batch_size": best_iteration.get("batch_size", 5),
                "progressive_accuracy": progressive_accuracy,
                "progressive_samples": progressive_samples,
                "combined_accuracy": combined_acc,
                "path": f"scripts/script_iteration_{best_iteration.get('iteration')}.py",
                "approach": best_iteration.get("approach_summary", ""),
                "rationale": result.get("rationale", "Highest overall accuracy")
            }
        except Exception as e:
            # Fallback method - use combined accuracy if available
            print(f"Error determining best script with LLM: {e}")
            print("Using fallback method to determine best script")

            try:
                # Helper function to calculate combined accuracy
                def combined_score(it):
                    if not it:
                        return 0

                    batch_acc = it.get("performance", {}).get("accuracy", 0)
                    batch_size = it.get("batch_size", 5)

                    prog_testing = it.get("progressive_testing", {})
                    if not prog_testing:
                        return batch_acc

                    prog_acc = prog_testing.get("accuracy", 0)
                    prog_size = prog_testing.get("total_examples", 0)

                    if prog_size > 0:
                        # Calculate weighted accuracy based on sample counts
                        total_correct = (batch_acc * batch_size) + (prog_acc * prog_size)
                        total_samples = batch_size + prog_size
                        return total_correct / total_samples
                    else:
                        return batch_acc

                # Find the best iteration using the combined score
                best_iteration = max(iterations, key=combined_score)

                # Extract progressive testing data
                progressive_accuracy = None
                progressive_samples = 0
                if "progressive_testing" in best_iteration and best_iteration["progressive_testing"]:
                    progressive_accuracy = best_iteration["progressive_testing"].get("accuracy", None)
                    progressive_samples = best_iteration["progressive_testing"].get("total_examples", 0)

                # Calculate combined accuracy
                combined_acc = combined_score(best_iteration)

                return {
                    "iteration": best_iteration.get("iteration"),
                    "accuracy": best_iteration.get("performance", {}).get("accuracy", 0),
                    "batch_size": best_iteration.get("batch_size", 5),
                    "progressive_accuracy": progressive_accuracy,
                    "progressive_samples": progressive_samples,
                    "combined_accuracy": combined_acc,
                    "path": f"scripts/script_iteration_{best_iteration.get('iteration')}.py",
                    "approach": best_iteration.get("approach_summary", ""),
                    "rationale": "Selected based on combined accuracy across all tested examples"
                }
            except Exception as e2:
                print(f"Error with fallback method: {e2}")
                # Ultra fallback - just return the first iteration
                if iterations and iterations[0]:
                    return {
                        "iteration": iterations[0].get("iteration", 0),
                        "accuracy": iterations[0].get("performance", {}).get("accuracy", 0),
                        "batch_size": iterations[0].get("batch_size", 5),
                        "path": f"scripts/script_iteration_{iterations[0].get('iteration', 0)}.py",
                        "approach": iterations[0].get("approach_summary", ""),
                        "rationale": "Ultra fallback - first available iteration"
                    }
                else:
                    return {
                        "iteration": 0,
                        "accuracy": 0,
                        "batch_size": 5,
                        "path": "No valid scripts available",
                        "approach": "No approaches tried yet",
                        "rationale": "No valid iterations completed"
                    }
    
    def run_progressive_testing(self, script: str, max_examples: int = 20) -> Dict:
        """Run progressive testing on seen examples, up to a maximum limit"""
        # Load the dataset
        if not hasattr(self, 'dataset_loader') or not self.dataset_loader:
            return {"success": False, "error": "No dataset loader available"}

        # Get all seen examples
        all_samples = []
        for example_index in self.seen_examples:
            # Get sample at this index
            try:
                examples = self.dataset_loader.get_examples(1)
                if examples:
                    all_samples.append(examples[0])
            except Exception as e:
                print(f"Error retrieving example {example_index}: {e}")
                continue

        if not all_samples:
            return {"success": False, "error": "No examples seen yet"}

        # Limit to max_examples (most recent)
        samples = all_samples[-max_examples:] if len(all_samples) > max_examples else all_samples

        print(f"Running progressive testing on {len(samples)} seen examples (out of {len(all_samples)} total seen)...")

        # Execute script on selected samples
        results = []
        for i, sample in enumerate(samples):
            if i % 5 == 0:  # Status update every 5 samples
                print(f"  Processing sample {i+1}/{len(samples)}...")

            # Execute the script with the full sample
            result = self.execute_script(script, sample)

            # Evaluate the result if successful
            if result.get("success"):
                golden_answer = self.dataset_loader.get_example_output(sample)  # This already uses the universal interface
                system_answer = result.get("answer", "")

                # Use LLM-based evaluation
                evaluation = self.evaluate_answer_with_llm(system_answer, golden_answer)
                result["golden_answer"] = golden_answer
                result["evaluation"] = evaluation
                result["match"] = evaluation.get("match", False)
            else:
                result["match"] = False

            results.append(result)

        # Calculate overall statistics
        successful_runs = sum(1 for r in results if r.get("success", False))
        matches = sum(1 for r in results if r.get("match", False))

        return {
            "total_examples": len(samples),
            "successful_runs": successful_runs,
            "matches": matches,
            "accuracy": matches / len(samples) if samples else 0,
            "results": results
        }

    def validate_script(self,
        script_path: str = None,
        start_index: int = 0,
        end_index: int = 999) -> Dict:
        """Test a script on a specified range of examples"""
        # If no script path provided, use the best script
        if not script_path:
            best_info = self.get_best_script_info()
            if not best_info:
                return {"success": False, "error": "No scripts available"}
            script_path = best_info.get("path")
    
        # Load the script
        try:
            with open(script_path, 'r') as f:
                script = f.read()
        except Exception as e:
            return {
            "success": False,
            "error": f"Error loading script: {str(e)}"
            }
        
        # Instead of loading the dataset directly, use the dataset loader
        if not hasattr(self, 'dataset_loader') or not self.dataset_loader:
            return {"success": False, "error": "No dataset loader available"}
    
        # Get examples in the specified range
        samples = []
        current_index = 0
    
        # This is a simple approach - in a real implementation, you might want to
        # implement more sophisticated indexing in your dataset loader
        while current_index <= end_index and len(samples) < (end_index - start_index + 1):
            examples = self.dataset_loader.get_examples(1)
            if examples and current_index >= start_index:
                samples.append(examples[0])
            current_index += 1
    
        if not samples:
            return {
            "success": False,
            "error": f"No examples found in range {start_index}-{end_index}"
            }
    
        print(f"Validating script on {len(samples)} examples from range {start_index}-{end_index}...")
    
        # Execute script on all samples
        results = []
        for i, sample in enumerate(samples):
            if i % 10 == 0:  # Status update every 10 samples
                print(f"  Processing sample {i+1}/{len(samples)}...")

            question = self.dataset_loader.get_example_input(sample)
            result = self.execute_script(script, question)

            # Evaluate the result if successful
            if result.get("success"):
                golden_answer = self.dataset_loader.get_example_output(sample)
                system_answer = result.get("answer", "")

                # Use LLM-based evaluation
                evaluation = self.evaluate_answer_with_llm(system_answer, golden_answer)

                result["evaluation"] = evaluation
                result["match"] = evaluation.get("match", False)
            else:
                result["match"] = False

            results.append({"key": sample.get("id", f"example_{i}"), "result": result})
    
        # Calculate overall statistics
        successful_runs = sum(1 for r in results if r["result"].get("success", False))
        matches = sum(1 for r in results if r["result"].get("match", False))
    
        return {
            "script_path": script_path,
            "total_examples": len(samples),
            "successful_runs": successful_runs,
            "matches": matches,
            "accuracy": matches / len(samples) if samples else 0,
            "results": results
        }

    def run_iteration(self) -> Dict:
        """Run a single iteration of the agent system with capability tracking"""


        print(f"\n=== Starting Iteration {self.current_iteration} ===")
        print(f"Current explore/exploit balance: {self.explore_rate}/{self.exploit_rate}")
        print(f"Current batch size: {self.current_batch_size}")
        print(f"Total seen examples: {len(self.seen_examples)}")

        iteration_start_time = time.time()

        # Get capability report if available
        capability_report = None
        if hasattr(self, 'capability_tracker'):
            capability_report = self.capability_tracker.generate_report()
            if capability_report:
                print("\n=== Current Capability Status ===")

                # Display strengths and weaknesses...
                if capability_report.get("strengths"):
                    print("\n  Strengths:")
                    for strength in capability_report.get("strengths", [])[:2]:
                        print(f"    - {strength}")
                if capability_report.get("weaknesses"):
                    print("\n  Weaknesses:")
                    for weakness in capability_report.get("weaknesses", [])[:2]:
                        print(f"    - {weakness}")
                if capability_report.get("improvement_suggestions"):
                    print("\n  Suggested Improvements:")
                    for suggestion in capability_report.get("improvement_suggestions", [])[:2]:
                        print(f"    - {suggestion}")
                print("=" * 40)

        # Decide whether to explore or exploit
        if self.force_exploitation_next:
            is_exploration = False
            self.force_exploitation_next = False
            print("Forcing exploitation based on previous good performance")
        else:
            is_exploration = random.random() * 100 <= self.explore_rate

        # If we have capability data with clear trends, potentially override the strategy
        if capability_report and capability_report.get("trend") != "insufficient_data":
            weakest_capability = capability_report.get("improvement_focus", "")
            trends = capability_report.get("trend", {})
            weakest_trend = trends.get(weakest_capability) if isinstance(trends, dict) else None

            # Strategy overrides based on trends...
            if weakest_trend == "declining":
                if not is_exploration:
                    print(f"Strategy override: Forcing exploration to address declining capability: {weakest_capability}")
                    is_exploration = True
            if isinstance(trends, dict):
                improving_count = sum(1 for trend in trends.values() if trend == "improving")
                if improving_count == len(trends) and is_exploration and improving_count > 0:
                    print(f"Strategy override: Forcing exploitation to capitalize on improving capabilities")
                    is_exploration = False

        strategy = "Exploration" if is_exploration else "Exploitation"
        print(f"Strategy for this iteration: {strategy}")

        # Generate script using LLM BEFORE getting test samples
        print("Generating script with LLM...")

        # Get capability insights
        capability_guidance = ""
        if capability_report:
            capability_guidance = self._generate_capability_guidance(capability_report)
            print("Generated capability guidance based on performance analysis")
            if capability_report:
                capability_guidance = self._generate_capability_guidance(capability_report)

        # Generate script before getting any test samples - prevent data leakage
        script = self.generate_script_with_llm(is_exploration)

        print("Performing initial script verification and repair...")
        script = self.attempt_script_repair(script, max_attempts=3)

        # Generate a summary of the approach
        try:
            approach_summary = self.generate_approach_summary(script)
            if approach_summary.startswith("API_RATE_LIMIT_EXCEEDED"):
                approach_summary = "Approach summary not available due to API rate limit"
        except Exception as e:
            approach_summary = f"Error generating approach summary: {str(e)}"

        print(f"Approach summary: {approach_summary}")

        # AFTER script generation, get the test samples that the script hasn't seen
        samples_data = self.get_samples()
        samples = samples_data["samples"]

        if not samples:
            print("No samples available in dataset. Exiting iteration.")
            return {"success": False, "error": "No samples available"}

        print(f"Processing {len(samples)} examples (including {samples_data['new_examples_added']} new examples)")

        # Execute script on samples
        print("Executing script on samples...")
        results = []
        for i, sample in enumerate(samples):
            print(f"  Processing sample {i+1}/{len(samples)}...")
            #print ("***/nSAMPLE/n****", sample)
            # Execute the script with the full sample dictionary
            result = self.execute_script(script, sample)

            if result.get("success"):
                print(f"    Result: {result.get('answer')}")
                # Evaluate with LLM
                golden_answer = sample.get("answer", "")  # Use universal "answer" field
                system_answer = result.get("answer", "")

                try:
                    evaluation = self.evaluate_answer_with_llm(system_answer, golden_answer)
                    result["evaluation"] = evaluation
                    result["match"] = evaluation.get("match", False)
                    if result["match"]:
                        print(f"    ✅ Match (confidence: {evaluation.get('confidence', 0):.2f})")
                    else:
                        print(f"    ❌ No match: {evaluation.get('explanation', '')}")
                except Exception as e:
                    print(f"    ⚠️ Error evaluating answer: {str(e)}")
                    # Fallback to exact match
                    exact_match = system_answer.strip() == golden_answer.strip()
                    result["match"] = exact_match
                    result["evaluation"] = {
                        "match": exact_match,
                        "confidence": 1.0 if exact_match else 0.0,
                        "explanation": f"Error evaluating: {str(e)}"
                    }
                    print(f"    {'✅' if exact_match else '❌'} Fallback to exact match: {exact_match}")
            else:
                print(f"    Error: {result.get('error')}")
                result["match"] = False

            results.append(result)

        # Calculate basic performance metrics
        successful_runs = sum(1 for r in results if r.get("success", False))
        matches = sum(1 for r in results if r.get("match", False))
        accuracy = matches / len(samples) if samples else 0

        # Basic evaluation summary
        basic_evaluation = {
            "accuracy": accuracy,
            "correct_count": matches,
            "total_count": len(samples),
            "evaluations": results
        }

        print(f"Performance: {accuracy:.2f} accuracy " + f"({matches}/{len(samples)} correct)")

        # Use LLM for deeper error analysis
        try:
            print("Performing error analysis with LLM...")
            evaluation = self.evaluate_with_llm(samples, results)
            if evaluation.get('error_analysis'):
                primary_issue = evaluation.get('error_analysis', {}).get('primary_issue', 'None')
                print(f"Primary issue identified: {primary_issue}")
                # Display strengths, weaknesses, and improvement suggestions
                if "error_analysis" in evaluation and evaluation["error_analysis"]:
                    error_analysis = evaluation["error_analysis"]
                    if "strengths" in error_analysis and error_analysis["strengths"]:
                        print("\n=== Strengths Identified ===")
                        for strength in error_analysis["strengths"]:
                            print(f"  - {strength}")
                    if "weaknesses" in error_analysis and error_analysis["weaknesses"]:
                        print("\n=== Weaknesses Identified ===")
                        for weakness in error_analysis["weaknesses"]:
                            print(f"  - {weakness}")
                    if "bottlenecks" in error_analysis and error_analysis["bottlenecks"]:
                        print("\n=== Critical Bottlenecks ===")
                        for bottleneck in error_analysis["bottlenecks"]:
                            print(f"  - {bottleneck}")
                    if "improvement_suggestions" in error_analysis and error_analysis["improvement_suggestions"]:
                        print("\n=== Improvement Suggestions ===")
                        for suggestion in error_analysis["improvement_suggestions"][:3]:
                            print(f"  - {suggestion}")
                    print("=" * 40)
            else:
                print("No specific issues identified")
        except Exception as e:
            print(f"Error in error analysis: {str(e)}")
            evaluation = basic_evaluation
            evaluation["error_analysis"] = {
                "primary_issue": "Analysis error",
                "error_patterns": ["Error during analysis"],
                "improvement_suggestions": ["Fix error handling"],
                "root_causes": [str(e)]
            }

        # Run progressive testing on all seen examples for promising scripts
        progressive_testing_results = None
        if accuracy >= 0.6:
            self.force_exploitation_next = True
            try:
                print("Script looks promising! Running progressive testing on all seen examples...")
                progressive_testing_results = self.run_progressive_testing(script, max_examples=10)
                if progressive_testing_results:
                    prog_accuracy = progressive_testing_results.get("accuracy", 0)
                    prog_matches = progressive_testing_results.get("matches", 0)
                    prog_total = progressive_testing_results.get("total_examples", 0)
                    print(f"Progressive testing results: {prog_accuracy:.2f} accuracy " + 
                            f"({prog_matches}/{prog_total} correct)")
            except Exception as e:
                print(f"Error in progressive testing: {str(e)}")
                progressive_testing_results = None

        # Adjust explore/exploit balance for next iteration
        try:
            print("Adjusting explore/exploit balance...")
            new_explore, new_exploit = self.adjust_explore_exploit_with_llm()
            print(f"New explore/exploit balance: {new_explore}/{new_exploit}")
        except Exception as e:
            print(f"Error adjusting explore/exploit balance: {str(e)}")
            new_explore, new_exploit = self.explore_rate, self.exploit_rate
            print(f"Maintaining current explore/exploit balance: {new_explore}/{new_exploit}")

        # Adjust batch size for next iteration
        try:
            print("Adjusting batch size...")
            new_batch_size, batch_adjustment_rationale = self.adjust_batch_size_with_llm(basic_evaluation)
            print(f"New batch size: {new_batch_size} ({batch_adjustment_rationale})")
        except Exception as e:
            print(f"Error adjusting batch size: {str(e)}")
            new_batch_size = self.current_batch_size
            batch_adjustment_rationale = f"Error: {str(e)}"
            print(f"Maintaining current batch size: {new_batch_size}")

        # Identify current best script
        try:
            best_script_info = self.get_best_script_info()
            if best_script_info:
                print("\n=== Current Best Script ===")
                print(f"Iteration: {best_script_info.get('iteration')}")

                # Report batch testing results
                batch_acc = best_script_info.get('accuracy', 0)
                batch_size = best_script_info.get('batch_size', 0)
                print(f"Batch Accuracy: {batch_acc:.2f} (tested on {batch_size} examples)")

                # Report progressive testing results if available
                prog_acc = best_script_info.get('progressive_accuracy')
                if prog_acc is not None:
                    prog_samples = best_script_info.get('progressive_samples', 0)
                    print(f"Progressive Accuracy: {prog_acc:.2f} (tested on {prog_samples} examples)")

                # Report combined accuracy if available
                combined_acc = best_script_info.get('combined_accuracy')
                if combined_acc is not None:
                    total_samples = batch_size
                    if prog_acc is not None:
                        total_samples += best_script_info.get('progressive_samples', 0)
                    print(f"Combined Accuracy: {combined_acc:.2f} (across all {total_samples} examples)")

                print(f"Path: {best_script_info.get('path')}")
                print(f"Approach: {best_script_info.get('approach')}")
                print(f"Rationale: {best_script_info.get('rationale')}")

                # Display capability assessment for the best script if available
                if hasattr(self, 'capability_tracker') and capability_report:
                    print("\n  Capability Assessment:")
                    improvement_focus = capability_report.get("improvement_focus", "")
                    if improvement_focus:
                        print(f"    Focus area: {improvement_focus.upper().replace('_', ' ')}")
                    if capability_report.get("strengths"):
                        print("    Strengths:")
                        for strength in capability_report.get("strengths", [])[:2]:
                            print(f"      - {strength}")
                    if capability_report.get("weaknesses"):
                        print("    Weaknesses:")
                        for weakness in capability_report.get("weaknesses", [])[:2]:
                            print(f"      - {weakness}")
        except Exception as e:
            print(f"Error identifying best script: {e}")
            best_script_info = None


        # Prepare iteration data
        iteration_data = {
            "iteration": self.current_iteration,
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy": strategy,
            "explore_rate": self.explore_rate,
            "exploit_rate": self.exploit_rate,
            "batch_size": self.current_batch_size,
            "script": script,
            "approach_summary": approach_summary,
            "sample_count": len(samples),
            "samples": samples,  # Add the original samples
            "samples_metadata": [sample.get("meta", {}) for sample in samples],
            "example_indices": list(range(self.examples_processed - len(samples), self.examples_processed)),
            "results": results,
            "performance": evaluation,
            "progressive_testing": progressive_testing_results,
            "execution_time": time.time() - iteration_start_time,
            "capability_report": capability_report
        }

        # Create summary
        summary = {
            "iteration": self.current_iteration,
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy": strategy,
            "explore_rate": self.explore_rate,
            "exploit_rate": self.exploit_rate,
            "batch_size": self.current_batch_size,
            "approach_summary": approach_summary,
            "performance": {
                "accuracy": accuracy,
                "correct_count": matches,
                "total_count": len(samples)
            },
            "progressive_accuracy": progressive_testing_results.get("accuracy", None) if progressive_testing_results else None,
            "primary_issue": evaluation.get("error_analysis", {}).get("primary_issue", "None identified"),
            "new_explore_rate": new_explore,
            "new_exploit_rate": new_exploit,
            "new_batch_size": new_batch_size,
            "capability_report": capability_report
        }

        # Save to archive and perform trace analysis - FIX TIMING ISSUES
        try:
            # First save the iteration data to the archive
            self.save_to_archive(iteration_data, f"iteration_{self.current_iteration}.json")

            # Add a small delay to ensure file is written completely before trace analysis
            time.sleep(0.5)  # 500ms delay should be sufficient

            # Then perform trace analysis
            print("Analyzing execution traces...")
            trace_analyzer = TraceAnalyzer(self.archive_dir)
            trace_insights = trace_analyzer.analyze_traces_with_llm(self.current_iteration)

            if trace_insights and not trace_insights.startswith("Error"):
                print("Successfully extracted trace insights")
                iteration_data["trace_insights"] = trace_insights
                iteration_data["trace_analysis"] = {
                    "analyzed_at": datetime.datetime.now().isoformat(),
                    "insights": trace_insights,
                    "trace_file": f"trace_iteration_{self.current_iteration}.jsonl"
                }

                # Update the archive with the insights
                self.save_to_archive(iteration_data, f"iteration_{self.current_iteration}.json")
            else:
                print("Failed to extract trace insights")

            # Then update summaries
            self.update_summaries(summary)
        except Exception as e:
            print(f"Error saving iteration data: {str(e)}")

        # Update explore/exploit rates for next iteration
        self.explore_rate = new_explore
        self.exploit_rate = new_exploit

        # Update batch size for next iteration
        self.current_batch_size = new_batch_size

        # Generate execution flow visualization
        print("\nGenerating execution flow visualization...")
        script_path = self.scripts_dir / f"script_iteration_{self.current_iteration}.py"
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "script_flow_graph.py", str(script_path)],
                check=False  # Don't raise exception on non-zero exit
            )
            print("Visualization saved!")
        except Exception as e:
            print(f"Error generating visualization: {e}")

        # Increment iteration counter
        self.current_iteration += 1

        print(f"=== Completed Iteration {self.current_iteration - 1} ===")

        # Update learnings file with insights from this iteration
        try:
            print("Updating learnings file with insights from this iteration...")
            self.update_learnings(iteration_data)
        except Exception as e:
            print(f"Error updating learnings: {e}")

        return iteration_data

class TraceAnalyzer:
    def __init__(self, archive_dir):
        self.archive_dir = Path(archive_dir)
        self.client = None
        try:
            self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        except Exception as e:
            print(f"Error initializing Gemini client for TraceAnalyzer: {e}")

    def call_llm(self, prompt, system_instruction=None):
        """Call the Gemini LLM with a prompt and return the response"""
        if not self.client:
            return "Error: Gemini client not initialized"

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction if system_instruction else ""
                ),
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def parse_trace_file(self, iteration_number):
        """Parse a trace file for a given iteration with better error handling and diagnostics."""
        trace_file = self.archive_dir / f"trace_iteration_{iteration_number}.jsonl"

        # Check if file exists and report diagnostic info
        if not trace_file.exists():
            print(f"Trace file not found: {trace_file}")
            return []

        # Check file size
        file_size = os.path.getsize(trace_file)
        if file_size == 0:
            print(f"Trace file exists but is empty: {trace_file}")
            return []

        print(f"Parsing trace file: {trace_file} (size: {file_size} bytes)")

        events = []
        line_count = 0
        error_count = 0

        try:
            with open(trace_file, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    line_count += 1
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError as e:
                        error_count += 1
                        print(f"Error parsing JSON at line {line_number}: {e}")
                        print(f"Line content: {line[:100]}...")

                        # Try to salvage partial JSON
                        if '{' in line and '}' in line:
                            try:
                                partial_json = line[line.find('{'):line.rfind('}')+1]
                                event = json.loads(partial_json)
                                events.append(event)
                                print(f"Recovered partial JSON from line {line_number}")
                            except:
                                pass
        except Exception as e:
            print(f"Error reading trace file: {e}")

        # Print summary statistics
        print(f"Parsed {len(events)} events from {line_count} lines with {error_count} errors")

        # Print event types summary
        event_types = {}
        sample_ids = set()

        for event in events:
            event_type = event.get('event', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1

            # Collect unique sample IDs
            sample_id = event.get('sample_id')
            if sample_id:
                sample_ids.add(sample_id)

        print(f"Event types: {event_types}")
        print(f"Sample IDs found in trace file: {sample_ids}")

        return events

    def get_iteration_data(self, iteration_number):
        """Load iteration data from archive with better diagnostics."""
        iteration_file = self.archive_dir / f"iteration_{iteration_number}.json"

        if not iteration_file.exists():
            print(f"Iteration file not found: {iteration_file}")
            return None

        try:
            with open(iteration_file, 'r', encoding='utf-8') as f:
                iteration_data = json.load(f)

                # Print diagnostic info
                sample_count = len(iteration_data.get('samples', []))
                result_count = len(iteration_data.get('results', []))
                print(f"Loaded iteration {iteration_number} data with {sample_count} samples and {result_count} results")

                # Print sample IDs
                sample_ids = [sample.get('id', f"idx_{i}") for i, sample in enumerate(iteration_data.get('samples', []))]
                print(f"Sample IDs in iteration data: {sample_ids}")

                return iteration_data
        except json.JSONDecodeError as e:
            print(f"Error parsing iteration file {iteration_file}: {e}")
            return None
        except Exception as e:
            print(f"Error loading iteration file {iteration_file}: {e}")
            return None

    def categorize_traces_by_correctness(self, iteration_number):
        """
        Categorize traces as correct or incorrect based on match results.
        Now with better error handling and more flexible ID matching strategies.
        """
        iteration_data = self.get_iteration_data(iteration_number)
        if not iteration_data:
            print(f"No iteration data found for iteration {iteration_number}")
            return [], []

        traces = self.parse_trace_file(iteration_number)
        if not traces:
            print(f"No trace events found for iteration {iteration_number}")
            return [], []

        samples = iteration_data.get("samples", [])
        results = iteration_data.get("results", [])

        # Ensure we have something to analyze
        if not samples or not results:
            print(f"No samples or results found for iteration {iteration_number}")
            return [], []

        print(f"Categorizing traces for iteration {iteration_number}: {len(samples)} samples, {len(results)} results, {len(traces)} trace events")

        # Group trace events by sample ID
        traces_by_sample = {}

        # First collect all traces regardless of sample_id
        all_traces = []
        for trace in traces:
            sample_id = trace.get("sample_id", "unknown")
            if sample_id not in traces_by_sample:
                traces_by_sample[sample_id] = []
            traces_by_sample[sample_id].append(trace)
            all_traces.append(trace)

        # Debug output of grouped traces
        print(f"Grouped trace events by sample ID: {list(traces_by_sample.keys())}")

        # Handle special case: if there's only one sample and only "test_sample" traces
        if len(samples) == 1 and "test_sample" in traces_by_sample:
            print("Special case: Single sample with 'test_sample' traces")
            sample = samples[0]
            result = results[0]

            trace_data = {
                "sample_index": 0,
                "sample_id": sample.get("id", ""),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "system_answer": result.get("answer", "") if result.get("success", False) else "ERROR: " + result.get("error", "Unknown error"),
                "match": result.get("match", False),
                "success": result.get("success", False),
                "traces": traces_by_sample.get("test_sample", []),
                "llm_calls": [],
                "errors": []
            }

            # Extract LLM calls and errors
            for event in trace_data["traces"]:
                if event.get("event") == "llm_call":
                    try:
                        trace_data["llm_calls"].append({
                            "caller": event.get("caller", {}).get("function", "unknown"),
                            "prompt": event.get("input", {}).get("prompt", ""),
                            "system_instruction": event.get("input", {}).get("system_instruction", ""),
                            "output": event.get("output", "")
                        })
                    except Exception as e:
                        print(f"Error extracting LLM call data: {e}")
                elif event.get("event") == "execution_error":
                    trace_data["errors"].append({
                        "error": event.get("error", ""),
                        "traceback": event.get("traceback", "")
                    })

            # Categorize as correct or incorrect
            if result.get("match", False):
                print("Categorized as correct trace")
                return [trace_data], []
            else:
                print("Categorized as incorrect trace")
                return [], [trace_data]

        # Try to match traces with samples using various strategies
        correct_traces = []
        incorrect_traces = []

        # Match samples with results and their traces
        for i, (sample, result) in enumerate(zip(samples, results)):
            # Try multiple ways of identifying the sample
            sample_id = sample.get("id", "")
            if not sample_id:
                sample_id = f"example_{i}"

            # Potential ID formats to check
            potential_ids = [
                sample_id,
                f"example_{i}",
                f"test_sample_{i}",
                str(i)
            ]

            # Special case: if we have "test_sample", check if this is the first example
            if i == 0:
                potential_ids.append("test_sample")

            # Get all trace events for this sample across all potential IDs
            sample_traces = []
            matched_id = None

            for potential_id in potential_ids:
                if potential_id in traces_by_sample:
                    sample_traces.extend(traces_by_sample[potential_id])
                    matched_id = potential_id
                    break

            # If we still don't have traces and this is our only sample, assign all traces
            if not sample_traces and len(samples) == 1:
                print(f"Assigning all traces to the single sample")
                sample_traces = all_traces
                matched_id = "all_traces"
            # If we still don't have traces but we have a "test_sample" and multiple samples,
            # try to divide those traces among the samples based on the event timestamp order
            elif not sample_traces and "test_sample" in traces_by_sample and len(samples) > 1:
                test_sample_traces = traces_by_sample["test_sample"]
                # Divide traces evenly
                traces_per_sample = len(test_sample_traces) // len(samples)
                if traces_per_sample > 0:
                    start_idx = i * traces_per_sample
                    end_idx = start_idx + traces_per_sample
                    if i == len(samples) - 1:  # Last sample gets remaining traces
                        end_idx = len(test_sample_traces)
                    sample_traces = test_sample_traces[start_idx:end_idx]
                    matched_id = f"test_sample[{start_idx}:{end_idx}]"

            if not sample_traces:
                print(f"No traces found for sample {i} (ID: {sample_id}, tried: {potential_ids})")
                continue

            print(f"Found {len(sample_traces)} traces for sample {i} using ID: {matched_id}")

            # Create a rich trace data object
            trace_data = {
                "sample_index": i,
                "sample_id": sample_id,
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "system_answer": result.get("answer", "") if result.get("success", False) else "ERROR: " + result.get("error", "Unknown error"),
                "match": result.get("match", False),
                "success": result.get("success", False),
                "traces": sample_traces,
                "llm_calls": [],
                "errors": []
            }

            # Extract LLM calls and errors from trace events
            for event in sample_traces:
                if event.get("event") == "llm_call":
                    try:
                        trace_data["llm_calls"].append({
                            "caller": event.get("caller", {}).get("function", "unknown"),
                            "prompt": event.get("input", {}).get("prompt", ""),
                            "system_instruction": event.get("input", {}).get("system_instruction", ""),
                            "output": event.get("output", "")
                        })
                    except Exception as e:
                        print(f"Error extracting LLM call data: {e}")
                elif event.get("event") == "execution_error":
                    trace_data["errors"].append({
                        "error": event.get("error", ""),
                        "traceback": event.get("traceback", "")
                    })

            # Categorize as correct or incorrect
            if result.get("match", False):
                correct_traces.append(trace_data)
            else:
                incorrect_traces.append(trace_data)

        # If we still couldn't match any traces to samples, try one last fallback:
        # Just divide all traces among all samples evenly
        if not correct_traces and not incorrect_traces and all_traces:
            print("Fallback: Dividing all traces among all samples")
            traces_per_sample = len(all_traces) // len(samples)
            if traces_per_sample > 0:
                for i, (sample, result) in enumerate(zip(samples, results)):
                    start_idx = i * traces_per_sample
                    end_idx = start_idx + traces_per_sample
                    if i == len(samples) - 1:  # Last sample gets remaining traces
                        end_idx = len(all_traces)
                    sample_traces = all_traces[start_idx:end_idx]

                    trace_data = {
                        "sample_index": i,
                        "sample_id": sample.get("id", f"example_{i}"),
                        "question": sample.get("question", ""),
                        "answer": sample.get("answer", ""),
                        "system_answer": result.get("answer", "") if result.get("success", False) else "ERROR: " + result.get("error", "Unknown error"),
                        "match": result.get("match", False),
                        "success": result.get("success", False),
                        "traces": sample_traces,
                        "llm_calls": [],
                        "errors": []
                    }

                    # Extract LLM calls and errors
                    for event in sample_traces:
                        if event.get("event") == "llm_call":
                            try:
                                trace_data["llm_calls"].append({
                                    "caller": event.get("caller", {}).get("function", "unknown"),
                                    "prompt": event.get("input", {}).get("prompt", ""),
                                    "system_instruction": event.get("input", {}).get("system_instruction", ""),
                                    "output": event.get("output", "")
                                })
                            except Exception as e:
                                print(f"Error extracting LLM call data: {e}")
                        elif event.get("event") == "execution_error":
                            trace_data["errors"].append({
                                "error": event.get("error", ""),
                                "traceback": event.get("traceback", "")
                            })

                    # Categorize as correct or incorrect
                    if result.get("match", False):
                        correct_traces.append(trace_data)
                    else:
                        incorrect_traces.append(trace_data)

        print(f"Categorized into {len(correct_traces)} correct and {len(incorrect_traces)} incorrect traces")
        return correct_traces, incorrect_traces

    def analyze_traces_with_llm(self, iteration_number):
        """
        Use LLM to extract specific insights from execution traces with improved error handling
        and fallback mechanisms.
        """
        print(f"Beginning trace analysis for iteration {iteration_number}")
        correct_traces, incorrect_traces = self.categorize_traces_by_correctness(iteration_number)

        # If we don't have any traces categorized, provide a helpful error message
        if not correct_traces and not incorrect_traces:
            print(f"Warning: No traces categorized for iteration {iteration_number}")
            error_message = f"""
            No usable traces found for iteration {iteration_number}.

            This could be due to several issues:
            1. The trace file may be missing or corrupted
            2. The sample IDs in the trace file don't match those in the iteration data
            3. The iteration may have failed completely with no execution

            Please check:
            - Whether the trace_iteration_{iteration_number}.jsonl file exists
            - If samples were properly executed during this iteration
            - If the test harness correctly recorded trace events
            """
            return error_message

        # Get the full script for analysis
        script = ""
        iteration_data = self.get_iteration_data(iteration_number)
        if iteration_data:
            script = iteration_data.get("script", "")
            if not script:
                print("Warning: No script found in iteration data")

        # Build trace summary with appropriate sampling to avoid oversized prompts
        trace_summary = {
            "iteration": iteration_number,
            "correct_count": len(correct_traces),
            "incorrect_count": len(incorrect_traces),
            "script_excerpt": script[:2000] if len(script) > 2000 else script,
            "correct_samples": [],
            "incorrect_samples": []
        }

        # Add selected correct traces (limit to 2 for prompt size)
        for trace in correct_traces[:2]:
            formatted_llm_calls = []
            for call in trace.get("llm_calls", [])[:3]:  # Limit to first 3 LLM calls
                if call:
                    formatted_llm_calls.append({
                        "caller": call.get("caller", "unknown"),
                        "prompt_excerpt": call.get("prompt", "")[:300] + "..." if len(call.get("prompt", "")) > 300 else call.get("prompt", ""),
                        "output_excerpt": call.get("output", "")[:300] + "..." if len(call.get("output", "")) > 300 else call.get("output", "")
                    })

            sample_info = {
                "index": trace.get("sample_index", -1),
                "question_snippet": trace.get("question", "")[:200] + "..." if len(trace.get("question", "")) > 200 else trace.get("question", ""),
                "expected_answer": trace.get("answer", "")[:100] + "..." if len(trace.get("answer", "")) > 100 else trace.get("answer", ""),
                "system_answer": trace.get("system_answer", "")[:100] + "..." if len(trace.get("system_answer", "")) > 100 else trace.get("system_answer", ""),
                "llm_calls": formatted_llm_calls
            }
            trace_summary["correct_samples"].append(sample_info)

        # Add selected incorrect traces (limit to 3 for prompt size)
        for trace in incorrect_traces[:3]:
            formatted_llm_calls = []
            for call in trace.get("llm_calls", [])[:3]:  # Limit to first 3 LLM calls
                if call:
                    formatted_llm_calls.append({
                        "caller": call.get("caller", "unknown"),
                        "prompt_excerpt": call.get("prompt", "")[:300] + "..." if len(call.get("prompt", "")) > 300 else call.get("prompt", ""),
                        "output_excerpt": call.get("output", "")[:300] + "..." if len(call.get("output", "")) > 300 else call.get("output", "")
                    })

            # Format error messages
            errors = []
            for error in trace.get("errors", []):
                error_msg = error.get("error", "")
                errors.append(error_msg[:200] + "..." if len(error_msg) > 200 else error_msg)

            sample_info = {
                "index": trace.get("sample_index", -1),
                "question_snippet": trace.get("question", "")[:200] + "..." if len(trace.get("question", "")) > 200 else trace.get("question", ""),
                "expected_answer": trace.get("answer", "")[:100] + "..." if len(trace.get("answer", "")) > 100 else trace.get("answer", ""),
                "system_answer": trace.get("system_answer", "")[:100] + "..." if len(trace.get("system_answer", "")) > 100 else trace.get("system_answer", ""),
                "llm_calls": formatted_llm_calls,
                "errors": errors
            }
            trace_summary["incorrect_samples"].append(sample_info)

        # System instruction for trace analysis
        system_instruction = """
        You are an Expert Code Trace Analyzer specializing in debugging LLM-driven systems. 
        Your expertise is in examining execution traces and connecting failures to specific 
        code issues or prompt engineering problems. Focus on being extremely specific and actionable.
        """

        # Create focus prompt based on available data
        if correct_traces and incorrect_traces:
            # We have both correct and incorrect traces - focus on comparing them
            focus_section = """
            1. COMPARE SUCCESS VS FAILURE PATTERNS:
                - What specific differences exist between successful and failed executions?
                - Which prompt structures or function calls lead to success vs. failure?
                - What code paths differ between successful and failed cases?
            """
        elif correct_traces:
            # We only have correct traces - focus on what's working well
            focus_section = """
            1. SUCCESS PATTERN ANALYSIS:
                - What specific code patterns are working correctly?
                - Which prompt structures and LLM responses are effective?
                - What makes these successful cases work well?
            """
        else:
            # We only have incorrect traces - focus on failure analysis
            focus_section = """
            1. FAILURE PATTERN ANALYSIS:
                - What specific code issues are causing failures?
                - Which prompt structures or function calls are problematic?
                - What exact lines or components need to be fixed?
            """

        # Create a highly focused prompt
        prompt = f"""
        Perform a detailed analysis of execution traces from iteration {iteration_number}.

        SUMMARY:
        - Correct examples: {len(correct_traces)}
        - Incorrect examples: {len(incorrect_traces)}

        CODE CONTEXT (excerpt):
        ```python
        {trace_summary['script_excerpt']}
        ```

        EXECUTION DATA:
        {json.dumps(trace_summary, indent=2)}

        ANALYSIS FOCUS:
        {focus_section}

        2. CODE-LEVEL ANALYSIS:
           - Identify specific functions that succeeded or failed
           - Point out exact code patterns that led to success or failure
           - Find bugs or inefficiencies in the implementation

        3. PROMPT ENGINEERING ANALYSIS:
           - Analyze the prompts used in LLM calls
           - Identify effective or problematic prompt elements
           - Suggest specific prompt improvements

        4. FAILURE POINT ISOLATION:
           - For incorrect samples, pinpoint the exact step where execution failed
           - Connect errors to specific code pathways
           - Identify whether failures occurred in prompt construction, LLM response, or output processing

        5. CONCRETE RECOMMENDATIONS:
           - Suggest specific code changes (exact function names, logic adjustments)
           - Provide precise prompt engineering improvements
           - Recommend exact parsing or validation approaches

        6. HIGH LEVEL INSIGHTS:
           - Pretend you are a human expert reviewing the system's performance. You think qualitatively at a high level about the system's behavior and reasoning, focusing on making the system more intelligent and more capable, and analyzing failure points. For example, "this system doesn't seem to understand that what we really want is..."
           - What is the system doing wrong that it could be doing better?
           - What is the system doing right that it should continue to do?
           - Based on wrong answers compared to the correct answers, outline the reasoning steps that would have gotten the correct answer.
           - What patterns, techniques, or approaches would have gotten the correct answer?
           - At a high level (not code) and based on this iteration, what specific things could be done to improve the system?

        FORMAT YOUR RESPONSE AS:

        ## EXECUTION PATTERN ANALYSIS
        [Analysis of execution patterns]

        ## SUCCESS FACTORS
        [Specific elements leading to success]

        ## FAILURE POINTS
        [Exact points of failure with code references]

        ## CODE-LEVEL RECOMMENDATIONS
        [Specific code changes]

        ## PROMPT ENGINEERING RECOMMENDATIONS
        [Exact prompt improvements]

        ## HIGH LEVEL INSIGHTS
        [General human-like insights and suggestions]

        Be extremely specific - reference actual function names, quote problematic code or prompts, 
        and suggest exact fixes.
        """

        # Call LLM for analysis with appropriate error handling
        try:
            print("Sending trace data to LLM for analysis...")
            response = self.call_llm(prompt, system_instruction=system_instruction)
            print(f"Received trace analysis from LLM ({len(response)} chars)")
            return response
        except Exception as e:
            error_message = f"Error analyzing traces: {str(e)}"
            print(error_message)

            # Provide a fallback analysis if LLM call fails
            if correct_traces or incorrect_traces:
                fallback = self._generate_fallback_analysis(correct_traces, incorrect_traces)
                return fallback
            else:
                return error_message

    def _generate_fallback_analysis(self, correct_traces, incorrect_traces):
        """Generate a basic analysis without using LLM if the LLM call fails."""
        analysis = [
            "## EXECUTION PATTERN ANALYSIS",
            f"Found {len(correct_traces)} successful and {len(incorrect_traces)} failed executions."
        ]

        # Add success factors if we have successful traces
        if correct_traces:
            analysis.extend([
                "",
                "## SUCCESS FACTORS",
                "The following executions completed successfully:"
            ])

            for i, trace in enumerate(correct_traces):
                analysis.append(f"- Sample {trace.get('sample_index', i)}: {len(trace.get('llm_calls', []))} LLM calls")

        # Add failure points if we have failed traces
        if incorrect_traces:
            analysis.extend([
                "",
                "## FAILURE POINTS",
                "The following executions failed:"
            ])

            for i, trace in enumerate(incorrect_traces):
                errors = [e.get('error', 'Unknown error') for e in trace.get('errors', [])]
                error_summary = '; '.join(errors) if errors else 'No specific error recorded'
                analysis.append(f"- Sample {trace.get('sample_index', i)}: {error_summary}")

        # Add recommendations
        analysis.extend([
            "",
            "## CODE-LEVEL RECOMMENDATIONS",
            "- Add more detailed error handling and logging to identify specific failure points",
            "- Consider adding validation steps to verify LLM outputs before processing them",
            "",
            "## PROMPT ENGINEERING RECOMMENDATIONS",
            "- Review prompts for clarity and completeness",
            "- Consider adding more examples to guide the LLM's responses"
        ])

        return "\n".join(analysis)

    def build_insight_based_context(self, iterations, max_iterations=3):
        """Build context based on insights from traces."""
        context_parts = []
        context_parts.append("=== EXECUTION TRACE INSIGHTS ===\n")

        for iteration in iterations[-max_iterations:]:
            iteration_number = iteration.get("iteration")
            if iteration_number is None:
                continue

            # First check if we already have stored insights
            stored_insights = iteration.get("trace_insights")

            if stored_insights:
                context_parts.append(f"\n--- Iteration {iteration_number} Analysis ---")
                context_parts.append(stored_insights)
            else:
                # Generate new insights
                insights = self.analyze_traces_with_llm(iteration_number)

                if insights and not insights.startswith("Error") and not "No usable traces found" in insights:
                    context_parts.append(f"\n--- Iteration {iteration_number} Analysis ---")
                    context_parts.append(insights)
                else:
                    # Fallback to basic analysis if LLM call fails
                    correct_traces, incorrect_traces = self.categorize_traces_by_correctness(iteration_number)
                    context_parts.append(f"\n--- Iteration {iteration_number} (Basic Analysis) ---")
                    context_parts.append(f"Correct: {len(correct_traces)}, Failed: {len(incorrect_traces)}")

                    if incorrect_traces:
                        context_parts.append("Common errors:")
                        for trace in incorrect_traces[:2]:
                            errors = trace.get('errors', [])
                            for error in errors:
                                context_parts.append(f"  - {error.get('error', 'Unknown error')[:100]}...")
                                break

        return "\n".join(context_parts)

class CapabilityTracker:
    """
    Tracks and generates qualitative assessments of system capabilities across iterations.
    Uses natural language analysis rather than rigid scores for flexibility.
    """

    def __init__(self):
        # Define core capabilities that are domain-agnostic
        self.capabilities = [
            "information_extraction",
            "constraint_handling",
            "solution_generation", 
            "solution_verification",
            "decision_making"
        ]

        # Store assessment history
        self.history = []

        # Current assessment
        self.current_assessment = {
            "strengths": [],
            "weaknesses": [],
            "bottlenecks": [],
            "improvement_areas": [],
            "improvement_suggestions": []
        }

    def update_from_evaluations(self, evaluations, capability_insights, error_analysis):
        """
        Generate a qualitative assessment based on evaluation results and text reports.

        Args:
            evaluations: List of evaluation results
            capability_insights: Insights from LLM analysis (may be empty)
            error_analysis: Error analysis from LLM (now contains text_report)
        """
        print(f"Generating qualitative assessment from {len(evaluations)} evaluations")

        # Extract basic statistics
        success_count = sum(1 for eval_data in evaluations if eval_data.get("match", False))
        failure_count = len(evaluations) - success_count
        success_rate = success_count / len(evaluations) if evaluations else 0
        print(f"Success rate: {success_rate:.2f} ({success_count}/{len(evaluations)})")

        # Generate qualitative assessment
        assessment = {
            "timestamp": datetime.datetime.now().isoformat(),
            "success_rate": success_rate,
            "text_report": error_analysis.get("text_report", "No report available"),
            "strengths": error_analysis.get("strengths", []),
            "weaknesses": error_analysis.get("weaknesses", []),
            "improvement_areas": [],
            "improvement_suggestions": error_analysis.get("improvement_suggestions", [])
        }

        # If we have specific capabilities mentioned in weaknesses, track them
        for weakness in assessment["weaknesses"]:
            for capability in self.capabilities:
                if capability.replace("_", " ") in weakness.lower():
                    assessment["improvement_areas"].append(capability)

        # Ensure we have some data
        if not assessment["strengths"]:
            assessment["strengths"].append("Successfully generated output")

        if not assessment["weaknesses"] and success_rate < 1.0:
            assessment["weaknesses"].append("Some examples still failing")

        if not assessment["improvement_areas"] and self.capabilities:
            assessment["improvement_areas"].append(self.capabilities[0])

        # Store the assessment
        self.current_assessment = assessment
        self.history.append(assessment)

        print(f"Generated qualitative assessment with {len(assessment['strengths'])} strengths, " +
              f"{len(assessment['weaknesses'])} weaknesses")

        return assessment
    
    def get_improvement_focus(self):
        """
        Return the current most critical improvement area.
        """
        if self.current_assessment["improvement_areas"]:
            return self.current_assessment["improvement_areas"][0]
        elif self.current_assessment["weaknesses"]:
            # Extract capability from weakness description
            for weakness in self.current_assessment["weaknesses"]:
                for cap in self.capabilities:
                    if cap.replace("_", " ") in weakness.lower():
                        return cap

        # Default to information extraction if nothing else is identified
        #return "no specific focus, refer to text reports"
        return ""

    def generate_report(self):
        """
        Generate a comprehensive report on capability status.
        """
        # Get the text report from the current assessment
        report = {
            "text_report": self.current_assessment.get("text_report", "No report available"),
            "strengths": self.current_assessment.get("strengths", []),
            "weaknesses": self.current_assessment.get("weaknesses", []),
            "improvement_suggestions": self.current_assessment.get("improvement_suggestions", []),
        }

        # Add trend analysis if we have enough history
        if len(self.history) >= 2:
            report["trend"] = self._analyze_trend()
        else:
            report["trend"] = "insufficient_data"
            
        return report

    def _analyze_trend(self):
        """
        Generate a qualitative analysis of trends over time.
        """
        if len(self.history) < 2:
            return "insufficient_data"

        # Compare the most recent two assessments
        latest = self.history[-1]
        previous = self.history[-2]

        # Compare success rates
        latest_rate = latest.get("success_rate", 0)
        previous_rate = previous.get("success_rate", 0)
        rate_change = latest_rate - previous_rate

        trends = {}

        # Generate natural language trends for each capability
        # This is a simplified approach that could be enhanced
        for cap in self.capabilities:
            # Start with overall trend based on success rate
            if rate_change > 0.05:
                trends[cap] = "improving"
            elif rate_change < -0.05:
                trends[cap] = "declining"
            else:
                trends[cap] = "stable"

            # Check if this capability was mentioned in weaknesses in both assessments
            latest_weakness = any(cap.replace("_", " ") in w.lower() for w in latest.get("weaknesses", []))
            previous_weakness = any(cap.replace("_", " ") in w.lower() for w in previous.get("weaknesses", []))

            # Override based on specific capability mentions
            if previous_weakness and not latest_weakness:
                trends[cap] = "improving"
            elif not previous_weakness and latest_weakness:
                trends[cap] = "declining"

        return trends

if __name__ == "__main__":
    pass
