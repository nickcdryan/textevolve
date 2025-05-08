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
                # Use the universal "question" field instead of "prompt_0shot"
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

        prompt = f"""
        Extract specific, concrete learnings from this iteration's results, focusing on dataset-specific insights:

        Iteration: {iteration_data.get("iteration")}
        Strategy: {iteration_data.get("strategy", "Unknown")}
        Accuracy: {accuracy:.2f}
        Approach summary: {iteration_data.get("approach_summary", "No summary available")}

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
    
    def synthesize_learnings(self, current_learnings: str, new_batch_learnings: str) -> str:
        """
        Synthesize existing learnings with new batch learnings, emphasizing dataset-specific insights.
        Automatically condenses content when approaching token limits.
        """
        # Define character limit threshold (staying well under the 41,000 character limit)
        CHARACTER_LIMIT_THRESHOLD = 40000

        # Calculate current lengths
        current_length = len(current_learnings)
        new_length = len(new_batch_learnings)
        combined_length = current_length + new_length

        # Print length info for debugging
        print(f"Current learnings length: {current_length}")
        print(f"New batch learnings length: {new_length}")
        print(f"Combined length: {combined_length}")

        # Determine if we need to condense content
        approaching_limit = combined_length > CHARACTER_LIMIT_THRESHOLD

        # Choose appropriate system instruction based on our needs
        if approaching_limit:
            system_instruction = "You are a Knowledge Condenser and Synthesizer. Your role is to extract and distill the most critical insights from accumulated knowledge and new findings, creating a concise yet comprehensive summary while staying within strict length constraints."
            print(f"Approaching character limit ({combined_length}/{CHARACTER_LIMIT_THRESHOLD}). Activating condensing mode.")
        else:
            system_instruction = "You are a Knowledge Integrator. Your role is to synthesize accumulated dataset-specific knowledge with new insights, creating an evolving experiment log that captures concrete patterns, strategies, and findings about this specific task."

        # Base prompt content
        base_prompt = f"""
        You are tasked with synthesizing existing knowledge with new learnings from our latest experiment on this dataset.

        EXISTING ACCUMULATED LEARNINGS:
        {current_learnings}

        NEW LEARNINGS FROM LATEST BATCH:
        {new_batch_learnings}
        """

        # Common synthesis instructions
        synthesis_instructions = """
        Organize the information into these sections:
        1. DATASET PATTERNS & CHARACTERISTICS
        2. EFFECTIVE TASK-SPECIFIC STRATEGIES
        3. COMMON FAILURE MODES ON THIS DATASET
        4. EXPERIMENT LOG & FINDINGS
        5. NEXT RESEARCH DIRECTIONS

        The synthesized learnings should read like a detailed research log about THIS specific dataset and task, 
        not a general guide to system design. Each section should include specific examples and concrete findings.

        Your output will replace the current learnings file and serve as long-term memory for working specifically with this dataset.
        """

        # Condensing-specific instructions when approaching limit
        if approaching_limit:
            condensing_instructions = f"""
            CRITICAL: The combined length of existing learnings and new learnings is approaching our limit.

            Your task is to CONDENSE and SYNTHESIZE this information while:
            1. Preserving the most important and actionable insights
            2. Eliminating redundancies and generalizing similar findings
            3. Focusing on concrete patterns rather than general observations
            4. Prioritizing recent findings while incorporating key historical insights
            5. Keeping your output UNDER {CHARACTER_LIMIT_THRESHOLD} characters in total length

            When condensing:
            - Merge similar observations from different iterations
            - Remove lengthy examples that illustrate the same point
            - Summarize detailed error descriptions while preserving their core lessons
            - Focus on patterns and trends rather than isolated incidents
            - Retain specific error details only when they reveal unique insights

            Create an updated, condensed version of our learnings that:
            1. Maintains the most important DATASET PATTERNS we've identified
            2. Preserves our critical understanding about what makes this specific task challenging
            3. Retains the most effective STRATEGIES and important failure modes
            4. Summarizes the EXPERIMENT LOG to highlight key findings
            5. Prioritizes concrete, task-specific insights over general principles

            IMPORTANT: Your response MUST be shorter than {CHARACTER_LIMIT_THRESHOLD} characters while preserving the most critical information.
            """

            prompt = base_prompt + condensing_instructions + synthesis_instructions
        else:
            # Standard synthesis instructions when not approaching limit
            standard_instructions = """
            Create an updated, synthesized version of our learnings that:
            1. Maintains a comprehensive catalog of DATASET PATTERNS we've identified
            2. Tracks the evolution of our understanding about what makes this specific task challenging
            3. Documents concrete STRATEGIES that have proven effective or ineffective for this particular dataset
            4. Creates a running EXPERIMENT LOG tracking our attempts and findings specific to this task
            5. Prioritizes concrete, task-specific insights over general system design principles

            IMPORTANT: Preserve all concrete, specific insights from both the existing learnings and new batch. Don't lose valuable information. Preserve the details of runtime, execution, error, and processing problems so that we can learn from them in the future and DO NOT make the same mistakes again.
            """

            prompt = base_prompt + standard_instructions + synthesis_instructions

        try:
            print(f"Calling LLM to {'condense and synthesize' if approaching_limit else 'synthesize'} learnings...")
            response = self.call_llm(prompt, system_instruction=system_instruction)

            response_length = len(response.strip())
            print(f"Received synthesized learnings: {response_length} characters")

            # Check if we're still close to the limit after synthesis
            if approaching_limit and response_length > CHARACTER_LIMIT_THRESHOLD:
                print(f"WARNING: Synthesized content ({response_length} chars) still exceeds threshold ({CHARACTER_LIMIT_THRESHOLD} chars)")

                # If we're still over the limit, try one more time with stricter constraints
                if response_length > CHARACTER_LIMIT_THRESHOLD:
                    emergency_prompt = f"""
                    EMERGENCY CONDENSING REQUIRED: The synthesized learnings are still too long at {response_length} characters.

                    Please condense this content DRASTICALLY while preserving only the MOST CRITICAL insights:

                    {response.strip()}

                    Your output MUST be under {CHARACTER_LIMIT_THRESHOLD} characters. Be ruthless in removing less important details
                    while maintaining the core patterns, key strategies, and critical failure modes.

                    Focus only on the most important:
                    1. DATASET PATTERNS & CHARACTERISTICS (most distinctive ones only)
                    2. EFFECTIVE TASK-SPECIFIC STRATEGIES (only the proven ones)
                    3. COMMON FAILURE MODES ON THIS DATASET (only the recurring ones)
                    4. KEY FINDINGS (only the most significant discoveries)
                    5. NEXT RESEARCH DIRECTIONS (only the most promising)

                    MAXIMUM LENGTH: {CHARACTER_LIMIT_THRESHOLD} characters.
                    """

                    print("Attempting emergency condensing...")
                    response = self.call_llm(
                        emergency_prompt,
                        system_instruction="You are an Emergency Content Condenser. Your task is to drastically reduce content length while preserving critical information."
                    )
                    print(f"Emergency condensing result: {len(response.strip())} characters")

            return response.strip()

        except Exception as e:
            error_message = f"Error synthesizing learnings: {str(e)}"
            print(error_message)
            traceback.print_exc()  # Print full traceback for better debugging

            # Return a condensed fallback if we're approaching the limit
            if approaching_limit:
                fallback = f"=== EMERGENCY TRUNCATED LEARNINGS ===\n\n{current_learnings[:20000]}\n\n=== NEWEST LEARNINGS (TRUNCATED) ===\n\n{new_batch_learnings[:5000]}"
                print(f"Using emergency truncated fallback: {len(fallback)} characters")
                return fallback
            else:
                # Standard fallback when not approaching limit
                fallback = f"{current_learnings}\n\n=== NEWEST LEARNINGS (NOT SYNTHESIZED DUE TO ERROR) ===\n\n{new_batch_learnings}"
                print(f"Using fallback concatenation: {len(fallback)} characters")
                return fallback
    
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

    # Modify the generate_script_with_llm method in the AgentSystem class to include learnings

    def generate_script_with_llm(self, is_exploration: bool) -> str:
        """
        Use the LLM to generate a script to solve dataset problems.
        Modified to work with dataset loader approach and use standard field names.
        """
        # Get previous iterations and summaries
        iterations = self.get_all_iterations()
        summaries = self.get_summaries()

        # Instead of getting current test samples, use examples from past iterations
        example_problems = []

        if iterations:
            # Collect samples from previous iterations
            prev_samples = []
            for iteration in sorted(iterations, key=lambda x: x.get('iteration', 0) if x else 0, reverse=True)[:3]:
                if iteration and 'samples' in iteration:
                    prev_samples.extend(iteration.get('samples', [])[:3])  # Get up to 3 from each recent iteration

            # Use these previous samples as examples
            for i, sample in enumerate(prev_samples[:3]):  # Limit to 3 examples
                example_problems.append({
                    "id": i,
                    "question": sample.get("question", ""),  # Use universal "question" field
                    "answer": sample.get("answer", "")       # Use universal "answer" field
                })

        # Handle cold start (first iteration)
        if not example_problems:
            # For initial training, get examples from a fixed training set
            training_examples = self.get_training_examples(5)
            for i, example in enumerate(training_examples):
                example_problems.append({
                    "id": i,
                    "question": self.dataset_loader.get_example_input(example),
                    "answer": self.dataset_loader.get_example_output(example)
                })

        # ==== LOAD ACCUMULATED LEARNINGS ====
        accumulated_learnings = self._load_learnings()

        # ==== HISTORICAL ANALYSIS ====
        best_scripts = []
        if iterations:
            for iteration in sorted(
                    iterations,
                    key=lambda x: x.get('performance', {}).get('accuracy', 0),
                    reverse=True)[:3]:
                if iteration.get('script') and iteration.get(
                        'performance', {}).get('accuracy', 0) > 0:
                    best_scripts.append({
                        "iteration":
                        iteration.get('iteration'),
                        "accuracy":
                        iteration.get('performance', {}).get('accuracy', 0),
                        "approach_summary":
                        iteration.get('approach_summary',
                                      'No summary available'),
                        "performance":
                        iteration.get('performance', {})
                    })

        # Get top performing scripts for exploitation instead of just the best one
        top_scripts_to_exploit = []
        if not is_exploration and best_scripts:
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

        # Collect approach history
        approach_history = []
        for summary in summaries:
            approach_history.append({
                "iteration":
                summary.get("iteration"),
                "strategy":
                summary.get("strategy"),
                "accuracy":
                summary.get("performance", {}).get("accuracy", 0),
                "approach":
                summary.get("approach_summary", "No summary available")
            })

        # Aggregate error analyses
        error_patterns = []
        primary_issues = []
        targeted_improvements = []

        for iteration in iterations:
            if not iteration:
                continue

            error_analysis = iteration.get("performance",
                                           {}).get("error_analysis", {})
            if error_analysis.get("error_patterns"):
                error_patterns.extend(error_analysis.get("error_patterns", []))
            if error_analysis.get("primary_issue"):
                primary_issues.append({
                    "iteration":
                    iteration.get("iteration"),
                    "issue":
                    error_analysis.get("primary_issue")
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

        # Get capability insights
        capability_report = None
        improvement_focus = None
        capability_guidance = ""

        if hasattr(self, 'capability_tracker'):
            capability_report = self.capability_tracker.generate_report()
            improvement_focus = capability_report.get("improvement_focus")
            if capability_report:
                capability_guidance = self._generate_capability_guidance(capability_report)


        # ==== RETRIEVE LAST FIVE SCRIPTS FOR EXPLORATION CONTEXT ====
        last_scripts_context = ""
        if is_exploration and iterations:
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

        # ==== DETERMINE STRATEGY ====
        approach_type = "exploration" if is_exploration else "exploitation"
        best_script_to_exploit = None

        if not is_exploration and best_scripts:
            best_script_to_exploit = best_scripts[0]
            for iteration in iterations:
                if iteration.get("iteration") == best_script_to_exploit.get(
                        "iteration"):
                    best_script_to_exploit["script"] = iteration.get(
                        "script", "")
                    break

        # ==== PREPARE CONTEXT ====
        # API usage example
        gemini_api_example = 'def call_llm(prompt, system_instruction=None):\n    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""\n    try:\n        from google import genai\n        from google.genai import types\n\n        # Initialize the Gemini client\n        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))\n\n        # Call the API with system instruction if provided\n        if system_instruction:\n            response = client.models.generate_content(\n                model="gemini-2.0-flash", \n                config=types.GenerateContentConfig(\n                    system_instruction=system_instruction\n                ),\n                contents=prompt\n            )\n        else:\n            response = client.models.generate_content(\n                model="gemini-2.0-flash",\n                contents=prompt\n            )\n\n        return response.text\n    except Exception as e:\n        print(f"Error calling Gemini API: {str(e)}")\n        return f"Error: {str(e)}"'

        # Example 1: General-purpose information extraction with embedded examples
        extraction_example = 'def extract_information_with_examples(text):\n    """Extract key information from the input text using embedded examples."""\n    system_instruction = "You are an information extraction specialist focusing on identifying key entities and relationships."\n    \n    prompt = f"""\n    Extract key information from this text. Focus on identifying all entities, relationships, and important attributes.\n    \n    Example usage:\n    \n    Input Text:\n    The company XYZ Corp reported quarterly earnings of $3.5 million, which represents a 12% increase from last year. The CEO, Jane Smith, attributed this growth to their new product line launched in March, which has already captured 8% of the market share. They expect to expand their operations to Europe by Q2 2023.\n    \n    Let\'s think step by step.\n    \n    The key entities are:\n    - XYZ Corp (company)\n    - Jane Smith (person, CEO)\n    - New product line (product)\n    \n    The key information points are:\n    - Financial: Quarterly earnings of $3.5 million\n    - Performance: 12% increase from previous year\n    - Product: New product line launched in March\n    - Market: 8% market share for new product\n    - Plans: Expansion to Europe by Q2 2023\n    \n    Extracted Information:\n    {{\n      "entities": [\n        {{"name": "XYZ Corp", "type": "company"}},\n        {{"name": "Jane Smith", "type": "person", "role": "CEO"}},\n        {{"name": "New product line", "type": "product", "launch_date": "March"}}\n      ],\n      "financial_data": {{\n        "quarterly_earnings": "$3.5 million",\n        "growth_rate": "12%"\n      }},\n      "market_data": {{\n        "product_market_share": "8%"\n      }},\n      "future_plans": [\n        {{"type": "expansion", "region": "Europe", "timeline": "Q2 2023"}}\n      ]\n    }}\n    \n    Now, extract information from this new text:\n    {text}\n    """\n    \n    return call_llm(prompt, system_instruction)'

        # Example 2: General-purpose verification with embedded examples
        verification_example = 'def verify_solution_with_examples(problem, proposed_solution):\n    """Verify if the proposed solution satisfies all requirements using embedded examples."""\n    system_instruction = "You are a critical evaluator who verifies if solutions correctly address problems."\n    \n    prompt = f"""\n    Verify if this proposed solution correctly addresses all aspects of the problem.\n    \n    Example usage:\n    \n    Problem:\n    Design a data structure that can efficiently perform the following operations:\n    1. Insert a value\n    2. Delete a value\n    3. Get a random value with equal probability for all stored values\n    All operations should have average time complexity of O(1).\n    \n    Proposed Solution:\n    I\'ll use a combination of a hashmap and an array. The hashmap will store the value as the key and its index in the array as the value. The array will store all the inserted values.\n    \n    For insert: Add the value to the end of the array and update the hashmap with the value and its index. O(1) time.\n    \n    For delete: Look up the index of the value in the hashmap, swap the value with the last element in the array, update the hashmap for the swapped element, remove the last element from the array, and remove the value from the hashmap. O(1) time.\n    \n    For get random: Generate a random index within the array\'s bounds and return the value at that index. O(1) time.\n    \n    Verification:\n    Let me check each requirement:\n    1. Insert operation: The solution adds the value to the end of the array and updates the hashmap with O(1) time complexity ✓\n    2. Delete operation: The solution uses the hashmap to find the index, then swaps with the last element and updates accordingly with O(1) time complexity ✓\n    3. Get random operation: The solution generates a random index within the array bounds with O(1) time complexity ✓\n    4. All operations have O(1) average time complexity ✓\n    \n    Result: VALID - The solution correctly addresses all requirements with the specified time complexity.\n    \n    Problem:\n    {problem}\n    \n    Proposed Solution:\n    {proposed_solution}\n    \n    Verification:\n    """\n    \n    return call_llm(prompt, system_instruction)'

        # Example 3: Repeated validation with feedback loop
        validation_loop_example = 'def solve_with_validation_loop(problem, max_attempts=3):\n    """Solve a problem with iterative refinement through validation feedback loop."""\n    system_instruction_solver = "You are an expert problem solver who creates detailed, correct solutions."\n    system_instruction_validator = "You are a critical validator who carefully checks solutions against all requirements."\n    \n    # Initial solution generation\n    solution_prompt = f"""\n    Provide a detailed solution to this problem. Be thorough and ensure you address all requirements.\n    \n    Problem:\n    {problem}\n    """\n    \n    solution = call_llm(solution_prompt, system_instruction_solver)\n    \n    # Validation loop\n    for attempt in range(max_attempts):\n        # Validate the current solution\n        validation_prompt = f"""\n        Carefully validate if this solution correctly addresses all aspects of the problem.\n        If the solution is valid, respond with "VALID: [brief reason]".\n        If the solution has any issues, respond with "INVALID: [detailed explanation of issues]".\n        \n        Problem:\n        {problem}\n        \n        Proposed Solution:\n        {solution}\n        """\n        \n        validation_result = call_llm(validation_prompt, system_instruction_validator)\n        \n        # Check if solution is valid\n        if validation_result.startswith("VALID:"):\n            return solution\n        \n        # If invalid, refine the solution\n        refined_prompt = f"""\n        Your previous solution to this problem has some issues that need to be addressed.\n        \n        Problem:\n        {problem}\n        \n        Your previous solution:\n        {solution}\n        \n        Validation feedback:\n        {validation_result}\n        \n        Please provide a completely revised solution that addresses all the issues mentioned.\n        """\n        \n        solution = call_llm(refined_prompt, system_instruction_solver)\n    \n    return solution'

        # Example 4: Multi-perspective analysis with synthesis
        multi_perspective_example = 'def multi_perspective_analysis(problem):\n    """Analyze a problem from multiple specialized perspectives and synthesize the insights."""\n    # Define specialized analysis functions\n    def analyze_factual_content(problem):\n        system_instruction = "You are a factual analyst who focuses on identifying key facts and data points."\n        prompt = f"""\n        Analyze this problem for factual content only. Identify explicit facts, constraints, and requirements.\n        \n        Problem:\n        {problem}\n        """\n        return call_llm(prompt, system_instruction)\n    \n    def analyze_structure(problem):\n        system_instruction = "You are a structural analyst who specializes in problem organization and patterns."\n        prompt = f"""\n        Analyze the structure of this problem. Identify its components, relationships, and patterns.\n        \n        Problem:\n        {problem}\n        """\n        return call_llm(prompt, system_instruction)\n    \n    # Execute parallel analyses\n    factual_analysis = analyze_factual_content(problem)\n    structural_analysis = analyze_structure(problem)\n    \n    # Synthesize the results\n    synthesis_prompt = f"""\n    Synthesize these two different analyses of the same problem into a comprehensive understanding.\n    \n    Factual Analysis:\n    {factual_analysis}\n    \n    Structural Analysis:\n    {structural_analysis}\n    \n    Provide a unified analysis that leverages both perspectives.\n    """\n    \n    return call_llm(synthesis_prompt, "You are an insight synthesizer who combines multiple analyses.")'

        # Example 5: Best of n approach for solution selection
        best_of_n_example = 'def best_of_n_approach(problem, n=3):\n    """Generate multiple solutions and select the best one based on a quality evaluation."""\n    system_instruction_solver = "You are an expert problem solver who provides detailed, correct solutions."\n    system_instruction_evaluator = "You are a quality evaluator who assesses solutions based on correctness, completeness, and clarity."\n    \n    # Generate n different solutions\n    solutions = []\n    for i in range(n):\n        diversity_factor = f"Solution approach {i+1}/{n}: Use a different perspective from previous solutions."\n        solution_prompt = f"""\n        Provide a detailed solution to this problem.\n        {diversity_factor if i > 0 else ""}\n        \n        Problem:\n        {problem}\n        """\n        \n        solutions.append(call_llm(solution_prompt, system_instruction_solver))\n    \n    # Evaluate each solution\n    evaluations = []\n    for i, solution in enumerate(solutions):\n        evaluation_prompt = f"""\n        Evaluate this solution on correctness, completeness, and clarity (1-10 scale).\n        \n        Problem:\n        {problem}\n        \n        Solution {i+1}:\n        {solution}\n        \n        Provide your evaluation as a JSON with scores and explanation.\n        """\n        \n        evaluations.append(call_llm(evaluation_prompt, system_instruction_evaluator))\n    \n    # Find the best solution\n    comparison_prompt = f"""\n    Compare these solutions and their evaluations. Select the best one.\n    \n    Problem:\n    {problem}\n    \n    {["Solution " + str(i+1) + ": " + solutions[i] + "\\n\\nEvaluation: " + evaluations[i] for i in range(n)]}\n    \n    Which solution is best? Respond with the solution number and explanation.\n    """\n    \n    best_solution_index = int(call_llm(comparison_prompt, "You are a solution selector.").split()[1]) - 1\n    return solutions[best_solution_index]'

        # Example 6: ReAct pattern for interactive reasoning and action
        react_example = 'def solve_with_react_pattern(problem):\n    """Solve problems through iterative Reasoning and Acting (ReAct) approach."""\n    system_instruction = "You are a problem-solving agent that follows the ReAct pattern: Reason about the current state, take an Action, observe the result, and repeat until reaching a solution."\n    \n    # Initialize ReAct process\n    prompt = f"""\n    Solve this problem using the ReAct pattern - alternate between Reasoning and Acting until you reach a final answer.\n    \n    Example usage:\n    \n    Problem: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?\n    \n    Thought 1: I need to determine which country the Great Barrier Reef is in, then find its capital, and finally the population of that capital.\n    Action 1: Search[Great Barrier Reef location]\n    Observation 1: The Great Barrier Reef is located off the coast of Queensland in northeastern Australia.\n    \n    Thought 2: Now I know the Great Barrier Reef is in Australia. I need to find Australia\'s capital city.\n    Action 2: Search[capital of Australia]\n    Observation 2: The capital of Australia is Canberra.\n    \n    Thought 3: Now I need to find the population of Canberra.\n    Action 3: Search[population of Canberra]\n    Observation 3: As of 2021, the population of Canberra is approximately 431,500.\n    \n    Thought 4: I have found all the required information. The capital of Australia (where the Great Barrier Reef is located) is Canberra, and its population is approximately 431,500.\n    Action 4: Finish[The capital of Australia is Canberra, with a population of approximately 431,500.]\n    \n    Now solve this new problem:\n    {problem}\n    \n    Start with Thought 1:\n    """\n    \n    # Initial reasoning and action planning\n    react_response = call_llm(prompt, system_instruction)\n    \n    # Extract the action from the response\n    action = extract_action(react_response)\n    \n    # Continue the ReAct loop until we reach a "Finish" action\n    while not action["type"] == "Finish":\n        # Perform the requested action and get an observation\n        if action["type"] == "Search":\n            observation = perform_search(action["query"])\n        elif action["type"] == "Calculate":\n            observation = perform_calculation(action["expression"])\n        elif action["type"] == "Lookup":\n            observation = perform_lookup(action["term"])\n        else:\n            observation = f"Unknown action type: {action[\'type\']}"\n        \n        # Continue the ReAct process with the new observation\n        continuation_prompt = f"""\n        {react_response}\n        Observation {action["step_number"]}: {observation}\n        \n        Continue with the next thought and action:\n        """\n        \n        # Get the next reasoning step and action\n        react_response += "\\n" + call_llm(continuation_prompt, system_instruction)\n        \n        # Extract the next action\n        action = extract_action(react_response)\n    \n    # Extract the final answer from the Finish action\n    final_answer = action["answer"]\n    return final_answer\n\ndef extract_action(text):\n    """Parse the ReAct response to extract the current action."""\n    # Find the last action in the text\n    action_matches = re.findall(r"Action (\d+): (\\w+)\\[(.*?)\\]", text)\n    if not action_matches:\n        return {"type": "Error", "step_number": 0, "query": "No action found"}\n    \n    # Get the most recent action\n    last_action = action_matches[-1]\n    step_number = int(last_action[0])\n    action_type = last_action[1]\n    action_content = last_action[2]\n    \n    # Handle different action types\n    if action_type == "Finish":\n        return {"type": "Finish", "step_number": step_number, "answer": action_content}\n    elif action_type in ["Search", "Lookup", "Calculate"]:\n        return {"type": action_type, "step_number": step_number, "query": action_content}\n    else:\n        return {"type": "Unknown", "step_number": step_number, "query": action_content}\n\ndef perform_search(query):\n    """Simulate a search action in the ReAct pattern."""\n    # In a real implementation, this would call an actual search API\n    return call_llm(f"Provide a factual answer about: {query}", "You are a helpful search engine that provides concise, factual information.")\n\ndef perform_calculation(expression):\n    """Perform a calculation action in the ReAct pattern."""\n    try:\n        # Safely evaluate the expression\n        result = eval(expression, {"__builtins__": {}}, {"math": math})\n        return f"The result is {result}"\n    except Exception as e:\n        return f"Error in calculation: {str(e)}"\n\ndef perform_lookup(term):\n    """Simulate a lookup action for specific information."""\n    # In a real implementation, this would query a knowledge base or database\n    return call_llm(f"Provide specific information about: {term}", "You are a knowledge base that provides specific factual information.")'

        # Create few-shot examples context 
        few_shot_examples = f"EXAMPLE OF EFFECTIVE LLM USAGE PATTERNS:\n\n```python\n{extraction_example}\n```\n\n```python\n{verification_example}\n```\n\n```python\n{validation_loop_example}\n```\n\n```python\n{multi_perspective_example}\n```\n\n```python\n{best_of_n_example}\n```\n\n```python\n{react_example}\n```"

        # Historical context summary
        best_accuracy_str = f"{best_scripts[0].get('accuracy', 0):.2f} (iteration {best_scripts[0].get('iteration')})" if best_scripts else "None"

        historical_context = f"""
        ITERATION HISTORY SUMMARY:
        - Total iterations completed: {len(summaries)}
        - Current explore/exploit balance: {self.explore_rate}/{self.exploit_rate}
        - Best accuracy achieved: {best_accuracy_str}

        APPROACH HISTORY (last {min(10, len(approach_history))} iterations):
        {json.dumps(approach_history[-10:] if len(approach_history) > 10 else approach_history, indent=2)}

        COMMON ERROR PATTERNS:
        {json.dumps(list(set(error_patterns[-10:] if len(error_patterns) > 10 else error_patterns)), indent=2)}

        PRIMARY ISSUES (last {min(3, len(primary_issues))} iterations):
        {json.dumps(primary_issues[-10:] if len(primary_issues) > 10 else primary_issues, indent=2)}

        TARGETED IMPROVEMENTS:
        {json.dumps(list(set(targeted_improvements[-10:] if len(targeted_improvements) > 10 else targeted_improvements)), indent=2)}
        """

        # Add the few-shot examples to the context
        historical_context += f"\n\n{few_shot_examples}"

        historical_context += """MULTI-EXAMPLE PROMPTING GUIDANCE:
        1. CRITICAL: Use MULTIPLE examples (2-5) in EVERY LLM prompt, not just one
        2. Vary the number of examples based on task complexity - more complex tasks need more examples
        3. Select diverse examples that showcase different patterns and edge cases
        4. Structure your few-shot examples to demonstrate clear step-by-step reasoning
        5. Consider using both "easy" and "challenging" examples to help the LLM learn from contrasts
        6. The collection of examples should collectively cover all key aspects of the problem
        7. When available, use examples from previous iterations that revealed specific strengths or weaknesses.
        8. USE REAL EXAMPLES FROM THE DATASET WHERE POSSIBLE!!

        Example of poor single-example prompting:
        ```python
        def extract_entities(text):
            prompt = f'''
            Extract entities from this text.

            Example:
            Text: John will meet Mary at 3pm on Tuesday.
            Entities: {{"people": ["John", "Mary"], "time": "3pm", "day": "Tuesday"}}

            Text: {text}
            Entities:
            '''
            return call_llm(prompt)
        ```

        Example of effective multi-example prompting:
        ```python
        def extract_entities(text):
            prompt = f'''
            Extract entities from this text.

            Example 1:
            Text: John will meet Mary at 3pm on Tuesday.
            Entities: {{"people": ["John", "Mary"], "time": "3pm", "day": "Tuesday"}}

            Example 2:
            Text: The team needs to submit the report by Friday at noon.
            Entities: {{"people": ["the team"], "time": "noon", "day": "Friday", "object": "report"}}

            Example 3:
            Text: Alex cannot attend the conference from Jan 3-5 due to prior commitments.
            Entities: {{"people": ["Alex"], "event": "conference", "date_range": ["Jan 3-5"], "reason": "prior commitments"}}

            Text: {text}
            Entities:
            '''
            return call_llm(prompt)
        ```

        === DIRECT LLM REASONING APPROACH ===

        CRITICAL: Previous scripts have shown that complex code generation with JSON parsing and multi-step pipelines often 
        leads to errors and low performance. Instead, focus on leveraging the LLM's natural reasoning abilities:

        1. SIMPLIFY YOUR APPROACH:
           - Minimize the number of processing steps - simpler is better
           - Directly use LLM for pattern recognition rather than writing complex code
           - Avoid trying to parse or manipulate JSON manually - pass it as text to the LLM

        2. DIRECT TRANSFORMATION:
           - Instead of trying to extract features and then apply them, use the LLM to do the transformation directly
           - Use examples to teach the LLM the pattern, then have it apply that pattern to new inputs
           - Avoid attempting to write complex algorithmic solutions when pattern recognition will work better

        3. ROBUST ERROR HANDLING:
           - Include multiple approaches in case one fails (direct approach + fallback approach)
           - Use simple validation to check if outputs are in the expected format
           - Include a last-resort approach that will always return something valid

        4. AVOID COMMON PITFALLS:
           - Do NOT attempt to use json.loads() or complex JSON parsing - it often fails
           - Do NOT create overly complex Python pipelines that require perfect indentation
           - Do NOT create functions that generate or execute dynamic code
           - Do NOT create unnecessarily complex data transformations

        5. SUCCESSFUL EXAMPLES:
           - The most successful approaches have used direct pattern matching with multiple examples
           - Scripts with simple validation and fallback approaches perform better
           - Scripts with fewer processing steps have higher success rates
        
        IMPLEMENTATION STRATEGIES:
        1. Maintain a "example bank" of successful and failed examples to select from
        2. Implement n-shot prompting with n=3 as default, but adapt based on performance
        3. For complex tasks, use up to 5 examples; for simpler tasks, 2-3 may be sufficient
        4. Include examples with a range of complexity levels, rather than all similar examples



        VALIDATION AND VERIFICATION GUIDANCE:
        1. CRITICAL: Consider implementing validation loops for EACH key processing step, not just final outputs
        2. Design your system to detect, diagnose, and recover from specific errors. This will help future learnings
        3. For every LLM extraction or generation, add a verification step that checks:
           - Whether the output is well-formed and complete
           - Whether the output is logically consistent with the input
           - Whether all constraints are satisfied
        4. Add feedback loops that retry failures with specific feedback
        5. Include diagnostic outputs that reveal exactly where failures occur. Add print statements and intermediate outputs such that you can see them later to determine why things are going wrong.
        6. Include capability to trace through execution steps to identify failure points

        Example of pipeline without verification:
        ```python
        def process_question(question):
            entities = extract_entities(question)
            constraints = identify_constraints(question)
            solution = generate_solution(entities, constraints)
            return solution
        ```

        Example of robust pipeline with verification:
        ```python
        def process_question(question, max_attempts=3):
            # Step 1: Extract entities with verification
            entities_result = extract_entities_with_verification(question)
            if not entities_result.get("is_valid"):
                print(f"Entity extraction failed: {entities_result.get('validation_feedback')}")
                return f"Error in entity extraction: {entities_result.get('validation_feedback')}"

            # Step 2: Identify constraints with verification
            constraints_result = identify_constraints_with_verification(question, entities_result["entities"])
            if not constraints_result.get("is_valid"):
                print(f"Constraint identification failed: {constraints_result.get('validation_feedback')}")
                return f"Error in constraint identification: {constraints_result.get('validation_feedback')}"

            # Step 3: Generate solution with verification
            solution_result = generate_solution_with_verification(
                question, 
                entities_result["entities"], 
                constraints_result["constraints"]
            )
            if not solution_result.get("is_valid"):
                print(f"Solution generation failed: {solution_result.get('validation_feedback')}")
                return f"Error in solution generation: {solution_result.get('validation_feedback')}"

            return solution_result["solution"]

        def extract_entities_with_verification(question, max_attempts=3):
            #Extract entities and verify their validity with feedback loop.
            system_instruction = "You are an expert at extracting and validating entities."

            for attempt in range(max_attempts):
                # First attempt at extraction
                extraction_prompt = f'''
                Extract key entities from this question. 
                Return a JSON object with the extracted entities.

                Example 1: [example with entities]
                Example 2: [example with different entities]
                Example 3: [example with complex entities]

                Question: {question}
                Extraction:
                '''

                extracted_data = call_llm(extraction_prompt, system_instruction)

                try:
                    # Parse the extraction
                    data = json.loads(extracted_data)

                    # Verification step
                    verification_prompt = f'''
                    Verify if these extracted entities are complete and correct:

                    Question: {question}
                    Extracted entities: {json.dumps(data, indent=2)}

                    Check if:
                    1. All relevant entities are extracted
                    2. No irrelevant entities are included
                    3. All entity values are correct

                    Return a JSON with:
                    {{
                      "is_valid": true/false,
                      "validation_feedback": "detailed explanation",
                      "missing_entities": ["entity1", "entity2"],
                      "incorrect_entities": ["entity3"]
                    }}
                    '''

                    verification_result = call_llm(verification_prompt, system_instruction)
                    verification_data = json.loads(verification_result)

                    if verification_data.get("is_valid", False):
                        data["is_valid"] = True
                        data["validation_feedback"] = "All entities are valid."
                        return data

                    # If not valid and we have attempts left, refine with feedback
                    if attempt < max_attempts - 1:
                        feedback = verification_data.get("validation_feedback", "")
                        print(f"Validation failed (attempt {attempt+1}/{max_attempts}): {feedback}")
                        continue

                    # If we're out of attempts, return the best we have with validation info
                    data["is_valid"] = False
                    data["validation_feedback"] = verification_data.get("validation_feedback", "Unknown validation error")
                    return data

                except Exception as e:
                    print(f"Error in extraction/validation (attempt {attempt+1}/{max_attempts}): {str(e)}")
                    if attempt >= max_attempts - 1:
                        return {
                            "is_valid": False,
                            "validation_feedback": f"Error during processing: {str(e)}"
                        }

            return {
                "is_valid": False,
                "validation_feedback": "Failed to extract valid entities after multiple attempts."
            }
        ```

        VALIDATION IMPLEMENTATION STRATEGIES:
        1. Create detailed verification functions for each major processing step
        2. Implement max_attempts limits on all retry loops (typically 3-5 attempts)
        3. Pass specific feedback from verification to subsequent retry attempts
        4. Log all verification failures to help identify systemic issues
        5. Design fallback behaviors when verification repeatedly fails

        """

        # Add the accumulated learnings to the context
        learning_context = ""
        if accumulated_learnings:
            learning_context = f"""
        ACCUMULATED LEARNINGS FROM PREVIOUS ITERATIONS:
        {accumulated_learnings}
        """

        # Add capability-specific guidance if available
        capability_context = ""
        if capability_guidance:
            capability_context = f"""
        CAPABILITY ASSESSMENT & IMPROVEMENT GUIDANCE:
        {capability_guidance}
        """

        # Set specific system instruction for script generation
        script_generator_system_instruction = f"{self.system_prompt}\n\nYou are now acting as a Script Generator for an {approach_type} task. Your goal is to create a Python script that uses LLM-driven agentic approaches with chain-of-thought reasoning, agentic LLM patterns, and python to solve the problem examples provided."

        # Create appropriate prompt based on strategy
        if is_exploration:
            # Exploration prompt - now including last_scripts_context
            prompt = f"""
            You are developing a Python script to solve problems using LLM reasoning capabilities.
            You are in the EXPLORATION PHASE. You must generate a NEW approach that's different from previous approaches but informed by their successes and failures. With this approach, you will have a specific NEW HYPOTHESIS or variable you are trying to test. Your goal is to see if this new approach works, and you must add verification and validation steps to deduce if this new change is helpful. You may also test RADICAL NEW APPROACHES that are substantially different from previous approaches. 
            
            You should try NEW THINGS:
            
            Break down the problem into smaller pieces
            Think CREATIVELY about how to solve your problem if other approaches aren't working
            Transform data into different formats to see if it helps

            # YOUR TASK
            You are deeply familiar with prompting techniques and the agent works from the literature. 
            Your goal is to maximize the specified performance metrics by proposing interestingly new agents.
            Observe the past discovered agents and scripts carefully and think about what insights, lessons, or stepping stones can be learned from them.
            Be creative when thinking about the next interesting agent to try. You are encouraged to draw inspiration from related agent papers or academic papers from other research areas.
            Use the knowledge from the archive and inspiration from academic literature to propose the next interesting agentic system design.
            THINK OUTSIDE THE BOX.
            

            Here are example problems from previously seen data:
            {json.dumps(example_problems, indent=2)}

            HISTORICAL CONTEXT:
            {historical_context}

            PREVIOUSLY TRIED APPROACHES (LAST 5 SCRIPTS). YOUR APPROACH MUST BE SUBSTANTIVELY DIFFERENT THAN THESE:
            {last_scripts_context}

            LEARNINGS FROM PREVIOUS ITERATIONS:
            {learning_context}

            CAPABILITY ASSESSMENT & IMPROVEMENT GUIDANCE:
            {capability_context}

            EXPLORATION GUIDANCE:
            1. Review the historical approaches, error patterns, and accumulated learnings carefully
            2. Review the FULL CODE of previous scripts to understand what has already been tried
            3. Design a new approach that is DISTINCTLY DIFFERENT from previous attempts. This approach should have a specific NEW HYPOTHESIS or variable you are trying to test. 
            4. CRITICAL: Include EMBEDDED EXAMPLES directly within your LLM prompts
            5. For each key function, show a complete worked example, or include multiple examples, including:
               - Input example that resembles the dataset
               - Step-by-step reasoning through the example
               - Properly formatted output
            6. Apply the insights from the ACCUMULATED LEARNINGS section to avoid repeating past mistakes
            7. Pay SPECIAL ATTENTION to the weaknesses and improvement suggestions from the capability assessment
            8. Consider implementing one or more of these LLM usage patterns:
               - Repeated validation with feedback loops
               - Multi-perspective analysis with synthesis
               - Dynamic input-dependent routing with an orchestrator
               - Hybrid approaches combining LLM with deterministic functions
               - Best-of-n solution generation and selection
               - ReAct pattern for interactive reasoning and action
               - If it is unknown how successful a processing state or part of the pipeline is, include verification steps to different parts of the pipeline in order to help deduce which parts are successful and where the system is breaking
               - Answer checkers to validate the final answer against the problem statement. If the answer is incorrect, the checker can send the answer back to an earlier part of the system for for refinement with feedback

            Here's how to call the Gemini API. Use this example without modification and don't invent configuration options:
            {gemini_api_example}

            Since this is an EXPLORATION phase:
            - Try a fundamentally different approach to reasoning about the problem. Test a NEW HYPOTHESIS or variable, and add verification steps to deduce if this new change is helpful.
            - THIS IS KEY: Break down the problem into new, distinct reasoning steps based on past performance before you start coding
            - For EACH key LLM prompt, include a relevant example with:
              * Sample input similar to the dataset
              * Expected reasoning steps
              * Desired output format
            - Apply a verifier call to different parts of the pipeline in order to understand what parts of the pipeline of calls is successful and where the system is breaking
            - Pay special attention to addressing the primary issues from previous iterations
            - Ensure your new approach addresses the weaknesses identified in the capability assessment

            CRITICAL REQUIREMENTS:
            1. The script MUST properly handle all string literals - be extremely careful with quotes and triple quotes
            2. The script MUST NOT exceed 150 lines of code to prevent truncation
            3. Include detailed comments explaining your reasoning approach
            4. EVERY SINGLE LLM PROMPT must include at least one embedded example showing:
               - Sample input with reasoning
               - Desired output format
            5. Make proper use of error handling
            6. Implement robust capabilities to address the specific weaknesses identified in the capability assessment
            7. Do NOT use json.loads() in the LLM calls to process input data. JSON formatting is good to use to structure information as inputs and outputs, but attempting to have functions process JSON data explicitly with strict built-in functionality is error prone due to formatting issues and additional text that appears as documentation, reasoning, or comments. When passing data into another LLM call, you can read it as plain text rather than trying to load it in strict json format, is the better approach.

            Return a COMPLETE, RUNNABLE Python script that:
            1. Has a main function that takes a question string as input and returns the answer string
            2. Makes multiple LLM calls for different reasoning steps
            3. Has proper error handling for API calls
            4. Includes embedded examples in EVERY LLM prompt
            5. Is COMPLETE - no missing code, no "..." placeholders
            6. Closes all string literals properly

            This should be FUNDAMENTALLY DIFFERENT from all previous approaches. Do not reuse the same overall structure.

            BE EXTREMELY CAREFUL TO PROPERLY CLOSE ALL STRING QUOTES AND TRIPLE QUOTES!
            """
        else:
            # Exploitation prompt
            best_script_code = ""
            if best_script_to_exploit and 'script' in best_script_to_exploit:
                best_script_code = f"\nFULL SCRIPT TO REFINE:\n```python\n{best_script_to_exploit.get('script', '')}\n```"

            # FIX: Properly handle the accuracy formatting
            accuracy_str = ""
            if best_script_to_exploit:
                accuracy_value = best_script_to_exploit.get('accuracy', 0)
                accuracy_str = f"{accuracy_value:.2f}"
            else:
                accuracy_str = "N/A"

            # Generate content for multiple top scripts
            top_scripts_content = ""
            if top_scripts_to_exploit:
                for i, script_info in enumerate(top_scripts_to_exploit):
                    accuracy_value = script_info.get('accuracy', 0)
                    accuracy_str = f"{accuracy_value:.2f}"

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

            prompt = f"""
            You are improving a Python script that solves problems from a dataset.
            Your goal is to REFINE and ENHANCE the best performing approaches by combining their strengths and addressing specific weaknesses identified in error analysis.

            Here are example problems from previously seen data:
            {json.dumps(example_problems, indent=2)}

            {historical_context}

            {learning_context}

            {capability_context}

            TOP PERFORMING APPROACHES TO BUILD UPON:
            {top_scripts_content}
            {best_script_code}

            PREVIOUSLY ATTEMPTED VARIATIONS:
            {last_scripts_context}

            EXPLOITATION GUIDANCE:
            1. Review the error patterns, targeted improvements, and accumulated learnings carefully
            2. CRITICAL: Break down the problem into distinct reasoning steps before modifying code
            3. CRITICAL: Analyze the best scripts to identify which components are working well and which are failing. Focus your improvements on the weak points while preserving successful components.
            4. Maintain the core successful elements of the best approaches
            5. Consider how you can combine strengths from multiple top-performing approaches
            6. CRITICAL: Add EMBEDDED EXAMPLES to EVERY LLM prompt that illustrate:
               - Sample input that resembles the dataset
               - Step-by-step reasoning through the example
               - Properly formatted output
            7. Focus on fixing specific issues identified in previous error analyses. Create an explicit HYPOTHESIS for each targeted improvement, as well as a way to verify if it's successful.
            8. Enhance chain-of-thought reasoning and verification steps. Verification steps should be added to different parts of the pipeline in order to help deduce which parts are successful and where the system is breaking
            9. Apply the key insights from ACCUMULATED LEARNINGS to enhance the approach
            10. Pay SPECIAL ATTENTION to the weaknesses and improvement suggestions from the capability assessment

            IMPROVEMENT STRATEGY:
            Analyze why the top approaches succeeded where others failed. Identify the key differentiators and strengthen them further.

            SYSTEMATIC ENHANCEMENT APPROACH:
            1. First, identify which specific function or component is underperforming based on error analysis
            2. Examine how error cases differ from successful cases
            3. For each identified weakness, implement a targeted enhancement
            4. Add additional verification steps around modified components
            5. Consider how components interact - ensure improvements don't break successful parts

            Consider enhancing the script with one or more of these patterns:
            - Repeated validation with feedback loops
            - Multi-perspective analysis with synthesis
            - Dynamic input-dependent routing
            - Hybrid approaches combining LLM with deterministic functions
            - Best-of-n solution generation and selection
            - ReAct pattern for interactive reasoning and action
            - If it is unknown how successful a processing state or part of the pipeline is, include verification steps to different parts of the pipeline in order to help deduce which parts are successful and where the system is breaking
            - Answer checkers to validate the final answer against the problem statement. If the answer is incorrect, the checker can send the answer back to an earlier part of the system for refinement with feedback

            Here's how to call the Gemini API. Use this example without modification and don't invent configuration options:
            {gemini_api_example}

            Since this is an EXPLOITATION phase:
            - Build upon what's working well in the best approaches
            - Consider creative combinations of successful techniques from different scripts
            - Make TARGETED improvements to address specific error patterns
            - For EACH key LLM prompt, include a relevant example with:
              * Sample input similar to the dataset
              * Expected reasoning steps
              * Desired output format
            - Apply the knowledge from our accumulated learnings
            - Significantly enhance the script to address weaknesses identified in the capability assessment

            CRITICAL REQUIREMENTS:
            1. The script MUST properly handle all string literals - be extremely careful with quotes and triple quotes
            2. The script MUST NOT exceed 150 lines of code to prevent truncation
            3. Include detailed comments explaining your improvements
            4. EVERY SINGLE LLM PROMPT must include at least one embedded example showing:
               - Sample input with reasoning
               - Desired output format
            5. Make proper use of error handling
            6. Implement robust capabilities to address the specific weaknesses identified in the capability assessment
            7. Do NOT use json.loads() in the LLM calls to process input data. JSON formatting is good to use to structure information as inputs and outputs, but attempting to have functions process JSON data explicitly with strict built-in functionality is error prone due to formatting issues and additional text that appears as documentation, reasoning, or comments. When passing data into another LLM call, you can read it as plain text rather than trying to load it in strict json format, is the better approach.

            Return a COMPLETE, RUNNABLE Python script that:
            1. Has a main function that takes a question string as input and returns the answer string
            2. Makes multiple LLM calls for different reasoning steps
            3. Has proper error handling for API calls
            4. Includes embedded examples in EVERY LLM prompt
            5. Is COMPLETE - no missing code, no "..." placeholders
            6. Closes all string literals properly

            BE EXTREMELY CAREFUL TO PROPERLY CLOSE ALL STRING QUOTES AND TRIPLE QUOTES!
            """

        # ==== GENERATE SCRIPT WITH VALIDATION ====
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
                        "\n***\nFALLBACK: Maximum attempts reached. Returning a simple fallback script.\n***\n"
                    )
                    # Create a simple fallback script with embedded examples
                    fallback_script = """
    import os
    import json
    from google import genai
    from google.genai import types

    def call_llm(prompt, system_instruction=None):
        \"\"\"Call the Gemini LLM with a prompt and return the response\"\"\"
        try:
            # Initialize the Gemini client
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

            # Call the API with system instruction if provided
            if system_instruction:
                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                    contents=prompt
                )
            else:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )

            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return f"Error: {str(e)}"

    def extract_information(text):
        \"\"\"Extract key information from the input text\"\"\"
        system_instruction = "You are an information extraction specialist."

        prompt = f\"\"\"
        Extract key information from this text. Focus on identifying important elements and relationships.

        Example:
        Input: The project must be completed by June 15th and requires collaboration between the engineering and design teams.
        Output: {"deadline": "June 15th", "teams_involved": ["engineering", "design"], "requirement": "collaboration"}

        Now extract information from this input:
        {text}
        \"\"\"

        return call_llm(prompt, system_instruction)

    def generate_solution(problem):
        \"\"\"Generate a solution to the problem\"\"\"
        system_instruction = "You are a problem-solving expert."

        prompt = f\"\"\"
        Generate a detailed solution for this problem:

        Example:
        Problem: Design a simple notification system that sends alerts when a temperature sensor exceeds 30°C.
        Solution: Create a monitoring service that polls the temperature sensor every minute. When a reading exceeds 30°C, trigger the notification system to send an alert via email and SMS to registered users, including the current temperature value and timestamp.

        Now solve this problem:
        {problem}
        \"\"\"

        return call_llm(prompt, system_instruction)

    def main(question):
        \"\"\"Main function to solve problems\"\"\"
        try:
            # Step 1: Extract key information
            information = extract_information(question)

            # Step 2: Generate a solution
            solution = generate_solution(question)

            # Return the solution
            return solution
        except Exception as e:
            print(f"Error in main: {str(e)}")
            return "I couldn't generate a solution due to an error."
    """
                    script_path = self.scripts_dir / f"script_iteration_{self.current_iteration}.py"
                    with open(script_path, 'w', encoding='utf-8') as f:
                        f.write(fallback_script)

                    return fallback_script

                # Try again with a more explicit instruction about the error
                prompt = f"""
                You need to generate a complete, syntactically valid Python script. Your previous attempt had the following syntax error:
                {str(e)}

                Please generate a new script paying special attention to:
                1. Properly closing all string literals (quotes and triple quotes)
                2. Properly closing all parentheses and braces
                3. Keeping the script simple and short (under 150 lines)
                4. Using only syntactically valid Python code
                5. INCLUDING EMBEDDED EXAMPLES in all LLM prompts

                Generate a complete, runnable Python script that:
                1. Has a main function that takes a question string as input and returns the answer string
                2. Makes multiple LLM calls for different reasoning steps using the Gemini API
                3. Has proper error handling
                4. Includes a concrete example in EACH LLM prompt

                BE EXTREMELY CAREFUL WITH STRING LITERALS AND QUOTES!
                """

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
    

    def evaluate_answer_with_llm(self, system_answer: str, golden_answer: str) -> Dict:
        """Use LLM to determine if answers are semantically equivalent"""

        # Role-specific system instruction for the evaluator
        evaluator_system_instruction = "You are now acting as an Answer Evaluator. Your task is to determine if two answers convey the same meaning, even if they are worded or formatted differently."

        prompt = f"""
        You're evaluating two answers to determine if they convey the same information.

        System answer: {system_answer}
        Golden answer: {golden_answer}

        Do these answers communicate the same information, even if worded or formatted differently? The "Golden answer" is the reference answer. The "System answer" produced by our system may contain reasoning traces, but your job is to verify if  in the system answer there is an answer that is semantically equivalent to the golden answer.
        If this is a detailed numerical answer, where clearly precision is required, check very close, element by element, to ensure that the content is correct.
        Return only a JSON object with: {{"match": true/false, "confidence": 0-1, "explanation": "reason"}}
        """
        print ("SYSTEM ANSWER: ....", system_answer)
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

        # Save to archive
        try:
            self.save_to_archive(iteration_data, f"iteration_{self.current_iteration}.json")
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
