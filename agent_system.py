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
    """

    def __init__(self,
                 dataset_path: str = "calendar_scheduling.json",
                 example_prefix: str = "calendar_scheduling_example_"):
        """Initialize the agent system"""
        # Initialize configuration
        self.explore_rate = 70
        self.exploit_rate = 30
        self.dataset_path = dataset_path
        self.example_prefix = example_prefix

        # Initialize batch size and tracking for seen examples
        self.current_batch_size = 5  # Start with a small batch
        self.seen_examples = set()
        self.next_example_index = 0

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
            print(
                "No existing learnings found. Will start accumulating learnings."
            )

        # Load previous iterations if available
        self._load_previous_state()

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

        # Determine the next iteration number
        if summaries:
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

            # Reconstruct set of seen examples
            for iteration in iterations:
                if iteration and "sample_count" in iteration:
                    # Each iteration represents sample_count examples starting from some index
                    iter_num = iteration.get("iteration", 0)
                    sample_count = iteration.get("sample_count", 0)

                    # Calculate the approximate range of examples this iteration would have seen
                    for i in range(iter_num * sample_count,
                                   (iter_num + 1) * sample_count):
                        self.seen_examples.add(f"{self.example_prefix}{i}")

            # Set next example index to after the last seen example
            # This is approximate but better than starting from 0 again
            last_seen_index = max([
                int(ex.replace(self.example_prefix, ""))
                for ex in self.seen_examples
                if ex.startswith(self.example_prefix)
            ] or [0])
            self.next_example_index = last_seen_index + 1

            print(
                f"Loaded {len(self.seen_examples)} seen examples, next example index: {self.next_example_index}"
            )
        else:
            self.current_iteration = 0

    def call_llm(self, prompt: str, system_instruction: str = None) -> str:
        """Call the Gemini LLM with a prompt and return the response"""
        try:
            # Use provided system instruction or default to the loaded system prompt
            sys_instruction = system_instruction if system_instruction is not None else self.system_prompt

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
        """Get samples from the dataset, rotating through examples sequentially"""
        dataset = self.load_dataset()
        if not dataset:
            return {
                "samples": [],
                "new_examples_added": 0,
                "total_seen_examples": 0
            }

        samples = []
        new_examples_added = 0

        # Get current_batch_size samples
        for i in range(self.current_batch_size):
            example_key = f"{self.example_prefix}{self.next_example_index}"

            # Wrap around if we reach the end
            if example_key not in dataset:
                self.next_example_index = 0
                example_key = f"{self.example_prefix}{self.next_example_index}"

            if example_key in dataset:
                samples.append(dataset[example_key])

                # Track that we've seen this example
                if example_key not in self.seen_examples:
                    self.seen_examples.add(example_key)
                    new_examples_added += 1

                self.next_example_index += 1

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
            return []

        with open(summary_file, 'r', encoding='utf-8') as file:
            return json.load(file)

    def update_summaries(self, new_summary: Dict) -> None:
        """Add a new summary to the summaries file"""

        # Add capability data to summary
        if hasattr(self, 'capability_tracker'):
            new_summary[
                "capability_report"] = self.capability_tracker.generate_report(
                )

        summaries = self.get_summaries()
        summaries.append(new_summary)

        with open(self.archive_dir / "summaries.json", 'w',
                  encoding='utf-8') as file:
            json.dump(summaries, file, indent=2)

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
        batch_optimizer_system_instruction = f"{self.system_prompt}\n\nYou are a Batch Size Optimizer. Your task is to analyze performance trends and recommend the optimal batch size for testing, balancing between stability and throughput."

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
        - Batch size should be between 5 and 25
        - Increase batch size when performance is stable and good
        - Decrease batch size when performance suddenly drops
        - Keep batch size stable when exploring new approaches

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
            new_batch_size = max(5, min(25, new_batch_size))

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
        learnings_generator_system_instruction = f"{self.system_prompt}\n\nYou are a Knowledge Synthesizer. Your role is to extract concrete, dataset-specific insights from experiment results, focusing on patterns in the data, effective strategies for this specific task, and precise failure modes."

        # Get full original samples from iteration_data
        samples = []
        if "samples" in iteration_data:
            samples = iteration_data.get("samples", [])

        # Get example questions - prefer direct samples if available
        sample_questions = []
        for i in range(min(3, len(samples))):
            if i < len(samples):
                sample_questions.append(samples[i].get("prompt_0shot", "N/A"))
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
                    sample_question = samples[i].get("prompt_0shot", "N/A")
                    golden_answer = samples[i].get("golden_plan", "N/A")

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
        """Synthesize existing learnings with new batch learnings, emphasizing dataset-specific insights"""
        learnings_synthesizer_system_instruction = f"{self.system_prompt}\n\nYou are a Knowledge Integrator. Your role is to synthesize accumulated dataset-specific knowledge with new insights, creating an evolving experiment log that captures concrete patterns, strategies, and findings about this specific task."

        # Print length info for debugging
        print(f"Current learnings length: {len(current_learnings)}")
        print(f"New batch learnings length: {len(new_batch_learnings)}")

        prompt = f"""
        You are tasked with synthesizing existing knowledge with new learnings from our latest experiment on this dataset.

        EXISTING ACCUMULATED LEARNINGS:
        {current_learnings}

        NEW LEARNINGS FROM LATEST BATCH:
        {new_batch_learnings}

        Create an updated, synthesized version of our learnings that:

        1. Maintains a comprehensive catalog of DATASET PATTERNS we've identified
        2. Tracks the evolution of our understanding about what makes this specific task challenging
        3. Documents concrete STRATEGIES that have proven effective or ineffective for this particular dataset
        4. Creates a running EXPERIMENT LOG tracking our attempts and findings specific to this task
        5. Prioritizes concrete, task-specific insights over general system design principles

        The synthesized learnings should read like a detailed research log about THIS specific dataset and task, 
        not a general guide to system design. Each section should include specific examples and concrete findings.

        Organize the information into these sections:

        1. DATASET PATTERNS & CHARACTERISTICS
        2. EFFECTIVE TASK-SPECIFIC STRATEGIES
        3. COMMON FAILURE MODES ON THIS DATASET
        4. EXPERIMENT LOG & FINDINGS
        5. NEXT RESEARCH DIRECTIONS

        Your output will replace the current learnings file and serve as long-term memory for working specifically with this dataset.

        IMPORTANT: Preserve all concrete, specific insights from both the existing learnings and new batch. Don't lose valuable information.
        """

        try:
            print("Calling LLM to synthesize learnings...")
            response = self.call_llm(
                prompt,
                system_instruction=learnings_synthesizer_system_instruction)
            print(f"Received synthesized learnings: {len(response)} characters")
            return response.strip()
        except Exception as e:
            error_message = f"Error synthesizing learnings: {str(e)}"
            print(error_message)
            traceback.print_exc()  # Print full traceback for better debugging

            # Return a concatenated version as fallback
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
        strategy_optimizer_system_instruction = f"{self.system_prompt}\n\nYou are a Strategy Optimizer. Your role is to analyze performance patterns and determine the optimal balance between exploration (trying new approaches) and exploitation (refining successful approaches)."

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
        Includes embedded example prompts to improve performance.
        """
        # Get previous iterations and samples
        iterations = self.get_all_iterations()
        summaries = self.get_summaries()
        samples_data = self.get_samples()
        samples = samples_data["samples"]

        # Extract example problems for context (limit to 2-3 for space)
        example_problems = []
        for i, sample in enumerate(samples[:3]):
            example_problems.append({
                "id": i,
                "question": sample.get("prompt_0shot", ""),
                "answer": sample.get("golden_plan", "")
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
                print(f"Focusing script improvement on: {improvement_focus.upper().replace('_', ' ') if improvement_focus else 'No specific focus'}")

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
        gemini_api_example = 'def call_llm(prompt, system_instruction=None):\n    """Call the Gemini LLM with a prompt and return the response"""\n    try:\n        from google import genai\n        from google.genai import types\n\n        # Initialize the Gemini client\n        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))\n\n        # Call the API with system instruction if provided\n        if system_instruction:\n            response = client.models.generate_content(\n                model="gemini-2.0-flash", \n                config=types.GenerateContentConfig(\n                    system_instruction=system_instruction\n                ),\n                contents=prompt\n            )\n        else:\n            response = client.models.generate_content(\n                model="gemini-2.0-flash",\n                contents=prompt\n            )\n\n        return response.text\n    except Exception as e:\n        print(f"Error calling Gemini API: {str(e)}")\n        return f"Error: {str(e)}"'

        # Example 1: Information extraction with embedded example
        extraction_example = 'def extract_information_with_examples(problem):\n    """Extract key information from the problem statement using embedded examples."""\n    system_instruction = "You are an information extraction specialist focusing on identifying key entities and constraints."\n    \n    prompt = f"""\n    Extract key information from this problem statement. Focus on identifying all entities, relationships, and constraints.\n    \n    Example usage:\n    \n    Question:\n    You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. \n    Here are the existing schedules for everyone during the days: \n    John has no meetings the whole week.\n    Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 11:30, 12:00 to 17:00, Wednesday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00.\n    John would like to avoid more meetings on Monday after 14:30. Find a time that works for everyone\'s schedule and constraints.\n    \n    Let\'s think step by step.\n    \n    The key entities are:\n    - John (participant)\n    - Jennifer (participant)\n    \n    The key constraints are:\n    - Meeting duration: 30 minutes (half an hour)\n    - Valid meeting hours: 9:00-17:00\n    - Valid days: Monday, Tuesday, or Wednesday\n    - John\'s availability: All week (no meetings)\n    - Jennifer\'s availability:\n      * Monday: Busy 9:00-11:00, 11:30-13:00, 13:30-14:30, 15:00-17:00\n      * Tuesday: Busy 9:00-11:30, 12:00-17:00\n      * Wednesday: Busy 9:00-11:30, 12:00-12:30, 13:00-14:00, 14:30-16:00, 16:30-17:00\n    - Preferences: John prefers to avoid meetings on Monday after 14:30\n    \n    Extracted Information:\n    {{\n      "participants": ["John", "Jennifer"],\n      "duration": "30 minutes",\n      "valid_hours": "9:00-17:00",\n      "valid_days": ["Monday", "Tuesday", "Wednesday"],\n      "availability": {{\n        "John": "All times",\n        "Jennifer": {{\n          "Monday": ["11:00-11:30", "13:00-13:30", "14:30-15:00"],\n          "Tuesday": ["11:30-12:00"],\n          "Wednesday": ["11:30-12:00", "12:30-13:00", "14:00-14:30", "16:00-16:30"]\n        }}\n      }},\n      "preferences": {{\n        "John": "Avoid Monday after 14:30"\n      }}\n    }}\n    \n    Now, extract information from this new problem:\n    {problem}\n    """\n    \n    return call_llm(prompt, system_instruction)'

        # Example 2: Verification with embedded example
        verification_example = 'def verify_solution_with_examples(problem, proposed_solution):\n    """Verify if the proposed solution satisfies all constraints using embedded examples."""\n    system_instruction = "You are a critical evaluator who verifies if solutions satisfy all constraints."\n    \n    prompt = f"""\n    Verify if this proposed solution satisfies all constraints in the problem.\n    \n    Example usage:\n    \n    Problem:\n    You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. \n    Here are the existing schedules for everyone during the days: \n    John has no meetings the whole week.\n    Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 11:30, 12:00 to 17:00, Wednesday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00.\n    John would like to avoid more meetings on Monday after 14:30.\n    \n    Proposed Solution:\n    Schedule the meeting on Wednesday from 13:00 to 13:30.\n    \n    Verification:\n    Let me check each constraint:\n    1. Duration: The meeting is scheduled for 30 minutes (13:00-13:30) ✓\n    2. Work hours: Meeting time 13:00-13:30 is within 9:00-17:00 ✓\n    3. Valid day: Wednesday is one of the allowed days ✓\n    4. John\'s availability: John has no meetings all week ✓\n    5. Jennifer\'s availability on Wednesday:\n       - Jennifer is busy 9:00-11:30, 12:00-12:30, 13:00-14:00, 14:30-16:00, 16:30-17:00\n       - The proposed time 13:00-13:30 overlaps with Jennifer\'s busy time 13:00-14:00 ✗\n    6. John\'s preference: Not applicable (not Monday after 14:30) ✓\n    \n    Result: INVALID - The solution conflicts with Jennifer\'s schedule on Wednesday from 13:00-14:00.\n    \n    Problem:\n    {problem}\n    \n    Proposed Solution:\n    {proposed_solution}\n    \n    Verification:\n    """\n    \n    return call_llm(prompt, system_instruction)'

        # Create few-shot examples context
        few_shot_examples = f"EXAMPLE OF EFFECTIVE FEW-SHOT PROMPTING:\n\n```python\n{extraction_example}\n```\n\n```python\n{verification_example}\n```"

        # Historical context summary
        best_accuracy_str = f"{best_scripts[0].get('accuracy', 0):.2f} (iteration {best_scripts[0].get('iteration')})" if best_scripts else "None"

        historical_context = f"""
        ITERATION HISTORY SUMMARY:
        - Total iterations completed: {len(summaries)}
        - Current explore/exploit balance: {self.explore_rate}/{self.exploit_rate}
        - Best accuracy achieved: {best_accuracy_str}

    APPROACH HISTORY (last {min(5, len(approach_history))} iterations):
    {json.dumps(approach_history[-5:] if len(approach_history) > 5 else approach_history, indent=2)}

    COMMON ERROR PATTERNS:
    {json.dumps(list(set(error_patterns[-10:] if len(error_patterns) > 10 else error_patterns)), indent=2)}

    PRIMARY ISSUES (last {min(3, len(primary_issues))} iterations):
    {json.dumps(primary_issues[-3:] if len(primary_issues) > 3 else primary_issues, indent=2)}

    TARGETED IMPROVEMENTS:
    {json.dumps(list(set(targeted_improvements[-10:] if len(targeted_improvements) > 10 else targeted_improvements)), indent=2)}
        """

        # Add the few-shot examples to the context
        historical_context += f"\n\n{few_shot_examples}"

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
        script_generator_system_instruction = f"{self.system_prompt}\n\nYou are now acting as a Script Generator for an {approach_type} task. Your goal is to create a Python script that uses LLM-driven agentic approaches with chain-of-thought reasoning to solve the problem examples provided."

        # Create appropriate prompt based on strategy
        if is_exploration:
            # Exploration prompt
            prompt = f"""
            You are developing a Python script to solve dataset problems using LLM reasoning capabilities.
            You must generate a NEW approach that's different from previous approaches but informed by their successes and failures.

            Here are example problems from the dataset:
            {json.dumps(example_problems, indent=2)}

            {historical_context}

            {learning_context}

            {capability_context}

            EXPLORATION GUIDANCE:
            1. Review the historical approaches, error patterns, and accumulated learnings carefully
            2. Design a new approach that specifically addresses common error patterns
            3. Take inspiration from successful aspects of previous approaches but create something distinct
            4. CRITICAL: Include EMBEDDED EXAMPLES directly within your LLM prompts
            5. For each key function, show a complete worked example including:
               - Input example that resembles the dataset
               - Step-by-step reasoning through the example
               - Properly formatted output
            6. Apply the insights from the ACCUMULATED LEARNINGS section to avoid repeating past mistakes
            7. Pay SPECIAL ATTENTION to the weaknesses and improvement suggestions from the capability assessment

            Here's how to call the Gemini API:
            {gemini_api_example}

            Since this is an EXPLORATION phase:
            - Try a fundamentally different approach to reasoning about the problem
            - Implement chain-of-thought reasoning in a new way
            - For EACH key LLM prompt, include a relevant example with:
              * Sample input similar to the dataset
              * Expected reasoning steps
              * Desired output format
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

            Return a COMPLETE, RUNNABLE Python script that:
            1. Has a main function that takes a question string as input and returns the answer string
            2. Makes multiple LLM calls for different reasoning steps
            3. Has proper error handling for API calls
            4. Includes embedded examples in EVERY LLM prompt
            5. Is COMPLETE - no missing code, no "..." placeholders
            6. Closes all string literals properly

            BE EXTREMELY CAREFUL TO PROPERLY CLOSE ALL STRING QUOTES AND TRIPLE QUOTES!
            """
        else:
            # Exploitation prompt
            best_script_code = ""
            if best_script_to_exploit and 'script' in best_script_to_exploit:
                best_script_code = f"\nFULL SCRIPT TO REFINE:\n```python\n{best_script_to_exploit.get('script', '')}\n```"

            prompt = f"""
            You are improving a Python script that solves problems from a dataset.
            Your goal is to REFINE and ENHANCE the current best approach based on detailed error analysis and accumulated learnings.

            Here are example problems from the dataset:
            {json.dumps(example_problems, indent=2)}

            {historical_context}

            {learning_context}

            {capability_context}

            BEST PERFORMING APPROACH TO REFINE:
            Iteration: {best_script_to_exploit.get('iteration') if best_script_to_exploit else 'None'}
            Accuracy: {best_script_to_exploit.get('accuracy', 0):.2f if best_script_to_exploit else 'N/A'}
            Approach Summary: {best_script_to_exploit.get('approach_summary') if best_script_to_exploit else 'No approach to refine'}
            {best_script_code}

            EXPLOITATION GUIDANCE:
            1. Review the error patterns, targeted improvements, and accumulated learnings carefully
            2. Maintain the core successful elements of the best approach
            3. CRITICAL: Add EMBEDDED EXAMPLES to EVERY LLM prompt that illustrate:
               - Sample input that resembles the dataset
               - Step-by-step reasoning through the example
               - Properly formatted output
            4. Focus on fixing specific issues identified in previous error analyses
            5. Enhance chain-of-thought reasoning and verification steps
            6. Apply the key insights from ACCUMULATED LEARNINGS to enhance the approach
            7. Pay SPECIAL ATTENTION to the weaknesses and improvement suggestions from the capability assessment

            Here's how to call the Gemini API:
            {gemini_api_example}

            Since this is an EXPLOITATION phase:
            - Build upon what's working well in the best approach
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
                        "Maximum attempts reached. Returning a simple fallback script."
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

    def analyze_problem(question):
        \"\"\"Analyze the scheduling problem\"\"\"
        system_instruction = "You are a scheduling assistant."

        prompt = f\"\"\"
        Analyze this scheduling problem: 

        Example:
        Question: You need to schedule a meeting for Alice and Bob for 1 hour. Alice is free on Monday from 2-4pm and Tuesday all day. Bob is free on Monday and Tuesday mornings.
        Analysis: This problem involves scheduling a 1-hour meeting for Alice and Bob. Alice is available Monday 2-4pm and all day Tuesday. Bob is available Monday morning and Tuesday morning. The only overlapping time is Tuesday morning.

        Now analyze this problem:
        {question}
        \"\"\"

        return call_llm(prompt, system_instruction)

    def generate_solution(question):
        \"\"\"Generate a solution to the scheduling problem\"\"\"
        system_instruction = "You are a scheduling assistant."

        prompt = f\"\"\"
        Generate a scheduling solution for this problem:

        Example:
        Question: You need to schedule a meeting for Alice and Bob for 1 hour. Alice is free on Monday from 2-4pm and Tuesday all day. Bob is free on Monday and Tuesday mornings.
        Solution: Schedule the meeting for Tuesday at 10am. This works because Alice is available all day Tuesday, and Bob is available Tuesday morning.

        Now solve this problem:
        {question}
        \"\"\"

        return call_llm(prompt, system_instruction)

    def main(question):
        \"\"\"Main function to solve scheduling problems\"\"\"
        try:
            # Step 1: Analyze the problem
            analysis = analyze_problem(question)

            # Step 2: Generate a solution
            solution = generate_solution(question)

            # Return the solution
            return solution
        except Exception as e:
            print(f"Error in main: {str(e)}")
            return "I couldn't generate a scheduling plan due to an error."
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
    
    def execute_script(self, script: str, question: str) -> Dict:
        """
        Execute the generated script on a question and return the result.
        Uses automatic debugging if the script fails with specific errors.
        """
        # Create a temporary script file
        script_path = self.scripts_dir / f"current_script_{self.current_iteration}.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

        # Create a test harness for the script
        test_script = f"""
import sys
import traceback
import os

# Add the scripts directory to the path
sys.path.append("{self.scripts_dir}")

# Ensure the Gemini API key is available to the script
os.environ["GEMINI_API_KEY"] = "{os.environ.get('GEMINI_API_KEY')}"

try:
    # Import the script as a module
    from current_script_{self.current_iteration} import main

    # Execute the main function with the question
    question = {repr(question)}
    answer = main(question)

    # Print the answer for capture
    print("ANSWER_START")
    print(answer)
    print("ANSWER_END")

except Exception as e:
    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
"""

        test_path = self.scripts_dir / f"test_script_{self.current_iteration}.py"
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_script)

        # Execute the test script and capture output
        debug_attempts = 0
        max_debug_attempts = 3

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
                        "output": output
                    }
                elif "ERROR_START" in output and "ERROR_END" in output:
                    error = output.split("ERROR_START")[1].split(
                        "ERROR_END")[0].strip()

                    # If we've reached max debug attempts or this isn't a "missing main" error, return the error
                    if debug_attempts >= max_debug_attempts or "cannot import name 'main'" not in error:
                        return {
                            "success": False,
                            "error": error,
                            "output": output
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
                        "output": output
                    }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": "Script execution timed out (60 seconds)",
                    "output": "Timeout"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "output": traceback.format_exc()
                }

        # If we get here, we've exhausted our debug attempts
        return {
            "success": False,
            "error": "Maximum debug attempts reached. Could not fix script.",
            "output": "Debug failure"
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
                                or "schedule" in function_name or
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
        Generates natural language analysis of strengths, weaknesses, and improvements.
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
                    "match": False,
                    "capability_failures": ["execution"]  # Track execution failures
                })
                continue

            # Compare with golden answer using LLM
            if not result.get("evaluation"):
                golden_answer = sample.get("golden_plan", "").strip()
                system_answer = result.get("answer", "").strip()
                evaluation = self.evaluate_answer_with_llm(system_answer, golden_answer)
                result["evaluation"] = evaluation
                result["match"] = evaluation.get("match", False)

            if result.get("match", False):
                correct_count += 1
                # For successful cases, we consider all capabilities successful
                evaluations.append({
                    "sample_id": i,
                    "success": True,
                    "system_answer": result.get("answer", "").strip(),
                    "golden_answer": sample.get("golden_plan", "").strip(),
                    "match": True,
                    "evaluation": result.get("evaluation", {})
                })
            else:
                # For failed cases, we'll determine which capabilities failed later
                evaluations.append({
                    "sample_id": i,
                    "success": True,
                    "system_answer": result.get("answer", "").strip(),
                    "golden_answer": sample.get("golden_plan", "").strip(),
                    "match": False,
                    "evaluation": result.get("evaluation", {}),
                    "capability_failures": []  # Placeholder, will be populated after error analysis
                })

        # Calculate accuracy
        accuracy = correct_count / len(samples) if samples else 0

        # For deeper analysis, use LLM to analyze error patterns
        error_samples = []
        for i, eval_data in enumerate(evaluations):
            if not eval_data.get("match"):
                sample = samples[i]
                error_samples.append({
                    "sample_id": i,
                    "question": sample.get("prompt_0shot", ""),
                    "system_answer": eval_data.get("system_answer", ""),
                    "golden_answer": eval_data.get("golden_answer", ""),
                    "error_message": eval_data.get("error", ""),
                    "explanation": eval_data.get("evaluation", {}).get("explanation", "")
                })

        print(f"Found {len(error_samples)} error samples for analysis")

        # NEW APPROACH: Generate a text-based capability report instead of trying to parse complex JSON
        capability_report_text = ""
        error_analysis_text = ""

        if error_samples:
            # Modified system instruction for error analyzer that produces text instead of JSON
            error_analyzer_system_instruction = f"{self.system_prompt}\n\nYou are a Forensic Error Analyzer specializing in debugging complex reasoning systems. Your task is to perform deep, deliberate analysis of errors to identify specific failure points and propose targeted improvements."

            # Modified prompt for LLM to generate a natural language analysis
            prompt = f"""
            Perform a thorough forensic analysis of these error cases in our AI problem-solving system.

            For each error case, think step-by-step through what happened:

            ERROR CASES:
            {json.dumps(error_samples, indent=2)}

            ANALYSIS INSTRUCTIONS:
            1. TRACE THE REASONING PATH: For each error case, reconstruct the likely reasoning path the system took. Where precisely did the reasoning go wrong?

            2. IDENTIFY SPECIFIC FAILURE POINTS: What exact component, reasoning step, or assumption failed? 

            3. COMPARE WITH CORRECT SOLUTION: Analyze how the golden answer's reasoning differs from the system's approach

            4. FIND PATTERNS ACROSS ERRORS: Are there common failure modes or recurring issues?

            5. MAP FAILURES TO SYSTEM CAPABILITIES: For each error, identify which of these capabilities failed:
               - information_extraction: Extracting relevant information from the problem statement
               - constraint_handling: Identifying and applying constraints correctly
               - solution_generation: Generating valid potential solutions
               - solution_verification: Verifying solutions against constraints
               - decision_making: Making a final decision on the best solution

            FORMAT YOUR RESPONSE AS A STRUCTURED TEXT REPORT with the following sections:

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

            BE EXTREMELY SPECIFIC IN YOUR ANALYSIS. Avoid generic recommendations like "improve parsing" - instead identify exactly what type of parsing is failing and how to fix that specific issue.
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

        # Generate a capability report text
        try:
            # Define a system instruction for the capability reporter
            capability_reporter_system_instruction = f"{self.system_prompt}\n\nYou are a System Capability Analyst who specializes in providing actionable insights for improving AI systems."

            capability_prompt = f"""
            Generate a comprehensive capability report for our AI system based on its performance.

            PERFORMANCE SUMMARY:
            - Accuracy: {accuracy:.2f} ({correct_count}/{len(samples)})
            - Error samples: {len(error_samples)}/{len(samples)}

            ERROR ANALYSIS REPORT:
            {error_analysis_text}

            Please provide a thorough capability assessment that includes:

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
            """

            capability_report_text = self.call_llm(capability_prompt, system_instruction=capability_reporter_system_instruction)
            print("Generated capability report text successfully")

            # Extract the improvement focus for the dictionary return
            improvement_focus = "information_extraction"  # Default
            if "## IMPROVEMENT FOCUS" in capability_report_text:
                focus_section = capability_report_text.split("## IMPROVEMENT FOCUS")[1].split("##")[0].strip()
                # Look for capability names in the focus section
                for capability in ["information_extraction", "constraint_handling", "solution_generation", 
                                  "solution_verification", "decision_making"]:
                    if capability.replace("_", " ") in focus_section.lower():
                        improvement_focus = capability
                        break

            # Create a structured capability report for the dictionary return
            capability_report = {
                "text_report": capability_report_text,
                "improvement_focus": improvement_focus,
                "strengths": error_analysis.get("strengths", []),
                "weaknesses": error_analysis.get("weaknesses", []),
                "improvement_suggestions": error_analysis.get("improvement_suggestions", [])
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
        improvement_focus = capability_report.get("improvement_focus", "")

        # Build guidance
        guidance = "SYSTEM ANALYSIS & GUIDANCE\n\n"

        if improvement_focus:
            guidance += f"PRIMARY FOCUS AREA: {improvement_focus.upper().replace('_', ' ')}\n\n"

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
        evaluator_system_instruction = f"{self.system_prompt}\n\nYou are now acting as an Answer Evaluator. Your task is to determine if two answers convey the same meaning, even if they are worded differently."

        prompt = f"""
        You're evaluating two answers to determine if they convey the same information.

        System answer: {system_answer}
        Golden answer: {golden_answer}

        Do these answers effectively communicate the same information, even if worded differently?
        Return only a JSON object with: {{"match": true/false, "confidence": 0-1, "explanation": "reason"}}
        """

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
        summarizer_system_instruction = f"{self.system_prompt}\n\nYou are an Approach Summarizer. Your task is to analyze code and provide concise explanations of the techniques and methods used."

        prompt = f"""
        You're given a Python script that processes input and generates output using LLM-driven techniques.
        Provide a brief summary of the approach used in this script in 2-3 sentences.

        Focus on:
        1. What LLM-based techniques are used (chain-of-thought, verification, etc.)
        2. How the problem is decomposed
        3. What agent roles are involved
        4. The overall workflow

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
        accuracy and testing coverage
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
            if "progressive_testing" in it and it["progressive_testing"]:
                progressive_accuracy = it["progressive_testing"].get(
                    "accuracy", None)

            iteration_data.append({
                "iteration":
                it.get("iteration"),
                "accuracy":
                it.get("performance", {}).get("accuracy", 0),
                "batch_size":
                it.get("batch_size", 5),
                "progressive_accuracy":
                progressive_accuracy,
                "approach":
                it.get("approach_summary", "Unknown approach"),
                "strategy":
                it.get("strategy", "Unknown")
            })

        if not iteration_data:
            # Fallback if no valid iteration data
            return {
                "iteration": 0,
                "accuracy": 0,
                "batch_size": 5,
                "path": "No valid scripts available",
                "approach": "No approaches tried yet",
                "rationale": "No valid iterations completed"
            }

        # Role-specific system instruction for script evaluator
        script_evaluator_system_instruction = f"{self.system_prompt}\n\nYou are a Script Evaluator. Your task is to analyze performance metrics of different script iterations and determine which one represents the best overall approach."

        # Handle API rate limit issues - don't try to use LLM if we've hit limits
        try:
            # Use LLM to determine best script
            prompt = f"""
            As an AI system, determine which iteration produced the best script.

            Here is data about all iterations:
            {json.dumps(iteration_data, indent=2)}

            Consider both accuracy and testing coverage:
            - A script tested on more examples may be more robust
            - Recent iterations may reflect learned improvements
            - Higher accuracy is generally better
            - Scripts with good performance on progressive testing (across many examples) are particularly valuable

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
                # Fallback: just get the highest accuracy
                best_iteration = max(
                    iterations,
                    key=lambda x: x.get("performance", {}).get("accuracy", 0))

            return {
                "iteration":
                best_iteration.get("iteration"),
                "accuracy":
                best_iteration.get("performance", {}).get("accuracy", 0),
                "batch_size":
                best_iteration.get("batch_size", 5),
                "path":
                f"scripts/script_iteration_{best_iteration.get('iteration')}.py",
                "approach":
                best_iteration.get("approach_summary", ""),
                "rationale":
                result.get("rationale", "Highest overall accuracy")
            }
        except Exception as e:
            # Fallback method - don't use LLM, just pick highest accuracy
            print(f"Error determining best script with LLM: {e}")
            print("Using fallback method to determine best script")

            try:
                # Find the best iteration by accuracy
                best_iteration = max(
                    iterations,
                    key=lambda x: x.get("performance", {}).get("accuracy", 0)
                    if x else 0)

                return {
                    "iteration":
                    best_iteration.get("iteration"),
                    "accuracy":
                    best_iteration.get("performance", {}).get("accuracy", 0),
                    "batch_size":
                    best_iteration.get("batch_size", 5),
                    "path":
                    f"scripts/script_iteration_{best_iteration.get('iteration')}.py",
                    "approach":
                    best_iteration.get("approach_summary", ""),
                    "rationale":
                    "Fallback selection based on highest accuracy"
                }
            except Exception as e2:
                print(f"Error with fallback method: {e2}")
                # Ultra fallback - just return the first iteration
                if iterations and iterations[0]:
                    return {
                        "iteration":
                        iterations[0].get("iteration", 0),
                        "accuracy":
                        iterations[0].get("performance",
                                          {}).get("accuracy", 0),
                        "batch_size":
                        iterations[0].get("batch_size", 5),
                        "path":
                        f"scripts/script_iteration_{iterations[0].get('iteration', 0)}.py",
                        "approach":
                        iterations[0].get("approach_summary", ""),
                        "rationale":
                        "Ultra fallback - first available iteration"
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

    def run_progressive_testing(self,
                                script: str,
                                max_examples: int = 20) -> Dict:
        """Run progressive testing on seen examples, up to a maximum limit"""
        # Load the dataset
        dataset = self.load_dataset()

        # Get all seen examples
        all_samples = []
        for example_key in self.seen_examples:
            if example_key in dataset:
                all_samples.append(dataset[example_key])

        if not all_samples:
            return {"success": False, "error": "No examples seen yet"}

        # Limit to max_examples (most recent)
        samples = all_samples[-max_examples:] if len(
            all_samples) > max_examples else all_samples

        print(
            f"Running progressive testing on {len(samples)} seen examples (out of {len(all_samples)} total seen)..."
        )

        # Execute script on selected samples
        results = []
        for i, sample in enumerate(samples):
            if i % 5 == 0:  # Status update every 5 samples
                print(f"  Processing sample {i+1}/{len(samples)}...")

            question = sample.get("prompt_0shot", "")
            result = self.execute_script(script, question)

            # Evaluate the result if successful
            if result.get("success"):
                golden_answer = sample.get("golden_plan", "")
                system_answer = result.get("answer", "")

                # Use LLM-based evaluation
                evaluation = self.evaluate_answer_with_llm(
                    system_answer, golden_answer)

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

        # Load the dataset
        dataset = self.load_dataset()

        # Get examples in the specified range
        samples = []
        for i in range(start_index, end_index + 1):
            example_key = f"{self.example_prefix}{i}"
            if example_key in dataset:
                samples.append({
                    "key": example_key,
                    "data": dataset[example_key]
                })

        if not samples:
            return {
                "success": False,
                "error":
                f"No examples found in range {start_index}-{end_index}"
            }

        print(
            f"Validating script on {len(samples)} examples from range {start_index}-{end_index}..."
        )

        # Execute script on all samples
        results = []
        for i, sample in enumerate(samples):
            if i % 10 == 0:  # Status update every 10 samples
                print(f"  Processing sample {i+1}/{len(samples)}...")

            question = sample["data"].get("prompt_0shot", "")
            result = self.execute_script(script, question)

            # Evaluate the result if successful
            if result.get("success"):
                golden_answer = sample["data"].get("golden_plan", "")
                system_answer = result.get("answer", "")

                # Use LLM-based evaluation
                evaluation = self.evaluate_answer_with_llm(
                    system_answer, golden_answer)

                result["evaluation"] = evaluation
                result["match"] = evaluation.get("match", False)
            else:
                result["match"] = False

            results.append({"key": sample["key"], "result": result})

        # Calculate overall statistics
        successful_runs = sum(1 for r in results
                              if r["result"].get("success", False))
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

        # Get samples from dataset
        samples_data = self.get_samples()
        samples = samples_data["samples"]

        if not samples:
            print("No samples available in dataset. Exiting iteration.")
            return {"success": False, "error": "No samples available"}

        print(f"Processing {len(samples)} examples (including {samples_data['new_examples_added']} new examples)")

        # Get capability report if available
        capability_report = None
        if hasattr(self, 'capability_tracker'):
            capability_report = self.capability_tracker.generate_report()
            if capability_report:
                print("\n=== Current Capability Status ===")

                # Display improvement focus
                improvement_focus = capability_report.get("improvement_focus", "")
                if improvement_focus:
                    print(f"  Focus area: {improvement_focus.upper().replace('_', ' ')}")

                # Display strengths
                if capability_report.get("strengths"):
                    print("\n  Strengths:")
                    for strength in capability_report.get("strengths", [])[:2]:  # Show top 2
                        print(f"    - {strength}")

                # Display weaknesses
                if capability_report.get("weaknesses"):
                    print("\n  Weaknesses:")
                    for weakness in capability_report.get("weaknesses", [])[:2]:  # Show top 2
                        print(f"    - {weakness}")

                # Display improvement suggestions if available
                if capability_report.get("improvement_suggestions"):
                    print("\n  Suggested Improvements:")
                    for suggestion in capability_report.get("improvement_suggestions", [])[:2]:
                        print(f"    - {suggestion}")

                print("=" * 40)

        # Decide whether to explore or exploit
        is_exploration = (self.explore_rate > self.exploit_rate) or (random.random() * 100 <= self.explore_rate)

        # If we have capability data with clear trends, potentially override the strategy
        if capability_report and capability_report.get("trend") != "insufficient_data":
            weakest_capability = capability_report.get("improvement_focus", "")

            # Check trend for the weakest capability
            trends = capability_report.get("trend", {})
            weakest_trend = trends.get(weakest_capability) if isinstance(trends, dict) else None

            # If our weakest capability is declining, force exploration
            if weakest_trend == "declining":
                if not is_exploration:
                    print(f"Strategy override: Forcing exploration to address declining capability: {weakest_capability}")
                    is_exploration = True

            # If all capabilities are improving, consider more exploitation
            if isinstance(trends, dict):
                improving_count = sum(1 for trend in trends.values() if trend == "improving")
                if improving_count == len(trends) and is_exploration and improving_count > 0:
                    print(f"Strategy override: Forcing exploitation to capitalize on improving capabilities")
                    is_exploration = False

        strategy = "Exploration" if is_exploration else "Exploitation"
        print(f"Strategy for this iteration: {strategy}")

        # Generate script using LLM
        print("Generating script with LLM...")

        # Get capability insights
        capability_guidance = ""
        improvement_focus = None

        if hasattr(self, 'capability_tracker'):
            capability_report = self.capability_tracker.generate_report()
            improvement_focus = capability_report.get("improvement_focus", "")
            if capability_report:
                capability_guidance = self._generate_capability_guidance(capability_report)
                print(f"Focusing script improvement on: {improvement_focus.upper().replace('_', ' ')}")

        script = self.generate_script_with_llm(is_exploration)

        # Generate a summary of the approach
        try:
            approach_summary = self.generate_approach_summary(script)
            if approach_summary.startswith("API_RATE_LIMIT_EXCEEDED"):
                approach_summary = "Approach summary not available due to API rate limit"
        except Exception as e:
            approach_summary = f"Error generating approach summary: {str(e)}"

        print(f"Approach summary: {approach_summary}")

        # Execute script on samples
        print("Executing script on samples...")
        results = []
        for i, sample in enumerate(samples):
            print(f"  Processing sample {i+1}/{len(samples)}...")
            question = sample.get("prompt_0shot", "")
            result = self.execute_script(script, question)

            if result.get("success"):
                print(f"    Result: {result.get('answer')}")

                # Evaluate with LLM
                golden_answer = sample.get("golden_plan", "")
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
                        for suggestion in error_analysis["improvement_suggestions"][:3]:  # Top 3
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
        if accuracy >= 0.7:  # Only run progressive testing if current batch performance is good
            try:
                print("Script looks promising! Running progressive testing on all seen examples...")
                progressive_testing_results = self.run_progressive_testing(script, max_examples=20)

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
            # Maintain current values if error occurs
            new_explore, new_exploit = self.explore_rate, self.exploit_rate
            print(f"Maintaining current explore/exploit balance: {new_explore}/{new_exploit}")

        # Adjust batch size for next iteration
        try:
            print("Adjusting batch size...")
            new_batch_size, batch_adjustment_rationale = self.adjust_batch_size_with_llm(basic_evaluation)
            print(f"New batch size: {new_batch_size} ({batch_adjustment_rationale})")
        except Exception as e:
            print(f"Error adjusting batch size: {str(e)}")
            # Maintain current batch size if error occurs
            new_batch_size = self.current_batch_size
            batch_adjustment_rationale = f"Error: {str(e)}"
            print(f"Maintaining current batch size: {new_batch_size}")

        # Identify current best script
        try:
            best_script_info = self.get_best_script_info()
            if best_script_info:
                print("\n=== Current Best Script ===")
                print(f"Iteration: {best_script_info.get('iteration')}")
                print(f"Accuracy: {best_script_info.get('accuracy', 0):.2f} (tested on {best_script_info.get('batch_size', 0)} examples)")
                print(f"Path: {best_script_info.get('path')}")
                print(f"Approach: {best_script_info.get('approach')}")
                print(f"Rationale: {best_script_info.get('rationale')}")

                # Display capability assessment for the best script if available
                if hasattr(self, 'capability_tracker') and capability_report:
                    print("\n  Capability Assessment:")

                    # Display improvement focus
                    improvement_focus = capability_report.get("improvement_focus", "")
                    if improvement_focus:
                        print(f"    Focus area: {improvement_focus.upper().replace('_', ' ')}")

                    # Display strengths (top 2)
                    if capability_report.get("strengths"):
                        print("    Strengths:")
                        for strength in capability_report.get("strengths", [])[:2]:
                            print(f"      - {strength}")

                    # Display weaknesses (top 2)
                    if capability_report.get("weaknesses"):
                        print("    Weaknesses:")
                        for weakness in capability_report.get("weaknesses", [])[:2]:
                            print(f"      - {weakness}")
        except Exception as e:
            print(f"Error identifying best script: {str(e)}")
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
        return "information_extraction"

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
            "improvement_focus": self.get_improvement_focus()
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
