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

    # Add these methods to the AgentSystem class in agent_system.py

    def _load_learnings(self) -> str:
        """Load accumulated learnings from the learnings.txt file"""
        learnings_path = Path("learnings.txt")
        if not learnings_path.exists():
            print(
                "No existing learnings file found. Starting with fresh learnings."
            )
            return ""

        try:
            with open(learnings_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading learnings: {e}")
            return ""

    def _save_learnings(self, learnings: str) -> None:
        """Save updated learnings to the learnings.txt file"""
        learnings_path = Path("learnings.txt")
        try:
            with open(learnings_path, 'w', encoding='utf-8') as f:
                f.write(learnings)
            print(f"Learnings updated and saved to {learnings_path}")
        except Exception as e:
            print(f"Error saving learnings: {e}")

    def generate_batch_learnings(self, iteration_data: Dict) -> str:
        """Generate learnings from the current batch results with focus on dataset-specific insights"""
        learnings_generator_system_instruction = f"{self.system_prompt}\n\nYou are a Knowledge Synthesizer. Your role is to extract concrete, dataset-specific insights from experiment results, focusing on patterns in the data, effective strategies for this specific task, and precise failure modes."

        # Extract more detailed information from iteration_data
        examples = iteration_data.get("results", [])
        sample_questions = [
            samples[i].get("prompt_0shot", "N/A")
            for i in range(min(3, len(examples)))
        ]
        approach_details = iteration_data.get(
            "script", "")[:500]  # Get first 500 chars of script for context

        prompt = f"""
        Extract specific, concrete learnings from this iteration's results, focusing on dataset-specific insights:

        Iteration: {iteration_data.get("iteration")}
        Strategy: {iteration_data.get("strategy", "Unknown")}
        Accuracy: {iteration_data.get("performance", {}).get("accuracy", 0):.2f}
        Approach summary: {iteration_data.get("approach_summary", "No summary available")}

        Sample questions from dataset:
        {json.dumps(sample_questions, indent=2)}

        Primary issue identified: {iteration_data.get("performance", {}).get("error_analysis", {}).get("primary_issue", "None identified")}

        Error patterns:
        {json.dumps(iteration_data.get("performance", {}).get("error_analysis", {}).get("error_patterns", []), indent=2)}

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
            print(f"Error generating batch learnings: {e}")
            return f"--- LEARNINGS FROM ITERATION {iteration_data.get('iteration')} ---\nError generating learnings: {str(e)}\n\n"

    def synthesize_learnings(self, current_learnings: str,
                             new_batch_learnings: str) -> str:
        """Synthesize existing learnings with new batch learnings, emphasizing dataset-specific insights"""
        learnings_synthesizer_system_instruction = f"{self.system_prompt}\n\nYou are a Knowledge Integrator. Your role is to synthesize accumulated dataset-specific knowledge with new insights, creating an evolving experiment log that captures concrete patterns, strategies, and findings about this specific task."

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
        """

        try:
            response = self.call_llm(
                prompt,
                system_instruction=learnings_synthesizer_system_instruction)
            return response.strip()
        except Exception as e:
            print(f"Error synthesizing learnings: {e}")
            return f"{current_learnings}\n\n{new_batch_learnings}"

    def update_learnings(self, iteration_data: Dict) -> None:
        """Update the learnings file with insights from the current iteration"""
        try:
            # Load existing learnings
            current_learnings = self._load_learnings()

            # Generate learnings from current batch
            batch_learnings = self.generate_batch_learnings(iteration_data)

            # If this is the first iteration, just use the batch learnings
            if not current_learnings:
                updated_learnings = batch_learnings
            else:
                # Synthesize existing learnings with new batch learnings
                updated_learnings = self.synthesize_learnings(
                    current_learnings, batch_learnings)

            # Save updated learnings
            self._save_learnings(updated_learnings)

        except Exception as e:
            print(f"Error updating learnings: {e}")

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
            if improvement_focus:
                capability_guidance = self._generate_capability_guidance(
                    improvement_focus)
                print(f"Focusing script improvement on: {improvement_focus}")

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
        if improvement_focus and capability_guidance:
            capability_context = f"""
    CAPABILITY IMPROVEMENT FOCUS:
    The system currently needs the most improvement in: {improvement_focus.upper()}

    SPECIFIC GUIDANCE FOR IMPROVING {improvement_focus.upper()}:
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
            {f"7. Pay SPECIAL ATTENTION to improving the {improvement_focus} capability" if improvement_focus else ""}

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
            {f"- Ensure your new approach excels at {improvement_focus}" if improvement_focus else ""}

            CRITICAL REQUIREMENTS:
            1. The script MUST properly handle all string literals - be extremely careful with quotes and triple quotes
            2. The script MUST NOT exceed 150 lines of code to prevent truncation
            3. Include detailed comments explaining your reasoning approach
            4. EVERY SINGLE LLM PROMPT must include at least one embedded example showing:
               - Sample input with reasoning
               - Desired output format
            5. Make proper use of error handling
            {f"6. Implement robust {improvement_focus} capabilities as outlined in the guidance above" if improvement_focus else ""}

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
            {f"7. PRIORITIZE improving the {improvement_focus} capability as outlined in the guidance above" if improvement_focus else ""}

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
            {f"- Significantly enhance the {improvement_focus} component of the system" if improvement_focus else ""}

            CRITICAL REQUIREMENTS:
            1. The script MUST properly handle all string literals - be extremely careful with quotes and triple quotes
            2. The script MUST NOT exceed 150 lines of code to prevent truncation
            3. Include detailed comments explaining your improvements
            4. EVERY SINGLE LLM PROMPT must include at least one embedded example showing:
               - Sample input with reasoning
               - Desired output format
            5. Make proper use of error handling
            {f"6. Implement robust {improvement_focus} capabilities as outlined in the guidance above" if improvement_focus else ""}

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

    def evaluate_with_llm(self, samples: List[Dict],
                          results: List[Dict]) -> Dict:
        """
        Use the LLM to evaluate results and perform detailed error analysis.
        Integrates with capability tracking to identify specific system weaknesses.
        """
        evaluations = []

        # First, perform semantic evaluation with LLM
        correct_count = 0
        for i, (sample, result) in enumerate(zip(samples, results)):
            if not result.get("success"):
                evaluations.append({
                    "sample_id":
                    i,
                    "success":
                    False,
                    "error":
                    result.get("error", "Unknown error"),
                    "match":
                    False,
                    "capability_failures":
                    ["execution"]  # Track execution failures
                })
                continue

            # Compare with golden answer using LLM
            if not result.get("evaluation"):
                golden_answer = sample.get("golden_plan", "").strip()
                system_answer = result.get("answer", "").strip()
                evaluation = self.evaluate_answer_with_llm(
                    system_answer, golden_answer)
                result["evaluation"] = evaluation
                result["match"] = evaluation.get("match", False)

            if result.get("match", False):
                correct_count += 1
                # For successful cases, we consider all capabilities successful
                capability_success = [
                    "information_extraction", "constraint_handling",
                    "solution_generation", "solution_verification",
                    "decision_making"
                ]
                evaluations.append({
                    "sample_id":
                    i,
                    "success":
                    True,
                    "system_answer":
                    result.get("answer", "").strip(),
                    "golden_answer":
                    sample.get("golden_plan", "").strip(),
                    "match":
                    True,
                    "evaluation":
                    result.get("evaluation", {}),
                    "capability_successes":
                    capability_success
                })
            else:
                # For failed cases, we'll determine which capabilities failed later
                evaluations.append({
                    "sample_id":
                    i,
                    "success":
                    True,
                    "system_answer":
                    result.get("answer", "").strip(),
                    "golden_answer":
                    sample.get("golden_plan", "").strip(),
                    "match":
                    False,
                    "evaluation":
                    result.get("evaluation", {}),
                    "capability_failures":
                    []  # Placeholder, will be populated after error analysis
                })

        # Calculate accuracy
        accuracy = correct_count / len(samples) if samples else 0

        # For deeper analysis, use LLM to analyze error patterns
        error_samples = []
        for i, eval_data in enumerate(evaluations):
            if not eval_data.get("match"):
                sample = samples[i]
                error_samples.append({
                    "sample_id":
                    i,
                    "question":
                    sample.get("prompt_0shot", ""),
                    "system_answer":
                    eval_data.get("system_answer", ""),
                    "golden_answer":
                    eval_data.get("golden_answer", ""),
                    "error_message":
                    eval_data.get("error", ""),
                    "explanation":
                    eval_data.get("evaluation", {}).get("explanation", "")
                })

        error_analysis = {}
        capability_insights = {}

        if error_samples:
            # Enhanced system instruction for error analyzer that includes capability identification
            error_analyzer_system_instruction = f"{self.system_prompt}\n\nYou are a Forensic Error Analyzer specializing in debugging complex reasoning systems. Your task is to perform deep, deliberate analysis of errors to identify specific failure points and propose targeted improvements. You will also map errors to specific system capabilities."

            # Enhanced prompt for LLM to perform capability-aware detailed error analysis
            prompt = f"""
            Perform a thorough forensic analysis of these error cases in our AI problem-solving system.

            For each error case, think step-by-step through what happened:

            ERROR CASES:
            {json.dumps(error_samples, indent=2)}

            ANALYSIS INSTRUCTIONS:
            1. TRACE THE REASONING PATH: For each error case, reconstruct the likely reasoning path the system took. Where precisely did the reasoning go wrong?

            2. IDENTIFY SPECIFIC FAILURE POINTS: What exact component, reasoning step, or assumption failed? This might be:
               - Misunderstanding a specific aspect of the problem
               - Missing a key constraint or requirement
               - Applying the wrong reasoning technique
               - Failing to extract critical information
               - Making incorrect logical inferences

            3. COMPARE WITH CORRECT SOLUTION: Analyze how the golden answer's reasoning differs from the system's approach

            4. FIND PATTERNS ACROSS ERRORS: Are there common failure modes or recurring issues? Look for:
               - Types of problems the system struggles with
               - Specific reasoning patterns that consistently fail
               - Blind spots in the system's analysis approach
               - LLM call issues (prompting, role instructions, etc.)

            5. MAP FAILURES TO SYSTEM CAPABILITIES: For each error, identify which of these capabilities failed:
               - information_extraction: Extracting relevant information from the problem statement
               - constraint_handling: Identifying and applying constraints correctly
               - solution_generation: Generating valid potential solutions
               - solution_verification: Verifying solutions against constraints
               - decision_making: Making a final decision on the best solution

            6. PROPOSE TARGETED IMPROVEMENTS: Suggest specific, practical changes to fix these issues:
               - Explicit changes to reasoning steps
               - New verification procedures to catch specific errors
               - Different prompting strategies for the LLM
               - Additional agent roles or specialized agents for difficult aspects

            FORMAT YOUR RESPONSE AS A JSON OBJECT WITH THESE FIELDS:
            1. "detailed_analysis": [List of detailed per-case analyses explaining exactly what went wrong in each case]
            2. "failure_points": [Specific components or reasoning steps that are failing]
            3. "error_patterns": [Recurring patterns across multiple errors]
            4. "primary_issue": The single most critical problem to fix (be very specific)
            5. "targeted_improvements": [Specific, actionable changes to fix the identified issues]
            6. "root_causes": [Underlying causes behind the errors]
            7. "capability_mapping": {
               "sample_0": ["information_extraction", "constraint_handling"],
               "sample_1": ["information_extraction"]
            }
            8. "capability_insights": {
               "information_extraction": {"score": float 0-1, "issues": ["list of issues"], "improvements": ["list of improvements"]},
               "constraint_handling": {"score": float 0-1, "issues": ["list of issues"], "improvements": ["list of improvements"]},
               ...
            }

            BE EXTREMELY SPECIFIC IN YOUR ANALYSIS. Avoid generic recommendations like "improve parsing" - instead identify exactly what type of parsing is failing and how to fix that specific issue.
            """

            # Call LLM for detailed error analysis
            try:
                response = self.call_llm(
                    prompt,
                    system_instruction=error_analyzer_system_instruction)

                # Extract JSON from response
                response = response.strip()
                if response.startswith("```json"):
                    response = response.split("```json")[1]
                elif response.startswith("```"):
                    response = response.split("```")[1]

                if response.endswith("```"):
                    response = response.split("```")[0]

                error_analysis = json.loads(response)

                # Extract capability insights
                capability_insights = error_analysis.get(
                    "capability_insights", {})

                # Update each evaluation with capability failures
                if "capability_mapping" in error_analysis:
                    for sample_id_str, failed_capabilities in error_analysis[
                            "capability_mapping"].items():
                        try:
                            # Convert sample_id from string to int (remove "sample_" prefix)
                            sample_index = int(
                                sample_id_str.replace("sample_", ""))
                            # Find the matching evaluation
                            for eval_data in evaluations:
                                if eval_data[
                                        "sample_id"] == sample_index and not eval_data.get(
                                            "match", False):
                                    eval_data[
                                        "capability_failures"] = failed_capabilities
                        except (ValueError, KeyError) as e:
                            print(
                                f"Error processing capability mapping for {sample_id_str}: {e}"
                            )

                # Print out some of the detailed analysis for visibility
                if "detailed_analysis" in error_analysis and error_analysis[
                        "detailed_analysis"]:
                    print("\nDetailed Error Analysis Highlights:")
                    for i, analysis in enumerate(
                            error_analysis["detailed_analysis"]
                        [:2]):  # Show first 2 analyses
                        print(f"  Error {i+1}: {analysis[:150]}..."
                              if len(analysis) > 150 else analysis)

                if "primary_issue" in error_analysis:
                    print(
                        f"\nPrimary Issue: {error_analysis['primary_issue']}")

                if "targeted_improvements" in error_analysis and error_analysis[
                        "targeted_improvements"]:
                    print("\nKey Targeted Improvements:")
                    for i, improvement in enumerate(
                            error_analysis["targeted_improvements"]
                        [:3]):  # Show first 3 improvements
                        print(f"  {i+1}. {improvement}")

                # Print capability insights
                if capability_insights:
                    print("\nCapability Insights:")
                    for capability, data in capability_insights.items():
                        score = data.get("score", 0.0)
                        print(f"  {capability}: {score:.2f}")

            except Exception as e:
                print(f"Error in LLM error analysis: {e}")
                error_analysis = {
                    "error_patterns": ["Analysis failed"],
                    "primary_issue": "Unable to analyze errors with LLM",
                    "recommendations": ["Retry analysis in next iteration"],
                    "root_causes": ["LLM error analysis failed: " + str(e)]
                }

        # Update capability tracker if it exists
        if hasattr(self, 'capability_tracker'):
            self.capability_tracker.update_from_evaluations(
                evaluations, capability_insights, error_analysis)

            # Generate capability report
            capability_report = self.capability_tracker.generate_report()
        else:
            capability_report = None

        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(samples),
            "evaluations": evaluations,
            "error_analysis": error_analysis,
            "capability_report": capability_report,
            "capability_insights": capability_insights
        }

    def _generate_capability_guidance(self, capability):
        """Generate specific guidance for improving a capability."""
        guidance = {
            "information_extraction":
            """
            Focus on robust information extraction:
            1. Use explicit chain-of-thought reasoning to extract participants, schedules, and constraints
            2. Implement verification steps to confirm extracted information
            3. Use structured formats (JSON) for extracted information
            """,
            "constraint_handling":
            """
            Focus on comprehensive constraint handling:
            1. Clearly categorize constraints as hard (must satisfy) vs soft (preferences)
            2. Implement logic to check all constraints are satisfied before proposing a solution
            3. Handle conflicting constraints gracefully
            """,
            "solution_generation":
            """
            Focus on robust solution generation:
            1. Generate multiple candidate solutions
            2. Ensure comprehensive exploration of the solution space
            3. Implement methods to avoid premature termination of search
            """,
            "solution_verification":
            """
            Focus on rigorous solution verification:
            1. Implement explicit verification against each constraint
            2. Add detailed explanation of verification steps
            3. Double-check boundary conditions and edge cases
            """,
            "decision_making":
            """
            Focus on decisive solution selection:
            1. Implement clear criteria for selecting the best solution
            2. Add logic to commit to a single answer even with uncertainty
            3. Provide confidence assessment for the selected solution
            """
        }

        return guidance.get(capability, "")

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
        print(
            f"Current explore/exploit balance: {self.explore_rate}/{self.exploit_rate}"
        )
        print(f"Current batch size: {self.current_batch_size}")
        print(f"Total seen examples: {len(self.seen_examples)}")

        iteration_start_time = time.time()

        # Get samples from dataset
        samples_data = self.get_samples()
        samples = samples_data["samples"]

        if not samples:
            print("No samples available in dataset. Exiting iteration.")
            return {"success": False, "error": "No samples available"}

        print(
            f"Processing {len(samples)} examples (including {samples_data['new_examples_added']} new examples)"
        )

        # Get capability report if available
        capability_report = None
        if hasattr(self, 'capability_tracker'):
            capability_report = self.capability_tracker.generate_report()
            if capability_report:
                print("\n=== Current Capability Status ===")
                for capability, score in capability_report.get(
                        "capability_scores", {}).items():
                    print(f"  {capability}: {score:.2f}")

                if "improvement_focus" in capability_report:
                    print(
                        f"  Focus area: {capability_report['improvement_focus']}"
                    )

                if "trend" in capability_report and capability_report[
                        "trend"] != "insufficient_data":
                    print("  Capability trends:")
                    for cap, trend in capability_report["trend"].items():
                        print(f"    {cap}: {trend}")
                print("=" * 40)

        # Decide whether to explore or exploit
        is_exploration = (self.explore_rate
                          > self.exploit_rate) or (random.random() * 100
                                                   <= self.explore_rate)

        # If we have capability data with clear trends, potentially override the strategy
        if capability_report and capability_report.get(
                "trend") != "insufficient_data":
            weakest_capability = capability_report.get("weakest_capabilities",
                                                       [{}])[0].get("name")
            weakest_trend = capability_report.get("trend",
                                                  {}).get(weakest_capability)

            # If our weakest capability is declining, force exploration
            if weakest_trend == "declining":
                if not is_exploration:
                    print(
                        f"Strategy override: Forcing exploration to address declining capability: {weakest_capability}"
                    )
                    is_exploration = True

            # If all capabilities are improving, consider more exploitation
            improving_count = sum(
                1 for trend in capability_report.get("trend", {}).values()
                if trend == "improving")
            if improving_count == len(
                    capability_report.get("capability_scores",
                                          {})) and is_exploration:
                print(
                    f"Strategy override: Forcing exploitation to capitalize on improving capabilities"
                )
                is_exploration = False

        strategy = "Exploration" if is_exploration else "Exploitation"
        print(f"Strategy for this iteration: {strategy}")

        # Generate script using LLM
        print("Generating script with LLM...")
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
                    evaluation = self.evaluate_answer_with_llm(
                        system_answer, golden_answer)
                    result["evaluation"] = evaluation
                    result["match"] = evaluation.get("match", False)

                    if result["match"]:
                        print(
                            f"    ✅ Match (confidence: {evaluation.get('confidence', 0):.2f})"
                        )
                    else:
                        print(
                            f"    ❌ No match: {evaluation.get('explanation', '')}"
                        )
                except Exception as e:
                    print(f"    ⚠️ Error evaluating answer: {str(e)}")
                    # Fallback to exact match
                    exact_match = system_answer.strip() == golden_answer.strip(
                    )
                    result["match"] = exact_match
                    result["evaluation"] = {
                        "match": exact_match,
                        "confidence": 1.0 if exact_match else 0.0,
                        "explanation": f"Error evaluating: {str(e)}"
                    }
                    print(
                        f"    {'✅' if exact_match else '❌'} Fallback to exact match: {exact_match}"
                    )
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

        print(f"Performance: {accuracy:.2f} accuracy " +
              f"({matches}/{len(samples)} correct)")

        # Use LLM for deeper error analysis
        try:
            print("Performing error analysis with LLM...")
            evaluation = self.evaluate_with_llm(samples, results)

            if evaluation.get('error_analysis'):
                primary_issue = evaluation.get('error_analysis',
                                               {}).get('primary_issue', 'None')
                print(f"Primary issue identified: {primary_issue}")

                # If we have capability insights, print them
                if "capability_insights" in evaluation:
                    print("\n=== Capability Insights from This Iteration ===")
                    for capability, data in evaluation.get(
                            "capability_insights", {}).items():
                        score = data.get("score", 0.0)
                        print(f"  {capability}: {score:.2f}")
                        if "issues" in data and data["issues"]:
                            print(f"    Issues: {data['issues'][0]}")
                        if "improvements" in data and data["improvements"]:
                            print(
                                f"    Improvements: {data['improvements'][0]}")
                    print("=" * 40)
            else:
                print("No specific issues identified")

        except Exception as e:
            print(f"Error in error analysis: {str(e)}")
            evaluation = basic_evaluation
            evaluation["error_analysis"] = {
                "primary_issue": "Analysis error",
                "error_patterns": ["Error during analysis"],
                "recommendations": ["Fix error handling"],
                "root_causes": [str(e)]
            }

        # Run progressive testing on all seen examples for promising scripts
        progressive_testing_results = None
        if accuracy >= 0.7:  # Only run progressive testing if current batch performance is good
            try:
                print(
                    "Script looks promising! Running progressive testing on all seen examples..."
                )
                progressive_testing_results = self.run_progressive_testing(
                    script, max_examples=20)

                if progressive_testing_results:
                    prog_accuracy = progressive_testing_results.get(
                        "accuracy", 0)
                    prog_matches = progressive_testing_results.get(
                        "matches", 0)
                    prog_total = progressive_testing_results.get(
                        "total_examples", 0)

                    print(
                        f"Progressive testing results: {prog_accuracy:.2f} accuracy "
                        + f"({prog_matches}/{prog_total} correct)")
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
            print(
                f"Maintaining current explore/exploit balance: {new_explore}/{new_exploit}"
            )

        # Adjust batch size for next iteration
        try:
            print("Adjusting batch size...")
            new_batch_size, batch_adjustment_rationale = self.adjust_batch_size_with_llm(
                basic_evaluation)
            print(
                f"New batch size: {new_batch_size} ({batch_adjustment_rationale})"
            )
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
                print(
                    f"Accuracy: {best_script_info.get('accuracy', 0):.2f} (tested on {best_script_info.get('batch_size', 0)} examples)"
                )
                print(f"Path: {best_script_info.get('path')}")
                print(f"Approach: {best_script_info.get('approach')}")
                print(f"Rationale: {best_script_info.get('rationale')}")

                # If we have capability data for the best script, show it
                if hasattr(self, 'capability_tracker'
                           ) and "capability_scores" in capability_report:
                    print("\n  Capability Profile:")
                    for capability, score in capability_report.get(
                            "capability_scores", {}).items():
                        print(f"    {capability}: {score:.2f}")
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
            "results": results,
            "performance": evaluation,
            "progressive_testing": progressive_testing_results,
            "execution_time": time.time() - iteration_start_time,
            "capability_report": capability_report
        }

        # Create summary
        summary = {
            "iteration":
            self.current_iteration,
            "timestamp":
            datetime.datetime.now().isoformat(),
            "strategy":
            strategy,
            "explore_rate":
            self.explore_rate,
            "exploit_rate":
            self.exploit_rate,
            "batch_size":
            self.current_batch_size,
            "approach_summary":
            approach_summary,
            "performance": {
                "accuracy": accuracy,
                "correct_count": matches,
                "total_count": len(samples)
            },
            "progressive_accuracy":
            progressive_testing_results.get("accuracy", None)
            if progressive_testing_results else None,
            "primary_issue":
            evaluation.get("error_analysis", {}).get("primary_issue",
                                                     "None identified"),
            "new_explore_rate":
            new_explore,
            "new_exploit_rate":
            new_exploit,
            "new_batch_size":
            new_batch_size,
            "capability_report":
            capability_report
        }

        # Save to archive
        try:
            self.save_to_archive(iteration_data,
                                 f"iteration_{self.current_iteration}.json")
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
            print(
                "Updating learnings file with insights from this iteration...")
            self.update_learnings(iteration_data)
        except Exception as e:
            print(f"Error updating learnings: {e}")

        return iteration_data


class CapabilityTracker:
    """
    Tracks and evaluates specific capabilities of the system across iterations.
    Allows for targeted improvement rather than general exploration/exploitation.
    """

    def __init__(self):
        # Define core capabilities that are domain-agnostic
        self.capabilities = {
            "information_extraction": {
                "score": 0.0,
                "examples": [],
                "weight": 1.0
            },
            "constraint_handling": {
                "score": 0.0,
                "examples": [],
                "weight": 1.0
            },
            "solution_generation": {
                "score": 0.0,
                "examples": [],
                "weight": 1.0
            },
            "solution_verification": {
                "score": 0.0,
                "examples": [],
                "weight": 1.0
            },
            "decision_making": {
                "score": 0.0,
                "examples": [],
                "weight": 1.0
            }
        }
        self.history = []

        # Initialize with small non-zero scores to avoid cold-start issues
        for cap in self.capabilities:
            self.capabilities[cap]["score"] = 0.2  # Start with 0.2 instead of 0.0

    def update_capability_scores(self, results, examples, error_analysis):
        """
        Update capability scores based on test results and error analysis.

        Args:
            results: List of test results (success/failure)
            examples: The examples used for testing
            error_analysis: Error analysis data from LLM
        """
        # Create a snapshot of current scores for history
        snapshot = {
            cap: data["score"]
            for cap, data in self.capabilities.items()
        }
        snapshot["timestamp"] = datetime.datetime.now().isoformat()

        # Analyze which capabilities contributed to success/failure
        capability_contributions = self._analyze_capability_contributions(
            results, error_analysis)

        # Update scores for each capability
        for capability, contribution in capability_contributions.items():
            if capability in self.capabilities:
                # Use weighted moving average to update scores
                old_score = self.capabilities[capability]["score"]
                # Weight recent results more heavily (0.3 is the learning rate)
                new_score = (0.7 * old_score) + (0.3 * contribution["score"])
                self.capabilities[capability]["score"] = new_score

                # Add example IDs that influenced this capability
                for example_id in contribution["example_ids"]:
                    if example_id not in self.capabilities[capability][
                            "examples"]:
                        self.capabilities[capability]["examples"].append(
                            example_id)

        # Add snapshot to history
        self.history.append(snapshot)


    def update_from_evaluations(self, evaluations, capability_insights, error_analysis):
        """
        Update capability scores based on evaluation results.
        """
        # Print debug info to verify data structure
        print(f"Updating capabilities with {len(evaluations)} evaluations")

        # FIXED: Extract actual results with proper structure
        # Evaluations is a list of dictionaries with "match" nested inside
        success_count = 0
        for eval_data in evaluations:
            if eval_data.get("match", False):
                success_count += 1

        overall_success = success_count / len(evaluations) if evaluations else 0
        print(f"Overall success rate: {overall_success:.2f}")

        # Set initial contribution scores based on overall success
        capability_contributions = {}
        for capability in self.capabilities:
            capability_contributions[capability] = {
                "score": overall_success,
                "example_ids": []
            }

        # Process capability insights if available
        if capability_insights:
            print(f"Processing capability insights: {list(capability_insights.keys())}")
            for capability, insights in capability_insights.items():
                if capability in self.capabilities and isinstance(insights, dict) and "score" in insights:
                    # Convert score to float in case it's a string
                    try:
                        score = float(insights.get("score", 0.0))
                        # Direct assignment instead of blending initially
                        self.capabilities[capability]["score"] = score
                        print(f"Updated {capability} score to {score}")
                    except (ValueError, TypeError):
                        print(f"Invalid score for {capability}: {insights.get('score')}")

        # Track failures by capability
        for i, eval_data in enumerate(evaluations):
            if "capability_failures" in eval_data and not eval_data.get("match", False):
                for cap in eval_data["capability_failures"]:
                    if cap in self.capabilities:
                        self.capabilities[cap]["examples"].append(i)

        # Create a snapshot for trend analysis
        snapshot = {cap: data["score"] for cap, data in self.capabilities.items()}
        snapshot["timestamp"] = datetime.datetime.now().isoformat()
        self.history.append(snapshot)
    
    def _analyze_capability_contributions(self, results, error_analysis):
        """
        Analyze how each capability contributed to success or failure.
        Uses the error analysis from LLM to attribute failures to specific capabilities.
        """
        contributions = {
            cap: {
                "score": 0.0,
                "example_ids": []
            }
            for cap in self.capabilities
        }

        # Initialize with overall success rate
        overall_success = sum(
            1 for r in results if r.get("match", False)) / len(results)
        for cap in contributions:
            contributions[cap]["score"] = overall_success

        # Use error analysis to attribute failures to specific capabilities
        if error_analysis and "error_patterns" in error_analysis:
            for pattern in error_analysis["error_patterns"]:
                # Map error patterns to capabilities (simplified example)
                capability = self._map_error_to_capability(pattern)
                if capability:
                    # Lower the score for this capability
                    contributions[capability][
                        "score"] *= 0.8  # Penalize by 20%

        # Add example IDs to contributions
        for i, result in enumerate(results):
            for cap in contributions:
                if not result.get("match", False):
                    # For failed examples, only add to relevant capabilities
                    capability = self._map_error_to_capability(
                        result.get("explanation", "Unknown error"))
                    if capability == cap:
                        contributions[cap]["example_ids"].append(i)
                else:
                    # For successful examples, add to all capabilities
                    contributions[cap]["example_ids"].append(i)

        return contributions

    def _map_error_to_capability(self, error_description):
        """
        Map an error description to a specific capability.
        This uses simple keyword matching but could be enhanced with LLM-based mapping.
        """
        error_description = error_description.lower()

        # Map errors to capabilities based on keywords
        if any(
                kw in error_description for kw in
            ["extract", "parsing", "identification", "recognize", "detect"]):
            return "information_extraction"

        elif any(kw in error_description
                 for kw in ["constraint", "requirement", "condition", "rule"]):
            return "constraint_handling"

        elif any(kw in error_description for kw in
                 ["generate", "create", "propose", "suggest", "recommend"]):
            return "solution_generation"

        elif any(kw in error_description for kw in
                 ["verify", "validate", "check", "confirm", "ensure"]):
            return "solution_verification"

        elif any(
                kw in error_description
                for kw in ["decide", "select", "choose", "pick", "determine"]):
            return "decision_making"

        # Default to None if no clear mapping
        return None

    def identify_weakest_capabilities(self):
        """
        Identify the capabilities with the lowest scores.
        Returns them in ascending order (worst first).
        """
        sorted_capabilities = sorted(self.capabilities.items(),
                                     key=lambda x: x[1]["score"])
        return sorted_capabilities

    def get_improvement_focus(self):
        """
        Determine which capability should be the focus for improvement.
        Takes into account both score and importance weight.
        """
        weighted_scores = {
            cap: data["score"] * data["weight"]
            for cap, data in self.capabilities.items()
        }
        return min(weighted_scores.items(), key=lambda x: x[1])[0]

    def generate_report(self):
        """
        Generate a comprehensive report on capability status.
        """
        report = {
            "overall_status":
            self._calculate_overall_status(),
            "capability_scores": {
                cap: data["score"]
                for cap, data in self.capabilities.items()
            },
            "weakest_capabilities": [{
                "name": cap,
                "score": data["score"]
            } for cap, data in self.identify_weakest_capabilities()[:2]],
            "improvement_focus":
            self.get_improvement_focus(),
            "trend":
            self._analyze_trend()
        }
        return report

    def _calculate_overall_status(self):
        """Calculate an overall status score based on all capabilities."""
        weighted_sum = sum(data["score"] * data["weight"]
                           for data in self.capabilities.values())
        total_weight = sum(data["weight"]
                           for data in self.capabilities.values())
        return weighted_sum / total_weight if total_weight > 0 else 0

    def _analyze_trend(self):
        """Analyze the trend of capability scores over time."""
        if len(self.history) < 2:
            return "insufficient_data"

        # Compare the most recent two snapshots
        latest = self.history[-1]
        previous = self.history[-2]

        trends = {}
        for cap in self.capabilities:
            if cap in latest and cap in previous:
                diff = latest[cap] - previous[cap]
                if diff > 0.05:
                    trends[cap] = "improving"
                elif diff < -0.05:
                    trends[cap] = "declining"
                else:
                    trends[cap] = "stable"

        return trends


if __name__ == "__main__":
    pass
