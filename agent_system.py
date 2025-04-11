import os
import json
import time
import datetime
import traceback
import random
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from google import genai

class AgentSystem:
    """
    Agentic Learning System that uses LLM reasoning to continuously improve its approach
    to solving dataset problems through iterative exploration and exploitation.
    """

    def __init__(self, dataset_path: str = "calendar_scheduling.json", example_prefix: str = "calendar_scheduling_example_"):
        """Initialize the agent system"""
        # Initialize configuration
        self.explore_rate = 70
        self.exploit_rate = 30
        self.current_iteration = 0
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

        # Initialize Gemini API client
        try:
            self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            print("Gemini API client initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini API client: {e}")
            print("Make sure to set the GEMINI_API_KEY environment variable")
            raise

    def call_llm(self, prompt: str) -> str:
        """Call the Gemini LLM with a prompt and return the response"""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt
            )
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
            return {"samples": [], "new_examples_added": 0, "total_seen_examples": 0}

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
        summaries = self.get_summaries()
        summaries.append(new_summary)

        with open(self.archive_dir / "summaries.json", 'w', encoding='utf-8') as file:
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
                "iteration": iteration.get("iteration"),
                "batch_size": iteration.get("batch_size", 5),
                "accuracy": accuracy,
                "error_patterns": iteration.get("performance", {}).get("error_analysis", {}).get("error_patterns", [])
            })

        # Default response if no LLM available
        default_response = (self.current_batch_size, "Maintaining current batch size due to insufficient performance data")

        # If no performance history, just keep current batch size
        if not performance_history:
            return default_response

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
            response = self.call_llm(prompt)

            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.endswith("```"):
                response = response.split("```")[0]

            result = json.loads(response)

            # Validate and extract new batch size
            new_batch_size = int(result.get("new_batch_size", self.current_batch_size))

            # Ensure batch size is within reasonable limits
            new_batch_size = max(5, min(25, new_batch_size))

            return new_batch_size, result.get("rationale", "No rationale provided")
        except Exception as e:
            print(f"Error adjusting batch size: {e}")
            return default_response

    def adjust_explore_exploit_with_llm(self) -> Tuple[int, int]:
        """
        Use LLM reasoning to adjust the explore/exploit balance based on 
        performance history and current strategy.
        """
        iterations = self.get_all_iterations()
        summaries = self.get_summaries()

        # If there aren't enough iterations yet, make minimal adjustments
        if len(iterations) < 2:
            return self.explore_rate, self.exploit_rate

        # Prepare performance history for LLM reasoning
        performance_history = []
        for summary in summaries:
            performance_history.append({
                "iteration": summary.get("iteration"),
                "accuracy": summary.get("performance", {}).get("accuracy", 0),
                "batch_size": summary.get("batch_size", 5),
                "explore_rate": summary.get("explore_rate"),
                "exploit_rate": summary.get("exploit_rate")
            })

        # Create prompt for LLM to reason about explore/exploit adjustment
        prompt = f"""
        As an AI optimization system, you need to adjust the explore/exploit balance for 
        an iterative learning system. The system is currently solving problems with:

        Current explore rate: {self.explore_rate}%
        Current exploit rate: {self.exploit_rate}%
        Current batch size: {self.current_batch_size}
        Total examples seen: {len(self.seen_examples)}

        Here is the performance history:
        {json.dumps(performance_history, indent=2)}

        Based on this performance history, determine if and how the explore/exploit balance 
        should be adjusted. Consider:

        1. Performance trend (improving, stagnating, or declining)
        2. Size of performance changes between iterations
        3. Current balance of exploration vs exploitation
        4. Number of examples tested in each iteration

        Rules:
        - Explore + exploit must always sum to 100
        - If performance is significantly improving, favor exploitation
        - If performance is stagnating or declining, favor exploration
        - Make larger adjustments when the system is struggling
        - Make smaller adjustments when the system is performing well

        Return only a JSON object with two fields:
        {{"explore_rate": <new_explore_rate_as_integer>, "exploit_rate": <new_exploit_rate_as_integer>}}
        """

        # Call LLM to reason about adjustment
        try:
            response = self.call_llm(prompt)
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.endswith("```"):
                response = response.split("```")[0]

            result = json.loads(response)

            # Validate the result
            new_explore = int(result.get("explore_rate", self.explore_rate))
            new_exploit = int(result.get("exploit_rate", self.exploit_rate))

            # Ensure values are reasonable
            new_explore = max(10, min(90, new_explore))
            new_exploit = max(10, min(90, new_exploit))

            # Ensure they sum to 100
            if new_explore + new_exploit != 100:
                total = new_explore + new_exploit
                new_explore = int(round(new_explore * 100 / total))
                new_exploit = 100 - new_explore

            return new_explore, new_exploit
        except Exception as e:
            print(f"Error adjusting explore/exploit: {e}")
            # If there's an error, make a small adjustment favoring exploitation over time
            if self.explore_rate > 20:
                return self.explore_rate - 5, self.exploit_rate + 5
            return self.explore_rate, self.exploit_rate

    def generate_script_with_llm(self, is_exploration: bool) -> str:
        """
        Use the LLM to generate a script to solve the dataset problems,
        either exploring new approaches or exploiting/refining successful ones.
        """
        # Get previous iterations and samples
        iterations = self.get_all_iterations()
        samples_data = self.get_samples()
        samples = samples_data["samples"]

        # Extract example problems for context
        example_problems = []
        for i, sample in enumerate(samples):
            example_problems.append({
                "id": i,
                "question": sample.get("prompt_0shot", ""),
                "answer": sample.get("golden_plan", "")
            })

        # Prepare performance history context
        performance_context = ""
        if iterations:
            performance_context = "Previous iterations performance:\n"
            for iteration in iterations[-3:]:  # Just show the most recent 3
                performance_context += f"Iteration {iteration.get('iteration')}: Accuracy {iteration.get('performance', {}).get('accuracy', 0):.2f}, Approach: {iteration.get('approach_summary', 'No summary available')}\n"

        # Get successful scripts from past iterations
        successful_scripts = []
        if iterations:
            for iteration in sorted(iterations, key=lambda x: x.get('performance', {}).get('accuracy', 0), reverse=True)[:2]:
                if iteration.get('script') and iteration.get('performance', {}).get('accuracy', 0) > 0.5:
                    successful_scripts.append({
                        "iteration": iteration.get('iteration'),
                        "accuracy": iteration.get('performance', {}).get('accuracy', 0),
                        "script": iteration.get('script'),
                        "analysis": iteration.get('error_analysis', {})
                    })

        # Determine if this is exploration or exploitation
        approach_type = "exploration" if is_exploration else "exploitation"

        if is_exploration or not successful_scripts:
            # Exploration prompt: generate a novel approach
            prompt = f"""
            You are developing a Python script to solve dataset problems. You need to create a new approach that is different from previous attempts.

            Here are example problems from the dataset:
            {json.dumps(example_problems, indent=2)}

            {performance_context}

            For each problem, you need to analyze the input and generate the appropriate output.

            Since this is an {approach_type} phase, create a novel approach with different techniques than previous iterations.
            Be creative, innovative, and try something new.

            Return only a complete, runnable Python script that:
            1. Has a main function that takes a question string as input
            2. Returns the answer string
            3. Includes thorough error handling
            4. Uses well-structured, efficient code
            5. Comments explaining your approach and key functions

            The script should include no imports other than standard Python libraries.
            """
        else:
            # Exploitation prompt: refine the best existing approach
            best_script = successful_scripts[0]

            prompt = f"""
            You are improving a Python script that solves problems from a dataset. Your goal is to refine the current best approach to make it more accurate.

            Here are example problems from the dataset:
            {json.dumps(example_problems, indent=2)}

            The current best script (accuracy: {best_script['accuracy']:.2f}) is:

            ```python
            {best_script['script']}
            ```

            Error analysis from this script:
            {json.dumps(best_script.get('analysis', {}), indent=2)}

            {performance_context}

            Since this is an {approach_type} phase, your goal is to refine and improve this script while maintaining its core approach.
            Focus on fixing the specific issues identified in the error analysis and improving the handling of edge cases.

            Return only a complete, runnable Python script that:
            1. Has a main function that takes a question string as input
            2. Returns the answer string
            3. Includes thorough error handling
            4. Uses well-structured, efficient code
            5. Comments explaining your improvements

            The script should include no imports other than standard Python libraries.
            """

        # Call LLM to generate script
        response = self.call_llm(prompt)

        # Extract code block from response
        if "```python" in response:
            script = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            script = response.split("```")[1].split("```")[0].strip()
        else:
            script = response.strip()

        # Save the script to a file
        script_path = self.scripts_dir / f"script_iteration_{self.current_iteration}.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

        return script

    def execute_script(self, script: str, question: str) -> Dict:
        """
        Execute the generated script on a question and return the result.
        """
        # Create a temporary script file
        script_path = self.scripts_dir / f"current_script_{self.current_iteration}.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

        # Create a test harness for the script
        test_script = f"""
import sys
import traceback

# Add the scripts directory to the path
sys.path.append("{self.scripts_dir}")

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
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            # Parse the output
            output = result.stdout + result.stderr

            if "ANSWER_START" in output and "ANSWER_END" in output:
                answer = output.split("ANSWER_START")[1].split("ANSWER_END")[0].strip()
                return {
                    "success": True,
                    "answer": answer,
                    "output": output
                }
            elif "ERROR_START" in output and "ERROR_END" in output:
                error = output.split("ERROR_START")[1].split("ERROR_END")[0].strip()
                return {
                    "success": False,
                    "error": error,
                    "output": output
                }
            else:
                return {
                    "success": False,
                    "error": "Unknown execution error",
                    "output": output
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Script execution timed out (30 seconds)",
                "output": "Timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": traceback.format_exc()
            }

    def evaluate_with_llm(self, samples: List[Dict], results: List[Dict]) -> Dict:
        """
        Use the LLM to evaluate results and perform error analysis.
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
                    "match": False
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

            evaluations.append({
                "sample_id": i,
                "success": True,
                "system_answer": result.get("answer", "").strip(),
                "golden_answer": sample.get("golden_plan", "").strip(),
                "match": result.get("match", False),
                "evaluation": result.get("evaluation", {})
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
                    "error_message": eval_data.get("error", "")
                })

        error_analysis = {}
        if error_samples:
            # Create prompt for LLM to analyze errors
            prompt = f"""
            Analyze the errors in this system. You're given questions, the system's answers, and the correct answers.

            Here are the error cases:
            {json.dumps(error_samples, indent=2)}

            Identify patterns in these errors and categorize them. Consider issues like:
            1. Parsing problems (failing to extract relevant information)
            2. Logic errors in processing the input
            3. Output formatting issues
            4. Edge case handling

            For each error pattern you identify, suggest specific improvements to fix it.

            Return your analysis as a JSON object with these fields:
            1. "error_patterns": [List of identified error patterns]
            2. "primary_issue": The most critical issue to fix
            3. "recommendations": Specific technical recommendations to improve the system
            4. "root_causes": Underlying causes of the errors

            Format your response as a valid JSON object that can be parsed.
            """

            # Call LLM for error analysis
            try:
                response = self.call_llm(prompt)

                # Extract JSON from response
                response = response.strip()
                if response.startswith("```json"):
                    response = response.split("```json")[1]
                elif response.startswith("```"):
                    response = response.split("```")[1]

                if response.endswith("```"):
                    response = response.split("```")[0]

                error_analysis = json.loads(response)
            except Exception as e:
                print(f"Error in LLM error analysis: {e}")
                error_analysis = {
                    "error_patterns": ["Analysis failed"],
                    "primary_issue": "Unable to analyze errors with LLM",
                    "recommendations": ["Retry analysis in next iteration"],
                    "root_causes": ["LLM error analysis failed: " + str(e)]
                }

        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(samples),
            "evaluations": evaluations,
            "error_analysis": error_analysis
        }

    def evaluate_answer_with_llm(self, system_answer: str, golden_answer: str) -> Dict:
        """Use LLM to determine if answers are semantically equivalent"""
        prompt = f"""
        You're evaluating two answers to determine if they convey the same information.

        System answer: {system_answer}
        Golden answer: {golden_answer}

        Do these answers effectively communicate the same information, even if worded differently?
        Return only a JSON object with: {{"match": true/false, "confidence": 0-1, "explanation": "reason"}}
        """

        try:
            response = self.call_llm(prompt)

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
                "match": exact_match,
                "confidence": 1.0 if exact_match else 0.0,
                "explanation": f"Fallback to exact match comparison due to error: {str(e)}"
            }

        # For deeper analysis, use LLM to analyze error patterns
        error_samples = []
        for i, eval_data in enumerate(evaluations):
            if not eval_data.get("exact_match"):
                sample = samples[i]
                error_samples.append({
                    "sample_id": i,
                    "question": sample.get("prompt_0shot", ""),
                    "system_answer": eval_data.get("system_answer", ""),
                    "golden_answer": eval_data.get("golden_answer", ""),
                    "error_message": eval_data.get("error", "")
                })

        error_analysis = {}
        if error_samples:
            # Create prompt for LLM to analyze errors
            prompt = f"""
            Analyze the errors in this system. You're given questions, the system's answers, and the correct answers.

            Here are the error cases:
            {json.dumps(error_samples, indent=2)}

            Identify patterns in these errors and categorize them. Consider issues like:
            1. Parsing problems (failing to extract relevant information)
            2. Logic errors in processing the input
            3. Output formatting issues
            4. Edge case handling

            For each error pattern you identify, suggest specific improvements to fix it.

            Return your analysis as a JSON object with these fields:
            1. "error_patterns": [List of identified error patterns]
            2. "primary_issue": The most critical issue to fix
            3. "recommendations": Specific technical recommendations to improve the system
            4. "root_causes": Underlying causes of the errors

            Format your response as a valid JSON object that can be parsed.
            """

            # Call LLM for error analysis
            try:
                response = self.call_llm(prompt)

                # Extract JSON from response
                response = response.strip()
                if response.startswith("```json"):
                    response = response.split("```json")[1]
                elif response.startswith("```"):
                    response = response.split("```")[1]

                if response.endswith("```"):
                    response = response.split("```")[0]

                error_analysis = json.loads(response)
            except Exception as e:
                print(f"Error in LLM error analysis: {e}")
                error_analysis = {
                    "error_patterns": ["Analysis failed"],
                    "primary_issue": "Unable to analyze errors with LLM",
                    "recommendations": ["Retry analysis in next iteration"],
                    "root_causes": ["LLM error analysis failed: " + str(e)]
                }

        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(samples),
            "evaluations": evaluations,
            "error_analysis": error_analysis
        }

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
                progressive_accuracy = it["progressive_testing"].get("accuracy", None)

            iteration_data.append({
                "iteration": it.get("iteration"),
                "accuracy": it.get("performance", {}).get("accuracy", 0),
                "batch_size": it.get("batch_size", 5),
                "progressive_accuracy": progressive_accuracy,
                "approach": it.get("approach_summary", "Unknown approach"),
                "strategy": it.get("strategy", "Unknown")
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

            response = self.call_llm(prompt)

            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.endswith("```"):
                response = response.split("```")[0]

            result = json.loads(response)

            # Get detailed info about the best iteration
            best_iteration_number = int(result.get("best_iteration", -1))

            best_iteration = next((it for it in iterations 
                               if it.get("iteration") == best_iteration_number), None)

            if not best_iteration:
                # Fallback: just get the highest accuracy
                best_iteration = max(iterations, 
                                key=lambda x: x.get("performance", {}).get("accuracy", 0))

            return {
                "iteration": best_iteration.get("iteration"),
                "accuracy": best_iteration.get("performance", {}).get("accuracy", 0),
                "batch_size": best_iteration.get("batch_size", 5),
                "path": f"scripts/script_iteration_{best_iteration.get('iteration')}.py",
                "approach": best_iteration.get("approach_summary", ""),
                "rationale": result.get("rationale", "Highest overall accuracy")
            }
        except Exception as e:
            # Fallback method - don't use LLM, just pick highest accuracy
            print(f"Error determining best script with LLM: {e}")
            print("Using fallback method to determine best script")

            try:
                # Find the best iteration by accuracy
                best_iteration = max(iterations, 
                                key=lambda x: x.get("performance", {}).get("accuracy", 0) if x else 0)

                return {
                    "iteration": best_iteration.get("iteration"),
                    "accuracy": best_iteration.get("performance", {}).get("accuracy", 0),
                    "batch_size": best_iteration.get("batch_size", 5),
                    "path": f"scripts/script_iteration_{best_iteration.get('iteration')}.py",
                    "approach": best_iteration.get("approach_summary", ""),
                    "rationale": "Fallback selection based on highest accuracy"
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

    def generate_approach_summary(self, script: str) -> str:
        """
        Use the LLM to generate a brief summary of the approach used in the script.
        """
        prompt = f"""
        You're given a Python script that processes input and generates output.
        Provide a brief summary of the approach used in this script in 2-3 sentences.

        Focus on the key techniques, algorithms, or data structures used.

        Script:
        ```python
        {script}
        ```

        Return only the summary with no introduction or additional comments.
        """

        try:
            summary = self.call_llm(prompt)
            return summary.strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    def run_progressive_testing(self, script: str) -> Dict:
        """Run progressive testing on all seen examples"""
        # Load the dataset
        dataset = self.load_dataset()

        # Get all seen examples
        samples = []
        for example_key in self.seen_examples:
            if example_key in dataset:
                samples.append(dataset[example_key])

        if not samples:
            return {
                "success": False, 
                "error": "No examples seen yet"
            }

        print(f"Running progressive testing on {len(samples)} seen examples...")

        # Execute script on all samples
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
                evaluation = self.evaluate_answer_with_llm(system_answer, golden_answer)

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

    def validate_script(self, script_path: str = None, start_index: int = 0, end_index: int = 999) -> Dict:
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
            return {"success": False, "error": f"Error loading script: {str(e)}"}

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
            return {"success": False, "error": f"No examples found in range {start_index}-{end_index}"}

        print(f"Validating script on {len(samples)} examples from range {start_index}-{end_index}...")

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
                evaluation = self.evaluate_answer_with_llm(system_answer, golden_answer)

                result["evaluation"] = evaluation
                result["match"] = evaluation.get("match", False)
            else:
                result["match"] = False

            results.append({
                "key": sample["key"],
                "result": result
            })

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
        """Run a single iteration of the agent system"""
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

        # Decide whether to explore or exploit
        is_exploration = (self.explore_rate > self.exploit_rate) or (random.random() * 100 <= self.explore_rate)
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

        print(f"Performance: {accuracy:.2f} accuracy " +
              f"({matches}/{len(samples)} correct)")

        # Use LLM for deeper error analysis
        try:
            print("Performing error analysis with LLM...")
            evaluation = self.evaluate_with_llm(samples, results)

            if evaluation.get('error_analysis'):
                primary_issue = evaluation.get('error_analysis', {}).get('primary_issue', 'None')
                print(f"Primary issue identified: {primary_issue}")
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
                print("Script looks promising! Running progressive testing on all seen examples...")
                progressive_testing_results = self.run_progressive_testing(script)

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
            "execution_time": time.time() - iteration_start_time
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
            "new_batch_size": new_batch_size
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

        return iteration_data

if __name__ == "__main__":
    pass