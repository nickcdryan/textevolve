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

    def get_samples(self, n: int = 5) -> List[Dict]:
        """Get n samples from the dataset"""
        dataset = self.load_dataset()
        if not dataset:
            return []

        samples = []
        for i in range(n):
            example_key = f"{self.example_prefix}{i}"
            if example_key in dataset:
                samples.append(dataset[example_key])

        return samples

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
                "explore_rate": summary.get("explore_rate"),
                "exploit_rate": summary.get("exploit_rate")
            })

        # Create prompt for LLM to reason about explore/exploit adjustment
        prompt = f"""
        As an AI optimization system, you need to adjust the explore/exploit balance for 
        an iterative learning system. The system is currently solving problems with:

        Current explore rate: {self.explore_rate}%
        Current exploit rate: {self.exploit_rate}%

        Here is the performance history:
        {json.dumps(performance_history, indent=2)}

        Based on this performance history, determine if and how the explore/exploit balance 
        should be adjusted. Consider:

        1. Performance trend (improving, stagnating, or declining)
        2. Size of performance changes between iterations
        3. Current balance of exploration vs exploitation

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
        samples = self.get_samples(5)

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

        # First, perform basic exact match evaluation
        correct_count = 0
        for i, (sample, result) in enumerate(zip(samples, results)):
            if not result.get("success"):
                evaluations.append({
                    "sample_id": i,
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "exact_match": False
                })
                continue

            # Compare with golden answer
            golden_answer = sample.get("golden_plan", "").strip()
            system_answer = result.get("answer", "").strip()
            exact_match = golden_answer == system_answer

            if exact_match:
                correct_count += 1

            evaluations.append({
                "sample_id": i,
                "success": True,
                "system_answer": system_answer,
                "golden_answer": golden_answer,
                "exact_match": exact_match
            })

        # Calculate accuracy
        accuracy = correct_count / len(samples) if samples else 0

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

    def run_iteration(self) -> Dict:
        """Run a single iteration of the agent system"""
        print(f"\n=== Starting Iteration {self.current_iteration} ===")
        print(f"Current explore/exploit balance: {self.explore_rate}/{self.exploit_rate}")

        iteration_start_time = time.time()

        # Get samples from dataset
        samples = self.get_samples(5)
        if not samples:
            print("No samples available in dataset. Exiting iteration.")
            return {"success": False, "error": "No samples available"}

        # Decide whether to explore or exploit
        is_exploration = (self.explore_rate > self.exploit_rate) or (random.random() * 100 <= self.explore_rate)
        strategy = "Exploration" if is_exploration else "Exploitation"
        print(f"Strategy for this iteration: {strategy}")

        # Generate script using LLM
        print("Generating script with LLM...")
        script = self.generate_script_with_llm(is_exploration)

        # Generate a summary of the approach
        approach_summary = self.generate_approach_summary(script)
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
            else:
                print(f"    Error: {result.get('error')}")

            results.append(result)

        # Evaluate results with LLM
        print("Evaluating results with LLM...")
        evaluation = self.evaluate_with_llm(samples, results)

        print(f"Performance: {evaluation.get('accuracy', 0):.2f} accuracy " +
              f"({evaluation.get('correct_count', 0)}/{evaluation.get('total_count', 0)} correct)")

        if evaluation.get('error_analysis'):
            print("Primary issue identified:", evaluation.get('error_analysis', {}).get('primary_issue', 'None'))

        # Adjust explore/exploit balance for next iteration
        print("Adjusting explore/exploit balance...")
        new_explore, new_exploit = self.adjust_explore_exploit_with_llm()

        print(f"New explore/exploit balance: {new_explore}/{new_exploit}")

        # Prepare iteration data
        iteration_data = {
            "iteration": self.current_iteration,
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy": strategy,
            "explore_rate": self.explore_rate,
            "exploit_rate": self.exploit_rate,
            "script": script,
            "approach_summary": approach_summary,
            "sample_count": len(samples),
            "results": results,
            "performance": evaluation,
            "execution_time": time.time() - iteration_start_time
        }

        # Create summary
        summary = {
            "iteration": self.current_iteration,
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy": strategy,
            "explore_rate": self.explore_rate,
            "exploit_rate": self.exploit_rate,
            "approach_summary": approach_summary,
            "performance": {
                "accuracy": evaluation.get("accuracy", 0),
                "correct_count": evaluation.get("correct_count", 0),
                "total_count": evaluation.get("total_count", 0)
            },
            "primary_issue": evaluation.get("error_analysis", {}).get("primary_issue", "None identified"),
            "new_explore_rate": new_explore,
            "new_exploit_rate": new_exploit
        }

        # Save to archive
        self.save_to_archive(iteration_data, f"iteration_{self.current_iteration}.json")
        self.update_summaries(summary)

        # Update explore/exploit rates for next iteration
        self.explore_rate = new_explore
        self.exploit_rate = new_exploit

        # Increment iteration counter
        self.current_iteration += 1

        print(f"=== Completed Iteration {self.current_iteration - 1} ===")

        return iteration_data

if __name__ == "__main__":
    pass