"""
LLM prompting patterns and techniques.


CONTAINS:

as_example_code
extract_information_with_examples
verify_solution_with_examples
solve_with_validation_loop
best_of_n
solve_with_react_pattern

chain_of_thought_reasoning
verification_with_feedback
multi_perspective_analysis
self_consistency_approach
pattern_identification
wait_injection
solve_with_meta_programming
self_modifying_solver
debate_approach
adaptive_chain_solver
dynamic_memory_pattern
test_time_training
combination_example

"""


import inspect
from typing import List, Dict

# Helper function to convert any function to source code
def as_example_code(func):
    """Convert a function to its source code string for prompt inclusion"""
    return inspect.getsource(func)



def extract_information_with_examples(text):
    """Extract key information from the input text using embedded examples."""
    system_instruction = "You are an information extraction specialist focusing on identifying key entities and relationships."
    
    prompt = f"""
    Extract key information from this text. Focus on identifying all entities, relationships, and important attributes.
    
    Example usage:
    
    Input Text:
    The company XYZ Corp reported quarterly earnings of $3.5 million, which represents a 12% increase from last year. The CEO, Jane Smith, attributed this growth to their new product line launched in March, which has already captured 8% of the market share. They expect to expand their operations to Europe by Q2 2023.
    
    Let's think step by step.
    
    The key entities are:
    - XYZ Corp (company)
    - Jane Smith (person, CEO)
    - New product line (product)
    
    The key information points are:
    - Financial: Quarterly earnings of $3.5 million
    - Performance: 12% increase from previous year
    - Product: New product line launched in March
    - Market: 8% market share for new product
    - Plans: Expansion to Europe by Q2 2023
    
    Extracted Information:
    {{
      "entities": [
        {{"name": "XYZ Corp", "type": "company"}},
        {{"name": "Jane Smith", "type": "person", "role": "CEO"}},
        {{"name": "New product line", "type": "product", "launch_date": "March"}}
      ],
      "financial_data": {{
        "quarterly_earnings": "$3.5 million",
        "growth_rate": "12%"
      }},
      "market_data": {{
        "product_market_share": "8%"
      }},
      "future_plans": [
        {{"type": "expansion", "region": "Europe", "timeline": "Q2 2023"}}
      ]
    }}
    
    Now, extract information from this new text:
    {text}
    """
    
    return call_llm(prompt, system_instruction)



def verify_solution_with_examples(problem, proposed_solution):
    """Verify if the proposed solution satisfies all requirements using embedded examples."""
    system_instruction = "You are a critical evaluator who verifies if solutions correctly address problems."
  
    prompt = f"""
    Verify if this proposed solution correctly addresses all aspects of the problem.
  
    Example usage:
  
    Problem:
    Design a data structure that can efficiently perform the following operations:
    1. Insert a value
    2. Delete a value
    3. Get a random value with equal probability for all stored values
    All operations should have average time complexity of O(1).
  
    Proposed Solution:
    I'll use a combination of a hashmap and an array. The hashmap will store the value as the key and its index in the array as the value. The array will store all the inserted values.
  
    For insert: Add the value to the end of the array and update the hashmap with the value and its index. O(1) time.
  
    For delete: Look up the index of the value in the hashmap, swap the value with the last element in the array, update the hashmap for the swapped element, remove the last element from the array, and remove the value from the hashmap. O(1) time.
  
    For get random: Generate a random index within the array's bounds and return the value at that index. O(1) time.
  
    Verification:
    Let me check each requirement:
    1. Insert operation: The solution adds the value to the end of the array and updates the hashmap with O(1) time complexity ✓
    2. Delete operation: The solution uses the hashmap to find the index, then swaps with the last element and updates accordingly with O(1) time complexity ✓
    3. Get random operation: The solution generates a random index within the array bounds with O(1) time complexity ✓
    4. All operations have O(1) average time complexity ✓
  
    Result: VALID - The solution correctly addresses all requirements with the specified time complexity.
  
    Problem:
    {problem}
  
    Proposed Solution:
    {proposed_solution}
  
    Verification:
    """
  
    return call_llm(prompt, system_instruction)

def solve_with_validation_loop(problem, max_attempts=3):
    """Solve a problem with iterative refinement through validation feedback loop."""
    system_instruction_solver = "You are an expert problem solver who creates detailed, correct solutions."
    system_instruction_validator = "You are a critical validator who carefully checks solutions against all requirements."
  
    # Initial solution generation
    solution_prompt = f"""
    Provide a detailed solution to this problem. Be thorough and ensure you address all requirements.
  
    Problem:
    {problem}
    """
  
    solution = call_llm(solution_prompt, system_instruction_solver)
  
    # Validation loop
    for attempt in range(max_attempts):
        # Validate the current solution
        validation_prompt = f"""
        Carefully validate if this solution correctly addresses all aspects of the problem.
        If the solution is valid, respond with "VALID: [brief reason]".
        If the solution has any issues, respond with "INVALID: [detailed explanation of issues]".
  
        Problem:
        {problem}
  
        Proposed Solution:
        {solution}
        """
  
        validation_result = call_llm(validation_prompt, system_instruction_validator)
  
        # Check if solution is valid
        if validation_result.startswith("VALID:"):
            return solution
  
        # If invalid, refine the solution
        refined_prompt = f"""
        Your previous solution to this problem has some issues that need to be addressed.
  
        Problem:
        {problem}
  
        Your previous solution:
        {solution}
  
        Validation feedback:
        {validation_result}
  
        Please provide a completely revised solution that addresses all the issues mentioned.
        """
  
        solution = call_llm(refined_prompt, system_instruction_solver)
  
    return solution



def best_of_n(prompt, system_instruction="", n=5):
    """
    Calls an LLM n times to solve a task, and selects the best answer using another LLM call.
    The second LLM call performs majority voting if possible, or else selects the best candidate.

    Args:
        prompt (str): The task or question to solve.
        system_instruction (str): Optional system-level instruction for initial answer generation.
        n (int): Number of candidate answers to generate.

    Returns:
        str: The selected "best" answer.
    """

    # Step 1: Generate n candidate answers
    candidates = [call_llm(prompt, system_instruction) for _ in range(n)]

    # Step 2: Ask another LLM to pick the best based on majority or quality
    decision_prompt = f"""
    You are given a question and several answers generated by different LLM runs.
    
    Your task:
    - First, check if a majority of answers are identical or very similar. If so, return that common answer.
    - If there is no clear majority, then choose the answer that is most helpful, complete, and correct.
    
    Question:
    {prompt}
    
    Candidate Answers:
    {chr(10).join(f"{i+1}. {ans}" for i, ans in enumerate(candidates))}
    
    Return only the single best answer, verbatim.
    """.strip()

    return call_llm(decision_prompt, system_instruction="You are an expert answer selector.")



def solve_with_react_pattern(problem):
    """Solve problems through iterative Reasoning and Acting (ReAct) approach."""
    system_instruction = "You are a problem-solving agent that follows the ReAct pattern: Reason about the current state, take an Action, observe the result, and repeat until reaching a solution."



    # Initialize ReAct process
    prompt = f"""
    Solve this problem using the ReAct pattern - alternate between Reasoning and Acting until you reach a final answer.

    Example usage:

    Problem: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?

    Thought 1: I need to determine which country the Great Barrier Reef is in, then find its capital, and finally the population of that capital.
    Action 1: Search[Great Barrier Reef location]
    Observation 1: The Great Barrier Reef is located off the coast of Queensland in northeastern Australia.

    Thought 2: Now I know the Great Barrier Reef is in Australia. I need to find Australia's capital city.
    Action 2: Search[capital of Australia]
    Observation 2: The capital of Australia is Canberra.

    Thought 3: Now I need to find the population of Canberra.
    Action 3: Search[population of Canberra]
    Observation 3: As of 2021, the population of Canberra is approximately 431,500.

    Thought 4: I have found all the required information. The capital of Australia (where the Great Barrier Reef is located) is Canberra, and its population is approximately 431,500.
    Action 4: Finish[The capital of Australia is Canberra, with a population of approximately 431,500.]

    Now solve this new problem:
    {problem}

    Start with Thought 1:
    """

    # Initial reasoning and action planning
    react_response = call_llm(prompt, system_instruction)

    # Extract the action from the response
    action = extract_action(react_response)

    # Continue the ReAct loop until we reach a "Finish" action
    while not action["type"] == "Finish":
        # Perform the requested action and get an observation
        if action["type"] == "Search":
            observation = perform_search(action["query"])
        elif action["type"] == "Calculate":
            observation = perform_calculation(action["expression"])
        elif action["type"] == "Lookup":
            observation = perform_lookup(action["term"])
        else:
            observation = f"Unknown action type: {action['type']}"

        # Continue the ReAct process with the new observation
        continuation_prompt = f"""
        {react_response}
        Observation {action["step_number"]}: {observation}

        Continue with the next thought and action:
        """

        # Get the next reasoning step and action
        react_response += "\n" + call_llm(continuation_prompt, system_instruction)

        # Extract the next action
        action = extract_action(react_response)

    def extract_action(text):
        """Parse the ReAct response to extract the current action."""
        # Find the last action in the text
        action_matches = re.findall(r"Action (\d+): (\w+)\[(.*?)\]", text)
        if not action_matches:
            return {"type": "Error", "step_number": 0, "query": "No action found"}

        # Get the most recent action
        last_action = action_matches[-1]
        step_number = int(last_action[0])
        action_type = last_action[1]
        action_content = last_action[2]

        # Handle different action types
        if action_type == "Finish":
            return {"type": "Finish", "step_number": step_number, "answer": action_content}
        elif action_type in ["Search", "Lookup", "Calculate"]:
            return {"type": action_type, "step_number": step_number, "query": action_content}
        else:
            return {"type": "Unknown", "step_number": step_number, "query": action_content}

    def perform_search(query):
        """Simulate a search action in the ReAct pattern."""
        # In a real implementation, this would call an actual search API
        return call_llm(f"Provide a factual answer about: {query}", "You are a helpful search engine that provides concise, factual information.")

    def perform_calculation(expression):
        """Perform a calculation action in the ReAct pattern."""
        try:
            # Safely evaluate the expression
            result = eval(expression, {"__builtins__": {}}, {"math": math})
            return f"The result is {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"

    def perform_lookup(term):
        """Simulate a lookup action for specific information."""
        # In a real implementation, this would query a knowledge base or database
        return call_llm(f"Provide specific information about: {term}", "You are a knowledge base that provides specific factual information.")

    # Extract the final answer from the Finish action
    final_answer = action["answer"]
    return final_answer


def chain_of_thought_reasoning(problem: str) -> str:
    """
    Solve a problem using step-by-step reasoning.

    Uses a single example to demonstrate the chain-of-thought approach.
    """
    system_instruction = "You are an expert problem solver who breaks down problems step-by-step."

    prompt = f"""
    Solve this problem step-by-step:

    Example:
    Problem: If John has 5 apples and gives 2 to Mary, then buys 3 more, how many apples does John have?

    Step 1: Start with John's initial apples: 5 apples
    Step 2: Subtract the apples given to Mary: 5 - 2 = 3 apples
    Step 3: Add the newly purchased apples: 3 + 3 = 6 apples
    Therefore: John has 6 apples.

    Problem: {problem}

    Let's solve this step-by-step:
    """

    return call_llm(prompt, system_instruction)



def verification_with_feedback(problem: str, solution: str, max_attempts: int = 3) -> str:
    """
    Verify a solution and provide feedback for improvement.
    Uses moderate number of examples (2) to demonstrate verification criteria.
    """
    system_instruction = "You are a critical evaluator who verifies solutions and provides detailed feedback."

    # Initial verification with two examples
    verification_prompt = f"""
    Verify if this solution correctly addresses the problem:

    Example 1:
    Problem: Calculate the area of a rectangle with length 5m and width 3m.
    Solution: The area is 5m × 3m = 15m².
    Verification: VALID - The solution correctly calculates the area by multiplying length by width.

    Example 2:
    Problem: Find the next number in the sequence: 2, 4, 8, 16, ...
    Solution: The next number is 32 because each number is multiplied by 3.
    Verification: INVALID - The pattern is that each number is multiplied by 2, not 3. The correct next number is 32.

    Problem: {problem}
    Solution: {solution}

    Verify if the solution is valid and complete. Return:
    - "VALID: [brief explanation]" if the solution is correct
    - "INVALID: [detailed explanation of issues]" if there are any problems
    """

    verification_result = call_llm(verification_prompt, system_instruction)

    # Check if refinement is needed
    if "INVALID" in verification_result and max_attempts > 1:
        refinement_prompt = f"""
        Your solution needs improvement:

        Problem: {problem}

        Your solution:
        {solution}

        Feedback:
        {verification_result}

        Please provide a revised solution that addresses all the issues mentioned.
        """

        improved_solution = call_llm(refinement_prompt, system_instruction)

        # Recursive call with one fewer attempt
        return verification_with_feedback(problem, improved_solution, max_attempts - 1)

    return solution if "VALID" in verification_result else verification_result + "\n\n" + solution

def multi_perspective_analysis(problem: str, perspectives: List[str] = None) -> str:
    """
    Analyze a problem from multiple perspectives, with examples for only the first perspective.

    This demonstrates varying example usage - the first perspective has 2 examples,
    while others have none to show how to vary example density.
    """
    system_instruction = "You are an analytical thinker who can examine problems from diverse perspectives."

    if perspectives is None:
        perspectives = ["logical", "creative", "critical"]

    analyses = []

    # First perspective uses examples
    first_perspective = perspectives[0]
    first_perspective_prompt = f"""
    Analyze this problem from a {first_perspective} perspective:

    Example 1:
    Problem: A city is experiencing increasing traffic congestion.
    {first_perspective.capitalize()} perspective: This appears to be a resource allocation problem. We need to quantify current road capacity, traffic flow rates, peak usage times, and alternative route availability. With this data, we can identify bottlenecks and evaluate solutions like traffic light optimization, lane adjustments, or public transportation improvements.

    Example 2:
    Problem: A company's sales have declined for three consecutive quarters.
    {first_perspective.capitalize()} perspective: We should analyze the sales data by product line, region, and customer segment to identify specific decline patterns. We should compare against market trends, competitor performance, and economic indicators to determine internal versus external factors. Each potential cause should be tested against available evidence.

    Problem: {problem}

    Provide a thorough {first_perspective} perspective:
    """

    analyses.append({
        "perspective": first_perspective,
        "analysis": call_llm(first_perspective_prompt, system_instruction)
    })

    # Other perspectives don't use examples - demonstrating variation
    for perspective in perspectives[1:]:
        perspective_prompt = f"""
        Analyze this problem from a {perspective} perspective:

        Problem: {problem}

        Focus on aspects that a {perspective} thinker would notice.
        Provide a thorough {perspective} perspective:
        """

        analyses.append({
            "perspective": perspective,
            "analysis": call_llm(perspective_prompt, system_instruction)
        })

    # Synthesize the perspectives
    synthesis_prompt = f"""
    Synthesize these different perspectives into a comprehensive analysis:

    Problem: {problem}

    Perspectives:
    {chr(10).join([f"{p['perspective'].capitalize()} Perspective: {p['analysis']}" for p in analyses])}

    Create a unified analysis that incorporates insights from all perspectives.
    """

    return call_llm(synthesis_prompt, system_instruction)

def self_consistency_approach(problem: str, n_paths: int = 3) -> str:
    """
    Generate multiple reasoning paths and select the most consistent answer.

    Uses a moderate number of examples (2) to demonstrate the approach.
    """
    system_instruction = "You are a thorough problem solver who considers multiple approaches."

    # Generate multiple reasoning paths
    reasoning_paths = []

    # First path with examples
    first_path_prompt = f"""
    Solve this problem step by step:

    Example 1:
    Problem: If a train travels at 60 mph, how long will it take to travel 150 miles?
    Reasoning Path 1:
    Step 1: Identify the formula relating distance, speed, and time: time = distance ÷ speed
    Step 2: Substitute the values: time = 150 miles ÷ 60 mph
    Step 3: Calculate: time = 2.5 hours
    Therefore, it will take 2.5 hours to travel 150 miles.

    Example 2:
    Problem: What is the value of 3x + 5 = 20?
    Reasoning Path 1:
    Step 1: Subtract 5 from both sides: 3x = 15
    Step 2: Divide both sides by 3: x = 5
    Therefore, x = 5.

    Problem: {problem}

    Show your step-by-step reasoning to solve this problem:
    """

    reasoning_paths.append(call_llm(first_path_prompt, system_instruction))

    # Generate additional paths with fewer examples
    for i in range(1, n_paths):
        path_prompt = f"""
        Solve this problem using a different approach than before:

        Problem: {problem}

        Show your step-by-step reasoning using a unique approach:
        """

        reasoning_paths.append(call_llm(path_prompt, system_instruction))

    # Extract answers from each path
    answers = []
    for i, path in enumerate(reasoning_paths):
        extract_prompt = f"""
        Extract the final numerical or categorical answer from this reasoning:

        {path}

        Provide ONLY the final answer, with no explanation:
        """

        answers.append({
            "path_index": i,
            "reasoning": path,
            "answer": call_llm(extract_prompt, "Extract only the final answer.")
        })

    # Determine the most consistent answer
    newline = chr(10)
    consistency_prompt = f"""
    These are different approaches to solving the same problem:

    Problem: {problem}

    {chr(10).join([f"Approach {a['path_index']+1}:{newline}Reasoning: {a['reasoning']}{newline}Answer: {a['answer']}" for a in answers])}

    Which answer is most consistent across approaches? If there's disagreement, which reasoning path is most sound?
    Provide the final answer with explanation.
    """

    return call_llm(consistency_prompt, system_instruction)





def pattern_identification(examples: List[str], domain: str = "general") -> str:
    """
    Identify patterns across multiple examples.

    Uses a varying number of examples in the prompt based on domain.
    """
    system_instruction = "You are a pattern recognition specialist."

    # Format the user-provided examples
    newline = chr(10)
    formatted_examples = "{newline}".join([f"Example {i+1}:{newline}{ex}" for i, ex in enumerate(examples)])

    # Domain-specific patterns with varying example counts
    if domain == "sequence":
        prompt = f"""
        Examine these examples and identify underlying sequence patterns:

        Example Set 1:
        Sequence: 2, 4, 8, 16, 32, ...
        Pattern: Each number is multiplied by 2 to get the next number.

        Example Set 2:
        Sequence: 3, 6, 11, 18, 27, ...
        Pattern: The differences between consecutive numbers form an arithmetic sequence: 3, 5, 7, 9, ...

        Example Set 3:
        Sequence: 1, 4, 9, 16, 25, ...
        Pattern: These are perfect squares: 1², 2², 3², 4², 5², ...

        Your examples:
        {formatted_examples}

        Identify all possible patterns in these examples. For each pattern:
        1. Describe the pattern precisely
        2. Show how it applies to each example
        3. Predict the next items if the pattern continues
        """
    elif domain == "visual":
        prompt = f"""
        Examine these visual examples and identify underlying patterns:

        Example Set:
        Example 1: A triangle inside a circle
        Example 2: A square inside a circle
        Example 3: A pentagon inside a circle
        Pattern: Increasing number of sides for the shape inside the circle

        Your examples:
        {formatted_examples}

        Identify all possible visual patterns. For each pattern:
        1. Describe the pattern precisely
        2. Show how it applies to each example
        3. Predict what would come next in the pattern
        """
    else:  # general domain with single example
        prompt = f"""
        Examine these examples and identify all underlying patterns:

        Example Set:
        Items: Apple, Banana, Cherry, Date, Fig
        Pattern: Alphabetical order of fruit names

        Your examples:
        {formatted_examples}

        Identify all possible patterns. For each pattern:
        1. Describe the pattern precisely
        2. Show how it applies to each example
        3. Explain why this pattern is significant
        4. Predict the next items if the pattern continues
        """

    return call_llm(prompt, system_instruction)

def wait_injection(problem: str) -> str:
    """
    Use the 'wait' injection technique to improve reasoning.

    Inserting 'wait' and other intervention phrases is shown to encourage LLMs to reconsider 
    their initial conclusions and pursue new lines of thought
    """
    system_instruction = "You are a careful problem solver."

    # Get initial reasoning
    initial_prompt = f"""
    Solve this problem step by step:
    {problem}
    """

    initial_reasoning = call_llm(initial_prompt, system_instruction)

    # Find a good injection point - around 50-70% through the reasoning
    words = initial_reasoning.split()
    injection_point = len(words) // 2

    # Create parts for injection
    first_part = " ".join(words[:injection_point])

    # Inject wait and reconsideration
    wait_prompt = f"""
    Solve this problem step by step:
    {problem}

    {first_part}

    ...wait... let me reconsider this...
    """

    return call_llm(wait_prompt, system_instruction)



def solve_with_meta_programming(question):
    """
    Advanced: Script generates and executes its own code/prompts dynamically.
    The script acts as its own programmer and prompt engineer.
    """
    
    # Step 1: Analyze what approach is needed
    strategy_prompt = f"""
    For this problem: {question}
    
    What's the best approach?
    A) Generate Python code to calculate/process something
    B) Generate specialized LLM prompts for analysis  
    C) Use a hybrid approach with both code and LLM calls
    
    Explain your choice and what specific code or prompts I should generate.
    """
    
    
    analysis_system_prompt = """ 
    You are a problem analysis expert. You are a master of problem analysis and can 
    determine the best approach to solve a problem, understanding the strenghts and 
    weaknesses of LLMs for problem solving, when to delegate a more specific or problem 
    or subproblem to an additional LLM call, and when to write code to solve a problem.
    """
    strategy = call_llm(strategy_prompt, analysis_system_prompt)
    
    # Step 2: Generate and execute based on strategy
    if "###CODE_ONLY###" in strategy.lower():
        # Generate code dynamically
        code_gen_prompt = f"""
        Problem: {question}
        Strategy: {strategy}
        
        Write Python code to solve this problem. Include print statements for output.
        Return ONLY the Python code:
        """
        
        generated_code = call_llm(code_gen_prompt, "You are a Python programmer.")
        
        # Clean up code if wrapped in markdown
        import re
        code_match = re.search(r'```python\\s*\\n(.*?)\\n```', generated_code, re.DOTALL)
        if code_match:
          clean_code = code_match.group(1).strip()
        else:
          clean_code = generated_code.strip()
        
        # Execute the generated code
        execution_result = execute_code(clean_code)
        
        # Interpret the execution result
        interpretation_prompt = f"""
        Original problem: {question}
        Generated code: {clean_code}
        Execution result: {execution_result}
        
        What is the final answer based on these results?
        """
        
        final_answer = call_llm(interpretation_prompt, "You are a solution interpreter.")
        return final_answer
    
    elif "###PROMPT_ONLY###" in strategy.lower():
        # Generate specialized prompts dynamically
        prompt_design = f"""
        For this problem: {question}
        Strategy: {strategy}
        
        Design the most effective prompt to solve this problem:
        """
        
        specialized_prompt = call_llm(prompt_design, "You are a prompt engineer.")
        
        # Use the generated prompt
        solution = call_llm(specialized_prompt, "You are an expert problem solver.")
        return solution
    
    else:  # Hybrid approach
        # Chain code and LLM calls dynamically
        current_result = question
        
        for step in range(3):
            # Decide what to do at this step
            step_decision = call_llm(f"""
            Step {step + 1} of hybrid approach.
            Current state: {current_result}
            
            What should I do next?
            - Generate and execute code
            - Make an LLM analysis call
            - Provide final answer
            
            Choose one and explain exactly what to do.
            """, "You are a workflow coordinator.")
            
            if "final answer" in step_decision.lower():
                return current_result
            elif "code" in step_decision.lower():
                # Generate code for this step
                step_code_prompt = f"""
                Based on this decision: {step_decision}
                Current data: {current_result}
                
                Write Python code to process this. Return only the code:
                """
                step_code = call_llm(step_code_prompt, "You are a Python programmer.")
                code_result = execute_code(step_code)
                current_result = f"Previous: {current_result}\\nCode result: {code_result}"
            else:
                # Make LLM call for this step  
                step_analysis = call_llm(f"Analyze this data: {current_result}\\nBased on: {step_decision}", "You are an analyst.")
                current_result = f"Previous: {current_result}\\nAnalysis: {step_analysis}"
        
        return current_result



def self_modifying_solver(problem):
    """
    A solver that rewrites its own approach based on intermediate results.
    Advanced meta-programming where the script evolves its strategy.
    """

    strategy = "direct_analysis"
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        attempts += 1

        if strategy == "direct_analysis":
            # Try direct LLM analysis
            result = call_llm(f"Solve this problem: {problem}", "You are an expert problem solver.")

            # Evaluate if this worked
            evaluation_prompt = f"""
            Problem: {problem}
            My attempt: {result}

            Did this solve the problem correctly? If not, what approach should I try next?
            Options: computational_approach, step_by_step_breakdown, code_generation
            """

            evaluation = call_llm(evaluation_prompt, "You are a solution evaluator.")

            if "correct" in evaluation.lower() or "solved" in evaluation.lower():
                return result
            elif "computational" in evaluation.lower():
                strategy = "computational_approach"
            elif "step_by_step" in evaluation.lower():
                strategy = "step_by_step_breakdown"  
            else:
                strategy = "code_generation"

        elif strategy == "computational_approach":
            # Generate and execute computational code
            comp_prompt = f"""
            Problem: {problem}

            Write Python code to solve this computationally. Include:
            - Extract relevant numbers or data
            - Perform calculations
            - Print results clearly

            Return only the Python code:
            """

            comp_code = call_llm(comp_prompt, "You are a computational programmer.")
            comp_result = execute_code(comp_code)

            # Interpret computational result
            interpretation = call_llm(f"Problem: {problem}\\nComputation result: {comp_result}\\nFinal answer:", "You are an interpreter.")
            return interpretation

        elif strategy == "step_by_step_breakdown":
            # Generate step-by-step solution code
            breakdown_prompt = f"""
            Problem: {problem}

            Write Python code that breaks this problem into steps and solves it methodically:
            """

            breakdown_code = call_llm(breakdown_prompt, "You are a systematic programmer.")
            breakdown_result = execute_code(breakdown_code)

            # Build final solution based on breakdown
            final_solution = call_llm(f"Problem: {problem}\\nStep-by-step result: {breakdown_result}\\nFinal answer:", "You are a problem solver.")
            return final_solution

        else:  # code_generation strategy
            # Generate completely custom code for this problem
            custom_prompt = f"""
            Problem: {problem}

            Write custom Python code specifically designed to solve this exact problem type:
            """

            custom_code = call_llm(custom_prompt, "You are a custom code generator.")
            custom_result = execute_code(custom_code)

            return f"Custom solution result: {custom_result}"

    return "Could not solve after multiple strategy attempts"



def debate_approach(problem: str) -> str:
    """
    Simulate a debate between different viewpoints to explore a problem.

    Uses no explicit examples to demonstrate minimal example case.
    """
    system_instruction = "You can simulate a productive debate between different perspectives."

    # Generate initial position
    position_prompt = f"""
    Provide a solution to this problem:

    Problem: {problem}

    Offer a clear, well-reasoned solution approach.
    """

    initial_solution = call_llm(position_prompt, system_instruction)

    # Generate critique
    critique_prompt = f"""
    Critique this solution:

    Problem: {problem}

    Proposed solution:
    {initial_solution}

    Identify specific weaknesses, overlooked considerations, or potential issues with this approach.
    """

    critique = call_llm(critique_prompt, "You are a critical evaluator.")

    # Generate defense/refinement
    defense_prompt = f"""
    Respond to this critique of your solution:

    Problem: {problem}

    Your solution:
    {initial_solution}

    Critique:
    {critique}

    Either defend your approach or refine it to address the valid points in the critique.
    """

    defense = call_llm(defense_prompt, system_instruction)

    # Generate synthesis
    synthesis_prompt = f"""
    Based on this debate:

    Problem: {problem}

    Initial solution:
    {initial_solution}

    Critique:
    {critique}

    Defense/refinement:
    {defense}

    Provide an improved solution that incorporates valid points from both sides of the debate.
    """

    return call_llm(synthesis_prompt, system_instruction)


def adaptive_chain_solver(question):
    """
    Chains multiple code generations and LLM calls adaptively.
    Each step decides what the next step should be.
    """

    current_data = question
    step_count = 0
    max_steps = 5

    while step_count < max_steps:
        step_count += 1

        # Decide what to do at this step
        decision_prompt = f"""
        Step {step_count}: Working with: {current_data}

        What should I do next to solve this problem?
        A) Generate and execute Python code to process/calculate something
        B) Generate a specialized LLM prompt for analysis
        C) I have enough information - provide final answer

        Choose A, B, or C and explain exactly what to do:
        """

        decision = call_llm(decision_prompt, "You are an adaptive workflow coordinator.")

        if "C)" in decision or "final answer" in decision.lower():
            # Generate final answer
            final_prompt = f"""
            Original question: {question}
            Current data/results: {current_data}

            Based on all the processing done, what is the final answer?
            """
            return call_llm(final_prompt, "You are a solution synthesizer.")

        elif "A)" in decision or "code" in decision.lower():
            # Generate and execute code
            code_prompt = f"""
            Current data: {current_data}
            Decision: {decision}

            Write Python code to process this data as suggested. Return only the code:
            """

            code = call_llm(code_prompt, "You are a Python programmer.")

            # Execute and update current data
            code_result = execute_code(code)
            current_data = f"Step {step_count} result: {code_result}"

        else:  # Generate specialized LLM prompt
            # Create specialized prompt
            prompt_design = f"""
            Current data: {current_data}
            Decision: {decision}

            Design a specialized prompt for this analysis:
            """

            specialized_prompt = call_llm(prompt_design, "You are a prompt engineer.")

            # Use the specialized prompt
            analysis_result = call_llm(specialized_prompt, "You are a specialized analyst.")
            current_data = f"Step {step_count} analysis: {analysis_result}"

    return f"Final result after {max_steps} steps: {current_data}"





def dynamic_memory_pattern(problem: str, test_examples: List[Dict] = None, max_iterations: int = 3) -> str:
    """
    Use memory buffer to store and refine intermediate solutions iteratively.

    Uses a small number of examples embedded in the refinement prompts.
    """
    newline = chr(10)
    system_instruction = "You are an iterative problem solver who continually improves solutions."

    if not test_examples:
        test_examples = [{"input": "example input", "expected": "example output"}]

    # Initialize memory buffer
    memory_buffer = []

    # Generate initial candidate solutions with varying approaches
    initial_solutions = []

    # First solution with example
    first_solution_prompt = f"""
    Solve this problem with step-by-step reasoning:

    Example:
    Problem: Calculate the sum of the first 100 positive integers.
    Solution: I can use the formula for the sum of an arithmetic sequence: S = n(a₁ + aₙ)/2
    where n is the number of terms, a₁ is the first term, and aₙ is the last term.

    For the first 100 positive integers:
    n = 100
    a₁ = 1
    aₙ = 100

    S = 100(1 + 100)/2
    S = 100(101)/2
    S = 10100/2
    S = 5050

    Therefore, the sum of the first 100 positive integers is 5050.

    Problem: {problem}

    Provide a detailed step-by-step solution:
    """

    initial_solutions.append(call_llm(first_solution_prompt, system_instruction))

    # Second solution with different approach
    second_solution_prompt = f"""
    Solve this problem using a different approach than you would normally use:

    Problem: {problem}

    Try to approach this from an unusual or creative angle:
    """

    initial_solutions.append(call_llm(second_solution_prompt, system_instruction))

    # Third solution focusing on edge cases
    third_solution_prompt = f"""
    Solve this problem with special attention to edge cases:

    Problem: {problem}

    Be sure to address potential edge cases and corner conditions:
    """

    initial_solutions.append(call_llm(third_solution_prompt, system_instruction))

    # Evaluate and store each solution
    for i, solution in enumerate(initial_solutions):
        # Simulate evaluation
        evaluation_prompt = f"""
        Evaluate this solution:

        Problem: {problem}
        Solution: {solution}
        Test examples:
        {chr(10).join([f"Example {i+1}:{newline}Input: {ex.get('input', 'N/A')}{newline}Expected: {ex.get('expected', 'N/A')}" for i, ex in enumerate(test_examples)])}

        Rate this solution on:
        1. Correctness (1-10)
        2. Efficiency (1-10)
        3. Clarity (1-10)

        Provide specific feedback for improvement and an overall score (1-10).
        """

        evaluation = call_llm(evaluation_prompt, "You are a critical solution evaluator.")

        # Extract a score (simple text parsing)
        try:
            score_match = re.search(r"overall score[:\s]*(\d+)", evaluation, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 5
        except:
            score = 5

        # Store in memory buffer
        memory_buffer.append({
            'solution': solution,
            'evaluation': evaluation,
            'score': score,
            'iteration': 0,
            'approach_type': ["systematic", "creative", "edge_case_focused"][i]
        })

    # Iterative refinement using memory
    for iteration in range(1, max_iterations + 1):
        # Sort entries by score
        memory_buffer.sort(key=lambda x: x['score'], reverse=True)

        # Get top entries to refine
        top_entries = memory_buffer[:2]

        # Generate refined solutions based on memory
        refined_solutions = []
        for entry in top_entries:
            refinement_prompt = f"""
            Refine this solution based on evaluation feedback:

            Problem: {problem}

            Previous solution (score {entry['score']}/10):
            {entry['solution']}

            Evaluation feedback:
            {entry['evaluation']}

            Example of successful refinement:
            Original: The function should loop through the array and return the first element that matches the condition.
            Feedback: This approach doesn't handle empty arrays or cases where no element matches.
            Refined: The function should first check if the array is empty and return an appropriate default value. Then it should loop through the array and return the first matching element. If no element matches, it should return a specified default value.

            Now, provide an improved solution that specifically addresses the feedback points.
            """

            refined = call_llm(refinement_prompt, system_instruction)

            # Evaluate refined solution
            refined_eval_prompt = f"""
            Evaluate this refined solution:

            Problem: {problem}
            Solution: {refined}
            Test examples:
            {chr(10).join([f"Example {i+1}:{newline}Input: {ex.get('input', 'N/A')}{newline}Expected: {ex.get('expected', 'N/A')}" for i, ex in enumerate(test_examples)])}

            Rate this solution on:
            1. Correctness (1-10)
            2. Efficiency (1-10)
            3. Clarity (1-10)

            Provide specific feedback for further improvement and an overall score (1-10).
            """

            refined_evaluation = call_llm(refined_eval_prompt, "You are a critical solution evaluator.")

            # Extract a score
            try:
                score_match = re.search(r"overall score[:\s]*(\d+)", refined_evaluation, re.IGNORECASE)
                refined_score = int(score_match.group(1)) if score_match else 5
            except:
                refined_score = 5

            # Add to refined solutions
            refined_solutions.append({
                'solution': refined,
                'evaluation': refined_evaluation,
                'score': refined_score,
                'iteration': iteration,
                'parent': entry,
                'approach_type': entry['approach_type']
            })

        # Add refined solutions to memory
        memory_buffer.extend(refined_solutions)

    # Select best solutions based on performance
    memory_buffer.sort(key=lambda x: x['score'], reverse=True)
    top_solutions = memory_buffer[:3]

    # Synthesize final solution from top performers
    newline = chr(10)
    synthesis_prompt = f"""
    Create a final solution based on these top-performing approaches:

    Problem: {problem}

    {chr(10).join([f"Approach {i+1} (score {s['score']}/10):{newline}{s['solution']}" for i, s in enumerate(top_solutions)])}

    Example of good synthesis:
    Problem: Design an algorithm to find duplicates in an array.
    Approach 1: Using a nested loop (O(n²) complexity)
    Approach 2: Using a hash set (O(n) complexity but O(n) space)
    Approach 3: Sorting first, then linear scan (O(n log n) complexity, O(1) extra space)
    Synthesis: For this problem, Approach 2 offers the best time complexity. I'll use a hash set to track seen elements, which gives us O(n) time complexity. However, I'll incorporate the edge case handling from Approach 1 and the memory optimization technique from Approach 3 for large inputs.

    Create a solution that incorporates the strengths of all approaches while addressing their weaknesses.
    """

    final_solution = call_llm(synthesis_prompt, system_instruction)

    # Create a summary of the refinement process
    evolution_prompt = f"""
    Summarize how this solution evolved through iterations:

    Starting approaches:
    {initial_solutions[0][:100]}... (score: {memory_buffer[0]['score']})
    {initial_solutions[1][:100]}... (score: {memory_buffer[1]['score']})
    {initial_solutions[2][:100]}... (score: {memory_buffer[2]['score']})

    Final solution:
    {final_solution}

    Provide insights on how the solution improved across iterations.
    """

    evolution_summary = call_llm(evolution_prompt, system_instruction)

    return f"{final_solution}{newline}{newline}=== Solution Evolution Summary ==={newline}{evolution_summary}"






def test_time_training(problem_with_examples: str, max_iterations: int = 5) -> str:
    """
    Implement test-time training pattern: develop a hypothesis, test it on training examples,
    refine based on results, and apply to the test case only after verification.

    This pattern is essential when multiple examples demonstrate the same underlying pattern 
    that must be discovered and applied to a test case.

    Uses varied examples to demonstrate how incorrect hypotheses are detected and refined.
    """
    newline = chr(10)
    system_instruction = "You are a pattern recognition specialist who rigorously tests hypotheses against training examples."

    # Extract examples and identify test case
    extraction_prompt = f"""
    Extract the training examples and test case from this problem:

    {problem_with_examples}

    Format your response as follows:

    TRAINING_EXAMPLES:
    Example 1:
    Input: [first training input]
    Output: [first training output]

    Example 2:
    Input: [second training input]
    Output: [second training output]

    [Continue for all training examples]

    TEST_CASE:
    Input: [test input]

    DOMAIN:
    [problem domain]

    Be precise and comprehensive in extracting all information.
    """

    extraction_response = call_llm(extraction_prompt, system_instruction)

    # Parse the structured response
    training_examples = []
    test_case = {}
    domain = "unknown"

    # Extract training examples
    if "TRAINING_EXAMPLES:" in extraction_response:
        training_section = extraction_response.split("TRAINING_EXAMPLES:")[1].split("TEST_CASE:")[0].strip()
        example_blocks = re.split(r'\n\s*\n', training_section)

        for block in example_blocks:
            if not block.strip():
                continue

            input_match = re.search(r'Input: (.*?)(?:\n|$)', block)
            output_match = re.search(r'Output: (.*?)(?:\n|$)', block)

            if input_match and output_match:
                training_examples.append({
                    "input": input_match.group(1).strip(),
                    "output": output_match.group(1).strip()
                })

    # Extract test case
    if "TEST_CASE:" in extraction_response:
        test_section = extraction_response.split("TEST_CASE:")[1].split("DOMAIN:")[0].strip()
        input_match = re.search(r'Input: (.*?)(?:\n|$)', test_section)

        if input_match:
            test_case = {"input": input_match.group(1).strip()}

    # Extract domain
    if "DOMAIN:" in extraction_response:
        domain = extraction_response.split("DOMAIN:")[1].strip()

    # Generate initial hypothesis based on only the first example
    first_example_prompt = f"""
    Examine this SINGLE training example and formulate an initial hypothesis about the pattern:

    Example:
    Input: {training_examples[0]['input']}
    Output: {training_examples[0]['output']}

    Based ONLY on this example, what rule or pattern might explain it?
    Provide a detailed hypothesis about the transformation from input to output.
    """

    initial_hypothesis = call_llm(first_example_prompt, system_instruction)

    # Testing and refinement loop
    current_hypothesis = initial_hypothesis
    hypothesis_validated = False

    for iteration in range(max_iterations):
        # Test the hypothesis against ALL training examples
        testing_prompt = f"""
        Test this hypothesis against ALL of these training examples:

        Hypothesis:
        {current_hypothesis}

        Training Examples:
        {chr(10).join([f"Example {i+1}:{newline}Input: {ex['input']}{newline}Output: {ex['output']}" for i, ex in enumerate(training_examples)])}

        Example of thorough testing:

        Hypothesis: In the sequence, each number is doubled to get the next number.

        Training Examples:
        Example 1:
        Input: 2, 4, 8, 16
        Output: 32

        Example 2:
        Input: 5, 25, 125, 625
        Output: 3125

        Example 3:
        Input: 1, 1, 1, 1
        Output: 1

        Testing on Example 1: "2, 4, 8, 16" → expected "32"
        Analysis: If we double the last number: 16 × 2 = 32
        Result: ✓ Matches expected output "32"

        Testing on Example 2: "5, 25, 125, 625" → expected "3125"
        Analysis: If we double the last number: 625 × 2 = 1250
        Result: ✗ Does NOT match expected output "3125"

        Testing on Example 3: "1, 1, 1, 1" → expected "1"
        Analysis: If we double the last number: 1 × 2 = 2
        Result: ✗ Does NOT match expected output "1"

        Overall: The hypothesis fails on Examples 2 and 3. It needs refinement.

        Now, test your hypothesis on EACH training example:
        1. Apply the hypothesized rule to the input
        2. Check if the result matches the expected output
        3. Provide a detailed step-by-step analysis for each example

        Conclude whether your hypothesis explains ALL training examples or needs refinement.
        """

        test_results = call_llm(testing_prompt, system_instruction)

        # Check if hypothesis is validated
        validation_check = "correctly explains all" in test_results.lower() or "hypothesis is valid" in test_results.lower()
        validation_check = validation_check and not ("fails" in test_results.lower() or "does not match" in test_results.lower())

        if validation_check:
            hypothesis_validated = True
            break

        # Refine hypothesis based on test results
        refinement_prompt = f"""
        Your hypothesis needs refinement based on the test results:

        Current Hypothesis:
        {current_hypothesis}

        Test Results:
        {test_results}

        Example of good refinement:

        Original Hypothesis: In the sequence, each number is doubled to get the next number.

        Test Results: The hypothesis works for Example 1 ("2, 4, 8, 16" → "32") but fails on Examples 2 and 3:
        - For "5, 25, 125, 625" → expected "3125", doubling gives 1250, which is wrong
        - For "1, 1, 1, 1" → expected "1", doubling gives 2, which is wrong

        Refined Hypothesis: Each number in the sequence is multiplied by the first number in the sequence to get the next number.
        Testing:
        - Example 1: First number is 2. Last number is 16. 16 × 2 = 32 ✓
        - Example 2: First number is 5. Last number is 625. 625 × 5 = 3125 ✓
        - Example 3: First number is 1. Last number is 1. 1 × 1 = 1 ✓

        Now, refine your hypothesis to address the failures identified in the test results.
        Analyze patterns across ALL examples. Look for a single rule that works for EVERY case.
        Be creative in considering alternative patterns that might explain all examples.
        """

        current_hypothesis = call_llm(refinement_prompt, system_instruction)

    # Apply validated hypothesis to the test case
    if not hypothesis_validated:
        # Force a final hypothesis refinement if not validated after max iterations
        final_refinement_prompt = f"""
        After multiple iterations, we need a final refined hypothesis that best explains all training examples:

        Training Examples:
        {chr(10).join([f"Example {i+1}:{newline}Input: {ex['input']}{newline}Output: {ex['output']}" for i, ex in enumerate(training_examples)])}

        Current Hypothesis:
        {current_hypothesis}

        Analyze all examples together. Look for patterns across different sequences:
        - How does the first number relate to the pattern?
        - Is each sequence following its own internal logic?
        - What single rule could explain the transformation in EVERY example?

        Provide your best hypothesis that correctly explains ALL training examples.
        Test it against each example before submitting.
        """

        current_hypothesis = call_llm(final_refinement_prompt, system_instruction)

    # Apply the hypothesis to the test case
    application_prompt = f"""
    Now that we have a validated hypothesis, apply it to the test case:

    Hypothesis:
    {current_hypothesis}

    Test Case:
    Input: {test_case['input']}

    Example of detailed application:

    Hypothesis: Each number in the sequence is multiplied by the first number in the sequence to get the next number.

    Test Case: "3, 9, 27, 81"
    Analysis: 
    1. The first number in the sequence is 3
    2. The last number in the sequence is 81
    3. Applying our rule: 81 × 3 = 243

    Therefore, the next number is 243.

    Now, apply your hypothesis to the test case:
    1. Show your detailed step-by-step application of the rule
    2. Verify each step for accuracy
    3. Provide the final answer

    Be thorough and precise in your application.
    """

    application_result = call_llm(application_prompt, system_instruction)

    # Generate a comprehensive solution that explains the process
    final_solution_prompt = f"""
    Create a comprehensive solution that explains the entire test-time training process:

    Problem:
    {problem_with_examples}

    Initial Hypothesis (based on first example only):
    {initial_hypothesis}

    Testing and Refinement Process:
    {test_results}

    Final Validated Hypothesis:
    {current_hypothesis}

    Application to Test Case:
    {application_result}

    Provide a structured solution with these sections:

    1. INITIAL PATTERN RECOGNITION: How we formed our first hypothesis looking at only one example

    2. HYPOTHESIS TESTING: How we tested this hypothesis against ALL examples and discovered it didn't work for all cases

    3. HYPOTHESIS REFINEMENT: How we refined our thinking to find a rule that works across ALL examples

    4. VALIDATION: How we verified our refined hypothesis against all training examples

    5. APPLICATION: How we applied the validated rule to the test case

    6. ADVANTAGES OF TEST-TIME TRAINING: Explain how this approach prevented errors by confirming our hypothesis against multiple examples before submission

    7. FINAL ANSWER: The clear, concise answer to the test case

    Emphasize how the availability of multiple training examples allowed us to test and refine our hypotheses, preventing incorrect submissions.
    """

    return call_llm(final_solution_prompt, system_instruction)





def combination_example() -> str:
    """
    Provide a comprehensive example of effectively combining multiple LLM interaction patterns.

    Uses a single detailed example to demonstrate complex pattern combinations for general applications.
    """
    system_instruction = "You are an expert in LLM pattern design who creates sophisticated solutions by combining techniques."

    prompt = """
    Provide a detailed example of how to effectively combine multiple LLM interaction patterns to solve complex problems.

    # Example: Combining Multiple Patterns for Advanced Problem Solving

    ## Original Challenge
    Creating a system that can analyze complex text, identify key insights, and generate well-reasoned recommendations.

    ## Pattern Selection and Combination Strategy

    1. Start with Feature Extraction to identify key elements:
    ```python
    # Extract key information from input text
    extraction_prompt = f'''
    Analyze this text and extract key features:

    {input_text}

    Focus specifically on:
    - Main entities and their attributes
    - Relationships between entities
    - Explicit and implicit constraints
    - Quantitative data points
    - Key objectives and success criteria

    For each feature, explain why it's significant for the analysis.
    '''

    extracted_features = call_llm(extraction_prompt, system_instruction="You are a precise information extraction specialist.")
    ```

    2. Apply Multi-Perspective Analysis with domain experts:
    ```python
    # Generate analyses from different expert perspectives
    perspectives = ["data_analyst", "domain_expert", "strategic_advisor"]
    perspective_analyses = []

    for perspective in perspectives:
        perspective_prompt = f'''
        As a {perspective}, analyze this situation:

        Input text: {input_text}

        Key features identified:
        {extracted_features}

        Provide a thorough analysis focusing on aspects a {perspective} would prioritize.
        Highlight insights that might be missed by other perspectives.
        '''

        analysis = call_llm(perspective_prompt, 
                           system_instruction=f"You are an expert {perspective} with deep experience in this field.")
        perspective_analyses.append({"perspective": perspective, "analysis": analysis})

    # Synthesize the perspectives
    synthesis_prompt = f'''
    Combine these different expert analyses into a comprehensive understanding:

    {json.dumps(perspective_analyses, indent=2)}

    Identify:
    - Where the perspectives agree and disagree
    - Complementary insights that build on each other
    - Points of tension that require further investigation

    Create a unified analysis that leverages the strengths of each perspective.
    '''

    unified_analysis = call_llm(synthesis_prompt, system_instruction="You are a synthesis specialist.")
    ```

    3. Implement Chain-of-Thought with Self-Consistency:
    ```python
    # Generate multiple reasoning chains toward recommendations
    reasoning_paths = []

    for i in range(3):  # Generate 3 different reasoning paths
        reasoning_prompt = f'''
        Based on this unified analysis:
        {unified_analysis}

        Think step-by-step toward recommendation{i+1}.
        Focus on a different priority or approach than previous reasoning paths.

        Step 1: Identify key challenges and opportunities
        Step 2: Evaluate potential approaches
        Step 3: Consider implementation requirements
        Step 4: Assess risks and mitigations
        Step 5: Develop specific recommendations
        '''

        reasoning = call_llm(reasoning_prompt, 
                            system_instruction="You are a methodical problem solver who thinks step by step.")
        recommendations = extract_recommendations(reasoning)
        reasoning_paths.append({"reasoning": reasoning, "recommendations": recommendations})

    # Evaluate consistency across reasoning paths
    consistency_prompt = f'''
    Analyze these different reasoning approaches:
    {json.dumps(reasoning_paths, indent=2)}

    For each key recommendation:
    - Is it supported by multiple reasoning paths?
    - Are there contradictions between different paths?
    - Which path provides the strongest justification?

    Determine the most robust recommendations with their supporting rationale.
    '''

    consistent_recommendations = call_llm(consistency_prompt, 
                                        system_instruction="You are a critical evaluator.")
    ```

    4. Add Verification and Debate for rigorous testing:
    ```python
    # Simulate debate to stress-test recommendations
    debate_prompt = f'''
    Critique these recommendations from multiple perspectives:
    {consistent_recommendations}

    Perspective 1: Implementation Feasibility
    - What practical challenges might arise?
    - Are there resource or technical constraints?
    - How realistic is the timeline?

    Perspective 2: Potential Downsides
    - What negative outcomes might occur?
    - Are there ethical concerns?
    - What stakeholders might be adversely affected?

    Perspective 3: Alternatives Analysis
    - What alternative approaches weren't considered?
    - Are there simpler solutions?
    - What approaches have worked in similar situations?
    '''

    critique = call_llm(debate_prompt, system_instruction="You are a critical challenger.")

    # Refine recommendations based on critique
    for attempt in range(max_refinement_attempts):
        refinement_prompt = f'''
        Refine these recommendations based on critical feedback:

        Original recommendations:
        {consistent_recommendations}

        Critical feedback:
        {critique}

        Provide improved recommendations that address the valid concerns while
        maintaining the core value. Be specific about:
        - How each concern is addressed
        - What trade-offs are being made
        - Why this represents an improvement
        '''

        refined_recommendations = call_llm(refinement_prompt, 
                                         system_instruction="You are a solution refiner.")

        # Verify improvements
        verification_prompt = f'''
        Verify if these refined recommendations properly address the previous critiques:

        Original recommendations:
        {consistent_recommendations}

        Critiques:
        {critique}

        Refined recommendations:
        {refined_recommendations}

        For each major critique, indicate:
        - ADDRESSED: How the refinement addresses it
        - PARTIALLY ADDRESSED: What aspects still need work
        - NOT ADDRESSED: Why the critique wasn't adequately addressed

        Overall verification: Are the refined recommendations an improvement?
        '''

        verification = call_llm(verification_prompt, 
                               system_instruction="You are a verification specialist.")

        if "IMPROVEMENT: YES" in verification:
            break

        # Update critique for next refinement iteration
        critique = extract_unaddressed_critiques(verification)
    ```

    5. Final Synthesis with Best-of-N Selection:
    ```python
    # Generate multiple final versions
    final_versions = []

    for i in range(3):
        final_prompt = f'''
        Create a final recommendation report that integrates:

        1. The key insights from the unified analysis:
        {unified_analysis}

        2. The consistent recommendations from multiple reasoning paths:
        {consistent_recommendations}

        3. The refinements based on critical feedback:
        {refined_recommendations}

        Format {i+1}: {["concise executive summary", "detailed analysis", "action-oriented plan"][i]}

        Focus on creating a {["strategic", "comprehensive", "practical"][i]} set of recommendations.
        '''

        final_version = call_llm(final_prompt, system_instruction="You are a recommendation specialist.")

        # Evaluate version quality
        evaluation_prompt = f'''
        Evaluate this recommendation report on:
        - Clarity (1-10)
        - Comprehensiveness (1-10)
        - Actionability (1-10)
        - Persuasiveness (1-10)
        - Logical consistency (1-10)

        Recommendation report:
        {final_version}

        Provide numerical scores and brief justifications.
        '''

        evaluation = call_llm(evaluation_prompt, system_instruction="You are a quality evaluator.")
        scores = extract_scores(evaluation)

        final_versions.append({
            "version": final_version,
            "evaluation": evaluation,
            "total_score": sum(scores.values())
        })

    # Select best version
    final_versions.sort(key=lambda x: x["total_score"], reverse=True)
    best_version = final_versions[0]["version"]
    ```

    ## Key Integration Points
    - Feature Extraction provides structured input for Multi-Perspective Analysis
    - Multi-Perspective Analysis feeds unified context to Chain-of-Thought
    - Self-Consistency ensures robustness of reasoning paths
    - Debate and Verification rigorously test and improve recommendations
    - Best-of-N Selection optimizes the final output format and content

    ## Benefits of Pattern Combination
    - Each pattern addresses different aspects of the complex problem
    - Later patterns build upon the outputs of earlier patterns
    - Verification catches issues that might be missed in a linear approach
    - Multiple perspectives create more robust solutions
    - Self-consistency reduces likelihood of spurious reasoning

    This example demonstrates how combining patterns creates a solution pipeline that's much more powerful than any single pattern alone, particularly for complex analytical and recommendation tasks.
    """

    return call_llm(prompt, system_instruction)