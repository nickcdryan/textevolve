import inspect

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


def multi_perspective_analysis(problem):
    """Analyze a problem from multiple specialized perspectives and synthesize the insights."""
    # Define specialized analysis functions
    def analyze_factual_content(problem):
        system_instruction = "You are a factual analyst who focuses on identifying key facts and data points."
        prompt = f"""
        Analyze this problem for factual content only. Identify explicit facts, constraints, and requirements.
    
        Problem:
        {problem}
        """
        return call_llm(prompt, system_instruction)
    
    def analyze_structure(problem):
        system_instruction = "You are a structural analyst who specializes in problem organization and patterns."
        prompt = f"""
        Analyze the structure of this problem. Identify its components, relationships, and patterns.
    
        Problem:
        {problem}
        """
        return call_llm(prompt, system_instruction)
    
    # Execute parallel analyses
    factual_analysis = analyze_factual_content(problem)
    structural_analysis = analyze_structure(problem)

    # Synthesize the results
    synthesis_prompt = f"""
    Synthesize these two different analyses of the same problem into a comprehensive understanding.
  
    Factual Analysis:
    {factual_analysis}
  
    Structural Analysis:
    {structural_analysis}
  
    Provide a unified analysis that leverages both perspectives.
    """
  
    return call_llm(synthesis_prompt, "You are an insight synthesizer who combines multiple analyses.")


def best_of_n_approach(problem, n=3):
    """Generate multiple solutions and select the best one based on a quality evaluation."""
    system_instruction_solver = "You are an expert problem solver who provides detailed, correct solutions."
    system_instruction_evaluator = "You are a quality evaluator who assesses solutions based on correctness, completeness, and clarity."
  
    # Generate n different solutions
    solutions = []
    for i in range(n):
        diversity_factor = f"Solution approach {i+1}/{n}: Use a different perspective from previous solutions."
        solution_prompt = f"""
        Provide a detailed solution to this problem.
        {diversity_factor if i > 0 else ""}
  
        Problem:
        {problem}
        """
  
        solutions.append(call_llm(solution_prompt, system_instruction_solver))
  
    # Evaluate each solution
    evaluations = []
    for i, solution in enumerate(solutions):
        evaluation_prompt = f"""
        Evaluate this solution on correctness, completeness, and clarity (1-10 scale).
  
        Problem:
        {problem}
  
        Solution {i+1}:
        {solution}
  
        Provide your evaluation as a JSON with scores and explanation.
        """
  
        evaluations.append(call_llm(evaluation_prompt, system_instruction_evaluator))
  
    # Find the best solution
    comparison_prompt = f"""
    Compare these solutions and their evaluations. Select the best one.
  
    Problem:
    {problem}
  
    {["Solution " + str(i+1) + ": " + solutions[i] + "\n\nEvaluation: " + evaluations[i] for i in range(n)]}
  
    Which solution is best? Respond with the solution number and explanation.
    """
  
    best_solution_index = int(call_llm(comparison_prompt, "You are a solution selector.").split()[1]) - 1
    return solutions[best_solution_index]




def solve_with_react_pattern(problem):
    """Solve problems through iterative Reasoning and Acting (ReAct) approach."""
    system_instruction = "You are a problem-solving agent that follows the ReAct pattern: Reason about the current state, take an Action, observe the result, and repeat until reaching a solution."
  
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
    
    # Extract the final answer from the Finish action
    final_answer = action["answer"]
    return final_answer




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