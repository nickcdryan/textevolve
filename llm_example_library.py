"""
LLM Example Library

This module contains example code snippets for various LLM-based techniques.
These examples can be used as templates in LLM prompts.
"""

class APIExamples:
    """Examples of API usage for different LLM services."""

    @staticmethod
    def gemini_api():
        """Return example code for Gemini API usage."""
        return '''def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types

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
        return f"Error: {str(e)}"'''


class TechniqueExamples:
    """Examples of various LLM-based techniques."""

    @staticmethod
    def information_extraction():
        """Return example code for information extraction with embedded examples."""
        return '''def extract_information_with_examples(text):
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

    return call_llm(prompt, system_instruction)'''

    @staticmethod
    def verification():
        """Return example code for solution verification with embedded examples."""
        return '''def verify_solution_with_examples(problem, proposed_solution):
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

    return call_llm(prompt, system_instruction)'''

    @staticmethod
    def validation_loop():
        """Return example code for solution validation with feedback loop."""
        return '''def solve_with_validation_loop(problem, max_attempts=3):
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

    return solution'''

    @staticmethod
    def multi_perspective_analysis():
        """Return example code for multi-perspective analysis."""
        return '''def multi_perspective_analysis(problem):
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

    return call_llm(synthesis_prompt, "You are an insight synthesizer who combines multiple analyses.")'''

    @staticmethod
    def best_of_n_approach():
        """Return example code for best-of-n approach."""
        return '''def best_of_n_approach(problem, n=3):
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

    {["Solution " + str(i+1) + ": " + solutions[i] + "\\n\\nEvaluation: " + evaluations[i] for i in range(n)]}

    Which solution is best? Respond with the solution number and explanation.
    """

    best_solution_index = int(call_llm(comparison_prompt, "You are a solution selector.").split()[1]) - 1
    return solutions[best_solution_index]'''

    @staticmethod
    def react_pattern():
        """Return example code for ReAct pattern."""
        return '''def solve_with_react_pattern(problem):
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
        react_response += "\\n" + call_llm(continuation_prompt, system_instruction)

        # Extract the next action
        action = extract_action(react_response)

    # Extract the final answer from the Finish action
    final_answer = action["answer"]
    return final_answer

def extract_action(text):
    """Parse the ReAct response to extract the current action."""
    # Find the last action in the text
    action_matches = re.findall(r"Action (\d+): (\\w+)\\[(.*?)\\]", text)
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
    return call_llm(f"Provide specific information about: {term}", "You are a knowledge base that provides specific factual information.")'''


class ExamplePatterns:
    """Common example patterns to use in prompts."""

    @staticmethod
    def pipeline_without_verification():
        """Example of a pipeline without verification."""
        return '''def process_question(question):
    entities = extract_entities(question)
    constraints = identify_constraints(question)
    solution = generate_solution(entities, constraints)
    return solution'''

    @staticmethod
    def robust_pipeline_with_verification():
        """Example of a robust pipeline with verification steps."""
        return '''def process_question(question, max_attempts=3):
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

    return solution_result["solution"]'''

    @staticmethod
    def extraction_with_verification():
        """Example of entity extraction with verification."""
        return '''def extract_entities_with_verification(question, max_attempts=3):
    """Extract entities and verify their validity with feedback loop."""
    system_instruction = "You are an expert at extracting and validating entities."

    for attempt in range(max_attempts):
        # First attempt at extraction
        extraction_prompt = f\'\'\'
        Extract key entities from this question. 
        Return a JSON object with the extracted entities.

        Example 1: [example with entities]
        Example 2: [example with different entities]
        Example 3: [example with complex entities]

        Question: {question}
        Extraction:
        \'\'\'

        extracted_data = call_llm(extraction_prompt, system_instruction)

        try:
            # Parse the extraction
            data = json.loads(extracted_data)

            # Verification step
            verification_prompt = f\'\'\'
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
            \'\'\'

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
    }'''

    @staticmethod
    def poor_single_example_prompting():
        """Example of poor single-example prompting."""
        return '''def extract_entities(text):
    prompt = f\'\'\'
    Extract entities from this text.

    Example:
    Text: John will meet Mary at 3pm on Tuesday.
    Entities: {{"people": ["John", "Mary"], "time": "3pm", "day": "Tuesday"}}

    Text: {text}
    Entities:
    \'\'\'
    return call_llm(prompt)'''

    @staticmethod
    def effective_multi_example_prompting():
        """Example of effective multi-example prompting."""
        return '''def extract_entities(text):
    prompt = f\'\'\'
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
    \'\'\'
    return call_llm(prompt)'''


class ExampleSets:
    """Pre-configured sets of examples for different use cases."""

    @staticmethod
    def get_standard_examples():
        """Return a dictionary with the standard set of examples."""
        return {
            "gemini_api_example": APIExamples.gemini_api(),
            "extraction_example": TechniqueExamples.information_extraction(),
            "verification_example": TechniqueExamples.verification(),
            "validation_loop_example": TechniqueExamples.validation_loop(),
            "multi_perspective_example": TechniqueExamples.multi_perspective_analysis(),
            "best_of_n_example": TechniqueExamples.best_of_n_approach(),
            "react_example": TechniqueExamples.react_pattern()
        }

    @staticmethod
    def get_implementation_examples():
        """Return a dictionary with implementation pattern examples."""
        return {
            "pipeline_without_verification": ExamplePatterns.pipeline_without_verification(),
            "robust_pipeline": ExamplePatterns.robust_pipeline_with_verification(),
            "extraction_with_verification": ExamplePatterns.extraction_with_verification(),
            "poor_prompting": ExamplePatterns.poor_single_example_prompting(),
            "good_prompting": ExamplePatterns.effective_multi_example_prompting()
        }

    @staticmethod
    def get_few_shot_examples_text():
        """Return a formatted string with few-shot examples for inclusion in prompts."""
        examples = ExampleSets.get_standard_examples()

        # Create few-shot examples block
        few_shot_examples = f"""EXAMPLE OF EFFECTIVE LLM USAGE PATTERNS:

```python
{examples['extraction_example']}
```

```python
{examples['verification_example']}
```

```python
{examples['validation_loop_example']}
```

```python
{examples['multi_perspective_example']}
```

```python
{examples['best_of_n_example']}
```

```python
{examples['react_example']}
```"""

        return few_shot_examples

    @staticmethod
    def get_implementation_patterns_text():
        """Return a formatted string with implementation patterns for inclusion in prompts."""
        examples = ExampleSets.get_implementation_examples()

        # Create implementation patterns block
        implementation_patterns = f"""IMPLEMENTATION PATTERNS:

Without verification:
```python
{examples['pipeline_without_verification']}
```

With verification:
```python
{examples['robust_pipeline']}
```

Extraction with verification:
```python
{examples['extraction_with_verification']}
```"""

        return implementation_patterns

    @staticmethod
    def get_prompting_examples_text():
        """Return a formatted string with prompting examples for inclusion in prompts."""
        examples = ExampleSets.get_implementation_examples()

        # Create prompting examples block
        prompting_examples = f"""PROMPTING EXAMPLES:

Poor single-example prompting:
```python
{examples['poor_prompting']}
```

Effective multi-example prompting:
```python
{examples['good_prompting']}
```"""

        return prompting_examples


class FallbackScripts:
    """Collection of fallback scripts to use when generation fails."""

    @staticmethod
    def basic_fallback():
        """Return a basic fallback script with embedded examples."""
        return """
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