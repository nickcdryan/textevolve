"""
Static prompting guidance and instruction blocks.
These are reusable text blocks that don't require variable interpolation.
"""

multi_example_prompting_guide = """MULTI-EXAMPLE PROMPTING GUIDANCE:
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
"""

llm_reasoning_prompting_guide = """
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
"""

validation_prompting_guide = """
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


meta_programming_prompting_guide = """

=== ADVANCED CAPABILITY: DYNAMIC META-PROGRAMMING ===

Your scripts now have POWERFUL meta-programming capabilities through two key functions:

## execute_code(code_string) - Dynamic Code Execution
- Safely executes Python code you generate at runtime
- Has access to: math, re, json modules and basic Python builtins
- Returns results as strings
- Perfect for: calculations, data processing, algorithmic solutions

## call_llm(prompt, system_instruction) - Dynamic LLM Calls  
- Calls the LLM with prompts you generate at runtime
- You can create specialized prompts for specific tasks
- Perfect for: analysis, reasoning, prompt engineering

## META-PROGRAMMING TOOL PATTERNS YOU CAN USE, MODIFY, ADAPT, OR COMBINE:

### Pattern 1: Adaptive Code Generation
```python
def solve_adaptively(problem):
    # Script decides what code to write based on the problem
    code_prompt = f"Write Python code to solve: {problem}"
    generated_code = call_llm(code_prompt, "You are a programmer")

    # Execute the generated code
    result = execute_code(generated_code)

    # Interpret results
    return call_llm(f"Problem: {problem}, Code result: {result}, Final answer?")
```

### Pattern 2: Dynamic Prompt Engineering
```python
def analyze_with_specialized_prompts(data):
    # Generate the perfect prompt for this specific data
    prompt_design = call_llm(f"Design the best prompt to analyze: {data}")

    # Use the generated prompt
    return call_llm(prompt_design, "You are a specialist")
```

### Pattern 3: Self-Modifying Strategy
```python
def solve_with_strategy_evolution(problem):
    strategy = "initial_approach"

    while True:
        if strategy == "initial_approach":
            result = call_llm(f"Solve: {problem}")
            evaluation = call_llm(f"Did this work? {result} If not, what strategy next?")

            if "solved" in evaluation:
                return result
            elif "code" in evaluation:
                strategy = "code_approach"
        elif strategy == "code_approach":
            code = call_llm(f"Write code to solve: {problem}")
            return execute_code(code)
```

### Pattern 4: Chain Code and LLM Dynamically
```python
def chain_adaptively(input_data):
    current_data = input_data

    for step in range(3):
        # Decide what to do next
        decision = call_llm(f"Step {step}: What should I do with {current_data}?")

        if "code" in decision.lower():
            code = call_llm(f"Write code to process: {current_data}")
            current_data = execute_code(code)
        else:
            current_data = call_llm(f"Analyze: {current_data}")

    return current_data
```

## WHEN TO USE META-PROGRAMMING:

🎯 **Use Code Generation When:**
- Problem requires calculations or data processing
- You need algorithmic solutions
- Mathematical operations are involved
- Data transformation is needed
- Remember that code generation is more error-prone and should be used when you have a high confidence that the approach will work
- Remember that LLMs are powerful, and sometimes sufficient for algorithmic and data transformation tasks 

🎯 **Use Dynamic Prompts When:**
- Problem requires specialized analysis
- You need domain-specific reasoning
- Different problem types need different approaches
- You want to optimize prompts for specific inputs

🎯 **Use Hybrid Approaches When:**
- Complex problems need both reasoning and computation
- You want to chain multiple processing steps
- You need to verify results through different methods
- Problem-solving requires adaptive strategies

## KEY PRINCIPLES:

1. **Scripts Can Be Programmers**: Your script can write and execute its own code at runtime
2. **Scripts Can Be Prompt Engineers**: Your script can design and use specialized prompts
3. **Adaptive Problem Solving**: Let each step decide what the next step should be
4. **Self-Modification**: Scripts can change their own strategy based on results
5. **Chain Dynamically**: Combine code execution and LLM calls in flexible sequences

## EXAMPLE META-PROGRAMMING WORKFLOW:

```python
def meta_solve(question):
    # 1. Analyze problem type
    analysis = call_llm(f"What type of problem is this: {question}")

    # 2. Generate appropriate solution approach
    if "mathematical" in analysis:
        code = call_llm(f"Write math code for: {question}")
        result = execute_code(code)
        return call_llm(f"Interpret math result: {result}")
    else:
        specialized_prompt = call_llm(f"Design analysis prompt for: {question}")
        return call_llm(specialized_prompt)
```

This gives your scripts the power to be truly autonomous problem-solvers that adapt their approach in real-time!
If traditional approaches aren't working you should try these more expensive but advanced meta-programming techniques.
"""



code_execution_prompting_guide = """

🔥 CRITICAL: CODE EXECUTION CAPABILITY AVAILABLE 🔥

You have access to a powerful execute_code() function that can run Python code safely.

WHEN TO USE execute_code():
- ANY complex mathematical calculations (percentages, areas, arithmetic)
- Data processing or algorithmic problems  
- When you need precise computational results
- Problems involving numbers, formulas, or calculations
- You understand that using code execution is more reliable for these tasks than asking an LLM

HOW TO USE execute_code():
```python
# Generate code string
code = '''
result = 847293 * 0.15
print(f"15% of 847,293 = {result}")
'''

# Execute it
output = execute_code(code)
# output contains: "15% of 847,293 = 127093.95" 
```

Example pattern:
```python
def main(question):
    if any(char.isdigit() for char in question):
        # Has numbers - use code execution
        code = call_llm(f"Write Python code to solve: {question}")
        result = execute_code(code) 
        return result
    else:
        # No numbers - use reasoning
        return call_llm(f"Solve: {question}")
```

REMEMBER: execute_code() is available - use it for computational problems!

⛔ DO NOT DEFINE execute_code() or call_llm() - they are PROVIDED BY THE SYSTEM
⛔ Just USE them like built-in functions (like print() or len())

✅ CORRECT:
def main(question):
    result = execute_code("print('hello')")  # Just use it
    return result

❌ WRONG:
def execute_code(code):  # Don't define this!
    exec(code)

REMEMBER! If you want to execute code you must use the execute_code() function. Just saying 
you will execute code without calling the execute_code() function is not allowed.

⛔ DO NOT DEFINE execute_code() or call_llm() - they are PROVIDED BY THE SYSTEM
"""