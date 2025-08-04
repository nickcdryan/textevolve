"""
Static prompting guidance and instruction blocks.
These are reusable text blocks that don't require variable interpolation.


CONTAINS:

multi_example_prompting_guide
llm_reasoning_prompting_guide
validation_prompting_guide
meta_programming_prompting_guide
code_execution_prompting_guide

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


database_prompting_guide = """

🔥 CRITICAL: DATABASE ACCESS CAPABILITY AVAILABLE 🔥

You have access to a powerful call_database() function that can query SQLite databases safely.

WHEN TO USE call_database():
- When you need to retrieve structured data from a database
- Problems involving data analysis, lookups, or queries
- When you need to access pre-existing datasets in database format
- Questions that reference tables, records, or database-like operations
- You understand that using database queries is more reliable for data retrieval than asking an LLM

HOW TO USE call_database():
```python
# Query a database file
result = call_database("path/to/database.db", "SELECT * FROM table_name LIMIT 10")

# The result contains formatted query results as a string
# For SELECT queries: Returns formatted table with columns and rows
# For INSERT/UPDATE/DELETE: Returns success message with affected row count

# Example with specific query
customers = call_database("shop.db", "SELECT name, email FROM customers WHERE age > 25")

# Example with aggregation
stats = call_database("analytics.db", "SELECT COUNT(*) as total_users, AVG(score) as avg_score FROM users")
```

Example pattern:
```python
def main(question):
    if "database" in question.lower() or "table" in question.lower():
        # Extract what database/table is being asked about
        query_prompt = f"Generate a SQL query for this question: {question}"
        sql_query = call_llm(query_prompt, "You are a SQL expert")
        
        # Execute the database query
        db_result = call_database("data.db", sql_query)
        
        # Interpret the results
        return call_llm(f"Question: {question}, Database results: {db_result}, What's the answer?")
    else:
        # No database needed - use reasoning
        return call_llm(f"Solve: {question}")
```

DATABASE FEATURES:
- Automatic path resolution (works in sandbox and local environments)
- Handles both SELECT queries (returns formatted results) and modification queries (INSERT/UPDATE/DELETE)
- Built-in error handling for SQL syntax errors and file access issues
- Results limited to 50 rows for readability (shows total count if more)
- Column names included in results for easy interpretation

REMEMBER: call_database() is available - use it for data retrieval and analysis!

⛔ DO NOT DEFINE call_database() - it is PROVIDED BY THE SYSTEM
⛔ Just USE it like built-in functions (like print() or len())

✅ CORRECT:
def main(question):
    result = call_database("data.db", "SELECT * FROM products")  # Just use it
    return result

❌ WRONG:
def call_database(db_path, query):  # Don't define this!
    import sqlite3
    # ... implementation

DATABASE PATH NOTES:
- Relative paths like "data.db" work automatically (checks workspace and current directory)
- Absolute paths work if the database file is accessible
- In sandbox environments, databases should be in the workspace directory

⛔ DO NOT DEFINE call_database() - it is PROVIDED BY THE SYSTEM
"""


file_access_prompting_guide = """

🔥 CRITICAL: FILE ACCESS CAPABILITIES AVAILABLE 🔥

You have access to powerful file reading and searching functions for accessing local files safely.

## read_file(filepath, start_line=None, end_line=None) - File Reading

WHEN TO USE read_file():
- When you need to read configuration files, code files, or documentation
- Reading specific sections of large files using line ranges
- Accessing data files, logs, or any text-based content
- When you need the complete content or specific portions of a file

HOW TO USE read_file():
```python
# Read entire file
content = read_file("config.txt")

# Read specific line range (1-based line numbers)
content = read_file("large_file.log", start_line=100, end_line=200)

# Read from start to specific line
content = read_file("script.py", end_line=50)

# Read from specific line to end
content = read_file("data.txt", start_line=25)
```

## search_file(filepath, pattern, case_sensitive=True, show_line_numbers=True, context_lines=0, max_results=50) - File Search

WHEN TO USE search_file():
- Finding specific functions, classes, or variables in code files
- Searching for patterns, keywords, or errors in log files
- Locating specific content with context around matches
- When you need grep-like functionality with regex support

HOW TO USE search_file():
```python
# Find function definitions
results = search_file("script.py", "def ")

# Case-insensitive search
results = search_file("docs.md", "database", case_sensitive=False)

# Search with context lines around matches
results = search_file("error.log", "ERROR", context_lines=2)

# Use regex patterns
results = search_file("code.py", r"class \w+\(", case_sensitive=True)

# Limit results for performance
results = search_file("large.txt", "pattern", max_results=10)
```

Example pattern combining both functions:
```python
def main(question):
    if "find" in question.lower() or "search" in question.lower():
        # Extract what to search for
        search_prompt = f"What pattern should I search for based on: {question}"
        pattern = call_llm(search_prompt, "Extract the search pattern")
        
        # Search in relevant file
        search_results = search_file("target_file.py", pattern, context_lines=1)
        
        # Analyze results
        return call_llm(f"Question: {question}, Search results: {search_results}, What's the answer?")
    elif "read" in question.lower() or "content" in question.lower():
        # Read specific file content
        content = read_file("target_file.txt")
        
        # Process content
        return call_llm(f"Question: {question}, File content: {content}, What's the answer?")
    else:
        # No file access needed
        return call_llm(f"Solve: {question}")
```

## FILE ACCESS FEATURES:

### read_file() Features:
- Automatic path resolution (works in sandbox and local environments)
- Line range support for reading specific sections
- Line-numbered output for easy reference
- Handles large files with warnings
- Built-in error handling for missing files, permissions, binary files

### search_file() Features:
- Full regex pattern support with proper error handling
- Case-sensitive and case-insensitive search options
- Context lines around matches (like grep -A/-B)
- Line numbers and highlighted matching lines
- Result limiting to prevent overwhelming output
- Handles overlapping context intelligently

## PATH RESOLUTION:
- Relative paths like "config.txt" work automatically (checks workspace and current directory)
- Absolute paths work if the file is accessible
- In sandbox environments, files should be in the workspace directory

## ERROR HANDLING:
- Graceful handling of missing files, permission errors
- Binary file detection and appropriate error messages
- Invalid regex pattern detection and helpful error messages
- File vs directory validation

REMEMBER: read_file() and search_file() are available - use them for file operations!

⛔ DO NOT DEFINE read_file() or search_file() - they are PROVIDED BY THE SYSTEM
⛔ Just USE them like built-in functions (like print() or len())

✅ CORRECT:
def main(question):
    content = read_file("data.txt")  # Just use it
    matches = search_file("script.py", "def main")  # Just use it
    return content

❌ WRONG:
def read_file(filepath):  # Don't define this!
    with open(filepath) as f:
        return f.read()

def search_file(filepath, pattern):  # Don't define this!
    import re
    # ... implementation

⛔ DO NOT DEFINE read_file() or search_file() - they are PROVIDED BY THE SYSTEM

"""