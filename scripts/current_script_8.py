import os
import re

def main(question):
    """
    Solve the question by extracting relevant information from the passage and using chain-of-thought reasoning.
    This approach builds upon a prior attempt to use question decomposition, strengthens answer synthesis, and includes examples in all LLM prompts.
    It combines the strengths of Iteration 1 and 2, incorporating question-type determination and specialized processing.
    """
    try:
        # Step 1: Determine the question type (numerical or general).
        question_type_result = determine_question_type(question)
        if not question_type_result.get("is_valid"):
            return f"Error in determining question type: {question_type_result.get('validation_feedback')}"

        # Step 2: Process the question based on its type.
        if question_type_result["question_type"] == "numerical":
            process_result = process_numerical_question(question)
        else:
            process_result = process_general_question(question)

        return process_result

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def determine_question_type(question, max_attempts=3):
    """Determine if the question is numerical or general."""
    system_instruction = "You are an expert at classifying questions as either numerical or general."

    for attempt in range(max_attempts):
        type_prompt = f"""
        Classify the question as either "numerical" or "general". Numerical questions require numerical calculations. General questions do not.

        Example 1:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Type: numerical

        Example 2:
        Question: Who caught the final touchdown of the game?
        Type: general

        Question: {question}
        Type:
        """

        type_result = call_llm(type_prompt, system_instruction)

        verification_prompt = f"""
        Verify if the question type classification is correct.

        Question: {question}
        Classification: {type_result}

        Example:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Classification: numerical
        Validation: Valid

        Is the classification valid? Respond with 'Valid' or 'Invalid'.
        """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "valid" in verification_result.lower() and type_result in ("numerical", "general"):
            return {"is_valid": True, "question_type": type_result}
        else:
            print(f"Question type validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")

    return {"is_valid": False, "validation_feedback": "Failed to determine the question type successfully."}

def process_numerical_question(question):
    """Process a numerical question."""
    try:
        extracted_info_result = extract_numerical_info(question)
        if not extracted_info_result.get("is_valid"):
            return f"Error in extracting numerical info: {extracted_info_result.get('validation_feedback')}"

        calculation_result = calculate_answer(question, extracted_info_result["numbers"])
        if not calculation_result.get("is_valid"):
            return f"Error in calculation: {calculation_result.get('validation_feedback')}"

        return calculation_result["answer"]

    except Exception as e:
        return f"Error processing numerical question: {str(e)}"

def extract_numerical_info(question, max_attempts=3):
    """Extract numerical information from the question."""
    system_instruction = "You are an expert at extracting numerical information."

    for attempt in range(max_attempts):
        extraction_prompt = f"""
        Extract all numbers from the question and their corresponding entities.

        Example 1:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Numbers:
        6 yards (Chris Johnson's first touchdown)
        53 yards (Jason Hanson's first field goal)

        Example 2:
        Question: How many touchdowns did Brandon Jacobs rush for?
        Numbers:
        2 touchdowns (Brandon Jacobs)

        Question: {question}
        Numbers:
        """

        extracted_numbers = call_llm(extraction_prompt, system_instruction)

        verification_prompt = f"""
        Verify that the numbers extracted are correct and correspond to their entities.

        Question: {question}
        Extracted Numbers: {extracted_numbers}

        Example:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Extracted Numbers: 6 yards (Chris Johnson's first touchdown), 53 yards (Jason Hanson's first field goal)
        Validation: Valid

        Is the extraction valid? Respond with 'Valid' or 'Invalid'.
        """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "valid" in verification_result.lower():
            # Simple parsing
            numbers = re.findall(r'\d+', extracted_numbers)
            return {"is_valid": True, "numbers": numbers}
        else:
            print(f"Numerical info extraction validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")

    return {"is_valid": False, "validation_feedback": "Failed to extract numerical information successfully."}

def calculate_answer(question, numbers, max_attempts=3):
    """Calculate the answer to a numerical question."""
    system_instruction = "You are an expert calculator."

    for attempt in range(max_attempts):
        calculation_prompt = f"""
        Calculate the answer based on the extracted numbers and the original question. Show your work.

        Example:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Numbers: 6, 53
        Calculation: 6 + 53 = 59
        Answer: 59

        Question: {question}
        Numbers: {numbers}
        Calculation:
        """

        calculation_result = call_llm(calculation_prompt, system_instruction)

        verification_prompt = f"""
        Verify if the calculation is correct.

        Question: {question}
        Numbers: {numbers}
        Calculation: {calculation_result}

        Example:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Numbers: 6, 53
        Calculation: 6 + 53 = 59
        Validation: Valid

        Is the calculation valid? Respond with 'Valid' or 'Invalid'.
        """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "valid" in verification_result.lower():
            answer = re.search(r'\d+$', calculation_result)
            if answer:
                return {"is_valid": True, "answer": answer.group(0)}
            else:
                return {"is_valid": False, "validation_feedback": "Could not find numerical answer."}
        else:
            print(f"Calculation validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")

    return {"is_valid": False, "validation_feedback": "Failed to calculate the answer successfully."}

def process_general_question(question):
    """Process a general question."""
    try:
        # Step 1: Decompose the question into sub-questions.
        decomposition_result = decompose_question(question)
        if not decomposition_result.get("is_valid"):
            return f"Error in question decomposition: {decomposition_result.get('validation_feedback')}"
        
        # Step 2: Extract relevant information based on sub-questions.
        information_extraction_result = extract_information(question, decomposition_result["sub_questions"])
        if not information_extraction_result.get("is_valid"):
            return f"Error in information extraction: {information_extraction_result.get('validation_feedback')}"

        # Step 3: Synthesize the answer from extracted information.
        answer_synthesis_result = synthesize_answer(question, information_extraction_result["extracted_info"])
        if not answer_synthesis_result.get("is_valid"):
            return f"Error in answer synthesis: {answer_synthesis_result.get('validation_feedback')}"
        
        return answer_synthesis_result["answer"]

    except Exception as e:
        return f"Error processing general question: {str(e)}"

def decompose_question(question, max_attempts=3):
    """Decompose the main question into smaller, answerable sub-questions."""
    system_instruction = "You are an expert question decomposer."
    
    for attempt in range(max_attempts):
        decomposition_prompt = f"""
        Decompose the given question into smaller, self-contained sub-questions that, when answered, will fully answer the original question.

        Example 1:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Sub-questions:
        1. How many yards was Chris Johnson's first touchdown?
        2. How many yards was Jason Hanson's first field goal?
        3. What is the sum of those two values?

        Example 2:
        Question: Who caught the final touchdown of the game?
        Sub-questions:
        1. Who scored the final touchdown of the game?

        Question: {question}
        Sub-questions:
        """
        
        decomposition_result = call_llm(decomposition_prompt, system_instruction)
        
        # Verify if the decomposition is valid
        verification_prompt = f"""
        Verify if these sub-questions are valid and sufficient to answer the original question.

        Original Question: {question}
        Sub-questions: {decomposition_result}

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Sub-questions: 1. How many yards was Chris Johnson's first touchdown? 2. How many yards was Jason Hanson's first field goal? 3. What is the sum of those two values?
        Validation: Valid

        Is the decomposition valid and sufficient? Respond with 'Valid' or 'Invalid'.
        """
        
        verification_result = call_llm(verification_prompt, system_instruction)
        
        if "valid" in verification_result.lower():
            return {"is_valid": True, "sub_questions": decomposition_result}
        else:
            print(f"Decomposition validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")
            
    return {"is_valid": False, "validation_feedback": "Failed to decompose the question successfully."}

def extract_information(question, sub_questions, max_attempts=3):
    """Extract relevant information from the passage based on the sub-questions."""
    system_instruction = "You are an information extraction expert."
    
    for attempt in range(max_attempts):
        extraction_prompt = f"""
        Given the original question and its sub-questions, extract the relevant information from the passage required to answer the sub-questions.

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Sub-questions:
        1. How many yards was Chris Johnson's first touchdown?
        2. How many yards was Jason Hanson's first field goal?
        Extracted Information:
        Chris Johnson's first touchdown was 6 yards. Jason Hanson's first field goal was 53 yards.

        Original Question: {question}
        Sub-questions: {sub_questions}
        Extracted Information:
        """
        
        extracted_info = call_llm(extraction_prompt, system_instruction)
        
        # Validate information extraction
        verification_prompt = f"""
        Verify if the extracted information is relevant and sufficient to answer the sub-questions.

        Original Question: {question}
        Sub-questions: {sub_questions}
        Extracted Information: {extracted_info}

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Sub-questions: 1. How many yards was Chris Johnson's first touchdown? 2. How many yards was Jason Hanson's first field goal?
        Extracted Information: Chris Johnson's first touchdown was 6 yards. Jason Hanson's first field goal was 53 yards.
        Validation: Valid

        Is the extraction relevant and sufficient? Respond with 'Valid' or 'Invalid'.
        """
        
        verification_result = call_llm(verification_prompt, system_instruction)
        
        if "valid" in verification_result.lower():
            return {"is_valid": True, "extracted_info": extracted_info}
        else:
            print(f"Information extraction validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")
            
    return {"is_valid": False, "validation_feedback": "Failed to extract relevant information successfully."}

def synthesize_answer(question, extracted_info, max_attempts=3):
    """Synthesize the answer from the extracted information to answer the main question."""
    system_instruction = "You are an answer synthesis expert."

    for attempt in range(max_attempts):
        synthesis_prompt = f"""
        Given the original question and the extracted information, synthesize the final answer.

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Extracted Information: Chris Johnson's first touchdown was 6 yards. Jason Hanson's first field goal was 53 yards.
        Final Answer: 59

        Original Question: {question}
        Extracted Information: {extracted_info}
        Final Answer:
        """
        
        answer = call_llm(synthesis_prompt, system_instruction)

        # Answer checker
        verification_prompt = f"""
        Check if the answer is correct and answers the original question fully.

        Original Question: {question}
        Synthesized Answer: {answer}

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Synthesized Answer: 59
        Validation: Valid

        Is the answer correct and complete? Respond with 'Valid' or 'Invalid'.
        """
        
        verification_result = call_llm(verification_prompt, system_instruction)

        if "valid" in verification_result.lower():
            return {"is_valid": True, "answer": answer}
        else:
            print(f"Answer synthesis validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")
            
    return {"is_valid": False, "validation_feedback": "Failed to synthesize a valid answer."}

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types
        import os  # Import the os module

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