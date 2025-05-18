import os
import re

def main(question):
    """
    Solve the question by extracting relevant information from the passage and using chain-of-thought reasoning.
    This approach builds upon successful techniques from previous iterations: question decomposition, information extraction, and answer synthesis.
    Includes detailed embedded examples in every LLM prompt for improved performance and targeted error handling.
    """
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
        return f"An unexpected error occurred: {str(e)}"

def decompose_question(question, max_attempts=3):
    """Decompose the main question into smaller, answerable sub-questions. Includes an example for prompt engineering."""
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
    """Extract relevant information from the passage based on the sub-questions. Example is now present for prompt engineering."""
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
    """Synthesize the answer from the extracted information to answer the main question. Example is present."""
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
        import os

        # Initialize the Gemini client
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-pro')

        # Call the API with system instruction if provided
        if system_instruction:
            response = model.generate_content(
                [system_instruction, prompt]
            )
        else:
            response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"