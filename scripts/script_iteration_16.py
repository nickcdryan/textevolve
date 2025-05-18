import os
import re

def main(question):
    """
    This script takes a question and applies a "Dual Verification with Iterative Refinement" approach.
    The hypothesis is that by having two independent LLM calls verify the extracted information and synthesized answer,
    we can improve the overall accuracy and robustness of the system. Also, we will be trying a completely new decomposition, the QA decomposition.
    """
    try:
        # Step 1: Decompose the question into question/answer pairs to guide extraction
        decomposition_result = decompose_question_answer(question)
        if not decomposition_result.get("is_valid"):
            return f"Error in QA decomposition: {decomposition_result.get('validation_feedback')}"
        
        # Step 2: Extract relevant information based on QA pairs with verification
        information_extraction_result = extract_information(question, decomposition_result["qa_pairs"])
        if not information_extraction_result.get("is_valid"):
            return f"Error in information extraction: {information_extraction_result.get('validation_feedback')}"

        # Step 3: Synthesize the answer from extracted information with verification
        answer_synthesis_result = synthesize_answer(question, information_extraction_result["extracted_info"])
        if not answer_synthesis_result.get("is_valid"):
            return f"Error in answer synthesis: {answer_synthesis_result.get('validation_feedback')}"
        
        return answer_synthesis_result["answer"]

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def decompose_question_answer(question, max_attempts=3):
    """Decompose the main question into question/answer pairs to guide extraction."""
    system_instruction = "You are an expert at creating question/answer pairs from a question."
    
    for attempt in range(max_attempts):
        decomposition_prompt = f"""
        Decompose the given question into question and expected answer skeleton pairs.
        This decomposition helps to understand what information should be extracted.

        Example 1:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Question/Answer pairs:
        1. Q: How many yards was Chris Johnson's first touchdown? A: [number] yards
        2. Q: How many yards was Jason Hanson's first field goal? A: [number] yards
        3. Q: What is the sum of those two values? A: [number] yards

        Example 2:
        Question: Which happened later, Chinese invasion of tibet or the outbreak of the Xinhai Revolution?
        Question/Answer pairs:
        1. Q: When was the Chinese invasion of Tibet? A: [date]
        2. Q: When did the outbreak of the Xinhai Revolution occur? A: [date]
        3. Q: Which of the two dates is later? A: [event]

        Question: {question}
        Question/Answer pairs:
        """
        
        decomposition_result = call_llm(decomposition_prompt, system_instruction)
        
        verification_prompt = f"""
        Verify if these question and answer pairs are valid and sufficient to answer the original question.

        Original Question: {question}
        Question/Answer pairs: {decomposition_result}

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Question/Answer pairs: 1. Q: How many yards was Chris Johnson's first touchdown? A: [number] yards 2. Q: How many yards was Jason Hanson's first field goal? A: [number] yards 3. Q: What is the sum of those two values? A: [number] yards
        Validation: Valid

        Is the decomposition valid and sufficient? Respond with 'Valid' or 'Invalid'.
        """
        
        verification_result = call_llm(verification_prompt, system_instruction)
        
        if "valid" in verification_result.lower():
            return {"is_valid": True, "qa_pairs": decomposition_result}
        else:
            print(f"QA decomposition validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")
            
    return {"is_valid": False, "validation_feedback": "Failed to create valid question/answer pairs successfully."}

def extract_information(question, qa_pairs, max_attempts=3):
    """Extract relevant information from the passage based on the question/answer pairs with dual verification."""
    system_instruction = "You are an information extraction expert."
    
    for attempt in range(max_attempts):
        extraction_prompt = f"""
        Given the original question and its question/answer pairs, extract the relevant information.

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Question/Answer pairs:
        1. Q: How many yards was Chris Johnson's first touchdown? A: [number] yards
        2. Q: How many yards was Jason Hanson's first field goal? A: [number] yards
        3. Q: What is the sum of those two values? A: [number] yards
        Extracted Information:
        Chris Johnson's first touchdown was 6 yards. Jason Hanson's first field goal was 53 yards.

        Original Question: {question}
        Question/Answer pairs: {qa_pairs}
        Extracted Information:
        """
        
        extracted_info = call_llm(extraction_prompt, system_instruction)

        verification_prompt = f"""
        Verify if the extracted information is relevant and sufficient to answer the question/answer pairs.

        Original Question: {question}
        Question/Answer pairs: {qa_pairs}
        Extracted Information: {extracted_info}

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Question/Answer pairs: 1. Q: How many yards was Chris Johnson's first touchdown? A: [number] yards 2. Q: How many yards was Jason Hanson's first field goal? A: [number] yards 3. Q: What is the sum of those two values? A: [number] yards
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
    """Synthesize the answer from the extracted information with dual verification."""
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