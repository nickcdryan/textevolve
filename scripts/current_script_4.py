import os
import re
import math

def main(question):
    """
    Solve the question by extracting relevant information from the passage and using chain-of-thought reasoning.
    This approach builds upon a prior attempt to use question decomposition, strengthens answer synthesis, and includes examples in all LLM prompts.
    Includes arithmetic module and refines prompts for numerical questions.
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
        3. What arithmetic operation needs to be performed with the extracted data?

        Example 2:
        Question: Who caught the final touchdown of the game?
        Sub-questions:
        1. Who scored the final touchdown of the game?

        Example 3:
        Question: How many points did the Steelers score in the first quarter?
        Sub-questions:
        1. How many points did the Steelers score in the first quarter?

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
        Sub-questions: 1. How many yards was Chris Johnson's first touchdown? 2. How many yards was Jason Hanson's first field goal? 3. What arithmetic operation needs to be performed with the extracted data?
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
        Given the original question and its sub-questions, extract the relevant information from the passage required to answer the sub-questions. Explicitly identify if an arithmetic operation is needed.

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Sub-questions:
        1. How many yards was Chris Johnson's first touchdown?
        2. How many yards was Jason Hanson's first field goal?
        3. What arithmetic operation needs to be performed with the extracted data?
        Extracted Information:
        Chris Johnson's first touchdown was 6 yards. Jason Hanson's first field goal was 53 yards. Operation Needed: Addition

        Original Question: {question}
        Sub-questions: {sub_questions}
        Extracted Information:
        """
        
        extracted_info = call_llm(extraction_prompt, system_instruction)
        
        # Validate information extraction
        verification_prompt = f"""
        Verify if the extracted information is relevant and sufficient to answer the sub-questions. Also confirm if the need for arithmetic operation is stated.

        Original Question: {question}
        Sub-questions: {sub_questions}
        Extracted Information: {extracted_info}

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Sub-questions: 1. How many yards was Chris Johnson's first touchdown? 2. How many yards was Jason Hanson's first field goal? 3. What arithmetic operation needs to be performed with the extracted data?
        Extracted Information: Chris Johnson's first touchdown was 6 yards. Jason Hanson's first field goal was 53 yards. Operation Needed: Addition
        Validation: Valid

        Is the extraction relevant and sufficient? Also, is arithmetic operation needed specified? Respond with 'Valid' or 'Invalid'.
        """
        
        verification_result = call_llm(verification_prompt, system_instruction)
        
        if "valid" in verification_result.lower():
            return {"is_valid": True, "extracted_info": extracted_info}
        else:
            print(f"Information extraction validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")
            
    return {"is_valid": False, "validation_feedback": "Failed to extract relevant information successfully."}

def synthesize_answer(question, extracted_info, max_attempts=3):
    """Synthesize the answer from the extracted information to answer the main question."""
    system_instruction = "You are an answer synthesis expert. You can also perform arithmetic operations."

    for attempt in range(max_attempts):
        synthesis_prompt = f"""
        Given the original question and the extracted information, synthesize the final answer. If the extracted information includes numbers, perform any necessary arithmetic. If no numbers are present, respond directly based on the text.

        Example 1:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Extracted Information: Chris Johnson's first touchdown was 6 yards. Jason Hanson's first field goal was 53 yards. Operation Needed: Addition
        Final Answer: 59

        Example 2:
        Original Question: Who caught the final touchdown of the game?
        Extracted Information:  The passage states that Nate Burleson caught the final touchdown.
        Final Answer: Nate Burleson

        Original Question: {question}
        Extracted Information: {extracted_info}
        Final Answer:
        """
        
        answer = call_llm(synthesis_prompt, system_instruction)

        # Answer checker
        verification_prompt = f"""
        Check if the answer is correct, complete and based on the original question and extracted information.

        Original Question: {question}
        Synthesized Answer: {answer}

        Example 1:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Synthesized Answer: 59
        Validation: Valid

        Example 2:
        Original Question: Who caught the final touchdown of the game?
        Synthesized Answer: Nate Burleson
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