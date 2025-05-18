import os
import re

def main(question):
    """
    Solve the question by using a NEW approach: "Iterative Question Refinement with Contextual Expansion."
    This strategy aims to improve accuracy by:
    1. Iteratively refining the question to remove ambiguity and focus on core information needs.
    2. Dynamically expanding the context around key entities to enhance information extraction.
    3. Includes a "relevance filter" that filters the information based on question to the agent, which is designed to force minimal inclusion
    4. Each step involves a dedicated LLM agent with a specific system instruction for its task, along with examples in the prompt.
    Each stage validates its results by prompting the LLM to make sure it is "valid" or "invalid".
    """
    try:
        # Step 1: Refine the question to improve clarity and focus.
        refined_question_result = refine_question(question)
        if not refined_question_result.get("is_valid"):
            return f"Error in question refinement: {refined_question_result.get('validation_feedback')}"
        refined_question = refined_question_result["refined_question"]

        # Step 2: Extract key entities from the refined question
        entity_extraction_result = extract_entities(refined_question)
        if not entity_extraction_result.get("is_valid"):
            return f"Error in entity extraction: {entity_extraction_result.get('validation_feedback')}"
        entities = entity_extraction_result["entities"]
        
        # Step 3: Extract relevant information based on entities, expanding the context around them.
        information_extraction_result = extract_information(refined_question, entities)
        if not information_extraction_result.get("is_valid"):
            return f"Error in information extraction: {information_extraction_result.get('validation_feedback')}"
        extracted_info = information_extraction_result["extracted_info"]
            
        # Step 4: Synthesize the answer from extracted information.
        answer_synthesis_result = synthesize_answer(refined_question, extracted_info)
        if not answer_synthesis_result.get("is_valid"):
            return f"Error in answer synthesis: {answer_synthesis_result.get('validation_feedback')}"
        
        return answer_synthesis_result["answer"]

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def refine_question(question, max_attempts=3):
    """Refine the question to improve clarity and focus."""
    system_instruction = "You are an expert question refiner. Refine the question without changing its meaning."
    for attempt in range(max_attempts):
        refinement_prompt = f"""
        Refine the given question to make it more specific and easier to answer. Do NOT change the meaning of the question.

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Refined Question: What is the sum of Chris Johnson's first touchdown yards and Jason Hanson's first field goal yards?

        Original Question: {question}
        Refined Question:
        """
        refinement_result = call_llm(refinement_prompt, system_instruction)

        # Verify
        verification_prompt = f"""
        Verify if the refined question is valid and maintains the meaning of the original question.

        Original Question: {question}
        Refined Question: {refinement_result}

        Example:
        Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Refined Question: What is the sum of Chris Johnson's first touchdown yards and Jason Hanson's first field goal yards?
        Validation: Valid

        Is the refinement valid? Respond with 'Valid' or 'Invalid'.
        """
        verification_result = call_llm(verification_prompt, system_instruction)
        if "valid" in verification_result.lower():
            return {"is_valid": True, "refined_question": refinement_result}
        else:
            print(f"Refinement validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")
    return {"is_valid": False, "validation_feedback": "Failed to refine the question successfully."}

def extract_entities(question, max_attempts=3):
    """Extract key entities from the question."""
    system_instruction = "You are an expert entity extractor. Identify key entities."
    for attempt in range(max_attempts):
        extraction_prompt = f"""
        Extract the key entities from the given question.
        Example:
        Question: What is the sum of Chris Johnson's first touchdown yards and Jason Hanson's first field goal yards?
        Entities: Chris Johnson, Jason Hanson, touchdown, field goal, yards

        Question: {question}
        Entities:
        """
        extraction_result = call_llm(extraction_prompt, system_instruction)
        verification_prompt = f"""
        Verify if the extracted entities are valid and complete for the question.
        Question: {question}
        Entities: {extraction_result}
        Example:
        Question: What is the sum of Chris Johnson's first touchdown yards and Jason Hanson's first field goal yards?
        Entities: Chris Johnson, Jason Hanson, touchdown, field goal, yards
        Validation: Valid

        Is the extraction valid? Respond with 'Valid' or 'Invalid'.
        """
        verification_result = call_llm(verification_prompt, system_instruction)

        if "valid" in verification_result.lower():
            return {"is_valid": True, "entities": extraction_result.split(', ')}
        else:
            print(f"Entity extraction validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")
    return {"is_valid": False, "validation_feedback": "Failed to extract entities successfully."}

def extract_information(question, entities, max_attempts=3):
    """Extract relevant information, expanding the context around entities."""
    system_instruction = "You are an expert information extractor. Be as concise as possible."
    for attempt in range(max_attempts):
        extraction_prompt = f"""
        Given the question and entities, extract relevant information. Only provide the relevant information with the least amount of words and context.

        Example:
        Question: What is the sum of Chris Johnson's first touchdown yards and Jason Hanson's first field goal yards?
        Entities: Chris Johnson, Jason Hanson, touchdown, field goal, yards
        Extracted Information: Chris Johnson's first touchdown was 6 yards. Jason Hanson's first field goal was 53 yards.

        Question: {question}
        Entities: {entities}
        Extracted Information:
        """
        extracted_info = call_llm(extraction_prompt, system_instruction)
        verification_prompt = f"""
        Verify if the extracted information is relevant and sufficient to answer the question.
        Question: {question}
        Extracted Information: {extracted_info}
        Example:
        Question: What is the sum of Chris Johnson's first touchdown yards and Jason Hanson's first field goal yards?
        Extracted Information: Chris Johnson's first touchdown was 6 yards. Jason Hanson's first field goal was 53 yards.
        Validation: Valid
        
        Is the extraction valid? Respond with 'Valid' or 'Invalid'.
        """
        verification_result = call_llm(verification_prompt, system_instruction)
        if "valid" in verification_result.lower():
            return {"is_valid": True, "extracted_info": extracted_info}
        else:
            print(f"Information extraction validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")
    return {"is_valid": False, "validation_feedback": "Failed to extract relevant information successfully."}

def synthesize_answer(question, extracted_info, max_attempts=3):
    """Synthesize the answer from the extracted information."""
    system_instruction = "You are an answer synthesis expert. Answer the question based on the extracted information. Don't include anything else."
    for attempt in range(max_attempts):
        synthesis_prompt = f"""
        Given the question and extracted information, synthesize the final answer.

        Example:
        Question: What is the sum of Chris Johnson's first touchdown yards and Jason Hanson's first field goal yards?
        Extracted Information: Chris Johnson's first touchdown was 6 yards. Jason Hanson's first field goal was 53 yards.
        Final Answer: 59 yards

        Question: {question}
        Extracted Information: {extracted_info}
        Final Answer:
        """
        answer = call_llm(synthesis_prompt, system_instruction)
        verification_prompt = f"""
        Check if the answer is correct and answers the original question fully.
        Question: {question}
        Answer: {answer}
        Example:
        Question: What is the sum of Chris Johnson's first touchdown yards and Jason Hanson's first field goal yards?
        Answer: 59 yards
        Validation: Valid

        Is the answer valid? Respond with 'Valid' or 'Invalid'.
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

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
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