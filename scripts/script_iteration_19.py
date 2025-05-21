import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with enhanced error handling and examples.
    """
    try:
        # Step 1: Analyze question type and keywords
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return f"Error analyzing question: {question_analysis}"

        # Step 2: Extract relevant passage using identified keywords
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return f"Error extracting passage: {relevant_passage}"

        # Step 3: Generate answer using extracted passage and question type
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return f"Error generating answer: {answer}"

        # Step 4: Verify answer
        verified_answer = verify_answer(question, answer, relevant_passage)
        if "Error" in verified_answer:
            return f"Error verifying answer: {verified_answer}"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes the question to identify its type and keywords. Enhanced with examples."""
    system_instruction = "You are an expert at analyzing questions."
    prompt = f"""
    Analyze the question and identify its type (e.g., fact extraction, calculation, comparison) and keywords.

    Example 1: Question: Who caught the final touchdown of the game? Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"]}}
    Example 2: Question: How many running backs ran for a touchdown? Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"]}}
    Example 3: Question: Which player kicked the only field goal of the game? Analysis: {{"type": "fact extraction", "keywords": ["player", "field goal"]}}
    Example 4: Question: What was the attendance at the game? Analysis: {{"type": "fact extraction", "keywords": ["attendance", "game"]}}
    
    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts relevant passage based on keywords. Enhanced with more robust examples."""
    system_instruction = "You are an expert at extracting relevant passages from text."
    prompt = f"""
    Extract the relevant passage from the following text based on the question and keywords.

    Example 1: Question: Who caught the final touchdown of the game? Keywords: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"]}} Text: PASSAGE: ...Rodgers found Jarrett Boykin... final score 31-13. Passage: Rodgers found Jarrett Boykin... final score 31-13.
    Example 2: Question: How many running backs ran for a touchdown? Keywords: {{"type": "counting", "keywords": ["running backs", "touchdown"]}} Text: PASSAGE: ...Chris Johnson got a 6-yard TD run... LenDale White getting a 6-yard and a 2-yard TD run. Passage: Chris Johnson got a 6-yard TD run... LenDale White getting a 6-yard and a 2-yard TD run.
    Example 3: Question: Which player kicked the only field goal of the game? Keywords: {{"type": "fact extraction", "keywords": ["player", "field goal"]}} Text: PASSAGE: ...Josh Scobee nailed a 47-yard field goal. Passage: Josh Scobee nailed a 47-yard field goal.
    Example 4: Question: What was the attendance at the game? Keywords: {{"type": "fact extraction", "keywords": ["attendance", "game"]}} Text: PASSAGE: ...attendance of 75,000... Passage: attendance of 75,000

    Question: {question}
    Keywords: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer. Enhanced with explicit instructions to prioritize the relevant passage."""
    system_instruction = "You are an expert at generating answers based on provided text."
    prompt = f"""
    Generate a concise answer to the question, using ONLY the provided passage.

    Example 1: Question: Who caught the final touchdown? Passage: ...Boykin...final score... Answer: Jarrett Boykin
    Example 2: Question: How many running backs ran for a touchdown? Passage: ...Chris Johnson... LenDale White... Answer: 2
    Example 3: Question: Which player kicked the only field goal? Passage: ...Josh Scobee...field goal. Answer: Josh Scobee
    Example 4: Question: What was the attendance? Passage: ...attendance of 75,000... Answer: 75,000

    Question: {question}
    Passage: {relevant_passage}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage):
    """Verifies and enhances the answer based on the passage. Example to help. Returns extracted answer."""
    system_instruction = "You are an expert at verifying answers and extracting information."
    prompt = f"""
    Carefully verify the answer against the passage. Return the EXACT answer from the passage if it is correct. If incorrect, extract the CORRECT answer from the passage.

    Example 1: Question: Who caught the final touchdown? Answer: Boykin Passage: ...Jarrett Boykin... final score... Verification: Jarrett Boykin
    Example 2: Question: How many running backs ran for a touchdown? Answer: 2 Passage: ...Chris Johnson...LenDale White... Verification: 2
    Example 3: Question: Which player kicked the only field goal? Answer: Scobee Passage: ...Josh Scobee...field goal. Verification: Josh Scobee
    Example 4: Question: What was the attendance? Answer: 75,000 Passage: ...attendance of 75,000... Verification: 75,000

    Question: {question}
    Answer: {answer}
    Passage: {relevant_passage}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response.  This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types

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