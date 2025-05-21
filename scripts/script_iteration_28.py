import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with enhanced reasoning and validation.
    """
    try:
        # Step 1: Analyze question to identify type and keywords, including calculation need
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question"

        # Step 2: Extract relevant passage using keywords, now with unit awareness
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return "Error extracting passage"

        # Step 3: Generate answer, explicitly instructing to perform calculations if needed
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer"

        # Step 4: Verify answer, checking for calculation completeness and unit inclusion
        verified_answer = verify_answer(question, answer, relevant_passage, question_analysis)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes question, identifying type, keywords, and if calculation is needed. Includes examples."""
    system_instruction = "You are an expert question analyzer determining type, keywords, and calculation needs."
    prompt = f"""
    Analyze the question, identify its type (fact extraction, calculation, comparison), keywords, and if calculation is needed.

    Example 1:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_needed": false}}

    Example 2:
    Question: How many more yards did Chris Johnson gain than LenDale White?
    Analysis: {{"type": "comparison", "keywords": ["yards", "Chris Johnson", "LenDale White"], "calculation_needed": true}}

    Example 3:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "calculation_needed": false}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts relevant passage based on keywords. Focuses on unit awareness. Includes examples."""
    system_instruction = "You are an expert at extracting relevant passages, focusing on unit awareness."
    prompt = f"""
    Extract the relevant passage based on the question, keywords, and especially units.

    Example 1:
    Question: How many yards did Chris Johnson gain?
    Keywords: {{"keywords": ["yards", "Chris Johnson"]}}
    Text: PASSAGE: Chris Johnson ran for 120 yards.
    Passage: Chris Johnson ran for 120 yards.

    Example 2:
    Question: What was the final score?
    Keywords: {{"keywords": ["final score"]}}
    Text: PASSAGE: The final score was 27-14.
    Passage: The final score was 27-14.

     Example 3:
    Question: How many touchdowns were scored in the first quarter?
    Keywords: {{"keywords": ["touchdowns", "first quarter"]}}
    Text: PASSAGE: Two touchdowns were scored in the first quarter.
    Passage: Two touchdowns were scored in the first quarter.

    Question: {question}
    Keywords: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates answer, with explicit instruction to perform calculation if needed. Includes examples."""
    system_instruction = "You are an expert at generating answers, performing calculations if needed."
    prompt = f"""
    Generate the answer, performing any necessary calculations based on the question type.

    Example 1:
    Question: What was the final score of the game?
    Passage: The final score was 27-14.
    Answer: 27-14

    Example 2:
    Question: How many yards did Chris Johnson gain?
    Passage: Chris Johnson ran for 120 yards.
    Answer: 120 yards

    Example 3:
    Question: How many more yards did Chris Johnson gain than LenDale White?
    Passage: Chris Johnson ran for 120 yards. LenDale White ran for 80 yards.
    Answer: 40 yards

    Question: {question}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage, question_analysis):
    """Verifies answer, checking calculation completeness and unit inclusion. Includes examples."""
    system_instruction = "You are an expert at verifying answers, checking calculations and units."
    prompt = f"""
    Verify the answer, checking that all necessary calculations were completed and units are included. Return the answer if correct, else return the correct answer.

    Example 1:
    Question: What was the final score of the game?
    Answer: 27-14
    Passage: The final score was 27-14.
    Verification: 27-14

    Example 2:
    Question: How many yards did Chris Johnson gain?
    Answer: 120 yards
    Passage: Chris Johnson ran for 120 yards.
    Verification: 120 yards

    Example 3:
    Question: How many more yards did Chris Johnson gain than LenDale White?
    Answer: 40 yards
    Passage: Chris Johnson ran for 120 yards. LenDale White ran for 80 yards.
    Verification: 40 yards

    Question: {question}
    Answer: {answer}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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