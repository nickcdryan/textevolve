import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with enhanced verification.
    """
    try:
        # Step 1: Analyze question
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question"

        # Step 2: Extract relevant passage
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return "Error extracting passage"

        # Step 3: Generate answer
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer"

        # Step 4: Verify answer with enhanced checks
        verified_answer = verify_answer(question, answer, relevant_passage, question_analysis)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes question type, keywords, and if calculation is needed."""
    system_instruction = "Expert at analyzing questions for type, keywords, calculation needs."
    prompt = f"""
    Analyze the question, identify its type (fact extraction, calculation, comparison), keywords, and if calculation is needed (yes/no).

    Example 1:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_needed": "yes"}}

    Example 2:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "calculation_needed": "no"}}

    Example 3:
    Question: How many more yards did X gain than Y?
    Analysis: {{"type": "comparison", "keywords": ["yards", "X", "Y"], "calculation_needed": "yes"}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts relevant passage based on question and keywords."""
    system_instruction = "Expert at extracting relevant passages from text."
    prompt = f"""
    Extract the relevant passage from the text based on the question and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game? PASSAGE: Text about the game... Rodgers found Boykin on a touchdown.
    Keywords: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "calculation_needed": "no"}}
    Passage: Rodgers found Boykin on a touchdown.

    Example 2:
    Question: How many running backs ran for a touchdown? PASSAGE: Chris Johnson and LenDale White ran for touchdowns.
    Keywords: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_needed": "yes"}}
    Passage: Chris Johnson and LenDale White ran for touchdowns.

    Question: {question}
    Keywords: {question_analysis}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer based on the question, passage, and analysis."""
    system_instruction = "Expert at generating answers from provided text, be concise."
    prompt = f"""
    Generate a concise answer to the question based on the relevant passage and question analysis.

    Example 1:
    Question: Who caught the final touchdown of the game? Passage: Rodgers found Boykin.
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "calculation_needed": "no"}}
    Answer: Boykin

    Example 2:
    Question: How many running backs ran for a touchdown? Passage: Chris Johnson and LenDale White ran for touchdowns.
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_needed": "yes"}}
    Answer: 2

    Question: {question}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage, question_analysis):
    """Verifies the answer and performs calculation if needed."""
    system_instruction = "Expert at verifying answers and performing calculations if needed."
    prompt = f"""
    Verify if the answer is correct based on the passage. If calculation is needed, perform the calculation and provide the correct answer. Return ONLY the final correct answer.
    
    Example 1:
    Question: Who caught the final touchdown? Answer: Boykin. Passage: Rodgers found Boykin. Analysis: no calc
    Verification: Boykin

    Example 2:
    Question: How many running backs ran for a TD? Answer: 1. Passage: Only Johnson ran. Analysis: calc needed
    Verification: 1

    Example 3:
    Question: How many yards did X gain? Answer: 5. Passage: X and Y gained some yards.
    Analysis: no calculation needed.
    Verification: 5
    
    Question: {question}
    Answer: {answer}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
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
        return f"Error: {str(e)}"