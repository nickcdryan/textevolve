import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with enhanced calculation and verification.
    """
    try:
        # Step 1: Identify question type and keywords
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question"

        # Step 2: Extract relevant passage using identified keywords
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return "Error extracting passage"

        # Step 3: Generate answer using extracted passage and question type
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer"

        # Step 4: Verify answer and perform calculation if needed
        verified_answer = verify_answer(question, answer, relevant_passage, question_analysis)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes question, identifies type (fact extraction, calculation), keywords. Few-shot examples included."""
    system_instruction = "You are an expert at analyzing questions to determine type and keywords."
    prompt = f"""
    Analyze the question to identify its type (fact extraction, calculation, comparison) and keywords.

    Example 1: Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_needed": true}}

    Example 2: Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "calculation_needed": false}}

    Example 3: Question: How many more yards did A gain than B?
    Analysis: {{"type": "comparison", "keywords": ["yards", "A", "B"], "calculation_needed": true}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts the relevant passage from the question based on keywords. Includes multiple examples."""
    system_instruction = "You are an expert at extracting relevant passages from text."
    prompt = f"""
    Extract relevant passage based on question and keywords.
    Example 1: Question: Who caught the final touchdown? Keywords: touchdown, caught. Passage: ...Boykin caught a touchdown...
    Passage: Boykin caught a touchdown
    Example 2: Question: How many running backs...? Keywords: running backs, touchdown. Passage: Johnson and White ran for touchdowns...
    Passage: Johnson and White ran for touchdowns
    Question: {question}
    Keywords: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer based on the question, relevant passage, and question type. Includes multiple examples."""
    system_instruction = "You are an expert at generating answers to questions based on provided text."
    prompt = f"""
    Generate the answer based on passage and question type.
    Example 1: Question: Who caught the final touchdown? Passage: Boykin caught a touchdown. Answer: Boykin
    Example 2: Question: How many running backs...? Passage: Johnson and White ran for touchdowns. Answer: 2
    Question: {question}
    Passage: {relevant_passage}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage, question_analysis):
    """Verifies the generated answer and performs calculations if needed. Includes calculation example."""
    system_instruction = "You are an expert at verifying answers, performing calculations if needed."
    prompt = f"""
    Verify the answer. If calculation is needed (indicated in question_analysis), perform it and return the result.

    Example 1: Question: What is 5 + 3? Answer: 8. Calculation Needed: True. Verification: 8
    Example 2: Question: Who caught the touchdown? Answer: Boykin. Calculation Needed: False. Verification: Boykin

    Question: {question}
    Answer: {answer}
    Passage: {relevant_passage}
    Question Analysis: {question_analysis}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template."""
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