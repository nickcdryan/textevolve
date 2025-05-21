import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with enhanced arithmetic reasoning.
    """
    try:
        # Step 1: Analyze question and identify if calculation is needed
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question"

        # Step 2: Extract relevant passage
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return "Error extracting passage"

        # Step 3: Generate answer, performing calculation if required
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer"

        # Step 4: Verify answer
        verified_answer = verify_answer(question, answer, relevant_passage, question_analysis)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes question, identifying type, keywords, and need for calculation."""
    system_instruction = "Analyze questions, determine type, keywords, and if calculation is needed."
    prompt = f"""
    Analyze the question, identifying its type (fact extraction, calculation, comparison), keywords, and whether a calculation is required.

    Example 1:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_required": false}}

    Example 2:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Analysis: {{"type": "calculation", "keywords": ["Chris Johnson", "touchdown", "Jason Hanson", "field goal"], "calculation_required": true}}

    Example 3:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "calculation_required": false}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts relevant passage based on question and analysis."""
    system_instruction = "Extract relevant passages from text."
    prompt = f"""
    Extract the relevant passage from the following text based on the question, keywords and question type.

    Example 1:
    Question: How many running backs ran for a touchdown?
    Keywords: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_required": false}}
    Text: PASSAGE: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.

    Example 2:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Keywords: {{"type": "calculation", "keywords": ["Chris Johnson", "touchdown", "Jason Hanson", "field goal"], "calculation_required": true}}
    Text: PASSAGE: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. The Lions would respond with kicker Jason Hanson getting a 53-yard field goal.
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. The Lions would respond with kicker Jason Hanson getting a 53-yard field goal.

    Question: {question}
    Keywords: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer, performing calculation if needed."""
    system_instruction = "Generate answers based on provided text, performing calculations if required."
    prompt = f"""
    Generate the answer to the question based on the relevant passage and question analysis. If a calculation is required, perform the calculation and provide the result.

    Example 1:
    Question: How many running backs ran for a touchdown?
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_required": false}}
    Answer: 2

    Example 2:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. The Lions would respond with kicker Jason Hanson getting a 53-yard field goal.
    Analysis: {{"type": "calculation", "keywords": ["Chris Johnson", "touchdown", "Jason Hanson", "field goal"], "calculation_required": true}}
    Answer: 59

    Question: {question}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage, question_analysis):
    """Verifies the generated answer."""
    system_instruction = "Verify answers to questions and correct if needed."
    prompt = f"""
    Verify the answer to the question based on the relevant passage and the question analysis. If the answer is incorrect, provide the correct answer.

    Example 1:
    Question: How many running backs ran for a touchdown?
    Answer: 2
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_required": false}}
    Verification: 2

    Example 2:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Answer: 59
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. The Lions would respond with kicker Jason Hanson getting a 53-yard field goal.
        Analysis: {{"type": "calculation", "keywords": ["Chris Johnson", "touchdown", "Jason Hanson", "field goal"], "calculation_required": true}}
    Verification: 59

    Question: {question}
    Answer: {answer}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. """
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