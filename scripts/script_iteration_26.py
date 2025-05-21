import os
import re
import math

def main(question):
    """Solve the question using a multi-stage LLM approach with enhanced reasoning and error handling."""
    try:
        # Step 1: Analyze question (enhanced type and keyword identification)
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question: " + question_analysis

        # Step 2: Extract relevant passage (focused extraction)
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return "Error extracting passage: " + relevant_passage

        # Step 3: Generate answer (explicit calculation instruction)
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer: " + answer

        # Step 4: Verify answer (numerical accuracy and unit check)
        verified_answer = verify_answer(question, answer, relevant_passage, question_analysis)
        if "Error" in verified_answer:
            return "Error verifying answer: " + verified_answer
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes the question to identify its type, keywords, and whether calculation is needed."""
    system_instruction = "You are an expert question analyzer."
    prompt = f"""
    Analyze the question and identify its type, keywords, and if calculation is needed.

    Example 1:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_needed": false}}

    Example 2:
    Question: How many total points were scored in the game?
    Analysis: {{"type": "calculation", "keywords": ["total points", "scored"], "calculation_needed": true}}

    Example 3:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "calculation_needed": false}}
    
    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts the relevant passage based on keywords, with a focus on extracting all key information."""
    system_instruction = "You are an expert at extracting relevant passages from text to answer questions."
    prompt = f"""
    Extract the relevant passage containing the answer.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Keywords: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "calculation_needed": false}}
    Text: PASSAGE: ...Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Passage: Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.

    Example 2:
    Question: How many running backs ran for a touchdown?
    Keywords: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_needed": false}}
    Text: PASSAGE: ...Chris Johnson got a 6-yard TD run. LenDale White got a 6-yard and a 2-yard TD run.
    Passage: Chris Johnson got a 6-yard TD run. LenDale White got a 6-yard and a 2-yard TD run.

    Example 3:
    Question: How many total points were scored in the game?
    Keywords: {{"type": "calculation", "keywords": ["total points", "scored"], "calculation_needed": true}}
    Text: PASSAGE: ...The score was 24-17...They scored a touchdown and extra point...
    Passage: The score was 24-17...They scored a touchdown and extra point...

    Question: {question}
    Keywords: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer. If calculation is needed, explicitly instruct the LLM to calculate."""
    system_instruction = "You are an expert at generating answers to questions from provided text. If a calculation is needed, perform the calculation."
    prompt = f"""
    Generate the answer to the question. If the question requires a calculation, perform it.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Passage: Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Answer: Jarrett Boykin

    Example 2:
    Question: How many running backs ran for a touchdown?
    Passage: Chris Johnson got a 6-yard TD run. LenDale White got a 6-yard and a 2-yard TD run.
    Answer: 2

    Example 3:
    Question: How many total points were scored in the game?
    Passage: The score was 24-17...They scored a touchdown and extra point...(7 points)
    Answer: 48
    
    Question: {question}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage, question_analysis):
    """Verifies the generated answer, performing explicit numerical checks if needed."""
    system_instruction = "You are an expert at verifying answers, especially numerical accuracy."
    prompt = f"""
    Verify the answer. If the question involved calculation, check the numerical accuracy. Return the verified answer, or the correct answer if the original was wrong.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Answer: Jarrett Boykin
    Passage: Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Verification: Jarrett Boykin

    Example 2:
    Question: How many running backs ran for a touchdown?
    Answer: 2
    Passage: Chris Johnson got a 6-yard TD run. LenDale White got a 6-yard and a 2-yard TD run.
    Verification: 2

    Example 3:
    Question: How many total points were scored in the game?
    Answer: 48
    Passage: The score was 24-17...They scored a touchdown and extra point...(7 points)
    Verification: 48

    Question: {question}
    Answer: {answer}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        if system_instruction:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(system_instruction=system_instruction),
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