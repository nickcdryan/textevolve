import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach. Enhanced for ordinality and error handling.
    """
    try:
        # Step 1: Identify question type and keywords
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question"

        # Step 2: Extract relevant passage, enhance to find *all* relevant info
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return "Error extracting passage"

        # Step 3: Generate answer using extracted passage and question type
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer"

        # Step 4: Verify answer
        verified_answer = verify_answer(question, answer, relevant_passage)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes the question, identifying type and keywords. Now handles ordinal questions."""
    system_instruction = "You are an expert question analyzer."
    prompt = f"""
    Analyze the question to identify its type and keywords. Identify if it is ordinal.
    Example 1: Question: Who caught the final touchdown? Analysis: {{"type": "fact extraction", "keywords": ["final touchdown"], "ordinal": true}}
    Example 2: Question: How many RBs ran for a TD? Analysis: {{"type": "counting", "keywords": ["RB", "touchdown"], "ordinal": false}}
    Example 3: Question: Which player kicked the only FG? Analysis: {{"type": "fact extraction", "keywords": ["player", "FG"], "ordinal": false}}
    Question: {question} Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts the relevant passage. Now extracts *all* potentially relevant sentences."""
    system_instruction = "You are an expert at passage extraction."
    prompt = f"""
    Extract all sentences that *might* be relevant to the question.
    Example 1: Question: Who caught the final TD? Passage: "The final TD was caught by X."
    Example 2: Question: How many RBs scored? Passage: "RB A scored. RB B also scored."
    Example 3: Question: Which player kicked the only FG? Passage: "Player C kicked the only FG."
    Question: {question} Text: {question} Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer. Enhanced to handle ordinality and complex reasoning."""
    system_instruction = "You are an expert answer generator."
    prompt = f"""
    Generate the answer from the passage, using the question type.
    Example 1: Question: Who caught the final TD? Passage: "The final TD was caught by X." Answer: X
    Example 2: Question: How many RBs scored? Passage: "RB A scored. RB B also scored." Answer: 2
    Example 3: Question: Which player kicked the only FG? Passage: "Player C kicked the only FG." Answer: Player C
    Question: {question} Passage: {relevant_passage} Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage):
    """Verifies the generated answer. Enhanced for feedback if incorrect."""
    system_instruction = "You are an expert answer verifier."
    prompt = f"""
    Verify if the answer is correct based on the passage. If incorrect, provide the correct answer.
    Example 1: Question: Who caught the final TD? Answer: X. Passage: "The final TD was caught by X." Verification: X
    Example 2: Question: How many RBs scored? Answer: 3. Passage: "RB A scored. RB B also scored." Verification: 2
    Question: {question} Answer: {answer} Passage: {relevant_passage} Verification:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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