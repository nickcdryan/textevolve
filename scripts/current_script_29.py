import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with robust numerical reasoning.
    """
    try:
        # Step 1: Identify question type and keywords
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question"

        # Step 2: Extract relevant passage
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return "Error extracting passage"

        # Step 3: Generate answer with numerical reasoning capabilities
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer"

        # Step 4: Verify answer with enhanced numerical checks
        verified_answer = verify_answer(question, answer, relevant_passage, question_analysis)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes the question to identify type and keywords."""
    system_instruction = "You are an expert question analyzer."
    prompt = f"""
    Analyze the question. Identify the question type (fact extraction, calculation, comparison) and keywords.
    
    Example 1:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "requires_calculation": false}}

    Example 2:
    Question: How many more points did Team A score than Team B?
    Analysis: {{"type": "comparison", "keywords": ["more points", "Team A", "Team B"], "requires_calculation": true}}

    Example 3:
    Question: Which player kicked the only field goal of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["player", "field goal"], "requires_calculation": false}}
    
    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts the relevant passage based on question and keywords."""
    system_instruction = "You are an expert passage extractor."
    prompt = f"""
    Extract the relevant passage from the question based on keywords.
    
    Example 1:
    Question: Who caught the final touchdown of the game?
    Keywords: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"]}}
    Text: PASSAGE: The Packers won when Rodgers found Boykin on a 20-yard pass.
    Passage: The Packers won when Rodgers found Boykin on a 20-yard pass.
    
    Example 2:
    Question: How many running backs ran for a touchdown?
    Keywords: {{"type": "counting", "keywords": ["running backs", "touchdown"]}}
    Text: PASSAGE: Johnson and White scored touchdowns.
    Passage: Johnson and White scored touchdowns.

    Example 3:
    Question: Which player kicked the only field goal of the game?
    Keywords: {{"type": "fact extraction", "keywords": ["player", "field goal"]}}
    Text: PASSAGE: Scobee kicked a field goal.
    Passage: Scobee kicked a field goal.

    Question: {question}
    Keywords: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer based on the question, relevant passage, and question type."""
    system_instruction = "You are an expert answer generator with numerical reasoning."
    prompt = f"""
    Generate the answer based on the passage. If the question requires calculation, perform it.
    
    Example 1:
    Question: How many running backs ran for a touchdown?
    Passage: Johnson and White scored touchdowns.
    Answer: 2

    Example 2:
    Question: What was the final score of the game, Team A vs. Team B? Team A scored 2 touchdowns (7 points each) and 1 field goal (3 points). Team B scored 1 touchdown (7 points).
    Passage: Team A scored 2 touchdowns (7 points each) and 1 field goal (3 points). Team B scored 1 touchdown (7 points).
    Answer: Team A: 17, Team B: 7

    Example 3:
    Question: Which player kicked the only field goal of the game?
    Passage: Scobee kicked a field goal.
    Answer: Scobee

    Question: {question}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage, question_analysis):
    """Verifies the generated answer, includes numerical checks."""
    system_instruction = "You are an expert answer verifier with numerical reasoning."
    prompt = f"""
    Verify the answer based on the passage. If a calculation is required, check the calculation. Return the answer if it is correct. Return the correct answer if it is incorrect.

    Example 1:
    Question: How many running backs ran for a touchdown?
    Answer: 2
    Passage: Johnson and White scored touchdowns.
    Verification: 2

    Example 2:
    Question: What was the final score of the game, Team A vs. Team B? Team A scored 2 touchdowns (7 points each) and 1 field goal (3 points). Team B scored 1 touchdown (7 points).
    Answer: Team A: 17, Team B: 7
    Passage: Team A scored 2 touchdowns (7 points each) and 1 field goal (3 points). Team B scored 1 touchdown (7 points).
    Verification: Team A: 17, Team B: 7

    Example 3:
    Question: Which player kicked the only field goal of the game?
    Answer: Scobee
    Passage: Scobee kicked a field goal.
    Verification: Scobee

    Question: {question}
    Answer: {answer}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM. Use this example without modification."""
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