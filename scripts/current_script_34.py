import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with enhanced reasoning and verification.
    """
    try:
        # Step 1: Analyze question type and keywords
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

        # Step 4: Verify answer
        verified_answer = verify_answer(question, answer, relevant_passage, question_analysis)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes the question to identify its type and keywords."""
    system_instruction = "You are an expert at analyzing questions."
    prompt = f"""
    Analyze the question to identify its type (fact extraction, calculation, comparison) and keywords.
    Example 1: Question: Who caught the final touchdown of the game? Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"]}}
    Example 2: Question: How many running backs ran for a touchdown? Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"]}}
    Example 3: Question: Which player kicked the only field goal? Analysis: {{"type": "fact extraction", "keywords": ["player", "field goal"]}}
    Question: {question} Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts the relevant passage from the question based on keywords."""
    system_instruction = "You are an expert at extracting relevant passages."
    prompt = f"""
    Extract the relevant passage from the text based on the question and keywords.
    Example 1: Question: Who caught the final touchdown? Keywords: {{"keywords": ["final touchdown"]}} Text: PASSAGE: Rodgers found Boykin... final score 31-13. Passage: Rodgers found Boykin...31-13.
    Example 2: Question: How many running backs ran for a touchdown? Keywords: {{"keywords": ["running backs", "touchdown"]}} Text: PASSAGE: Johnson got a 6-yard TD run...White getting a 6-yard...TD run. Passage: Johnson got a 6-yard TD run...White getting a 6-yard...TD run.
    Question: {question} Keywords: {question_analysis} Text: {question} Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer based on the question, relevant passage, and question type."""
    system_instruction = "You are an expert at generating answers."
    prompt = f"""
    Generate the answer to the question based on the passage and question type.
    Example 1: Question: Who caught the final touchdown? Passage: Rodgers found Boykin... final score. Answer: Jarrett Boykin
    Example 2: Question: How many running backs ran for a touchdown? Passage: Johnson got a 6-yard TD run...White getting a 6-yard...TD run. Answer: 2
    Question: {question} Passage: {relevant_passage} Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage, question_analysis):
    """Verifies the generated answer and ensures completeness. Includes examples."""
    system_instruction = "You are an expert at verifying answers to questions, return the correct answer."
    prompt = f"""
    Verify the answer. If correct, return the answer, if not return the correct answer.
    Example 1: Question: Who caught the final touchdown? Answer: Boykin Passage: Rodgers found Boykin... final score. Verification: Jarrett Boykin
    Example 2: Question: How many running backs ran for a touchdown? Answer: 2 Passage: Johnson got a 6-yard TD run...White getting a 6-yard...TD run. Verification: 2
    Question: {question} Answer: {answer} Passage: {relevant_passage} Verification:
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