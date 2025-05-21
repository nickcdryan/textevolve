import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with enhanced question analysis and verification.
    """
    try:
        # Step 1: Analyze question with improved type and constraint identification
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question"

        # Step 2: Extract relevant passage using identified keywords and constraints
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return "Error extracting passage"

        # Step 3: Generate answer based on the passage and question analysis
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer"

        # Step 4: Verify answer, including explicit unit and logic checks
        verified_answer = verify_answer(question, answer, relevant_passage, question_analysis)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes question for type, keywords, and constraints. Includes more diverse examples."""
    system_instruction = "You are an expert at analyzing questions to determine their type, keywords, and constraints."
    prompt = f"""
    Analyze the question and identify its type (fact extraction, calculation, comparison, counting), keywords, and any constraints.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "constraints": []}}

    Example 2:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "constraints": []}}
    
    Example 3:
    Question: Which player kicked the only field goal of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["player", "field goal"], "constraints": ["only"]}}
    
    Example 4:
    Question: How many more modern artillery pieces than batteries did the 1919 Afghan regular army have?
    Analysis: {{"type": "comparison", "keywords": ["artillery pieces", "batteries"], "constraints": ["1919", "Afghan regular army"]}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts relevant passage based on keywords and constraints. Includes more diverse examples."""
    system_instruction = "You are an expert at extracting relevant passages from text, considering constraints."
    prompt = f"""
    Extract the relevant passage from the following text based on the question, keywords, and constraints.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "constraints": []}}
    Text: PASSAGE: After a tough loss at home, the Browns traveled to take on the Packers. ... The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    
    Example 2:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "constraints": []}}
    Text: PASSAGE: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. The Lions would respond with kicker Jason Hanson getting a 53-yard field goal. The Titans would answer with Johnson getting a 58-yard TD run.
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. 

    Question: {question}
    Analysis: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer based on passage, question type, and constraints. Includes examples."""
    system_instruction = "You are an expert at generating answers to questions, adhering to question constraints."
    prompt = f"""
    Generate the answer to the question based on the relevant passage and analysis. Consider constraints.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Answer: Jarrett Boykin

    Example 2:
    Question: How many running backs ran for a touchdown?
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run.
    Answer: 1

    Question: {question}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage, question_analysis):
    """Verifies answer for correctness, completeness, and constraint satisfaction. Adds example for calculation verification."""
    system_instruction = "You are an expert at verifying answers, ensuring they are correct, complete, and satisfy all constraints."
    prompt = f"""
    Verify the answer to the question based on the relevant passage and question analysis. Ensure correctness, completeness, and constraint satisfaction. Return the answer if correct, otherwise provide the correct answer.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Answer: Jarrett Boykin
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Verification: Jarrett Boykin
    
    Example 2:
    Question: How many running backs ran for a touchdown?
    Answer: 1
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. 
    Verification: 1
    
    Example 3:
    Question: How many more modern artillery pieces than batteries did the 1919 Afghan regular army have?
    Answer: 210
    Passage: In 1919 the Afghan regular army was not a very formidable force...with about 280 modern artillery pieces, organised into 70 batteries...
    Verification: 210

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