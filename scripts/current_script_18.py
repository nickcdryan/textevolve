import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach.
    Enhanced with robust error handling and embedded examples.
    """
    try:
        # Step 1: Analyze question type and keywords
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question: " + question_analysis

        # Step 2: Extract relevant passage using identified keywords
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return "Error extracting passage: " + relevant_passage

        # Step 3: Generate answer using extracted passage and question type
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer: " + answer

        # Step 4: Verify answer
        verified_answer = verify_answer(question, answer, relevant_passage)
        if "Error" in verified_answer:
            return "Error verifying answer: " + verified_answer
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes the question to identify its type and keywords. Now with multiple examples."""
    system_instruction = "You are an expert at analyzing questions to determine their type and keywords."
    prompt = f"""
    Analyze the following question and identify its type (e.g., fact extraction, calculation, comparison) and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught", "game"]}}

    Example 2:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown", "ran"]}}
    
    Example 3:
    Question: Which player kicked the only field goal of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["player", "field goal", "kicked"]}}

    Example 4:
    Question: How many total passing touchdown yards did Dalton have?
    Analysis: {{"type": "calculation", "keywords": ["total", "passing", "touchdown", "yards", "Dalton"]}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts the relevant passage from the question based on keywords. Now with more examples."""
    system_instruction = "You are an expert at extracting relevant passages from text."
    prompt = f"""
    Extract the most relevant passage from the following text to answer the question, based on its type and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game? PASSAGE: ... The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Keywords: {{"type": "fact extraction", "keywords": ["final touchdown", "caught", "game"]}}
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    
    Example 2:
    Question: How many running backs ran for a touchdown? PASSAGE: ... Chris Johnson got a 6-yard TD run... LenDale White getting a 6-yard and a 2-yard TD run.
    Keywords: {{"type": "counting", "keywords": ["running backs", "touchdown", "ran"]}}
    Passage: Chris Johnson got a 6-yard TD run. LenDale White getting a 6-yard and a 2-yard TD run.

    Example 3:
    Question: Which player kicked the only field goal of the game? PASSAGE: ...Josh Scobee nailed a 47-yard field goal.
    Keywords: {{"type": "fact extraction", "keywords": ["player", "field goal", "kicked"]}}
    Passage: Josh Scobee nailed a 47-yard field goal.

    Example 4:
    Question: How many total passing touchdown yards did Dalton have? PASSAGE: Andy Dalton was 24-of-34 for 372 yards and 3 touchdowns... First, Dalton hit Tyler Eifert for a 32-yard TD, and Stafford followed shortly after with a 27-yard TD toss to Calvin Johnson
    Keywords: {{"type": "calculation", "keywords": ["total", "passing", "touchdown", "yards", "Dalton"]}}
    Passage: Andy Dalton was 24-of-34 for 372 yards and 3 touchdowns. First, Dalton hit Tyler Eifert for a 32-yard TD

    Question: {question}
    Keywords: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer based on the question, relevant passage, and question type. Now with multiple examples."""
    system_instruction = "You are an expert at generating answers to questions based on provided text. Provide ONLY the direct answer. Do not include any extraneous reasoning or text."
    prompt = f"""
    Generate a direct answer to the question, using ONLY information from the relevant passage and question type.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Answer: Jarrett Boykin

    Example 2:
    Question: How many running backs ran for a touchdown?
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Answer: 2
    
    Example 3:
    Question: Which player kicked the only field goal of the game?
    Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Answer: Josh Scobee

    Example 4:
    Question: How many total passing touchdown yards did Dalton have?
    Passage: Andy Dalton was 24-of-34 for 372 yards and 3 touchdowns. First, Dalton hit Tyler Eifert for a 32-yard TD
    Answer: 372

    Question: {question}
    Passage: {relevant_passage}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage):
    """Verifies the generated answer and ensures format. Now with examples and more checks."""
    system_instruction = "You are an expert at verifying answers to questions. Return the correct answer EXACTLY as it appears in the relevant passage, if possible. If the answer requires a calculation, perform the calculation and return the result."
    prompt = f"""
    Carefully verify the provided answer against the relevant passage.  If the answer is directly stated in the passage, return it exactly as it appears. If the answer requires a calculation based on the passage, perform the calculation and return the result. Ensure the answer is complete and in the correct format.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Answer: Jarrett Boykin
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Verification: Jarrett Boykin
    
    Example 2:
    Question: How many running backs ran for a touchdown?
    Answer: 2
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Verification: 2

    Example 3:
    Question: Which player kicked the only field goal of the game?
    Answer: Josh Scobee
    Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Verification: Josh Scobee

    Example 4:
    Question: How many total passing touchdown yards did Dalton have?
    Answer: 372
    Passage: Andy Dalton was 24-of-34 for 372 yards and 3 touchdowns.
    Verification: 372

    Question: {question}
    Answer: {answer}
    Passage: {relevant_passage}
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