import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with enhanced analysis, extraction, and verification.
    Includes detailed examples in prompts and robust error handling.
    """
    try:
        # Step 1: Analyze question with enhanced type classification and keyword extraction
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question"

        # Step 2: Extract relevant passage with focused context
        relevant_passage = extract_relevant_passage(question, question_analysis, question) # Added question here for additional context in passage extraction
        if "Error" in relevant_passage:
            return "Error extracting passage"

        # Step 3: Generate answer using passage, question analysis, and enhanced reasoning
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer"

        # Step 4: Verify answer with contextual awareness and verification examples
        verified_answer = verify_answer(question, answer, relevant_passage)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes question to identify its type and keywords. Includes multiple examples for various question types."""
    system_instruction = "You are an expert at analyzing questions to determine their type and keywords."
    prompt = f"""
    Analyze the following question and identify its type (e.g., fact extraction, calculation, comparison, negation) and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught", "player"]}}

    Example 2:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown", "number"]}}
    
    Example 3:
    Question: Which player kicked the only field goal of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["player", "field goal", "kicked"]}}

    Example 4:
    Question: How many percent are not non-families?
    Analysis: {{"type": "calculation", "keywords": ["percent", "not", "non-families", "calculate"]}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis, text):
    """Extracts relevant passage based on keywords and context. Includes question for contextual understanding."""
    system_instruction = "You are an expert at extracting relevant passages from text based on a question's keywords and context."
    prompt = f"""
    Extract the most relevant passage from the following text that helps answer the question, considering the question's keywords and overall context.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Keywords: {{"type": "fact extraction", "keywords": ["final touchdown", "caught", "player"]}}
    Text: PASSAGE: After a tough loss at home, the Browns traveled to take on the Packers. ... The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.

    Example 2:
    Question: How many running backs ran for a touchdown?
    Keywords: {{"type": "counting", "keywords": ["running backs", "touchdown", "number"]}}
    Text: PASSAGE: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. ... In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    
    Example 3:
    Question: How many percent are not non-families?
    Keywords: {{"type": "calculation", "keywords": ["percent", "not", "non-families", "calculate"]}}
    Text: PASSAGE: In 2000 there were 79,667 households out of which 38.70% had children under the age of 18 living with them, 61.90% were married couples living together, 10.20% had a female householder with no husband present, and 24.20% were non-families.
    Passage: 24.20% were non-families.

    Question: {question}
    Keywords: {question_analysis}
    Text: {text}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer with enhanced reasoning and includes multiple examples."""
    system_instruction = "You are an expert at generating answers to questions based on provided text, demonstrating clear reasoning."
    prompt = f"""
    Generate the answer to the question based on the relevant passage, question type, and your expert reasoning.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Answer: Jarrett Boykin, as he caught the pass in the last scoring play.

    Example 2:
    Question: How many running backs ran for a touchdown?
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Answer: 2, Chris Johnson and LenDale White each ran for touchdowns.
    
    Example 3:
    Question: How many percent are not non-families?
    Passage: In 2000 there were ... and 24.20% were non-families.
    Answer: 75.8%, calculated by subtracting 24.20 from 100.

    Question: {question}
    Passage: {relevant_passage}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage):
    """Verifies the answer, correcting it if needed. Includes examples demonstrating different verification scenarios."""
    system_instruction = "You are an expert at verifying answers, correcting them based on provided context and reasoning."
    prompt = f"""
    Verify the following answer to the question based on the relevant passage.  
    If the answer is correct, return it as is. If it is incorrect, provide the correct answer with your reasoning.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Answer: Jarrett Boykin
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Verification: Jarrett Boykin is correct because the passage states 'Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score'.

    Example 2:
    Question: How many running backs ran for a touchdown?
    Answer: 2
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Verification: 2 is correct, because Chris Johnson and LenDale White are the only RBs mentioned scoring touchdowns.

    Example 3:
    Question: How many percent are not non-families?
    Answer: 24.2
    Passage: ... and 24.20% were non-families.
    Verification: Incorrect. The question asked 'how many percent are NOT non-families'. The correct answer is 75.8%, since 100 - 24.2 = 75.8.

    Question: {question}
    Answer: {answer}
    Passage: {relevant_passage}
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