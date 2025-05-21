import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach.
    This approach focuses on breaking down the problem into question type identification, 
    focused passage extraction, and direct answer generation with verification.
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

        # Step 4: Verify answer
        verified_answer = verify_answer(question, answer, relevant_passage)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes the question to identify its type and keywords. Includes multiple examples."""
    system_instruction = "You are an expert at analyzing questions to determine their type and keywords."
    prompt = f"""
    Analyze the following question and identify its type (e.g., fact extraction, calculation, comparison) and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught", "who"]}}

    Example 2:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown", "how many"]}}
    
    Example 3:
    Question: Which player kicked the only field goal of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["player", "field goal", "which"]}}

    Example 4:
    Question: How many more yards did Chris Johnson gain on his first touchdown run than LenDale White on his first touchdown run?
    Analysis: {{"type": "comparison/calculation", "keywords": ["Chris Johnson", "LenDale White", "first touchdown", "how many more yards"]}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts the relevant passage from the question based on keywords. Includes multiple examples."""
    system_instruction = "You are an expert at extracting relevant passages from text. Focus on complete sentences for context."
    prompt = f"""
    Extract the relevant passage from the following text based on the question and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Keywords: {{"type": "fact extraction", "keywords": ["final touchdown", "caught", "who"]}}
    Text: PASSAGE: After a tough loss at home, the Browns traveled to take on the Packers. ... The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.

    Example 2:
    Question: How many running backs ran for a touchdown?
    Keywords: {{"type": "counting", "keywords": ["running backs", "touchdown", "how many"]}}
    Text: PASSAGE: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. The Lions would respond with kicker Jason Hanson getting a 53-yard field goal. The Titans would answer with Johnson getting a 58-yard TD run, along with DE Dave Ball returning an interception 15 yards for a touchdown. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.

    Example 3:
    Question: Which player kicked the only field goal of the game?
    Keywords: {{"type": "fact extraction", "keywords": ["player", "field goal", "which"]}}
    Text: PASSAGE: Game SummaryComing off their Thanksgiving road win over the Falcons, the Colts went home for a Week 13 AFC South rematch with the Jacksonville Jaguars.  ... In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.

    Example 4:
    Question: How many more yards did Chris Johnson gain on his first touchdown run than LenDale White on his first touchdown run?
    Keywords: {{"type": "comparison/calculation", "keywords": ["Chris Johnson", "LenDale White", "first touchdown", "how many more yards"]}}
    Text: PASSAGE: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. ... In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.

    Question: {question}
    Keywords: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer based on the question, relevant passage, and question type. Includes multiple examples."""
    system_instruction = "You are an expert at generating accurate and complete answers to questions based on provided text. If calculation is needed, show your work."
    prompt = f"""
    Generate the answer to the question based on the relevant passage and question type.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Answer: Jarrett Boykin

    Example 2:
    Question: How many running backs ran for a touchdown?
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. The Lions would respond with kicker Jason Hanson getting a 53-yard field goal. The Titans would answer with Johnson getting a 58-yard TD run, along with DE Dave Ball returning an interception 15 yards for a touchdown. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Answer: 2
    
    Example 3:
    Question: Which player kicked the only field goal of the game?
    Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Answer: Josh Scobee

    Example 4:
    Question: How many more yards did Chris Johnson gain on his first touchdown run than LenDale White on his first touchdown run?
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. ... In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Answer: Chris Johnson gained 6 yards and LenDale White gained 6 yards. 6-6 = 0. So Chris Johnson gained 0 more yards than LenDale White.

    Question: {question}
    Passage: {relevant_passage}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage):
    """Verifies the generated answer. Includes multiple examples. If the answer is wrong and calculable, calculate and return the correct answer."""
    system_instruction = "You are an expert at verifying answers to questions. If the answer is incorrect and can be determined from the passage, correct it."
    prompt = f"""
    Verify the following answer to the question based on the relevant passage. Return the answer if it is correct. Return the correct answer if it is incorrect, *especially* if calculation is involved. Show your work and explain your answer.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Answer: Jarrett Boykin
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Verification: Jarrett Boykin is correct.

    Example 2:
    Question: How many running backs ran for a touchdown?
    Answer: 2
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Verification: 2 is correct. Chris Johnson and LenDale White both ran for touchdowns.

    Example 3:
    Question: Which player kicked the only field goal of the game?
    Answer: Josh Scobee
    Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Verification: Josh Scobee is correct.

    Example 4:
    Question: How many more yards did Chris Johnson gain on his first touchdown run than LenDale White on his first touchdown run?
    Answer: Chris Johnson gained 2 more yards than LenDale White.
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. ... In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Verification: Incorrect. Chris Johnson gained 6 yards on his first touchdown run. LenDale White gained 6 yards on his first touchdown run. 6 - 6 = 0. The correct answer is 0.

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