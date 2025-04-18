import os
import json
import re
import math

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
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

def extract_participants_with_repair(question, max_attempts=3):
    """Extract participant names and repair if necessary."""
    system_instruction = "You are an expert at extracting participant names from text."
    prompt = f"""
    Extract the names of the participants from the question.
    
    Example:
    Question: Schedule a meeting for John, Jane, and Mike.
    Participants: John, Jane, Mike
    
    Question: Schedule a meeting for Barbara and Jason.
    Participants: Barbara, Jason
    
    Question: {question}
    Participants:
    """
    participants = call_llm(prompt, system_instruction)

    for attempt in range(max_attempts):
        verification_prompt = f"""
        Verify that these names are participants from the question. If a name is not a participant, remove it.
        
        Example:
        Question: Schedule a meeting for John, the project manager, and Jane.
        Extracted Names: John, project manager, Jane
        Verified Names: John, Jane
        
        Question: {question}
        Extracted Names: {participants}
        Verified Names:
        """
        verified_participants = call_llm(verification_prompt, system_instruction)
        return verified_participants

    return participants  # Return original if verification fails

def extract_constraints_structured(question):
    """Extract constraints with structure for easier parsing."""
    system_instruction = "You are an expert at extracting scheduling constraints."
    prompt = f"""
    Identify all constraints, formatting output like this:
    
    Example:
    Question: John is busy Monday 9-10, Jane prefers Tuesdays.
    Constraints:
    {{
       "John": ["Monday 9-10"],
       "Jane": ["Prefers Tuesdays"]
    }}
    
    Question: {question}
    Constraints:
    """
    return call_llm(prompt, system_instruction)

def solve_meeting_problem_with_feedback(participants, constraints, max_attempts=3):
    """Solve the scheduling problem with iterative refinement."""
    system_instruction = "You are an expert at solving meeting scheduling problems."
    prompt = f"""
    Find a suitable meeting time, given the constraints.
    
    Example:
    Participants: John, Jane
    Constraints:
    {{
       "John": ["Monday 9-10"],
       "Jane": ["Prefers Tuesdays"]
    }}
    Solution: Tuesday 11:00 - 11:30
    
    Participants: {participants}
    Constraints: {constraints}
    Solution:
    """
    solution = call_llm(prompt, system_instruction)
    return solution

def main(question):
    """Main function to schedule meetings."""
    try:
        # Extract participants with repair
        participants = extract_participants_with_repair(question)
        if not participants:
            return "Error: Could not extract participants."
        
        # Extract constraints
        constraints = extract_constraints_structured(question)
        if not constraints:
            return "Error: Could not extract constraints."
        
        # Solve the meeting problem with iterative refinement
        solution = solve_meeting_problem_with_feedback(participants, constraints)
        if not solution:
            return "No suitable time slots found."

        return f"Here is the proposed time: {solution}"

    except Exception as e:
        return f"Error: {str(e)}"