import os
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

def find_available_time_slots(question, participants, schedules, preferences, duration="half an hour"):
    """Find available time slots considering schedules, preferences, and duration. Now returns a single best option using preferences."""
    system_instruction = "You are an expert at scheduling meetings, considering participant availability and preferences to find the best possible time."
    prompt = f"""
    Given the following scheduling problem, identify a single best time slot for a meeting, considering everyone's schedule and any preferences.  Prioritize the most suitable option based on the constraints and preferences. If no time is available, respond with "No suitable time found".

    Example 1:
    Question: Schedule a meeting for Alice and Bob for 30 minutes on Monday. Alice is busy 10:00-12:00. Bob prefers not to meet after 2:00 PM. His schedule is clear otherwise.
    Available Times: Monday, 13:00 - 13:30
    Example 2:
    Question: Schedule a meeting for Carol and David for 1 hour on Tuesday. Carol is busy all morning. David has a meeting 3:00-4:00. David would prefer to meet in the morning.
    Available Times: Tuesday, 14:00 - 15:00

    Question: {question}

    Available Times:
    """
    return call_llm(prompt, system_instruction)

def format_answer(time_slot):
    """Format the answer in a consistent way."""
    return f"Here is the proposed time: {time_slot} "

def main(question):
    """Main function to schedule a meeting given the question."""
    system_instruction = "You are an expert at understanding scheduling constraints and participant preferences for meetings."

    # Decompose the problem to ensure all requirements are met.
    problem_decomposition_prompt = f"""
    Decompose the scheduling problem into smaller steps and identify the key components. Ensure all explicit and implicit requirements are identified.
    
    Example 1:
    Question: Schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. John has no meetings. Jennifer has meetings on Monday during 9:00 to 11:00.
    Decomposition: Identify participants (John, Jennifer), duration (half an hour), days (Monday, Tuesday, Wednesday), work hours (9:00 to 17:00), Jennifer's blocked schedule on Monday.
    
    Question: {question}
    Decomposition:
    """
    decomposition = call_llm(problem_decomposition_prompt, system_instruction)

    # Find an available time slot using the LLM
    available_time = find_available_time_slots(question, [], {}, "")  # Placeholder arguments, improved with next LLM call
    return format_answer(available_time)