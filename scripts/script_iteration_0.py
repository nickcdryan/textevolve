import os

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

def extract_meeting_info(question):
    """Extract key meeting information using example-based prompting."""
    system_instruction = "You are an expert meeting scheduler, skilled at extracting key details from meeting requests."
    prompt = f"""
    Extract the participants, duration, available days, work hours, and any participant preferences from the meeting request.
    
    Example:
    Input: You need to schedule a meeting for Nicholas, Sara for half an hour between 9:00 to 17:00 on Monday. Sara does not want to meet before 10.
    Output:
    {{
      "participants": ["Nicholas", "Sara"],
      "duration": "0.5",
      "available_days": ["Monday"],
      "work_hours": ["9:00", "17:00"],
      "preferences": {{"Sara": "after 10:00"}}
    }}
    
    Now extract the details from this request:
    {question}
    """
    return call_llm(prompt, system_instruction)

def extract_schedules(question, participants):
    """Extract existing schedules using example-based prompting and verification."""
    system_instruction = "You are an expert schedule extractor, extracting availability from a text description."
    prompt = f"""
    Extract the existing schedules for each participant. Only provide schedules for the specified participants.
    
    Example:
    Input: Here are the existing schedules: Nicholas is busy on Monday during 9:00 to 9:30; Sara is busy on Monday during 10:00 to 10:30. Participants: Nicholas, Sara.
    Output:
    {{
      "Nicholas": [["Monday", "9:00", "9:30"]],
      "Sara": [["Monday", "10:00", "10:30"]]
    }}
    
    Now extract the schedules from this text, for these participants: {participants}
    {question}
    """
    schedule_info = call_llm(prompt, system_instruction)
    #Implement schedule verification
    verification_prompt = f"""
        Verify if the schedules were extracted correctly and completely, given the following original question and participant information:
        Question: {question}
        Extracted Schedules: {schedule_info}
        Participants: {participants}
        If there are any errors or omissions, explain what needs to be corrected. Otherwise, state "OK".
        """
    verification_result = call_llm(verification_prompt)
    if "OK" not in verification_result:
        print(f"Schedule Extraction Verification Failed: {verification_result}")
        return "Error: Schedule extraction failed."
    else:
        return schedule_info

def find_available_time(meeting_info, schedules):
    """Find an available meeting time using example-based prompting."""
    system_instruction = "You are an expert at finding available times given meeting constraints and schedules."
    prompt = f"""
    Given the meeting information and participant schedules, find a time that works for everyone.
    
    Example:
    Meeting Info: {{"participants": ["Nicholas", "Sara"], "duration": "0.5", "available_days": ["Monday"], "work_hours": ["9:00", "17:00"]}}
    Schedules: {{"Nicholas": [["Monday", "9:00", "9:30"]], "Sara": [["Monday", "10:00", "10:30"]]}}
    Output: Monday, 9:30 - 10:00
    
    Now, using this information:
    Meeting Info: {meeting_info}
    Schedules: {schedules}
    Find a valid meeting time.
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        participants_start_index = question.find("schedule a meeting for ") + len("schedule a meeting for ")
        participants_end_index = question.find(" for", participants_start_index)
        participants_string = question[participants_start_index:participants_end_index]
        participants = [p.strip() for p in participants_string.split(",")]
        
        meeting_info = extract_meeting_info(question)
        schedules = extract_schedules(question, participants)
        available_time = find_available_time(meeting_info, schedules)
        return "Here is the proposed time: " + available_time
    except Exception as e:
        return f"Error: {str(e)}"