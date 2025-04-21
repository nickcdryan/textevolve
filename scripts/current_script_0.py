import json
import re
import os
import math

# Overall reasoning:
# This iteration will focus on enhancing the information extraction step by using a multi-stage extraction and verification process.
# The hypothesis is that by breaking down the information extraction into smaller, verifiable steps, we can improve the overall accuracy of the extracted data.
# Specifically, we will extract the participants, the existing schedules, and the constraints separately, and then verify each extraction before proceeding to the next step.
# After extracting all the information with each step verified, we will generate a solution plan using LLM and execute the plan step by step.
# We use multiple examples in every prompt and a ReAct pattern for generating solutions.

# Error handling and validation are implemented at each extraction and generation stage.

def main(question):
    """Main function to schedule a meeting based on the given question."""
    try:
        # Step 1: Extract participants with verification
        participants_result = extract_participants_with_verification(question)
        if not participants_result.get("is_valid"):
            print(f"Participant extraction failed: {participants_result.get('validation_feedback')}")
            return f"Error in participant extraction: {participants_result.get('validation_feedback')}"
        participants = participants_result["participants"]

        # Step 2: Extract existing schedules with verification
        schedules_result = extract_schedules_with_verification(question, participants)
        if not schedules_result.get("is_valid"):
            print(f"Schedule extraction failed: {schedules_result.get('validation_feedback')}")
            return f"Error in schedule extraction: {schedules_result.get('validation_feedback')}"
        schedules = schedules_result["schedules"]

        # Step 3: Extract constraints with verification
        constraints_result = extract_constraints_with_verification(question, participants)
        if not constraints_result.get("is_valid"):
            print(f"Constraint extraction failed: {constraints_result.get('validation_feedback')}")
            return f"Error in constraint extraction: {constraints_result.get('validation_feedback')}"
        constraints = constraints_result["constraints"]

        # Step 4: Generate a solution plan
        solution_plan_result = generate_solution_plan(question, participants, schedules, constraints)
        if not solution_plan_result.get("is_valid"):
            print(f"Solution plan generation failed: {solution_plan_result.get('validation_feedback')}")
            return f"Error in solution plan generation: {solution_plan_result.get('validation_feedback')}"
        solution_plan = solution_plan_result["solution_plan"]

        # Step 5: Execute the solution plan using ReAct
        solution = solve_with_react_pattern(question, solution_plan)

        return solution

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"Error: {str(e)}"

def extract_participants_with_verification(question, max_attempts=3):
    """Extract participants from the question and verify the extraction."""
    system_instruction = "You are an expert at extracting participants' names from a scheduling request."

    for attempt in range(max_attempts):
        extraction_prompt = f"""
            Extract the names of all participants mentioned in the following scheduling request.
            Return the names as a JSON array.

            Example 1:
            Question: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour...
            Participants: ["Joyce", "Christine", "Alexander"]

            Example 2:
            Question: Schedule a meeting for Betty and Scott for half an hour...
            Participants: ["Betty", "Scott"]

            Example 3:
            Question: You need to schedule a meeting for David, Ethan, Bradley and Natalie...
            Participants: ["David", "Ethan", "Bradley", "Natalie"]

            Question: {question}
            Participants:
            """

        extracted_data = call_llm(extraction_prompt, system_instruction)

        try:
            participants = json.loads(extracted_data)

            # Verification step
            verification_prompt = f"""
                Verify if the extracted participants are correct and complete.

                Question: {question}
                Extracted Participants: {json.dumps(participants)}

                Are all participants' names correctly extracted? Are there any missing or incorrect names?
                Respond with a JSON object indicating whether the extraction is valid.

                Example of Valid Response:
                {{
                  "is_valid": true,
                  "validation_feedback": "All names are correctly extracted."
                }}

                Example of Invalid Response:
                {{
                  "is_valid": false,
                  "validation_feedback": "Missing participant 'Alexander'."
                }}

                Verification:
                """

            verification_result = call_llm(verification_prompt, system_instruction)
            verification_data = json.loads(verification_result)

            if verification_data.get("is_valid", False):
                return {"is_valid": True, "participants": participants}
            else:
                print(f"Participant extraction validation failed: {verification_data.get('validation_feedback')}")

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")

    return {"is_valid": False, "validation_feedback": "Failed to extract participants after multiple attempts."}

def extract_schedules_with_verification(question, participants, max_attempts=3):
    """Extract existing schedules for each participant and verify the extraction."""
    system_instruction = "You are an expert at extracting existing schedules for participants from a scheduling request."

    for attempt in range(max_attempts):
        extraction_prompt = f"""
            Extract the existing schedules for each participant from the following scheduling request.
            Return the schedules as a JSON object, where the keys are the participant names and the values are their schedules.

            Example 1:
            Question: Joyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00; Christinehas no meetings the whole day. Alexander has meetings on Monday during 9:00 to 11:00...
            Participants: ["Joyce", "Christine", "Alexander"]
            Schedules: {{
                "Joyce": "Monday during 11:00 to 11:30, 13:30 to 14:00",
                "Christine": "no meetings the whole day",
                "Alexander": "Monday during 9:00 to 11:00"
            }}

            Example 2:
            Question: Betty is busy on Monday during 10:00 to 10:30... Scott is busy on Monday during 9:30 to 15:00...
            Participants: ["Betty", "Scott"]
            Schedules: {{
                "Betty": "Monday during 10:00 to 10:30",
                "Scott": "Monday during 9:30 to 15:00"
            }}

            Question: {question}
            Participants: {json.dumps(participants)}
            Schedules:
            """

        extracted_data = call_llm(extraction_prompt, system_instruction)

        try:
            schedules = json.loads(extracted_data)

            # Verification step
            verification_prompt = f"""
                Verify if the extracted schedules are correct and complete for each participant.

                Question: {question}
                Participants: {json.dumps(participants)}
                Extracted Schedules: {json.dumps(schedules)}

                Are all participants' schedules correctly extracted? Are there any missing or incorrect schedules?
                Respond with a JSON object indicating whether the extraction is valid.

                Example of Valid Response:
                {{
                  "is_valid": true,
                  "validation_feedback": "All schedules are correctly extracted."
                }}

                Example of Invalid Response:
                {{
                  "is_valid": false,
                  "validation_feedback": "Missing schedule for participant 'Alexander'."
                }}

                Verification:
                """

            verification_result = call_llm(verification_prompt, system_instruction)
            verification_data = json.loads(verification_result)

            if verification_data.get("is_valid", False):
                return {"is_valid": True, "schedules": schedules}
            else:
                print(f"Schedule extraction validation failed: {verification_data.get('validation_feedback')}")

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")

    return {"is_valid": False, "validation_feedback": "Failed to extract schedules after multiple attempts."}

def extract_constraints_with_verification(question, participants, max_attempts=3):
    """Extract constraints from the question and verify the extraction."""
    system_instruction = "You are an expert at extracting constraints from a scheduling request."

    for attempt in range(max_attempts):
        extraction_prompt = f"""
            Extract the constraints from the following scheduling request.
            Return the constraints as a JSON array.

            Example 1:
            Question: Christine can not meet on Monday before 12:00.
            Participants: ["Joyce", "Christine", "Alexander"]
            Constraints: ["Christine can not meet on Monday before 12:00"]

            Example 2:
            Question: Betty can not meet on Monday. Tuesday. Thursday before 15:00. Scott would like to avoid more meetings on Wednesday.
            Participants: ["Betty", "Scott"]
            Constraints: ["Betty can not meet on Monday before 15:00", "Betty can not meet on Tuesday before 15:00", "Betty can not meet on Thursday before 15:00", "Scott would like to avoid more meetings on Wednesday"]

            Question: {question}
            Participants: {json.dumps(participants)}
            Constraints:
            """

        extracted_data = call_llm(extraction_prompt, system_instruction)

        try:
            constraints = json.loads(extracted_data)

            # Verification step
            verification_prompt = f"""
                Verify if the extracted constraints are correct and complete.

                Question: {question}
                Participants: {json.dumps(participants)}
                Extracted Constraints: {json.dumps(constraints)}

                Are all constraints correctly extracted? Are there any missing or incorrect constraints?
                Respond with a JSON object indicating whether the extraction is valid.

                Example of Valid Response:
                {{
                  "is_valid": true,
                  "validation_feedback": "All constraints are correctly extracted."
                }}

                Example of Invalid Response:
                {{
                  "is_valid": false,
                  "validation_feedback": "Missing constraint 'Scott would like to avoid more meetings on Wednesday'."
                }}

                Verification:
                """

            verification_result = call_llm(verification_prompt, system_instruction)
            verification_data = json.loads(verification_result)

            if verification_data.get("is_valid", False):
                return {"is_valid": True, "constraints": constraints}
            else:
                print(f"Constraint extraction validation failed: {verification_data.get('validation_feedback')}")

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")

    return {"is_valid": False, "validation_feedback": "Failed to extract constraints after multiple attempts."}

def generate_solution_plan(question, participants, schedules, constraints, max_attempts=3):
    """Generate a plan to solve the problem of scheduling a meeting."""
    system_instruction = "You are an expert at generating solution plans for scheduling meetings."

    for attempt in range(max_attempts):
        plan_prompt = f"""
            Generate a detailed solution plan for scheduling a meeting based on the following information.
            The plan should outline the steps required to find a suitable time slot that satisfies all constraints.

            Question: {question}
            Participants: {json.dumps(participants)}
            Schedules: {json.dumps(schedules)}
            Constraints: {json.dumps(constraints)}

            Example Solution Plan:
            1. Identify the available time slots for each participant based on their schedules.
            2. Consider the constraints and eliminate any time slots that violate the constraints.
            3. Find a common time slot that works for all participants.
            4. Propose the meeting time.

            Generate a JSON object with the solution plan in a list of strings.

            Solution Plan:
            """

        plan_data = call_llm(plan_prompt, system_instruction)

        try:
            solution_plan = json.loads(plan_data)

            # Verification step
            verification_prompt = f"""
                Verify if the generated solution plan is complete and covers all necessary steps.

                Question: {question}
                Solution Plan: {json.dumps(solution_plan)}

                Does the solution plan address all aspects of the scheduling problem?
                Are there any missing steps?
                Respond with a JSON object indicating whether the plan is valid.

                Example of Valid Response:
                {{
                  "is_valid": true,
                  "validation_feedback": "The solution plan is complete and covers all necessary steps."
                }}

                Example of Invalid Response:
                {{
                  "is_valid": false,
                  "validation_feedback": "The solution plan does not consider the constraints."
                }}

                Verification:
                """

            verification_result = call_llm(verification_prompt, system_instruction)
            verification_data = json.loads(verification_result)

            if verification_data.get("is_valid", False):
                return {"is_valid": True, "solution_plan": solution_plan}
            else:
                print(f"Solution plan validation failed: {verification_data.get('validation_feedback')}")

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")

    return {"is_valid": False, "validation_feedback": "Failed to generate a valid solution plan after multiple attempts."}

def solve_with_react_pattern(problem, solution_plan, max_iterations=10):
    """Solve problems through iterative Reasoning and Acting (ReAct) approach."""
    system_instruction = "You are a problem-solving agent that follows the ReAct pattern: Reason about the current state, take an Action, observe the result, and repeat until reaching a solution."
    
    # Initialize ReAct process
    prompt = f"""
    Solve this problem using the ReAct pattern and the following solution plan. Alternate between Reasoning and Acting until you reach a final answer.
    
    Problem: {problem}
    Solution Plan: {json.dumps(solution_plan)}
    
    Example usage:
    
    Problem: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?
    
    Solution Plan:
    1. Determine which country the Great Barrier Reef is in.
    2. Find the capital of that country.
    3. Find the population of the capital.
    4. Combine the information and provide the answer.
    
    Thought 1: I need to determine which country the Great Barrier Reef is in, according to the solution plan.
    Action 1: Search[Great Barrier Reef location]
    Observation 1: The Great Barrier Reef is located off the coast of Queensland in northeastern Australia.
    
    Thought 2: Now I know the Great Barrier Reef is in Australia. I need to find Australia's capital city.
    Action 2: Search[capital of Australia]
    Observation 2: The capital of Australia is Canberra.
    
    Thought 3: Now I need to find the population of Canberra.
    Action 3: Search[population of Canberra]
    Observation 3: As of 2021, the population of Canberra is approximately 431,500.
    
    Thought 4: I have found all the required information. The capital of Australia (where the Great Barrier Reef is located) is Canberra, and its population is approximately 431,500.
    Action 4: Finish[The capital of Australia is Canberra, with a population of approximately 431,500.]
    
    Now solve this new problem:
    {problem}
    
    Start with Thought 1: Let's begin by following the solution plan to solve this scheduling problem.
    """
    
    # Initial reasoning and action planning
    react_response = call_llm(prompt, system_instruction)
    
    # Extract the action from the response
    action = extract_action(react_response)
    
    # Continue the ReAct loop until we reach a "Finish" action
    while not action["type"] == "Finish":
        # Perform the requested action and get an observation
        if action["type"] == "Search":
            observation = perform_search(action["query"])
        elif action["type"] == "Calculate":
            observation = perform_calculation(action["expression"])
        elif action["type"] == "Lookup":
            observation = perform_lookup(action["term"])
        else:
            observation = f"Unknown action type: {action['type']}"
        
        # Continue the ReAct process with the new observation
        continuation_prompt = f"""
        {react_response}
        Observation {action["step_number"]}: {observation}
        
        Continue with the next thought and action:
        """
        
        # Get the next reasoning step and action
        react_response += "\n" + call_llm(continuation_prompt, system_instruction)
        
        # Extract the next action
        action = extract_action(react_response)
    
    # Extract the final answer from the Finish action
    final_answer = action["answer"]
    return final_answer

def extract_action(text):
    """Parse the ReAct response to extract the current action."""
    # Find the last action in the text
    action_matches = re.findall(r"Action (\d+): (\w+)\[(.*?)\]", text)
    if not action_matches:
        return {"type": "Error", "step_number": 0, "query": "No action found"}
    
    # Get the most recent action
    last_action = action_matches[-1]
    step_number = int(last_action[0])
    action_type = last_action[1]
    action_content = last_action[2]
    
    # Handle different action types
    if action_type == "Finish":
        return {"type": "Finish", "step_number": step_number, "answer": action_content}
    elif action_type in ["Search", "Lookup", "Calculate"]:
        return {"type": action_type, "step_number": step_number, "query": action_content}
    else:
        return {"type": "Unknown", "step_number": step_number, "query": action_content}

def perform_search(query):
    """Simulate a search action in the ReAct pattern."""
    # In a real implementation, this would call an actual search API
    return call_llm(f"Provide a factual answer about: {query}", "You are a helpful search engine that provides concise, factual information.")

def perform_calculation(expression):
    """Perform a calculation action in the ReAct pattern."""
    try:
        # Safely evaluate the expression
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"The result is {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def perform_lookup(term):
    """Simulate a lookup action for specific information."""
    # In a real implementation, this would query a knowledge base or database
    return call_llm(f"Provide specific information about: {term}", "You are a knowledge base that provides specific factual information.")

# Example usage:
if __name__ == "__main__":
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nJoyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; \nChristinehas no meetings the whole day.\nAlexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; \n\nChristine can not meet on Monday before 12:00. Find a time that works for everyone's schedule and constraints. "
    answer = main(question)
    print(f"Final Answer: {answer}")