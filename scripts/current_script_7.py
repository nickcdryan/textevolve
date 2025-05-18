import os
import re

def main(question):
    """
    Solve the question by extracting relevant information from the passage and using chain-of-thought reasoning.
    This approach builds upon top-performing approaches by strengthening answer synthesis with a numerical reasoning module and improving verification,
    and uses multiple examples in all LLM prompts.
    """
    try:
        # Step 1: Determine question type
        question_type_result = determine_question_type(question)
        if not question_type_result.get("is_valid"):
            return f"Error in determining question type: {question_type_result.get('validation_feedback')}"

        # Step 2: Process question based on type
        if question_type_result["question_type"] == "numerical":
            process_result = process_numerical_question(question)
        else:
            process_result = process_general_question(question)

        return process_result

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def determine_question_type(question, max_attempts=3):
    """Determine if the question requires numerical reasoning or general information."""
    system_instruction = "You are an expert question type identifier."

    for attempt in range(max_attempts):
        type_prompt = f"""
        Determine if the question requires numerical reasoning (calculations) or general information extraction.

        Example 1:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Type: numerical

        Example 2:
        Question: Who caught the final touchdown of the game?
        Type: general

        Question: {question}
        Type:
        """

        type_result = call_llm(type_prompt, system_instruction)

        verification_prompt = f"""
        Verify if the identified question type is correct.

        Question: {question}
        Identified Type: {type_result}

        Example:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Identified Type: numerical
        Validation: Valid

        Is the identified type valid? Respond with 'Valid' or 'Invalid'.
        """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "valid" in verification_result.lower():
            return {"is_valid": True, "question_type": type_result.lower()}
        else:
            print(f"Question type validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")

    return {"is_valid": False, "validation_feedback": "Failed to determine question type successfully."}

def process_numerical_question(question):
    """Process numerical questions by extracting numbers and performing calculations."""
    try:
        # Step 1: Extract numerical information
        extraction_result = extract_numerical_info(question)
        if not extraction_result.get("is_valid"):
            return f"Error in numerical information extraction: {extraction_result.get('validation_feedback')}"

        # Step 2: Calculate the answer
        calculation_result = calculate_answer(question, extraction_result["extracted_info"])
        if not calculation_result.get("is_valid"):
            return f"Error in calculation: {calculation_result.get('validation_feedback')}"

        return calculation_result["answer"]

    except Exception as e:
        return f"Error in processing numerical question: {str(e)}"

def extract_numerical_info(question, max_attempts=3):
    """Extract numerical information and units from the question."""
    system_instruction = "You are an expert at extracting numerical information and their units from text."

    for attempt in range(max_attempts):
        extraction_prompt = f"""
        Extract all numerical values and their corresponding units from the question.

        Example 1:
        Question: How many yards did Chris Johnson's first touchdown (6 yards) and Jason Hanson's first field goal (53 yards) combine for?
        Extracted Info:
        - 6 yards (touchdown)
        - 53 yards (field goal)

        Example 2:
        Question: The population increased by 12%, from 1000 to what number?
        Extracted Info:
        - 12% (increase)
        - 1000 (initial population)

        Question: {question}
        Extracted Info:
        """

        extracted_info = call_llm(extraction_prompt, system_instruction)

        verification_prompt = f"""
        Verify if the extracted numerical information is complete and accurate.

        Question: {question}
        Extracted Info: {extracted_info}

        Example:
        Question: How many yards did Chris Johnson's first touchdown (6 yards) and Jason Hanson's first field goal (53 yards) combine for?
        Extracted Info: - 6 yards (touchdown) - 53 yards (field goal)
        Validation: Valid

        Is the extracted information valid? Respond with 'Valid' or 'Invalid'.
        """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "valid" in verification_result.lower():
            return {"is_valid": True, "extracted_info": extracted_info}
        else:
            print(f"Numerical info extraction failed (attempt {attempt+1}/{max_attempts}): {verification_result}")

    return {"is_valid": False, "validation_feedback": "Failed to extract numerical information successfully."}

def calculate_answer(question, extracted_info, max_attempts=3):
    """Calculate the answer based on the extracted numerical information."""
    system_instruction = "You are an expert calculator."

    for attempt in range(max_attempts):
        calculation_prompt = f"""
        Given the question and extracted numerical information, calculate the final answer.
        Identify the operation to perform (addition, subtraction, etc.) and then calculate it.

        Example:
        Question: How many yards did Chris Johnson's first touchdown (6 yards) and Jason Hanson's first field goal (53 yards) combine for?
        Extracted Info: - 6 yards (touchdown) - 53 yards (field goal)
        Calculation: 6 + 53 = 59
        Answer: 59

        Question: {question}
        Extracted Info: {extracted_info}
        Calculation:
        """

        calculation = call_llm(calculation_prompt, system_instruction)
        try:
            # Extract the numbers for the calculation from the LLM's calculation statement
            numbers = re.findall(r'\d+', calculation)
            if len(numbers) < 2:
                print("Not enough numbers were able to be extracted for the calculation")
                raise ValueError("Could not perform calculation with invalid numbers")
            num1 = int(numbers[0])
            num2 = int(numbers[1])

            # Extract the operator from the LLM's calculation statement
            operator_match = re.search(r'(\+|-|\*|/)', calculation)

            if not operator_match:
                print("No valid operator was able to be extracted for the calculation")
                raise ValueError("Invalid operator")
            operator = operator_match.group(1)

            if operator == "+":
                answer = num1 + num2
            elif operator == "-":
                answer = num1 - num2
            elif operator == "*":
                answer = num1 * num2
            elif operator == "/":
                answer = num1 / num2
            else:
                print("No known operator was selected")
                raise ValueError("Unknown operator")

            answer = str(answer)

        except Exception as e:
            print(f"Error performing calculation: {str(e)}")
            return {"is_valid": False, "validation_feedback": f"Failed to perform calculation: {str(e)}"}

        verification_prompt = f"""
        Verify if the calculated answer is correct based on the extracted information and question.

        Question: {question}
        Extracted Info: {extracted_info}
        Calculated Answer: {answer}

        Example:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Extracted Info: - 6 yards (touchdown) - 53 yards (field goal)
        Calculated Answer: 59
        Validation: Valid

        Is the calculated answer valid? Respond with 'Valid' or 'Invalid'.
        """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "valid" in verification_result.lower():
            return {"is_valid": True, "answer": answer}
        else:
            print(f"Calculation validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")

    return {"is_valid": False, "validation_feedback": "Failed to calculate a valid answer."}

def process_general_question(question):
    """Process general questions using decomposition, extraction, and synthesis."""
    try:
        # Step 1: Decompose the question into sub-questions.
        decomposition_result = decompose_question(question)
        if not decomposition_result.get("is_valid"):
            return f"Error in question decomposition: {decomposition_result.get('validation_feedback')}"

        # Step 2: Extract relevant information based on sub-questions.
        information_extraction_result = extract_information(question, decomposition_result["sub_questions"])
        if not information_extraction_result.get("is_valid"):
            return f"Error in information extraction: {information_extraction_result.get('validation_feedback')}"

        # Step 3: Synthesize the answer from extracted information.
        answer_synthesis_result = synthesize_answer(question, information_extraction_result["extracted_info"])
        if not answer_synthesis_result.get("is_valid"):
            return f"Error in answer synthesis: {answer_synthesis_result.get('validation_feedback')}"

        return answer_synthesis_result["answer"]

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"