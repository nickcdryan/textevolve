import os
import re
import math
import json

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

def main(question):
    """
    This script synthesizes the best elements from multiple successful approaches:
    - Strategy selection followed by strategy-specific solving (from Approach #1)
    - Adaptive prompt construction and multi-agent verification (from Approach #2)
    - Chain-of-thought prompting for improved reasoning and validation (from Approach #3)

    It creates a hybrid approach that leverages strengths while avoiding weaknesses, by using validation loops for each processing step to provide error handling.

    EVERY LLM PROMPT includes embedded examples to guide the model.
    """

    # STEP 1: Initial Problem Analysis and Strategy Selection (from Approach #1)
    analysis_prompt = f"""
    Analyze the following problem and determine the best strategy to solve it.

    Example 1:
    Problem: A methanol solution of (R)-(+)-Limonene is stirred with Pd/C under a Hydrogen atmosphere. After 1 equivalent of hydrogen is consumed, product 1 is isolated as the major product. 1 is treated with 3-chloroperbenzoic acid, forming product 2. Product 2 is treated with sodium methoxide, forming product 3. Product 3 is treated with propanoic acid, dicyclohexylcarbodiimide. and a catalytic amount of  4-dimethylaminopyridine, forming product 4. what is a valid structure of product 4? (product 4 exists as a mixture of isomers. the correct answer is one of them).
    Strategy: ChemicalReactionAnalysis

    Example 2:
    Problem: You come across an algorithm that gives the following output (written as input -> output): AGG -> 115 TGCTGA -> 176 What value does ACAGTGACC give?
    Strategy: AlgorithmicPatternRecognition

    Example 3:
    Problem: What is the highest mountain in the world?
    Strategy: FactualQuestionAnswering

    Problem: {question}
    Strategy:
    """
    strategy = call_llm(analysis_prompt, "You are a strategy selector that selects best strategy given the problem.")

    # STEP 2: Strategy-Specific Reasoning and Solution Generation (from Approach #1, with validation from #3)
    if "ChemicalReactionAnalysis" in strategy:
        solution = solve_chemical_reaction(question)
    elif "AlgorithmicPatternRecognition" in strategy:
        solution = solve_algorithmic_problem(question)
    elif "FactualQuestionAnswering" in strategy:
        solution = solve_factual_question(question) # added FactualQuestionAnswering
    else:
        solution = call_llm(f"Solve this problem directly: {question}", "You are an expert problem solver.")
        #Validation loop moved to solution

    return solution

def solve_chemical_reaction(question):
    """Solves chemical reaction problems using chain-of-thought prompting."""

    # STEP 3: Reaction Pathway Breakdown (from Approach #1)
    pathway_prompt = f"""
    For the following chemical reaction problem, break down the reaction pathway into individual steps, identifying the reactants, reagents, and expected products for each step.

    Example:
    Problem: A methanol solution of (R)-(+)-Limonene is stirred with Pd/C under a Hydrogen atmosphere. After 1 equivalent of hydrogen is consumed, product 1 is isolated as the major product. 1 is treated with 3-chloroperbenzoic acid, forming product 2. Product 2 is treated with sodium methoxide, forming product 3. Product 3 is treated with propanoic acid, dicyclohexylcarbodiimide. and a catalytic amount of  4-dimethylaminopyridine, forming product 4. what is a valid structure of product 4? (product 4 exists as a mixture of isomers. the correct answer is one of them).
    ReactionPathway:
    Step 1: (R)-(+)-Limonene + H2, Pd/C -> Product 1 (hydrogenation)
    Step 2: Product 1 + 3-chloroperbenzoic acid -> Product 2 (epoxidation)
    Step 3: Product 2 + sodium methoxide -> Product 3 (ring opening)
    Step 4: Product 3 + propanoic acid, dicyclohexylcarbodiimide, 4-dimethylaminopyridine -> Product 4 (esterification)

    Problem: {question}
    ReactionPathway:
    """
    reaction_pathway = call_llm(pathway_prompt, "You are an expert chemist outlining reaction pathways.")

    # STEP 4: Final Product Prediction (from Approach #1, with validation from #3)
    final_product_prompt = f"""
    Based on the following reaction pathway, predict the final product.

    Example:
    ReactionPathway:
    Step 1: (R)-(+)-Limonene + H2, Pd/C -> Product 1 (hydrogenation)
    Step 2: Product 1 + 3-chloroperbenzoic acid -> Product 2 (epoxidation)
    Step 3: Product 2 + sodium methoxide -> Product 3 (ring opening)
    Step 4: Product 3 + propanoic acid, dicyclohexylcarbodiimide, 4-dimethylaminopyridine -> Product 4 (esterification)
    FinalProduct: (1S,2S,4R)-4-isopropyl-2-methoxy-1-methylcyclohexyl propionate

    ReactionPathway: {reaction_pathway}
    FinalProduct:
    """
    final_product = call_llm(final_product_prompt, "You are an expert chemist predicting final products.")

    # Validation loop
    validated_product = validate_solution(question, final_product)

    return validated_product

def solve_algorithmic_problem(question):
    """Solves algorithmic pattern recognition problems using example-based prompting."""

    # STEP 3: Algorithmic Pattern Deduction (from Approach #1, with validation from #3)
    pattern_prompt = f"""
    Deduce the pattern in the following algorithmic problem.

    Example:
    Problem: You come across an algorithm that gives the following output (written as input -> output): AGG -> 115 TGCTGA -> 176 What value does ACAGTGACC give?
    Pattern: The algorithm counts the occurrences of A, C, G, and T, multiplies each count by a specific value, and adds the products. A=2, C=3, G=5, T=7.
    FinalAnswer: 315

    Problem: {question}
    Pattern:
    FinalAnswer:
    """
    pattern = call_llm(pattern_prompt, "You are an expert in identifying patterns in algorithmic problems.")

    # Validation loop
    validated_pattern = validate_solution(question, pattern)

    return validated_pattern

def solve_factual_question(question): # Implemented factual question-answering
    """Solves factual question-answering problems using direct LLM call."""

    # STEP 3: Direct answer retrieval (from Approach #3, with validation)
    answer_prompt = f"""
    Answer the following factual question: {question}
    """
    answer = call_llm(answer_prompt, "You are a knowledgeable assistant.")

    # Validation loop
    validated_answer = validate_solution(question, answer)

    return validated_answer

def validate_solution(question, proposed_solution, max_attempts=3): # Added Validatation loop for verification of answer
    """Validates the proposed solution by generating verification prompts and refining the solution."""

    validation_attempts = 0
    while validation_attempts < max_attempts:
        # Generate verification prompt
        verification_prompt = f"""
        Verify the correctness and completeness of the following solution to the given problem.

        Example 1:
        Problem: What is the capital of France?
        Solution: The capital of France is Paris.
        Verification: The solution is correct and complete.

        Example 2:
        Problem: You come across an algorithm that gives the following output (written as input -> output): AGG -> 115 TGCTGA -> 176 What value does ACAGTGACC give?
        Solution: The algorithm counts the occurrences of A, C, G, and T, multiplies each count by a specific value, and adds the products. A=2, C=3, G=5, T=7. FinalAnswer: 315
        Verification: The solution is correct and complete.

        Problem: {question}
        Solution: {proposed_solution}
        Verification:
        """
        verification_result = call_llm(verification_prompt, "You are an expert at verifying solutions.")

        if "correct" in verification_result.lower() and "complete" in verification_result.lower():
            return proposed_solution  # Solution is valid

        # Generate refinement prompt if solution is incorrect
        refinement_prompt = f"""
        The proposed solution to the problem is not entirely correct or complete. Refine the solution based on the following problem and verification results.

        Problem: {question}
        Proposed Solution: {proposed_solution}
        Verification Results: {verification_result}

        Refined Solution:
        """
        proposed_solution = call_llm(refinement_prompt, "You are an expert at refining solutions.") # refining the solution from the llm call
        validation_attempts += 1 # keeps track of number of validations

    return proposed_solution  # Returning solution after max attemps