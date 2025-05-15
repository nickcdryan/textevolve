import os
import re
import math

# EXPLORATION: LLM-Orchestrated Recursive Subdivision and Transformation
# HYPOTHESIS: By having the LLM recursively subdivide the grid into smaller regions,
# identify transformation rules for those subregions independently, and then
# stitch the transformed subregions back together, we can handle more complex
# transformations that apply differently to different parts of the grid. This leverages
# the LLM's ability to identify and apply patterns locally, while also maintaining
# a global understanding of the overall grid structure. This approach directly addresses
# the past weaknesses of failing to handle different transformations in the same grid or to perform transformations depending on location.

def solve_grid_transformation(question, max_recursion_depth=2):
    """Solves grid transformation problems by recursively subdividing and transforming."""

    def call_llm(prompt, system_instruction=None):
        """Call the Gemini LLM with a prompt and return the response."""
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

    def transform_subgrid(subgrid_question):
        """Transforms a single subgrid using LLM."""
        system_instruction = "You are an expert at transforming small grids based on training examples."

        prompt = f"""
        Given the following grid transformation problem, transform the test input subgrid according to the patterns observed in the training examples. Output only the transformed subgrid.

        Example:
        Problem:
        === TRAINING EXAMPLES ===
        Input Grid:
        [[1, 2], [3, 4]]
        Output Grid:
        [[4, 3], [2, 1]]
        === TEST INPUT ===
        [[5, 6], [7, 8]]
        Transformed Subgrid:
        [[8, 7], [6, 5]]

        Problem:
        {subgrid_question}

        Transformed Subgrid:
        """
        transformed_subgrid = call_llm(prompt, system_instruction)

        # Basic Validation: check the subgrid is not "Error" and not empty
        if "Error" in transformed_subgrid or not transformed_subgrid.strip():
            return None  # Indicate failure
        return transformed_subgrid

    def subdivide_and_transform(question, depth):
        """Recursively subdivides the grid and transforms subregions."""
        if depth <= 0:
            # Base case: transform the whole grid directly
            return transform_subgrid(question)

        # Ask the LLM if subdivision is needed
        system_instruction = "You are an expert at analyzing grids and determining if they should be subdivided for easier transformation."
        subdivision_prompt = f"""
        Given the grid transformation problem below, should the input grid be subdivided into smaller regions for easier transformation?
        Answer YES if different parts of the grid seem to be transformed differently, or NO if the same transformation applies to the whole grid.

        Example:
        Problem:
        === TRAINING EXAMPLES ===
        Input Grid:
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
        Output Grid:
        [[4, 4, 4, 4], [2, 2, 2, 2], [5, 5, 5, 5]]
        Should the input grid be subdivided?
        YES (because the top and bottom rows are changed, but the middle row is unchanged)

        Problem:
        {question}
        Should the input grid be subdivided? (YES/NO)
        """
        should_subdivide = call_llm(subdivision_prompt, system_instruction)

        if "YES" in should_subdivide.upper():
            # Implement a *simple* subdivision (e.g., into four quadrants)
            #This assumes the grids are "square" for simplicity

            input_grid_str = re.search(r'Input Grid:\n(.*?)(\nOutput Grid:|\nTransformation Rule:|\nTransformed Grid:|$)', question, re.DOTALL)
            if not input_grid_str:
              return None

            input_grid_str = input_grid_str.group(1).strip()

            try:
              input_grid = eval(input_grid_str) # Convert grid string into matrix
              rows = len(input_grid)
              cols = len(input_grid[0]) if rows > 0 else 0

              mid_row = rows // 2
              mid_col = cols // 2

              #Divide the input_grid string into four quadrants
              quadrant1_question = question.replace(input_grid_str, str([row[:mid_col] for row in input_grid[:mid_row]]))
              quadrant2_question = question.replace(input_grid_str, str([row[mid_col:] for row in input_grid[:mid_row]]))
              quadrant3_question = question.replace(input_grid_str, str([row[:mid_col] for row in input_grid[mid_row:]]))
              quadrant4_question = question.replace(input_grid_str, str([row[mid_col:] for row in input_grid[mid_row:]]))

              #Recursively call function to process each of the quadrant
              q1_transformed = subdivide_and_transform(quadrant1_question, depth - 1)
              q2_transformed = subdivide_and_transform(quadrant2_question, depth - 1)
              q3_transformed = subdivide_and_transform(quadrant3_question, depth - 1)
              q4_transformed = subdivide_and_transform(quadrant4_question, depth - 1)

              #If any of the quadrants transformations return error, stop and return error message
              if any(q is None for q in [q1_transformed, q2_transformed, q3_transformed, q4_transformed]):
                return None
              
              #Attempt to combine the transformed quadrants to construct the final transformed grid
              try:
                #Convert transformed string to list of lists and combine
                q1_transformed = eval(q1_transformed)
                q2_transformed = eval(q2_transformed)
                q3_transformed = eval(q3_transformed)
                q4_transformed = eval(q4_transformed)
                
                #Combine transformed quadrants
                top_half = [row1 + row2 for row1, row2 in zip(q1_transformed, q2_transformed)]
                bottom_half = [row1 + row2 for row1, row2 in zip(q3_transformed, q4_transformed)]
                transformed_grid = top_half + bottom_half

                return str(transformed_grid)
              except:
                return None

            except Exception as e:
              print(f"Error during grid parsing or processing: {e}")
              return None
        else:
            # No subdivision needed, transform directly
            return transform_subgrid(question)

    answer = subdivide_and_transform(question, max_recursion_depth)
    if answer is None:
        return "Error: Could not transform grid"

    return answer

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"