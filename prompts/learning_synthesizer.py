import json

def get_learning_synthesis_prompt(current_learnings, new_batch_learnings, approaching_limit=False):
  """
  Generate prompt and system instruction for synthesizing learnings.

  Args:
      current_learnings: String of existing accumulated learnings
      new_batch_learnings: String of new learnings from latest batch
      approaching_limit: Boolean indicating if we need to condense content

  Returns:
      tuple: (prompt, system_instruction)
  """
  system_instruction = """You are a Knowledge Integrator. Your role is to synthesize accumulated
      dataset-specific knowledge with new insights, creating an evolving experiment 
      log that captures concrete patterns, strategies, and findings about this specific task."""

  base_prompt = f"""
You are tasked with synthesizing existing knowledge with new learnings from our latest experiment on this dataset.

EXISTING ACCUMULATED LEARNINGS:
{current_learnings}

NEW LEARNINGS FROM LATEST BATCH:
{new_batch_learnings}

Create an updated, synthesized version of our learnings that:

1. Maintains a comprehensive catalog of DATASET PATTERNS we've identified
2. Tracks the evolution of our understanding about what makes this specific task challenging
3. Documents concrete STRATEGIES that have proven effective or ineffective for this particular dataset
4. Creates a running EXPERIMENT LOG tracking our attempts and findings specific to this task
5. Prioritizes concrete, task-specific insights over general system design principles

The synthesized learnings should read like a detailed research log about THIS specific dataset and task, 
not a general guide to system design. Each section should include specific examples and concrete findings.

Organize the information into these sections:

1. DATASET PATTERNS & CHARACTERISTICS
2. EFFECTIVE TASK-SPECIFIC STRATEGIES
3. COMMON FAILURE MODES ON THIS DATASET
4. EXPERIMENT LOG & FINDINGS
5. NEXT RESEARCH DIRECTIONS

Your output will replace the current learnings file and serve as long-term memory for working specifically with this dataset.

IMPORTANT: Preserve all concrete, specific insights from both the existing learnings and new batch. Don't lose valuable information. Preserve the details of runtime, execution, error, and processing problems so that we can learn from them in the future and DO NOT make the same mistakes again.
"""
  return base_prompt, system_instruction