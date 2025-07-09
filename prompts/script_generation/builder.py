def build_script_generation_prompt(strategy_mode, context):
  """
  Main orchestrator that combines all the pieces
  """
  # Get base elements
  api_example = get_api_example()
  core_instructions = get_core_instructions()

  # Select LLM patterns based on strategy or context
  if strategy_mode == "explore":
      patterns = get_exploration_patterns()
  elif context.get('needs_verification'):
      patterns = get_verification_patterns()
  else:
      patterns = get_standard_patterns()

  # Get strategy-specific content
  strategy_content = get_strategy_content(strategy_mode, context)

  # Combine everything
  prompt = f"""
  {strategy_content}

  {get_examples_section(context)}

  {get_historical_context(context)}

  {get_learning_context(context)}

  {patterns}

  {core_instructions}

  {api_example}

  {get_requirements_section(strategy_mode)}
  """

  return prompt