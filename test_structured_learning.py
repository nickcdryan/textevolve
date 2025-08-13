#!/usr/bin/env python3
"""
Test script for the enhanced structured learning system.
Demonstrates deep script analysis and conditional learning capabilities.
"""

from structured_learning_system import StructuredLearningSystem
import json

def test_structured_learning():
    """Test the structured learning system with sample data."""
    
    print("=== Testing Enhanced Structured Learning System ===\n")
    
    # Initialize the system
    sls = StructuredLearningSystem()
    
    # Sample script analysis (simulating real analysis results)
    sample_script_analysis = {
        "architectural_pattern": "database_first",
        "processing_stages": "2 stages: email extraction, database lookup",
        "implementation_quality": "GOOD",
        "tool_usage_efficiency": "OPTIMAL",
        "llm_integration_pattern": "single_call",
        "transferability": "HIGH",
        "failure_modes": ["API connection issues", "Email parsing errors"],
        "improvement_opportunities": ["Add retry logic", "Better error handling"]
    }
    
    # Sample complexity assessment
    sample_complexity_assessment = {
        "task_complexity": "SIMPLE",
        "approach_complexity": "SIMPLE",
        "match_assessment": "APPROPRIATE",
        "recommendation": "MAINTAIN"
    }
    
    # Sample error analysis
    sample_error_analysis = {
        "primary_issue": "No major issues identified",
        "runtime_errors": [],
        "strengths": ["Direct database access", "Clean code structure"],
        "weaknesses": ["Limited error handling"]
    }
    
    # Sample performance data
    sample_performance = {
        "accuracy": 0.85,
        "success_rate": 0.90,
        "error_count": 1
    }
    
    # Sample dataset context
    dataset_context = {
        "type": "TicketWorldSimpleDatasetLoader",
        "has_database": True,
        "requires_reasoning": False
    }
    
    # Add a successful learning
    print("1. Adding a successful learning example...")
    learning_id = sls.add_learning(
        iteration=1,
        script_analysis=sample_script_analysis,
        complexity_assessment=sample_complexity_assessment,
        error_analysis=sample_error_analysis,
        performance_data=sample_performance,
        dataset_context=dataset_context
    )
    print(f"   Added learning: {learning_id}\n")
    
    # Add a failed learning (over-engineered approach)
    print("2. Adding a failed learning example (over-engineered)...")
    failed_script_analysis = {
        "architectural_pattern": "multi_step_reasoning",
        "processing_stages": "5 stages: extraction, verification, lookup, validation, resolution",
        "implementation_quality": "FAIR",
        "tool_usage_efficiency": "INEFFICIENT",
        "llm_integration_pattern": "verification_loop",
        "transferability": "LOW",
        "failure_modes": ["Unnecessary complexity", "Multiple LLM calls"],
        "improvement_opportunities": ["Simplify approach", "Direct database access"]
    }
    
    failed_complexity_assessment = {
        "task_complexity": "SIMPLE",
        "approach_complexity": "COMPLEX",
        "match_assessment": "OVER_ENGINEERED",
        "recommendation": "SIMPLIFY"
    }
    
    failed_performance = {
        "accuracy": 0.20,
        "success_rate": 0.60,
        "error_count": 4
    }
    
    learning_id2 = sls.add_learning(
        iteration=2,
        script_analysis=failed_script_analysis,
        complexity_assessment=failed_complexity_assessment,
        error_analysis={"primary_issue": "Over-engineering for simple task"},
        performance_data=failed_performance,
        dataset_context=dataset_context
    )
    print(f"   Added learning: {learning_id2}\n")
    
    # Test conditional learning queries
    print("3. Testing conditional learning queries...")
    
    # Query for simple database tasks
    print("   Querying for simple database tasks:")
    simple_db_conditions = {
        "task_complexity": "SIMPLE",
        "has_database": True
    }
    
    relevant_learnings = sls.query_learnings(
        conditions=simple_db_conditions,
        min_evidence_strength="MEDIUM",
        limit=5
    )
    
    for learning in relevant_learnings:
        print(f"     - Iteration {learning['iteration']}: {learning['lesson'][:80]}...")
    print()
    
    # Get conditional guidance for current situation
    print("4. Getting conditional guidance for new iteration...")
    current_conditions = {
        "task_complexity": "SIMPLE",
        "has_database": True,
        "architectural_pattern": "unknown"
    }
    
    guidance = sls.get_conditional_guidance(current_conditions)
    print(f"   Confidence: {guidance['confidence']}")
    print("   Recommendations:")
    for rec in guidance['recommendations']:
        print(f"     - {rec}")
    print("\n   Supporting Evidence:")
    for evidence in guidance['supporting_evidence']:
        print(f"     - Iteration {evidence['iteration']}: {evidence['accuracy']:.2f} accuracy - {evidence['lesson'][:60]}...")
    print()
    
    # Get learning summary
    print("5. Learning system summary:")
    summary = sls.get_learning_summary()
    print(f"   Total learnings: {summary['total_learnings']}")
    print(f"   Learning types: {summary['learning_types']}")
    print(f"   Evidence strengths: {summary['evidence_strengths']}")
    print(f"   High confidence learnings: {summary['high_confidence_learnings']}")
    print()
    
    # Test pattern-based queries
    print("6. Testing pattern-based queries...")
    database_first_learnings = sls.query_learnings(
        conditions={"architectural_pattern": "database_first"},
        learning_types=["SUCCESSFUL_PATTERN"],
        limit=3
    )
    
    print(f"   Found {len(database_first_learnings)} database_first success patterns:")
    for learning in database_first_learnings:
        print(f"     - {learning['lesson'][:80]}...")
    print()
    
    print("=== Structured Learning System Test Complete ===")
    print(f"Created structured_learnings.json with {summary['total_learnings']} entries")

if __name__ == "__main__":
    test_structured_learning()