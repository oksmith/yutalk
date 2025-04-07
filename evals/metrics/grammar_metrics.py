"""
Grammar correction evaluation metrics for YuTalk.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from yutalk.graph import GRAMMAR_CHECK_PROMPT, GrammarCheckResult
from yutalk.utils import setup_logger

logger = setup_logger(name="grammar_metrics", level=logging.INFO)

load_dotenv()

DEFAULT_LLM = "gpt-4o-mini"


def parse_grammar_response(response_text: str) -> Tuple[bool, Optional[List[Dict]]]:
    """
    Parse the grammar check response to determine if text is correct or needs corrections.
    
    Args:
        response_text: Response from the grammar checker
    
    Returns:
        Tuple of (is_correct, corrections)
    """
    if response_text.startswith("CORRECT:"):
        return True, None
    
    # If not marked as correct, extract the corrections
    corrections = [{"explanation": response_text}]
    return False, corrections

async def check_grammar(text: str, model_name: str = DEFAULT_LLM) -> GrammarCheckResult:
    """
    Check grammar of Chinese text.
    
    Args:
        text: Chinese text to check
        model_name: Model to use for grammar checking
        
    Returns:
        GrammarCheckResult with grammar check results
    """
    llm = ChatOpenAI(model=model_name, temperature=0.3)
    
    # System message for grammar checking
    system_msg = SystemMessage(content=GRAMMAR_CHECK_PROMPT)
    
    # Prompt for grammar checking
    check_prompt = HumanMessage(content=text)
    
    try:
        response = await llm.ainvoke([system_msg, check_prompt])

        response_text = response.content
        is_correct, corrections = parse_grammar_response(response_text)
        
        return GrammarCheckResult(
            is_correct=is_correct,
            corrections=corrections,
            original_text=text
        )
    except Exception as e:
        logger.error(f"Error in grammar checking: {str(e)}")
        raise Exception(f"Error checking grammar: {str(e)}")

def check_expected_errors(
    detected_errors: Optional[List[Dict]], 
    expected_errors: List[Dict]
) -> Tuple[int, int, int]:
    """
    Check if detected errors match expected errors.
    
    Args:
        detected_errors: List of detected error explanations
        expected_errors: List of expected error types
        
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    if not expected_errors:
        # No errors expected
        if not detected_errors:
            # No errors detected (true negative)
            return 0, 0, 0
        else:
            # Errors detected when none expected (false positives)
            return 0, len(detected_errors), 0
    
    if not detected_errors:
        # Errors expected but none detected (false negatives)
        return 0, 0, len(expected_errors)
    
    # Simple approach: if any errors detected when errors expected, count as true positive
    # In a more sophisticated implementation, we could match specific error types
    return 1, 0, 0

async def evaluate_grammar_correction(
    test_cases_file: str, 
    model_name: str = DEFAULT_LLM
) -> Dict:
    """
    Evaluate grammar correction on test cases.
    
    Args:
        test_cases_file: Path to the YAML file with grammar test cases
        model_name: Model to use for grammar checking
        
    Returns:
        Dictionary with evaluation metrics
    """
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = yaml.safe_load(f)
    
    results = {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "accuracy": 0.0,
        "total_cases": 0,
        "correct_cases": 0,
        "categories": {},
        "details": []
    }
    
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_true_negatives = 0
    
    for test_case in test_cases.get("test_cases", []):
        category = test_case.get("category", "general")
        text = test_case.get("text", "")
        expected_result = test_case.get("expected_result", {})
        expected_has_errors = expected_result.get("has_errors", False)
        expected_errors = expected_result.get("errors", [])
        
        # Skip empty test cases
        if not text:
            continue
        
        # Initialize category if not exists
        if category not in results["categories"]:
            results["categories"][category] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "total_cases": 0,
                "correct_cases": 0
            }
        
        grammar_result = await check_grammar(text, model_name)
        
        if expected_has_errors:
            if grammar_result.is_correct:
                # Should have errors but none detected (false negative)
                true_positives, false_positives, false_negatives = 0, 0, 1
                correct = False
            else:
                # Check if detected errors match expected error types
                tp, fp, fn = check_expected_errors(grammar_result.corrections, expected_errors)
                true_positives, false_positives, false_negatives = tp, fp, fn
                correct = (tp > 0 and fp == 0 and fn == 0)
        else:
            if grammar_result.is_correct:
                # No errors expected, none detected (true negative)
                true_positives, false_positives, false_negatives = 0, 0, 0
                true_negatives = 1
                correct = True
            else:
                # No errors expected, but errors detected (false positive)
                true_positives, false_positives, false_negatives = 0, 1, 0
                true_negatives = 0
                correct = False
        
        # Update overall counters
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        total_true_negatives += (1 if expected_has_errors == False and grammar_result.is_correct else 0)
        
        # Update category-specific counters
        results["categories"][category]["total_cases"] += 1
        results["categories"][category]["correct_cases"] += (1 if correct else 0)
        
        case_result = {
            "category": category,
            "text": text,
            "expected_has_errors": expected_has_errors,
            "detected_has_errors": not grammar_result.is_correct,
            "correct": correct,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "corrections": grammar_result.corrections
        }
        results["details"].append(case_result)
        
        logger.info(f"Category: {category}")
        logger.info(f"Text: {text}")
        logger.info(f"Expected has errors: {expected_has_errors}")
        logger.info(f"Detected has errors: {not grammar_result.is_correct}")
        logger.info(f"Correct: {correct}")
    
    total_cases = len(test_cases.get("test_cases", []))
    results["total_cases"] = total_cases
    results["correct_cases"] = total_true_positives + total_true_negatives
    
    # Calculate precision, recall, F1
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (total_true_positives + total_true_negatives) / total_cases if total_cases > 0 else 0
    
    results["precision"] = precision
    results["recall"] = recall
    results["f1_score"] = f1
    results["accuracy"] = accuracy
    
    for category, cat_results in results["categories"].items():
        cat_total = cat_results["total_cases"]
        cat_correct = cat_results["correct_cases"]
        
        if cat_total > 0:
            cat_accuracy = cat_correct / cat_total
            cat_results["accuracy"] = cat_accuracy
        else:
            cat_results["accuracy"] = 0.0
    
    return results

def run_grammar_eval(test_cases_file: str, model_name: str = DEFAULT_LLM) -> Dict:
    """
    Synchronous wrapper for the async evaluate_grammar_correction function.
    This allows the eval to be called from synchronous code.
    """
    return asyncio.run(evaluate_grammar_correction(test_cases_file, model_name))

if __name__ == "__main__":
    # Example usage (for testing)
    async def test_grammar_check():
        test_text = "我每天吃饭三次。"  # Incorrect word order in Chinese
        
        grammar_result = await check_grammar(test_text)
        
        print(f"Text: {test_text}")
        print(f"Is correct: {grammar_result.is_correct}")
        
        if not grammar_result.is_correct and grammar_result.corrections:
            print("Corrections:")
            for correction in grammar_result.corrections:
                print(f"  {correction.get('explanation', '')}")
    
    asyncio.run(test_grammar_check())
