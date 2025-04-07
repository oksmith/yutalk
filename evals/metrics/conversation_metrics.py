"""
Conversation quality evaluation metrics for YuTalk.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from yutalk.graph import UserPreferences, YuTalkGraph
from yutalk.utils import setup_logger

logger = setup_logger(name="conversation_metrics", level=logging.INFO)

load_dotenv()

DEFAULT_LLM = "gpt-4o-mini"

CONVERSATION_EVAL_PROMPT = """
You are evaluating the quality of a language learning conversation assistant.
Analyze the following conversation between a language learner and an AI assistant.

Please rate the assistant's response on the following metrics on a scale from 1-5 (where 5 is best):

1. Coherence (1-5): How well does the assistant's response logically follow from the learner's message?
2. Relevance (1-5): How relevant is the assistant's response to the learner's message?
3. Language Level Appropriateness (1-5): How well does the assistant match the learner's language level?
4. Engagement (1-5): How engaging and conversation-continuing is the assistant's response?
5. Pedagogical Value (1-5): How helpful is this exchange for language learning?

Provide a JSON-formatted response with your ratings and brief explanations:
```json
{
  "coherence": {
    "score": 4,
    "explanation": "The response directly addresses the learner's question about..."
  },
  "relevance": {
    "score": 5,
    "explanation": "The assistant provides exactly the information requested..."
  },
  "language_level": {
    "score": 3,
    "explanation": "The vocabulary is appropriate but some structures may be too complex..."
  },
  "engagement": {
    "score": 4,
    "explanation": "The response encourages further conversation by asking..."
  },
  "pedagogical_value": {
    "score": 4,
    "explanation": "The assistant subtly reinforces key grammar patterns..."
  }
}
```

Remember, you're evaluating a Chinese language learning assistant, so consider both linguistic appropriateness and educational value in your assessment.

IMPORTANT: Your response MUST be valid JSON and nothing else. Do not include any text outside the JSON structure.
"""


async def evaluate_conversation_quality(
    user_message: str,
    assistant_response: str,
    metrics: List[str],
    model_name: str = "gpt-4o-mini"
) -> Dict:
    """
    Evaluate the quality of a single conversation exchange.
    
    Args:
        user_message: The user's message
        assistant_response: The assistant's response
        metrics: List of metrics to evaluate
        model_name: Model to use for evaluation
        
    Returns:
        Dictionary with evaluation scores
    """
    llm = ChatOpenAI(model=model_name, temperature=0.3)
    
    # Create sys prompt
    system_msg = SystemMessage(content=CONVERSATION_EVAL_PROMPT)
    
    # Create eval prompt for this particular conversation exchange
    eval_prompt = f"""
    User message: {user_message}
    
    Assistant response: {assistant_response}
    """
    human_msg = HumanMessage(content=eval_prompt)

    try:
        # Invoke the LLM for evaluation
        response = await llm.ainvoke([system_msg, human_msg])
        response_text = response.content
        
        # Extract JSON from the response, handling various formats
        json_str = ""

        # Case 1: Response is enclosed in ```json ... ``` blocks
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            if json_end > json_start:
                json_str = response_text[json_start:json_end].strip()
        
        # Case 2: Response is enclosed in ``` ... ``` blocks
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            if json_end > json_start:
                json_str = response_text[json_start:json_end].strip()
        
        # Case 3: Response might be a JSON object directly
        elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
            json_str = response_text.strip()
        
        # Fallback: Try to find JSON object within the text
        else:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1].strip()
            else:
                # If no JSON structure found, log the raw response for debugging
                logger.error(f"Could not extract JSON from response: {response_text}")
                return {metric: {"score": 0, "explanation": "Parsing failed"} for metric in metrics}
        
        try:
            scores = json.loads(json_str)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, log detailed error and response
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response text: {response_text}")
            logger.error(f"Extracted JSON string: {json_str}")
            
            # Try a last resort fix for common JSON issues
            fixed_json = json_str.replace("'", '"')
            fixed_json = re.sub(r',\s*}', '}', fixed_json)
            
            try:
                scores = json.loads(fixed_json)
                logger.info("JSON was fixed and successfully parsed after correction")
            except:
                return {metric: {"score": -1, "explanation": f"JSON parsing failed: {str(e)}"} for metric in metrics}
        
        
        # Extract only the requested metrics
        result = {}
        for metric in metrics:
            if metric in scores:
                result[metric] = scores[metric]
        
        return result
    except Exception as e:
        logger.error(f"Error in conversation evaluation: {str(e)}")
        return {metric: {"score": -1, "explanation": "Evaluation failed"} for metric in metrics}

async def simulate_conversation(
    test_case: Dict
) -> Dict:
    """
    Simulate a conversation exchange using YuTalk and evaluate the result.
    
    Args:
        test_case: Dictionary with test case details
        model_name: Model to use
        
    Returns:
        Dictionary with conversation exchange and evaluation
    """
    user_message = test_case.get("user_message", "")
    
    # Create user preferences
    user_prefs = UserPreferences(
        correction_level="balanced",
        skill_level="intermediate", # TODO: remove
        topics_of_interest=["daily life", "travel", "food"]
    )
    
    # Initialize YuTalk graph
    session_id = f"eval_{hash(user_message)}"
    yutalk = YuTalkGraph(user_preferences=user_prefs, session_id=session_id)
    
    try:
        result = []
        # TODO: really this should be wrapped into the same function as what YuTalk does. Don't need
        # to implement it twice.
        async for chunk in yutalk.process_user_input_stream(user_message):
            result.append(chunk)
        
        final_state = result[-1] if result else {}
        messages = final_state[1].get("messages", [])
        
        if messages and len(messages) > 0:
            # The last message should be the assistant's response
            assistant_response = messages[-1].content
        else:
            assistant_response = "No response generated"
        
        return {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error in conversation simulation: {str(e)}")
        return {
            "user_message": user_message,
            "assistant_response": "Error: " + str(e),
            "success": False
        }


async def evaluate_conversation(
    test_cases_file: str,
    model_name: str = DEFAULT_LLM,
    metrics: List[str] = ["coherence", "relevance", "engagement"]
) -> Dict:
    """
    Evaluate conversation quality using test cases.
    
    Args:
        test_cases_file: Path to the YAML file with conversation test cases
        model_name: Model to use for evaluation
        metrics: List of metrics to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = yaml.safe_load(f)
    
    results = {
        "overall": {metric: 0.0 for metric in metrics},
        "test_cases": []
    }
    
    for test_case in test_cases.get("test_cases", []):
        conversation = await simulate_conversation(test_case)
        
        if not conversation["success"]:
            logger.warning(f"Conversation simulation failed for test case: {test_case}")
            continue
        
        evaluation = await evaluate_conversation_quality(
            conversation["user_message"],
            conversation["assistant_response"],
            metrics,
            model_name
        )
        
        # Extract test case metrics
        case_result = {
            "user_message": conversation["user_message"],
            "assistant_response": conversation["assistant_response"],
            "evaluation": evaluation
        }
        results["test_cases"].append(case_result)
        
        # Keep track in a results object
        for metric in metrics:
            if metric in evaluation:
                results["overall"][metric] += evaluation[metric].get("score", 0)

    total_cases = len(results["test_cases"])
    if total_cases > 0:
        for metric in metrics:
            results["overall"][metric] /= total_cases
    
    return results

def run_conversation_eval(test_cases_file: str, model_name: str = "gpt-4o-mini") -> Dict:
    """
    Synchronous wrapper for the async evaluate_conversation function.
    This allows the eval to be called from synchronous code.
    """
    return asyncio.run(evaluate_conversation(test_cases_file, model_name))

if __name__ == "__main__":
    # Example usage (for testing)
    async def test_evaluation():
        example_exchange = {
            "user_message": "你好，我想学习中文。",
            "assistant_response": "你好！很高兴认识你。学习中文是一个很好的选择。你已经学习中文多久了？你对哪些中文话题感兴趣？"
        }
        
        evaluation = await evaluate_conversation_quality(
            example_exchange["user_message"],
            example_exchange["assistant_response"],
            ["coherence", "relevance", "engagement"]
        )
        
        print(json.dumps(evaluation, indent=2, ensure_ascii=False))
    
    asyncio.run(test_evaluation())

    # test_case = {"user_message ": '你好！我叫大卫。'}
    # asyncio.run(simulate_conversation(test_case))

