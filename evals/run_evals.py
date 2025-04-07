#!/usr/bin/env python
"""
Evaluation script for YuTalk Chinese language learning chatbot.
Runs automated tests to evaluate speech recognition, grammar correction,
and conversation capabilities.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.metrics.conversation_metrics import evaluate_conversation
from evals.metrics.grammar_metrics import evaluate_grammar_correction
from evals.metrics.speech_metrics import evaluate_speech_recognition
from yutalk.utils import setup_logger

logger = setup_logger(name="yutalk_evals", level=logging.INFO, log_file="evals.log")

def load_config(config_path: str) -> Dict:
    """Load evaluation configuration from a YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_results(results: Dict, output_dir: str, run_name: Optional[str] = None) -> str:
    """Save evaluation results to a JSON file."""
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"eval_run_{timestamp}"
    
    # Create the output directory if it doesn't exist
    output_path = Path(output_dir) / "runs" / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the results as JSON
    results_file = output_path / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    return str(results_file)

def compare_with_baseline(current_results: Dict, baseline_path: str) -> Dict:
    """Compare current evaluation results with the baseline."""
    try:
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        
        comparison = {"metrics": {}}
        
        # Compare each metric
        for category, metrics in current_results["metrics"].items():
            comparison["metrics"][category] = {}
            for metric_name, current_value in metrics.items():
                if category in baseline["metrics"] and metric_name in baseline["metrics"][category]:
                    baseline_value = baseline["metrics"][category][metric_name]
                    
                    # For metrics where higher is better
                    is_higher_better = metric_name not in ["wer", "cer", "error_rate"]
                    
                    if is_higher_better:
                        change = current_value - baseline_value
                        change_percent = (change / baseline_value) * 100 if baseline_value != 0 else float('inf')
                        improvement = change > 0
                    else:
                        change = baseline_value - current_value
                        change_percent = (change / baseline_value) * 100 if baseline_value != 0 else float('inf')
                        improvement = change > 0
                    
                    comparison["metrics"][category][metric_name] = {
                        "baseline": baseline_value,
                        "current": current_value,
                        "change": change,
                        "change_percent": change_percent,
                        "improved": improvement
                    }
        
        return comparison
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error comparing with baseline: {str(e)}")
        return {"error": str(e)}

async def run_all_evals_async(
    config_path: str, 
    output_dir: str,
    run_name: Optional[str] = None,
    compare_baseline: bool = True
) -> Dict:
    """Run all evaluations based on the provided configuration (async version)."""
    # Load the evaluation configuration
    config = load_config(config_path)
    
    # Initialize results dictionary
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config_path,
        "metrics": {}
    }
    
    # TODO: check all of these configs! What model to use in each one, for example. Probably just want to
    # use the same as what YuTalk uses for grammar correction?

    # Run speech recognition evaluation if configured
    if config.get("speech_recognition", {}).get("enabled", False):
        logger.info("Running speech recognition evaluation...")
        speech_results = evaluate_speech_recognition(
            audio_dir=config["speech_recognition"]["audio_dir"],
            reference_file=config["speech_recognition"]["reference_file"],
        )
        results["metrics"]["speech_recognition"] = speech_results
    
    # Run grammar correction evaluation if configured
    if config.get("grammar_correction", {}).get("enabled", False):
        logger.info("Running grammar correction evaluation...")
        grammar_results = await evaluate_grammar_correction(
            test_cases_file=config["grammar_correction"]["test_cases_file"],
            model_name=config["grammar_correction"].get("model_name", "gpt-4o-mini")
        )
        results["metrics"]["grammar_correction"] = grammar_results
    
    # Run conversation evaluation if configured
    if config.get("conversation", {}).get("enabled", False):
        logger.info("Running conversation evaluation...")
        conversation_results = await evaluate_conversation(
            test_cases_file=config["conversation"]["test_cases_file"],
            model_name=config["conversation"].get("model_name", "gpt-4o-mini"),
            metrics=config["conversation"].get("metrics", ["coherence", "relevance", "engagement"])
        )
        results["metrics"]["conversation"] = conversation_results
    
    # Save the results
    results_file = save_results(results, output_dir, run_name)
    
    # Compare with baseline if required
    if compare_baseline:
        baseline_path = Path(output_dir) / "baseline" / "results.json"
        if baseline_path.exists():
            logger.info("Comparing with baseline...")
            comparison = compare_with_baseline(results, str(baseline_path))
            comparison_file = Path(results_file).parent / "comparison.json"
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            logger.info(f"Comparison saved to {comparison_file}")
        else:
            logger.warning(f"Baseline file not found at {baseline_path}. Skipping comparison.")
    
    return results

def run_all_evals(
    config_path: str, 
    output_dir: str,
    run_name: Optional[str] = None,
    compare_baseline: bool = True
) -> Dict:
    """Synchronous wrapper for the async run_all_evals_async function."""
    return asyncio.run(run_all_evals_async(
        config_path, 
        output_dir, 
        run_name, 
        compare_baseline
    ))

async def create_baseline_async(
    config_path: str, 
    output_dir: str,
    force: bool = False
) -> None:
    """Create a baseline by running evaluations and saving results as baseline (async version)."""
    baseline_dir = Path(output_dir) / "baseline"
    baseline_file = baseline_dir / "results.json"
    
    if baseline_file.exists() and not force:
        logger.warning(f"Baseline already exists at {baseline_file}. Use --force to overwrite.")
        return
    
    logger.info("Creating baseline...")
    results = await run_all_evals_async(config_path, output_dir, "baseline", compare_baseline=False)
    
    # Save as baseline
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with open(baseline_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Baseline created at {baseline_file}")

def create_baseline(
    config_path: str, 
    output_dir: str,
    force: bool = False
) -> None:
    """Synchronous wrapper for the async create_baseline_async function."""
    asyncio.run(create_baseline_async(config_path, output_dir, force))



def main():
    parser = argparse.ArgumentParser(description="Run evaluations for YuTalk")
    parser.add_argument(
        "--config", 
        type=str, 
        default="evals/config.yaml", 
        help="Path to the evaluation configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evals/results", 
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--run-name", 
        type=str, 
        help="Name for this evaluation run (default: timestamp)"
    )
    parser.add_argument(
        "--create-baseline", 
        action="store_true", 
        help="Create a baseline from this evaluation run"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force overwrite of existing baseline"
    )
    parser.add_argument(
        "--no-compare", 
        action="store_true", 
        help="Do not compare with baseline"
    )
    parser.add_argument(
        "--speech-only",
        action="store_true",
        help="Run only speech recognition evaluation"
    )
    parser.add_argument(
        "--grammar-only",
        action="store_true",
        help="Run only grammar correction evaluation"
    )
    parser.add_argument(
        "--conversation-only",
        action="store_true",
        help="Run only conversation evaluation"
    )
    
    args = parser.parse_args()
    
    # Modify config based on specific evaluation flags
    if args.speech_only or args.grammar_only or args.conversation_only:
        config = load_config(args.config)
        
        # Disable all evals by default if any specific one is selected
        config["speech_recognition"]["enabled"] = False
        config["grammar_correction"]["enabled"] = False
        config["conversation"]["enabled"] = False
        
        # Enable only the selected evals
        if args.speech_only:
            config["speech_recognition"]["enabled"] = True
        if args.grammar_only:
            config["grammar_correction"]["enabled"] = True
        if args.conversation_only:
            config["conversation"]["enabled"] = True
        
        # Create a temporary config file
        temp_config_path = os.path.join(os.path.dirname(args.config), "temp_config.yaml")
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        config_path = temp_config_path
    else:
        config_path = args.config
    
    try:
        if args.create_baseline:
            create_baseline(config_path, args.output_dir, args.force)
        else:
            run_all_evals(
                config_path, 
                args.output_dir, 
                args.run_name, 
                not args.no_compare
            )
    finally:
        # Clean up temporary config file if created
        if args.speech_only or args.grammar_only or args.conversation_only:
            try:
                os.remove(temp_config_path)
            except:
                pass

if __name__ == "__main__":
    main()