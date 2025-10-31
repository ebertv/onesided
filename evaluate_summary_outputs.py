import asyncio
import json
import logging
from pathlib import Path
import random
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
import os
import sys
from openai import OpenAI
from rouge_score import rouge_scorer
from statistics import mean, stdev
import re
import aiohttp
from asyncio import Semaphore
import time
from tqdm import tqdm# Add data root and conversational data loader
DATA_ROOT = os.getenv("DATA_ROOT", "./data")
sys.path.append(".")
sys.path.append("./data")
try:
    from conversational_dataloader import ConversationalDataLoader
except ImportError:
    try:
        sys.path.append(str(Path(__file__).parent))
        from conversational_dataloader import ConversationalDataLoader
    except ImportError:
        print("Warning: Could not import ConversationalDataLoader")
        ConversationalDataLoader = None

# Constants - Optimized for maximum speed (can be overridden by environment variables)
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '30'))  # Reduced to avoid event loop issues
OPENAI_RATE_LIMIT_CALLS = int(os.getenv('OPENAI_RATE_LIMIT_CALLS', '50'))  # Reduced to avoid event loop issues
OPENAI_RATE_LIMIT_WINDOW = int(os.getenv('OPENAI_RATE_LIMIT_WINDOW', '60'))
OPENAI_RATE_LIMIT_DELAY = float(os.getenv('OPENAI_RATE_LIMIT_DELAY', '2'))  # Increased delay for stability
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '15'))  # Smaller batches for stability

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv('.env')

# Get API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in .env file")

class RateLimiter:
    def __init__(self, min_interval: float = 0.4):
        self.min_interval = min_interval
        self.lock = asyncio.Lock()
        self.last_call = 0.0

    async def acquire(self):
        async with self.lock:
            now = time.time()
            wait_time = self.min_interval - (now - self.last_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call = time.time()

class PredictionEvaluator:
    def __init__(self, openai_api_key: str):
        """Initialize the comparator with API clients and optimized rate limiting"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        
        # API call counters
        self.openai_api_calls = 0
        
        # Separate semaphores for better concurrency control
        self.openai_sem = Semaphore(MAX_CONCURRENT_REQUESTS)  # All requests can use either API
        
        # Separate rate limiters for Claude and OpenAI
        self.openai_rate_limiter = RateLimiter(min_interval=0.4)
    

    async def evaluate_summaries(self, masked_summary: str, full_summary: str, predicted_summary: str, dialogue_context: str) -> Dict[str, Any]:
        """Evaluate the difference between masked, predicted and full summaries using ROUGE and blind LLM judge"""
        try:
            # Calculate ROUGE scores only for masked_vs_full and predicted_vs_full
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge_scores = {
                "masked_vs_full": scorer.score(masked_summary, full_summary),
                "predicted_vs_full": scorer.score(predicted_summary, full_summary)
            }
            
            rouge_metrics = {
                "masked_vs_full": {
                    "rouge1": rouge_scores["masked_vs_full"]['rouge1'].fmeasure,
                    "rouge2": rouge_scores["masked_vs_full"]['rouge2'].fmeasure,
                    "rougeL": rouge_scores["masked_vs_full"]['rougeL'].fmeasure
                },
                "predicted_vs_full": {
                    "rouge1": rouge_scores["predicted_vs_full"]['rouge1'].fmeasure,
                    "rouge2": rouge_scores["predicted_vs_full"]['rouge2'].fmeasure,
                    "rougeL": rouge_scores["predicted_vs_full"]['rougeL'].fmeasure
                }
            }
            
            # Create randomized order for blind evaluation
            summaries_dict = {
                "complete": full_summary,
                "masked": masked_summary, 
                "predicted": predicted_summary
            }
            
            # Create randomized mapping: [(label, summary_type), ...]
            summary_types = list(summaries_dict.keys())
            random.shuffle(summary_types)
            blind_labels = ["A", "B", "C"]
            randomized_mapping = [(blind_labels[i], summary_types[i]) for i in range(len(summary_types))]
            
            # Log the randomization for debugging
            logging.info(f"=== BLIND EVALUATION RANDOMIZATION ===")
            logging.info(f"Original order: {list(summaries_dict.keys())}")
            logging.info(f"Randomized mapping: {randomized_mapping}")
            
            # Create the summaries section with randomized order
            blind_summaries_text = ""
            for label, summary_type in randomized_mapping:
                summary_text = summaries_dict[summary_type]
                blind_summaries_text += f"Summary {label}: {summary_text}\n\n"
            
            blind_evaluation_prompt = f"""
You are evaluating three summaries of the same dialogue. You will evaluate each summary individually with detailed reasoning, then rank them.

Context: These are three different summaries of the same dialogue, labeled A, B, and C. You do not know which method was used to generate each summary. Use the original dialogue below as your reference to evaluate how well each summary captures the actual conversation content.

## Evaluation Process
For each summary, provide specific reasoning FIRST, then assign a score (1-5) for each criterion:

1. **Content Coverage** – How well does the summary capture all the key specific information and main points from the original dialogue?
2. **Dialogue Flow** – How well does the summary reflect the natural progression and interaction between speakers?
3. **Information Accuracy** – How accurate and faithful is the summary to the available information?
4. **Purpose & Outcome** – How clearly does the summary convey the dialogue's goals and results?
5. **Detail Balance** – How well does the summary balance important details from both speakers?

**IMPORTANT**: Do NOT penalize summaries for using "XXXXXXX" placeholders.
These represent unknown specific information (like names, numbers, addresses) that was not available in the original context. 
Using XXXXXXX appropriately (when info is not in context) should be considered the same as using the actual correct info.

## Scoring Scale
1 – Poor/Inadequate
2 – Fair/Partial
3 – Good/Adequate
4 – Very Good/Comprehensive
5 – Excellent/Complete

## Original Dialogue (for reference)
{dialogue_context}

## Summaries to Evaluate
{blind_summaries_text.strip()}

## Output Format
Provide reasoning (max 30 words) FIRST, then assign 1-5 scores. Respond with valid JSON in this EXACT format.
{{
"reasoning_and_scores": {{
    "summary_a": {{
        "content_coverage_reasoning": "<string: your specific reasoning>",
        "content_coverage": <integer 1-5>,
        "dialogue_flow_reasoning": "<string: your specific reasoning>",
        "dialogue_flow": <integer 1-5>,
        "information_accuracy_reasoning": "<string: your specific reasoning>",
        "information_accuracy": <integer 1-5>,
        "purpose_outcome_reasoning": "<string: your specific reasoning>",
        "purpose_outcome": <integer 1-5>,
        "detail_balance_reasoning": "<string: your specific reasoning>",
        "detail_balance": <integer 1-5>,
        "total_score": <integer: sum of all 5 scores above>
    }},
    "summary_b": {{
        "content_coverage_reasoning": "<string: your specific reasoning>",
        "content_coverage": <integer 1-5>,
        "dialogue_flow_reasoning": "<string: your specific reasoning>",
        "dialogue_flow": <integer 1-5>,
        "information_accuracy_reasoning": "<string: your specific reasoning>",
        "information_accuracy": <integer 1-5>,
        "purpose_outcome_reasoning": "<string: your specific reasoning>",
        "purpose_outcome": <integer 1-5>,
        "detail_balance_reasoning": "<string: your specific reasoning>",
        "detail_balance": <integer 1-5>,
        "total_score": <integer: sum of all 5 scores above>
    }},
    "summary_c": {{
        "content_coverage_reasoning": "<string: your specific reasoning>",
        "content_coverage": <integer 1-5>,
        "dialogue_flow_reasoning": "<string: your specific reasoning>",
        "dialogue_flow": <integer 1-5>,
        "information_accuracy_reasoning": "<string: your specific reasoning>",
        "information_accuracy": <integer 1-5>,
        "purpose_outcome_reasoning": "<string: your specific reasoning>",
        "purpose_outcome": <integer 1-5>,
        "detail_balance_reasoning": "<string: your specific reasoning>",
        "detail_balance": <integer 1-5>,
        "total_score": <integer: sum of all 5 scores above>
    }}
}},
"ranking": [<"A" or "B" or "C">, <"A" or "B" or "C">, <"A" or "B" or "C">],
"ranking_explanation": "<string: your explanation>",
"comparative_analysis": "<string: your analysis>"
}}"""

            # Run both evaluations concurrently instead of sequentially
            evaluation_tasks = [
                self.get_blind_evaluation(blind_evaluation_prompt),
                self.evaluate_precision_recall(predicted_summary, full_summary),
                self.evaluate_masked_precision_recall(masked_summary, full_summary)
            ]
            
            blind_eval_raw, predicted_precision_recall_eval, masked_precision_recall_eval = await asyncio.gather(*evaluation_tasks)
            
            if not blind_eval_raw:
                logging.error("Blind evaluation failed")
                return None
            
            try:
                # Create label-to-evaluation dict (Dict 3)
                logging.info(f"=== MAPPING BACK EVALUATION RESULTS ===")
                logging.info(f"Available keys in GPT response: {list(blind_eval_raw['reasoning_and_scores'].keys())}")
                logging.info(f"Using randomized mapping: {randomized_mapping}")
                
                # Dict 3: label -> evaluation_scores
                label_to_eval = {}
                for label, _ in randomized_mapping:
                    gpt_key = f"summary_{label.lower()}"
                    if gpt_key not in blind_eval_raw["reasoning_and_scores"]:
                        logging.error(f"GPT key '{gpt_key}' not found in response")
                        return None
                    label_to_eval[label] = blind_eval_raw["reasoning_and_scores"][gpt_key]
                
                # Map back to summary types using randomized_mapping
                blind_eval = {"scores": {}}
                for label, summary_type in randomized_mapping:
                    blind_eval["scores"][summary_type] = label_to_eval[label]
                    logging.info(f"Mapped {label} -> {summary_type} ✓")
                
                # Map ranking back to summary types
                if "ranking" not in blind_eval_raw:
                    logging.error("No ranking found in GPT response")
                    return None
                
                # Create label-to-type lookup from randomized_mapping
                label_to_type = {label: summary_type for label, summary_type in randomized_mapping}
                
                mapped_ranking = []
                for label in blind_eval_raw["ranking"]:
                    if label in label_to_type:
                        mapped_type = label_to_type[label]
                        mapped_ranking.append(mapped_type)
                        logging.info(f"Ranking: {label} -> {mapped_type} ✓")
                    else:
                        logging.error(f"Unknown ranking label '{label}'")
                        return None
                
                blind_eval.update({
                    "ranking": mapped_ranking,
                    "ranking_explanation": blind_eval_raw.get("ranking_explanation", ""),
                    "comparative_analysis": blind_eval_raw.get("comparative_analysis", "")
                })
                
                # Validate scores are numeric and in valid range
                for summary_type, scores in blind_eval["scores"].items():
                    for metric, value in scores.items():
                        if metric == "total_score":
                            if not isinstance(value, (int, float)) or not (5 <= value <= 25):
                                logging.warning(f"Invalid total_score for {summary_type}: {value} (should be 5-25)")
                        elif metric in ["content_coverage", "dialogue_flow", "information_accuracy", "purpose_outcome", "detail_balance"]:
                            if not isinstance(value, (int, float)) or not (1 <= value <= 5):
                                logging.warning(f"Invalid score for {summary_type}.{metric}: {value} (should be 1-5)")
                
                logging.info(f"Mapped scores - Complete: {blind_eval['scores']['complete'].get('total_score', 'N/A')}, "
                           f"Masked: {blind_eval['scores']['masked'].get('total_score', 'N/A')}, "
                           f"Predicted: {blind_eval['scores']['predicted'].get('total_score', 'N/A')}")
                logging.info(f"Ranking: {blind_eval['ranking']}")
                
                # # Add precision/recall evaluation if available
                # print("ADDING PRECISION/RECALL EVALS")
                # if predicted_precision_recall_eval:
                #     blind_eval["precision_recall"]["predicted"] = predicted_precision_recall_eval
                #     print("ADDED PREDICTED PRECISION/RECALL")
                # if masked_precision_recall_eval:
                #     blind_eval["precision_recall"]["masked"] = masked_precision_recall_eval
                #     print("ADDED MASKED PRECISION/RECALL")
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse GPT-4 response: {e}")
                return None
 
            return {
                "rouge_scores": rouge_metrics,
                "blind_evaluation": blind_eval,
                "precision_recall_evaluations": {
                    "predicted": predicted_precision_recall_eval,
                    "masked": masked_precision_recall_eval
                },
                "summary_lengths": {
                    "masked_words": len(masked_summary.split()),
                    "full_words": len(full_summary.split()),
                    "predicted_words": len(predicted_summary.split()),
                    "length_differences": {
                        "masked_vs_full": len(full_summary.split()) - len(masked_summary.split()),
                        "predicted_vs_full": len(full_summary.split()) - len(predicted_summary.split())
                    }
                },
                "randomization_info": {
                    "original_order": list(summaries_dict.keys()),
                    "randomized_mapping": randomized_mapping,
                    "label_to_type": {label: summary_type for label, summary_type in randomized_mapping}
                },
                "prompts_sent": {
                    "blind_evaluation_prompt": blind_evaluation_prompt
                }
            }
                
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return None

    async def evaluate_precision_recall(self, predicted_summary: str, full_summary: str) -> Dict[str, Any]:
        """Separate precision/recall evaluator that compares predicted vs full summary (labeled, not blind)"""
        try:
            precision_recall_prompt = f"""
You are evaluating the precision and recall of a predicted summary compared to a complete summary.

## Task
Compare the predicted summary against the complete summary and extract concrete, specific, verifiable details for precision/recall calculation.

## Details Extraction and Precision/Recall Calculation
- Extract **actual_details**: list of concrete, specific, verifiable details in the complete summary
- Extract **predicted_details**: list of concrete, specific, verifiable details in the predicted summary
- Compare the lists:
  - **TP** = number of predicted_details also in actual_details
  - **FP** = number of predicted_details not in actual_details  
  - **FN** = number of actual_details not in predicted_details
- Calculate:
  - **precision_fraction** = TP / max(1, TP + FP)
  - **recall_fraction** = TP / max(1, TP + FN)

**IMPORTANT**: Do NOT penalize summaries for using "XXXXXXX" placeholders.
These represent unknown specific information (like names, numbers, addresses) that was not available in the original context. 
Using XXXXXXX appropriately should be considered the same as using the actual correct info.

## Summaries to Compare
Complete Summary (Reference): {full_summary}

Predicted Summary (To Evaluate): {predicted_summary}

## Output Format
Respond with valid JSON in this EXACT format:

{{
"detail_extraction": {{
    "actual_details": [<list of strings: details from complete summary>],
    "predicted_details": [<list of strings: details from predicted summary>],
    "tp": <integer: true positives count>,
    "fp": <integer: false positives count>,
    "fn": <integer: false negatives count>,
    "precision_fraction": <float: TP / (TP + FP)>,
    "recall_fraction": <float: TP / (TP + FN)>
}},
"analysis": "<string: your analysis>"
}}"""

            async with self.openai_sem:
                await self.openai_rate_limiter.acquire()
                logging.info(f"OpenAI API call starting at {time.time():.3f}")
                
                def _call():
                    return self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert in text analysis and information extraction. Respond with valid JSON only."},
                            {"role": "user", "content": precision_recall_prompt}
                        ],
                        temperature=0.1
                    )
                
                pr_response = await asyncio.to_thread(_call)
                self.openai_api_calls += 1  # Increment counter after successful call
                logging.info(f"OpenAI API call completed at {time.time():.3f} (Total: {self.openai_api_calls})")
            
            try:
                content = pr_response.choices[0].message.content
                # Extract JSON from markdown code blocks if present
                if content.startswith('```json'):
                    content = content.split('```json')[1].split('```')[0].strip()
                elif content.startswith('```'):
                    content = content.split('```')[1].split('```')[0].strip()
                
                pr_eval = json.loads(content)
                # Add the evaluation prompt to the result
                pr_eval["evaluation_prompt"] = precision_recall_prompt
                return pr_eval
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse precision/recall response: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Precision/recall evaluation failed: {e}")
            return None

    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate aggregate metrics across all dialogue comparisons"""
        if not results:
            return {}
        
        # Check if evaluations were skipped
        evaluations_skipped = all(r.get("evaluation", {}).get("evaluation_type") == "skipped" for r in results if r.get("evaluation"))
        
        if evaluations_skipped:
            # Return simple metrics when evaluation was skipped
            return {
                "evaluation_type": "skipped",
                "num_dialogues": len(results),
                "message": "Detailed metrics not available - evaluation was skipped",
                "summary_info": {
                    "dialogues_processed": len(results),
                    "contexts_generated": sum(1 for r in results if r.get("contexts"))
                }
            }
        
        # All evaluations should be full evaluations with predictions
        has_full_evaluation = any(r.get("evaluation", {}).get("blind_evaluation") for r in results if r.get("evaluation"))
            
        # Extract ROUGE metrics for specified pairs only
        rouge_metrics = {}
        rouge_pairs = ["masked_vs_full", "predicted_vs_full"]
        
        for pair in rouge_pairs:
            try:
                rouge1_scores = [r["evaluation"]["rouge_scores"][pair]["rouge1"] for r in results 
                               if r.get("evaluation") and r["evaluation"].get("rouge_scores", {}).get(pair)]
                rouge2_scores = [r["evaluation"]["rouge_scores"][pair]["rouge2"] for r in results 
                               if r.get("evaluation") and r["evaluation"].get("rouge_scores", {}).get(pair)]
                rougeL_scores = [r["evaluation"]["rouge_scores"][pair]["rougeL"] for r in results 
                               if r.get("evaluation") and r["evaluation"].get("rouge_scores", {}).get(pair)]
                
                rouge_metrics[pair] = {
                    "rouge1": self.calc_stats(rouge1_scores),
                    "rouge2": self.calc_stats(rouge2_scores),
                    "rougeL": self.calc_stats(rougeL_scores)
                }
            except KeyError as e:
                logging.warning(f"Missing ROUGE scores for pair {pair}: {e}")
                rouge_metrics[pair] = {
                    "rouge1": {"mean": 0, "stdev": 0},
                    "rouge2": {"mean": 0, "stdev": 0},
                    "rougeL": {"mean": 0, "stdev": 0}
                }
        
        # Extract blind evaluation metrics (requires predictions)
        evaluation_metrics = {}
        
        if has_full_evaluation:
            # Extract blind evaluation metrics for each summary type
            blind_metrics = {}
            blind_summary_types = ["complete", "masked", "predicted"]
            
            for summary_type in blind_summary_types:
                try:
                    blind_scores = {
                        "total_score": [r["evaluation"]["blind_evaluation"]["scores"][summary_type]["total_score"] for r in results if r.get("evaluation") and r["evaluation"].get("blind_evaluation")],
                        "content_coverage": [r["evaluation"]["blind_evaluation"]["scores"][summary_type]["content_coverage"] for r in results if r.get("evaluation") and r["evaluation"].get("blind_evaluation")],
                        "dialogue_flow": [r["evaluation"]["blind_evaluation"]["scores"][summary_type]["dialogue_flow"] for r in results if r.get("evaluation") and r["evaluation"].get("blind_evaluation")],
                        "information_accuracy": [r["evaluation"]["blind_evaluation"]["scores"][summary_type]["information_accuracy"] for r in results if r.get("evaluation") and r["evaluation"].get("blind_evaluation")],
                        "purpose_outcome": [r["evaluation"]["blind_evaluation"]["scores"][summary_type]["purpose_outcome"] for r in results if r.get("evaluation") and r["evaluation"].get("blind_evaluation")],
                        "detail_balance": [r["evaluation"]["blind_evaluation"]["scores"][summary_type]["detail_balance"] for r in results if r.get("evaluation") and r["evaluation"].get("blind_evaluation")]
                    }
                    blind_metrics[summary_type] = {
                        metric: self.calc_stats(scores) for metric, scores in blind_scores.items()
                    }
                except KeyError as e:
                    logging.warning(f"Missing key in blind evaluation for {summary_type}: {e}")
                    blind_metrics[summary_type] = {}
            
            # Extract ranking analysis for blind evaluation
            try:
                blind_rankings = [r["evaluation"]["blind_evaluation"]["ranking"] for r in results if r.get("evaluation") and r["evaluation"].get("blind_evaluation")]
                blind_ranking_counts = self.analyze_rankings(blind_rankings, ["complete", "masked", "predicted"])
            except KeyError as e:
                logging.warning(f"Missing key in blind ranking analysis: {e}")
                blind_ranking_counts = {}
            
            evaluation_metrics = {
                "type": "blind_evaluation",
                "scores": blind_metrics,
                "ranking_analysis": blind_ranking_counts
            }
        
        # Extract separate precision/recall metrics from the dedicated evaluator (only available for full evaluation)
        precision_recall_metrics = {}
        if has_full_evaluation:
            try:
                predicted_precision_scores = []
                predicted_recall_scores = []

                masked_precision_scores = []
                masked_recall_scores = []
                
                for r in results:
                    if (r.get("evaluation") and 
                        r["evaluation"].get("precision_recall_evaluations")):

                        predicted = r["evaluation"]["precision_recall_evaluations"].get("predicted")
                        masked = r["evaluation"]["precision_recall_evaluations"].get("masked")

                        predicted_extraction = predicted["detail_extraction"] if predicted else {}
                        masked_extraction = masked["detail_extraction"] if masked else {}
                        
                        
                        if predicted_extraction and "precision_fraction" in predicted_extraction:
                            predicted_precision_scores.append(predicted_extraction["precision_fraction"])
                        if predicted_extraction and "recall_fraction" in predicted_extraction:
                            predicted_recall_scores.append(predicted_extraction["recall_fraction"])
                        
                        if masked_extraction and "precision_fraction" in masked_extraction:
                            masked_precision_scores.append(masked_extraction["precision_fraction"])
                        if masked_extraction and "recall_fraction" in masked_extraction:
                            masked_recall_scores.append(masked_extraction["recall_fraction"])
                    
                
                logging.info(f"Found {len(predicted_precision_scores)} predicted precision scores and {len(predicted_recall_scores)} predicted recall scores")
                logging.info(f"Found {len(masked_precision_scores)} masked precision scores and {len(masked_recall_scores)} masked recall scores")

                
                predicted_precision_recall_metrics = {
                    "precision": self.calc_stats(predicted_precision_scores),
                    "recall": self.calc_stats(predicted_recall_scores)
                }
                masked_precision_recall_metrics = {
                    "precision": self.calc_stats(masked_precision_scores),
                    "recall": self.calc_stats(masked_recall_scores)
                }
            except (KeyError, TypeError) as e:
                logging.warning(f"Error in precision/recall extraction: {e}")
                predicted_precision_recall_metrics = {"precision": {"mean": 0, "stdev": 0}, "recall": {"mean": 0, "stdev": 0}}
                masked_precision_recall_metrics = {"precision": {"mean": 0, "stdev": 0}, "recall": {"mean": 0, "stdev": 0}}
        
        # Extract length differences for specified pairs only
        length_diffs = {}
        length_pairs = ["masked_vs_full", "predicted_vs_full"]
        
        for pair in length_pairs:
            try:
                length_scores = [r["evaluation"]["summary_lengths"]["length_differences"][pair] for r in results 
                               if r.get("evaluation") and r["evaluation"].get("summary_lengths", {}).get("length_differences", {}).get(pair) is not None]
                length_diffs[pair] = self.calc_stats(length_scores)
            except KeyError as e:
                logging.warning(f"Missing key in length differences for {pair}: {e}")
                length_diffs[pair] = {"mean": 0, "stdev": 0}
        
        result = {
            "rouge_scores": rouge_metrics,
            "evaluation": evaluation_metrics,
            "summary_length_differences": length_diffs,
            "num_dialogues": len(results)
        }
        
        # Add precision/recall only if available
        if predicted_precision_recall_metrics and masked_precision_recall_metrics:
            result["precision_recall"] = {}
            result["precision_recall"]["predicted"] = predicted_precision_recall_metrics
            result["precision_recall"]["masked"] = masked_precision_recall_metrics
            
        return result

    def calc_stats(self, scores):
        """Calculate mean and standard deviation for a list of scores"""
        if not scores:
            return {"mean": 0, "stdev": 0}
        return {
            "mean": mean(scores),
            "stdev": stdev(scores) if len(scores) > 1 else 0
        }

    def analyze_rankings(self, rankings, summary_labels):
        """Analyze ranking patterns across evaluations"""
        if not rankings:
            return {}
        
        # Count how often each summary appears in each position
        max_positions = max(len(ranking) for ranking in rankings) if rankings else 2
        position_names = ["1st", "2nd", "3rd"][:max_positions]
        position_counts = {label: {pos: 0 for pos in position_names} for label in summary_labels}
        
        for ranking in rankings:
            for i, summary_type in enumerate(ranking):
                if i < len(position_names) and summary_type in position_counts:
                    position_counts[summary_type][position_names[i]] += 1
        
        # Calculate percentages
        total_rankings = len(rankings)
        position_percentages = {}
        for label in summary_labels:
            position_percentages[label] = {}
            for pos in position_names:
                position_percentages[label][pos] = (
                    position_counts[label][pos] / total_rankings * 100 if total_rankings > 0 else 0
                )
        
        return {
            "counts": position_counts,
            "percentages": position_percentages,
            "total_rankings": total_rankings
        }

    async def run_comparison(self, summaries, output_file):
        """Run full comparison evaluation across summaries"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        results = []
        
        try:
            logger.info(f"Processing {len(summaries)} summaries in batches of {BATCH_SIZE}")
            
            # Process summaries in batches concurrently
            for batch_start in range(0, len(summaries), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(summaries))
                batch = summaries[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(len(summaries)-1)//BATCH_SIZE + 1}")
                batch_results = await self.process_comparison_batch(batch, batch_start)
                
                # Add batch results to overall results
                results.extend(batch_results)
                
                # No delay between batches - rate limiting handled by RateLimiter class

            logger.info(f"Completed processing all batches. Total summaries processed: {len(results)}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            logger.info(f"Results written to {output_file}")
           
            
        except Exception as e:
            logger.error(f"❌ Comparison failed: {str(e)}", exc_info=True)
            raise

    async def process_comparison_batch(self, dialogues: List[Dict], start_idx: int) -> List[Dict]:
        """Process a batch of dialogues concurrently"""
        results, tasks = [], []

        for i, dialogue in enumerate(dialogues, start=start_idx):
            
            async def one_job(dialogue=dialogue):
                try:
                    result = await self.process_single_comparison(dialogue)
                    if result:
                        return result
                    else:
                        logging.warning(f"No result for dialogue {dialogue['dialogue_id']}")
                        return None
                except Exception as e:
                    logging.error(f"Error processing dialogue {dialogue['dialogue_id']}: {e}")
                    return None

            tasks.append(asyncio.create_task(one_job()))

        for finished in await asyncio.gather(*tasks, return_exceptions=True):
            if finished and not isinstance(finished, Exception):
                results.append(finished)
            elif isinstance(finished, Exception):
                logging.error(f"Task exception in batch processing: {finished}")
        
        return results

    async def process_single_comparison(self, dialogue: Dict) -> Dict:
        """Process a single dialogue comparison"""
        try:
            masked_summary = dialogue.get("masked_summary", "").strip()
            masked_context = dialogue.get("masked_context", "").strip()
            full_summary = dialogue.get("full_summary", "").strip()
            full_context = dialogue.get("full_context", "").strip()
            predicted_summary = dialogue.get("predicted_summary", "").strip()
            predicted_context = dialogue.get("predicted_context", "").strip()
            
            
            if not masked_summary or not full_summary:  # Must have masked and full summaries
                logging.warning(f"Missing required summaries for dialogue {dialogue.get('dialogue_id', 0)}")
                return None
            
            # Error out if no predicted summary is available
            if not predicted_summary:
                logging.error(f"No predicted summary available for dialogue {dialogue.get('dialogue_id', 0)} - this is required for evaluation")
                return None
            else:
                evaluation = await self.evaluate_summaries(masked_summary, full_summary, predicted_summary, full_context)
                
                if evaluation:
                    comparison = {
                        "dialogue_id": dialogue["dialogue_id"],
                        "contexts": {
                            "masked": {
                                "text": masked_context.replace('\\n', '\n'),
                                "summary": masked_summary
                            },
                            "full": {
                                "text": full_context.replace('\\n', '\n'),
                                "summary": full_summary
                            },
                            "predicted": {
                                "text": predicted_context.replace('\\n', '\n'),
                                "summary": predicted_summary
                            }
                        },
                        "evaluation": evaluation
                    }
                    return comparison
            
        except Exception as e:
            logging.error(f"Error processing dialogue {dialogue.get('dialogue_id', 0)}: {e}")
        
        return None

    def log_summary_metrics(self, results: Dict):
        """Log summary metrics from the results dictionary."""
        print("\n=== Summarization Comparison Results ===")
        print(f"Processed Dialogues: {results.get('num_dialogues', 0)}")
        
        # Log ROUGE scores for specified pairs only
        rouge = results.get("rouge_scores", {})
        if rouge:
            print("\nROUGE Scores:")
            for pair in ["masked_vs_full", "predicted_vs_full"]:
                if pair in rouge:
                    print(f"\n{pair.replace('_', ' ').title()}:")
                    print(f"ROUGE-1: {rouge[pair]['rouge1']['mean']:.3f} ± {rouge[pair]['rouge1']['stdev']:.3f}")
                    print(f"ROUGE-2: {rouge[pair]['rouge2']['mean']:.3f} ± {rouge[pair]['rouge2']['stdev']:.3f}")
                    print(f"ROUGE-L: {rouge[pair]['rougeL']['mean']:.3f} ± {rouge[pair]['rougeL']['stdev']:.3f}")
        
        # Log blind evaluation scores (requires predictions)
        evaluation = results.get("evaluation", {})
        if evaluation and evaluation.get("type") == "blind_evaluation":
            eval_scores = evaluation.get("scores", {})
            print("\nBlind Evaluation Scores (Randomized Order):")
            for summary_type in ["complete", "masked", "predicted"]:
                if summary_type in eval_scores and eval_scores[summary_type]:
                    display_name = f"{summary_type.title()} Summary"
                    scores = eval_scores[summary_type]
                    print(f"\n{display_name}:")
                    print(f"Overall Score: {scores.get('total_score', {}).get('mean', 0):.2f} ± {scores.get('total_score', {}).get('stdev', 0):.2f}")
                    print(f"Content Coverage: {scores.get('content_coverage', {}).get('mean', 0):.2f} ± {scores.get('content_coverage', {}).get('stdev', 0):.2f}")
                    print(f"Information Accuracy: {scores.get('information_accuracy', {}).get('mean', 0):.2f} ± {scores.get('information_accuracy', {}).get('stdev', 0):.2f}")
                    print(f"Dialogue Flow: {scores.get('dialogue_flow', {}).get('mean', 0):.2f} ± {scores.get('dialogue_flow', {}).get('stdev', 0):.2f}")
                    print(f"Purpose & Outcome: {scores.get('purpose_outcome', {}).get('mean', 0):.2f} ± {scores.get('purpose_outcome', {}).get('stdev', 0):.2f}")
                    print(f"Detail Balance: {scores.get('detail_balance', {}).get('mean', 0):.2f} ± {scores.get('detail_balance', {}).get('stdev', 0):.2f}")
        
        # Log separate precision/recall metrics (predicted vs complete) - only available for blind evaluation
        precision_recall = results.get("precision_recall", {})
        if precision_recall:
            print("\nPrecision & Recall (Predicted vs Complete):")
            print(f"Precision: {precision_recall['predicted'].get('precision', {}).get('mean', 0):.2f} ± {precision_recall['predicted'].get('precision', {}).get('stdev', 0):.2f}")
            print(f"Recall: {precision_recall['predicted'].get('recall', {}).get('mean', 0):.2f} ± {precision_recall['predicted'].get('recall', {}).get('stdev', 0):.2f}")
            
            print("\nPrecision & Recall (Masked vs Complete):")
            print(f"Precision: {precision_recall['masked'].get('precision', {}).get('mean', 0):.2f} ± {precision_recall['masked'].get('precision', {}).get('stdev', 0):.2f}")
            print(f"Recall: {precision_recall['masked'].get('recall', {}).get('mean', 0):.2f} ± {precision_recall['masked'].get('recall', {}).get('stdev', 0):.2f}")

        
        # Log ranking analysis
        ranking = evaluation.get("ranking_analysis", {})
        if ranking:
            print("\nRanking Analysis:")
            
            if "percentages" in ranking:
                print("\nBlind Rankings (Randomized Order):")
                for summary_type in ["complete", "masked", "predicted"]:
                    if summary_type in ranking["percentages"]:
                        percentages = ranking["percentages"][summary_type]
                        print(f"{summary_type.title()} Summary:")
                        print(f"  1st: {percentages.get('1st', 0):.1f}%")
                        print(f"  2nd: {percentages.get('2nd', 0):.1f}%")
                        print(f"  3rd: {percentages.get('3rd', 0):.1f}%")
            
            # Log length differences for specified pairs only
            length_diffs = results.get("summary_length_differences", {})
            if length_diffs:
                print("\nSummary Length Differences (in words):")
                for pair, stats in length_diffs.items():
                    print(f"{pair.replace('_', ' ').title()}: {stats.get('mean', 0):.1f} ± {stats.get('stdev', 0):.1f}")



    async def evaluate_predicted_vs_full_conversation(self, predicted_context: str, full_context: str) -> Dict[str, Any]:
        """Evaluate predicted conversation directly against full conversation using same metrics as predictions_with_claude.py"""
        try:
            evaluation_prompt = f"""
You are evaluating two dialogue conversations from a task-oriented conversation. Compare how similar they are:

For the predicted and actual conversations, provide detailed reasoning for each evaluation criterion FIRST, then assign a **1–5 score for each factor** below.  

## Evaluation Criteria
1. **Semantic Similarity** – Do the responses convey the same overall meaning?
2. **Intent Preservation** – Do they serve the same conversational function (e.g., offer help, confirm, ask)?
3. **XXXXXXX Masking Compliance** – Does the prediction correctly mask all specific details that weren't previously mentioned in the conversation with "XXXXXXX"? 
4. **Contextual Appropriateness** – Does the predicted response fit smoothly in the conversation flow?
5. **Summary Alignment** – If you summarized both responses, would the summaries essentially match?

## Details Extraction and Precision/Recall Calculation
- Extract **actual_details**: list of concrete, specific, verifiable details in the actual conversation.
- Extract **predicted_details**: list of concrete, specific, verifiable details in the predicted conversation. Treat "XXXXXXX" as correct when replacing an unknown specific.
- Compare the lists:
    - **TP** = number of predicted_details also in actual_details
    - **FP** = number of predicted_details not in actual_details
    - **FN** = number of actual_details not in predicted_details
- Calculate:
    - **precision_fraction** = TP / max(1, TP + FP)
    - **recall_fraction** = TP / max(1, TP + FN)

## XXXXXXX Masking Analysis
Count masking behavior by comparing against the conversation context:
- **actual_specific_info_count**: How many pieces of specific info are in the actual conversation that are NOT available in the previous conversation context
- **xxx_used_count**: How many times does the predicted conversation use "XXXXXXX" to mask unavailable details

## Conversations to Compare

### Actual Conversation (Reference)
{full_context}

### Predicted Conversation (To Evaluate)
{predicted_context}

## Scoring Scale
5 = Excellent, 4 = Good, 3 = Adequate, 2 = Poor, 1 = Very poor

## Instructions
- Provide reasoning for each evaluation criterion 
- Then assign a **1–5 score for each factor** above  
- Fill in the "Details Extraction and Precision/Recall Calculation" section e.g. "I want to book a train to Stevenage on Friday" = ["book a train", "to Stevenage", "on Friday"]
- Focus XXXXXXX masking evaluation on whether the prediction correctly masks specific details (names, numbers, addresses, times, etc.) that are NOT available in the previous conversation context
- Count specific information carefully, ensuring it's NOT in the previous context before counting as requiring masking
- Return valid JSON with reasoning, scores, counts, and explanation            

## Output Format
Provide BRIEF reasoning (max 30 words) for each metric, then assign 1-5 scores. Respond with valid JSON in this EXACT format.
IMPORTANT: Keep reasoning text simple and avoid quotes, apostrophes, or special characters:

{{
  "detail_extraction": {{
    "actual_details": ["detail1", "detail2"],
    "predicted_details": ["detail1", "detail3"],
    "tp": 1,
    "fp": 1,
    "fn": 1,
    "precision_fraction": 0.5,
    "recall_fraction": 0.5
  }},
  "reasoning_and_scores": {{
    "semantic_similarity_reasoning": "reasoning for semantic similarity",
    "semantic_similarity": 1-5,
    "intent_preservation_reasoning": "reasoning for intent preservation",
    "intent_preservation": 1-5,
    "xxx_masking_compliance_reasoning": "reasoning for XXXXXXX masking compliance",
    "xxx_masking_compliance": 1-5,
    "contextual_appropriateness_reasoning": "reasoning for context fit",
    "contextual_appropriateness": 1-5,
    "summary_alignment_reasoning": "reasoning for summary alignment",
    "summary_alignment": 1-5
  }},
  "analysis_counts": {{
    "actual_specific_info_count": 0,
    "xxx_used_count": 0
  }}
}}"""

            async with self.openai_sem:
                await self.openai_rate_limiter.acquire()
                logging.info(f"OpenAI API call starting at {time.time():.3f}")
                
                def _call():
                    return self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert in dialogue analysis and conversation evaluation. Respond with valid JSON only."},
                            {"role": "user", "content": evaluation_prompt}
                        ],
                        temperature=0.1
                    )
                
                response = await asyncio.to_thread(_call)
                self.openai_api_calls += 1  # Increment counter after successful call
                logging.info(f"OpenAI API call completed at {time.time():.3f} (Total: {self.openai_api_calls})")
            
            try:
                content = response.choices[0].message.content
                # Extract JSON from markdown code blocks if present
                if content.startswith('```json'):
                    content = content.split('```json')[1].split('```')[0].strip()
                elif content.startswith('```'):
                    content = content.split('```')[1].split('```')[0].strip()
                
                evaluation_result = json.loads(content)
                # Add the evaluation prompt to the result for debugging/analysis
                evaluation_result["evaluation_prompt"] = evaluation_prompt
                return evaluation_result
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse evaluation response: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Conversation evaluation failed: {e}")
            return None

    def calculate_prediction_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate aggregate metrics for prediction evaluation results"""
        if not results:
            return {}
        
        # Check if evaluations were skipped
        evaluations_skipped = all(r.get("evaluation", {}).get("evaluation_type") == "skipped" for r in results if r.get("evaluation"))
        
        if evaluations_skipped:
            return {
                "evaluation_type": "skipped", 
                "num_dialogues": len(results),
                "message": "Detailed metrics not available - evaluation was skipped"
            }
        
        # Extract evaluation scores using the same structure as predictions_with_claude.py
        evaluation_metrics = {}
        try:
            score_categories = ["semantic_similarity", "intent_preservation", "xxx_masking_compliance", "contextual_appropriateness", "summary_alignment"]
            
            for category in score_categories:
                scores = [r["evaluation"]["reasoning_and_scores"][category] for r in results 
                         if r.get("evaluation") and r["evaluation"].get("reasoning_and_scores", {}).get(category) is not None]
                evaluation_metrics[category] = self.calc_stats(scores)
                
        except KeyError as e:
            logging.warning(f"Missing evaluation scores: {e}")
            evaluation_metrics = {}
        
        # Extract precision/recall metrics from detail_extraction
        precision_recall_metrics = {}
        try:
            precision_scores = [r["evaluation"]["detail_extraction"]["precision_fraction"] for r in results 
                              if r.get("evaluation") and r["evaluation"].get("detail_extraction", {}).get("precision_fraction") is not None]
            recall_scores = [r["evaluation"]["detail_extraction"]["recall_fraction"] for r in results 
                           if r.get("evaluation") and r["evaluation"].get("detail_extraction", {}).get("recall_fraction") is not None]
            
            precision_recall_metrics = {
                "precision": self.calc_stats(precision_scores),
                "recall": self.calc_stats(recall_scores)
            }
        except (KeyError, TypeError) as e:
            logging.warning(f"Error in precision/recall extraction: {e}")
            precision_recall_metrics = {
                "precision": {"mean": 0, "stdev": 0},
                "recall": {"mean": 0, "stdev": 0}
            }
        
        # Extract XXXXXXX masking analysis counts
        masking_analysis = {}
        try:
            actual_specific_info_counts = [r["evaluation"]["analysis_counts"]["actual_specific_info_count"] for r in results 
                                         if r.get("evaluation") and r["evaluation"].get("analysis_counts", {}).get("actual_specific_info_count") is not None]
            xxx_used_counts = [r["evaluation"]["analysis_counts"]["xxx_used_count"] for r in results 
                             if r.get("evaluation") and r["evaluation"].get("analysis_counts", {}).get("xxx_used_count") is not None]
            
            masking_analysis = {
                "actual_specific_info": self.calc_stats(actual_specific_info_counts),
                "xxx_used": self.calc_stats(xxx_used_counts)
            }
        except (KeyError, TypeError) as e:
            logging.warning(f"Error in masking analysis extraction: {e}")
            masking_analysis = {
                "actual_specific_info": {"mean": 0, "stdev": 0},
                "xxx_used": {"mean": 0, "stdev": 0}
            }
        
        return {
            "evaluation_scores": evaluation_metrics,
            "precision_recall": precision_recall_metrics,
            "masking_analysis": masking_analysis,
            "num_dialogues": len(results)
        }



    async def process_prediction_evaluation_batch(self, dialogues: List[Dict], start_idx: int, predictions_data: Dict) -> List[Dict]:
        """Process a batch of dialogues for prediction evaluation concurrently - optimized like predictions_with_claude.py"""
        results, tasks = [], []

        for i, dialogue in enumerate(dialogues, start=start_idx):
            dialogue["dialogue_id"] = i
            
            async def one_job(dialogue=dialogue, predictions_data=predictions_data):
                try:
                    # Prepare contexts
                    full_context = self.prepare_full_scenario(dialogue)
                    predicted_context = None
                    
                    if predictions_data:
                        predicted_context = self.prepare_predicted_scenario(dialogue, predictions_data)
                        if not predicted_context:
                            logging.warning(f"Could not prepare predicted context for dialogue {dialogue.get('dialogue_id', 0)}")
                            return None
                    else:
                        logging.warning(f"No predictions data available for dialogue {dialogue.get('dialogue_id', 0)}")
                        return None
                    
                    # Direct evaluation call - optimized for speed
                    evaluation = await self.evaluate_predicted_vs_full_conversation(predicted_context, full_context)
                    
                    if evaluation:
                        comparison = {
                            "dialogue_id": dialogue["dialogue_id"],
                            "contexts": {
                                "full": {
                                    "text": full_context.replace('\\n', '\n')
                                },
                                "predicted": {
                                    "text": predicted_context.replace('\\n', '\n')
                                }
                            },
                            "evaluation": evaluation
                        }
                        return comparison
                    else:
                        logging.warning(f"No evaluation result for dialogue {dialogue['dialogue_id']}")
                        return None
                        
                except Exception as e:
                    logging.error(f"Error processing dialogue {dialogue['dialogue_id']}: {e}")
                    return None

            tasks.append(asyncio.create_task(one_job()))

        # Use the same pattern as predictions_with_claude.py for maximum speed
        for finished in await asyncio.gather(*tasks, return_exceptions=False):
            if finished:
                results.append(finished)
        
        return results

    def log_prediction_summary_metrics(self, results: Dict):
        """Log summary metrics for prediction evaluation results"""
        
        logger = logging.getLogger(__name__)
        logger.info("\n=== Prediction Evaluation Results ===")
        logger.info(f"Processed Dialogues: {results['aggregate_metrics'].get('num_dialogues', 0)}")
        
        # Log evaluation scores using the same metrics as predictions_with_claude.py
        evaluation_scores = results["aggregate_metrics"].get("evaluation_scores", {})
        if evaluation_scores:
            logger.info("\nEvaluation Scores:")
            score_names = {
                "semantic_similarity": "Semantic Similarity",
                "intent_preservation": "Intent Preservation", 
                "xxx_masking_compliance": "XXXXXXX Masking Compliance",
                "contextual_appropriateness": "Contextual Appropriateness",
                "summary_alignment": "Summary Alignment"
            }
            
            for score_key, display_name in score_names.items():
                if score_key in evaluation_scores:
                    stats = evaluation_scores[score_key]
                    logger.info(f"{display_name}: {stats.get('mean', 0):.2f} ± {stats.get('stdev', 0):.2f}")
        
        # Log precision/recall metrics
        precision_recall = results["aggregate_metrics"].get("precision_recall", {})
        if precision_recall:
            logger.info("\nPrecision & Recall:")
            logger.info(f"Precision: {precision_recall.get('precision', {}).get('mean', 0):.2f} ± {precision_recall.get('precision', {}).get('stdev', 0):.2f}")
            logger.info(f"Recall: {precision_recall.get('recall', {}).get('mean', 0):.2f} ± {precision_recall.get('recall', {}).get('stdev', 0):.2f}")
        
        # Log XXXXXXX masking analysis
        masking_analysis = results["aggregate_metrics"].get("masking_analysis", {})
        if masking_analysis:
            logger.info("\nXXXXXXX Masking Analysis:")
            logger.info(f"Actual Specific Info Count: {masking_analysis.get('actual_specific_info', {}).get('mean', 0):.2f} ± {masking_analysis.get('actual_specific_info', {}).get('stdev', 0):.2f}")
            logger.info(f"XXXXXXX Used Count: {masking_analysis.get('xxx_used', {}).get('mean', 0):.2f} ± {masking_analysis.get('xxx_used', {}).get('stdev', 0):.2f}")

    async def run_prediction_evaluation(self, dataset_name: str, num_samples: int = 10, predictions_file_path: str = None, test_split_file: str = None):
        """Run prediction evaluation comparing predicted conversations against full conversations"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        results = {
            "metadata": {
                "dataset": dataset_name if not test_split_file else "test_split",
                "num_samples": num_samples,
                "timestamp": datetime.now().isoformat(),
                "description": "Evaluation of predicted conversations against full conversations",
                "test_split_file": test_split_file if test_split_file else None,
                "concurrent_requests": MAX_CONCURRENT_REQUESTS,
                "batch_size": BATCH_SIZE,
                "api_call_counts": {
                    "claude_api_calls": 0,  # Will be updated at the end
                    "openai_api_calls": 0   # Will be updated at the end
                }
            },
            "evaluations": []
        }
        
        try:
            if test_split_file:
                logger.info(f"Loading dialogues from test split file: {test_split_file}")
                dialogues = self.load_test_split(test_split_file, max_dialogues=num_samples * 3)
            else:
                logger.info(f"Loading {dataset_name} dataset...")
                dialogues = self.load_dataset(dataset_name, max_dialogues=num_samples * 3)
            
            if not dialogues:
                source = test_split_file if test_split_file else f"{dataset_name} dataset"
                logger.error(f"No dialogues found in {source}")
                return results
            
            # Load predictions - required for this evaluation
            predictions_dataset_name = Path(test_split_file).stem if test_split_file else dataset_name
            predictions_data = self.load_predictions(predictions_dataset_name, num_samples, predictions_file_path)
            if not predictions_data:
                logger.error("Predictions are required for this evaluation but none were found")
                return results
            
            if num_samples > 0:
                dialogues = dialogues[:num_samples]
            logger.info(f"Processing {len(dialogues)} dialogues in batches of {BATCH_SIZE}")
            
            # Process dialogues in batches concurrently
            for batch_start in range(0, len(dialogues), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(dialogues))
                batch = dialogues[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(len(dialogues)-1)//BATCH_SIZE + 1}")
                batch_results = await self.process_prediction_evaluation_batch(batch, batch_start, predictions_data)
                
                # Add batch results to overall results
                results["evaluations"].extend([r for r in batch_results if r is not None])
            
            # Update API call counts in results metadata
            results["metadata"]["api_call_counts"]["claude_api_calls"] = self.claude_api_calls
            results["metadata"]["api_call_counts"]["openai_api_calls"] = self.openai_api_calls
            
            # Log API call summary
            total_api_calls = self.claude_api_calls + self.openai_api_calls
            logger.info(f"\n📊 API CALL SUMMARY:")
            logger.info(f"   Claude API calls: {self.claude_api_calls}")
            logger.info(f"   OpenAI API calls: {self.openai_api_calls}")
            logger.info(f"   Total API calls: {total_api_calls}")
            
            # Determine output file path
            if test_split_file:
                test_split_name = Path(test_split_file).stem
                output_file = Path(f"{test_split_name}_conversation_evaluation.json")
            else:
                output_file = Path(f"{dataset_name}_prediction_evaluation.json")
            
            # Save results immediately (without aggregate metrics)
            logger.info(f"\n💾 Saving conversation evaluation results to {output_file}...")
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Conversation evaluation results saved successfully!")
            
            # Calculate and add aggregate metrics separately
            logger.info(f"\n📊 Calculating aggregate metrics...")
            try:
                aggregate_metrics = self.calculate_prediction_aggregate_metrics(results["evaluations"])
                results["aggregate_metrics"] = aggregate_metrics
                
                # Update the file with aggregate metrics
                logger.info(f"💾 Updating file with aggregate metrics...")
                with open(output_file, "w", encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Aggregate metrics calculated and saved!")
                
            except Exception as metrics_error:
                logger.error(f"❌ Error calculating aggregate metrics: {metrics_error}")
                logger.error(f"📁 Conversation evaluation results are still saved in {output_file}")
                logger.info(f"💡 You can manually inspect the results file to debug the metrics calculation")
                # Don't re-raise the error - results are already saved
            
            # Log summary metrics
            self.log_prediction_summary_metrics(results)
        
            
            logger.info(f"\n🎉 Conversation evaluation completed! Final results in {output_file}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Prediction evaluation failed: {str(e)}", exc_info=True)
            raise

    def load_test_split(self, test_split_file: str, max_dialogues: int = None) -> List[Dict[str, Any]]:
        """Load dialogues from test split JSON file"""
        logging.info(f"Loading test split from {test_split_file}...")
        
        try:
            test_split_path = Path(test_split_file)
            if not test_split_path.exists():
                logging.error(f"Test split file not found: {test_split_file}")
                return []
            
            with open(test_split_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            if not test_data:
                logging.warning(f"No dialogues found in test split file: {test_split_file}")
                return []
            
            # Apply max_dialogues limit if specified
            if max_dialogues and len(test_data) > max_dialogues:
                test_data = test_data[:max_dialogues]
            
            # Convert test split format to expected format
            formatted_dialogues = []
            for i, dialogue in enumerate(test_data):
                # Map string dialogue_id to integer index for predictions matching
                dialogue_id_str = dialogue.get('dialogue_id', f'test_dialogue_{i+1}')
                dialogue_id_int = i  # Use 0-based index for predictions matching
                
                formatted_dialogue = {
                    'speakers': dialogue.get('speakers', []),
                    'utterances': dialogue.get('utterances', []),
                    'dataset': dialogue.get('dataset', 'test_split'),
                    'dialogue_id': dialogue_id_int,  # Integer for predictions matching
                    'original_dialogue_id': dialogue_id_str  # Keep original for reference
                }
                
                formatted_dialogues.append(formatted_dialogue)
            
            logging.info(f"Loaded {len(formatted_dialogues)} dialogues from test split")
            return formatted_dialogues
            
        except Exception as e:
            logging.error(f"Error loading test split {test_split_file}: {e}")
            return []

    async def get_blind_evaluation(self, blind_evaluation_prompt: str) -> Dict[str, Any]:
        """Get blind evaluation from OpenAI with proper rate limiting"""
        try:
            async with self.openai_sem:
                await self.openai_rate_limiter.acquire()
                logging.info(f"OpenAI API call starting at {time.time():.3f}")
                
                def _call():
                    return self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert in dialogue analysis and summarization evaluation. Respond with valid JSON only."},
                            {"role": "user", "content": blind_evaluation_prompt}
                        ],
                        temperature=0.1  # Small temperature for slight variation while maintaining consistency
                    )
                
                blind_response = await asyncio.to_thread(_call)
                self.openai_api_calls += 1  # Increment counter after successful call
                logging.info(f"OpenAI API call completed at {time.time():.3f} (Total: {self.openai_api_calls})")
            
            try:
                content = blind_response.choices[0].message.content
                # Extract JSON from markdown code blocks if present
                if content.startswith('```json'):
                    content = content.split('```json')[1].split('```')[0].strip()
                elif content.startswith('```'):
                    content = content.split('```')[1].split('```')[0].strip()
                
                blind_eval_raw = json.loads(content)
                return blind_eval_raw
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse blind evaluation response: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Blind evaluation API call failed: {e}")
            return None

    async def evaluate_masked_precision_recall(self, masked_summary: str, full_summary: str) -> Dict[str, Any]:
        """Separate precision/recall evaluator that compares masked vs full summary (labeled, not blind)"""
        try:
            precision_recall_prompt = f"""
You are evaluating the precision and recall of a masked summary compared to a complete summary.

## Task
Compare the masked summary against the complete summary and extract concrete, specific, verifiable details for precision/recall calculation.

## Details Extraction and Precision/Recall Calculation
- Extract **actual_details**: list of concrete, specific, verifiable details in the complete summary
- Extract **masked_details**: list of concrete, specific, verifiable details in the masked summary
- Compare the lists:
  - **TP** = number of masked_details also in actual_details
  - **FP** = number of masked_details not in actual_details  
  - **FN** = number of actual_details not in masked_details
- Calculate:
  - **precision_fraction** = TP / max(1, TP + FP)
  - **recall_fraction** = TP / max(1, TP + FN)

**IMPORTANT**: The masked summary may have missing information due to masked speaker turns. This is expected behavior and should not be penalized in the evaluation.

## Summaries to Compare
Complete Summary (Reference): {full_summary}

Masked Summary (To Evaluate): {masked_summary}

## Output Format
Respond with valid JSON in this EXACT format:

{{
"detail_extraction": {{
    "actual_details": [<list of strings: details from complete summary>],
    "masked_details": [<list of strings: details from masked summary>],
    "tp": <integer: true positives count>,
    "fp": <integer: false positives count>,
    "fn": <integer: false negatives count>,
    "precision_fraction": <float: TP / (TP + FP)>,
    "recall_fraction": <float: TP / (TP + FN)>
}},
"analysis": "<string: your analysis>"
}}"""

            async with self.openai_sem:
                await self.openai_rate_limiter.acquire()
                logging.info(f"OpenAI API call starting at {time.time():.3f}")
                
                def _call():
                    return self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert in text analysis and information extraction. Respond with valid JSON only."},
                            {"role": "user", "content": precision_recall_prompt}
                        ],
                        temperature=0.1
                    )
                
                pr_response = await asyncio.to_thread(_call)
                self.openai_api_calls += 1  # Increment counter after successful call
                logging.info(f"OpenAI API call completed at {time.time():.3f} (Total: {self.openai_api_calls})")
            
            try:
                content = pr_response.choices[0].message.content
                # Extract JSON from markdown code blocks if present
                if content.startswith('```json'):
                    content = content.split('```json')[1].split('```')[0].strip()
                elif content.startswith('```'):
                    content = content.split('```')[1].split('```')[0].strip()
                
                pr_eval = json.loads(content)
                # Add the evaluation prompt to the result
                pr_eval["evaluation_prompt"] = precision_recall_prompt
                return pr_eval
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse masked precision/recall response: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Masked precision/recall evaluation failed: {e}")
            return None

    async def run_masked_precision_recall_evaluation(self, input_json_file: str, output_json_file: str = None, batch_size: int = 5) -> Dict[str, Any]:
        """
        Load an existing summarization JSON file and create a new file with masked vs full precision/recall evaluation.
        
        Args:
            input_json_file: Path to existing JSON file with summaries
            output_json_file: Path to save new JSON file with masked precision/recall results
            batch_size: Number of dialogues to process concurrently
    
        Returns:
            Dictionary with results and metrics
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        try:
            # Load existing JSON file
            logger.info(f"Loading existing summarization results from {input_json_file}")
            with open(input_json_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            if "comparisons" not in existing_data:
                logger.error("Invalid JSON format - missing 'comparisons' key")
                return None
            
            comparisons = existing_data["comparisons"]
            logger.info(f"Found {len(comparisons)} dialogues to process")
            
            # Create new results structure focused on masked precision/recall
            results = {
                "metadata": {
                    "source_file": input_json_file,
                    "evaluation_type": "masked_precision_recall",
                    "num_samples": len(comparisons),
                    "timestamp": datetime.now().isoformat(),
                    "description": "Masked vs Full Precision/Recall Evaluation"
                },
                "masked_precision_recall_results": []
            }
            
            # Process dialogues in batches
            for batch_start in range(0, len(comparisons), batch_size):
                batch_end = min(batch_start + batch_size, len(comparisons))
                batch = comparisons[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(comparisons)-1)//batch_size + 1}")
                
                # Process batch concurrently
                batch_tasks = []
                for comparison in batch:
                    task = self.process_single_masked_precision_recall(comparison)
                    batch_tasks.append(asyncio.create_task(task))
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Add results to masked precision/recall results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error processing comparison: {result}")
                    elif result:  # Only add non-None results
                        results["masked_precision_recall_results"].append(result)
            
            # Calculate aggregate metrics for masked precision/recall
            logger.info("Calculating aggregate metrics for masked precision/recall...")
            aggregate_metrics = self.calculate_masked_precision_recall_metrics(results["masked_precision_recall_results"])
            results["aggregate_metrics"] = aggregate_metrics
            
            # Determine output file path
            if output_json_file is None:
                input_path = Path(input_json_file)
                output_json_file = input_path.parent / f"{input_path.stem}_masked_precision_recall.json"
            
            # Save new file
            logger.info(f"Saving masked precision/recall results to {output_json_file}")
            with open(output_json_file, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Log summary of masked precision/recall metrics
            self.log_masked_precision_recall_summary(aggregate_metrics)
            
            # Output token usage summary if in token counting mode
            if self.token_count_only and self.token_counter:
                self.token_counter.log_summary()
                
                # Save token usage to file
                token_summary = self.token_counter.get_summary()
                token_output_file = Path(output_json_file).parent / f"{Path(output_json_file).stem}_token_usage.json"
                prompts_output_file = Path(output_json_file).parent / f"{Path(output_json_file).stem}_prompts_verification.txt"
                
                with open(token_output_file, "w") as f:
                    json.dump(token_summary, f, indent=2)
                
                # Save all prompts for verification
                self.token_counter.save_prompts_to_file(str(prompts_output_file))
                
                logger.info(f"\n✅ Token usage saved to {token_output_file}")
                logger.info(f"✅ Prompts saved to {prompts_output_file}")
            
            logger.info(f"✅ Successfully created masked precision/recall results in {output_json_file}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to create masked precision/recall evaluation: {str(e)}", exc_info=True)
            raise

    async def process_single_masked_precision_recall(self, comparison: Dict) -> Dict:
        """Process a single comparison for masked precision/recall evaluation"""
        logger = logging.getLogger(__name__)
        try:
            # Extract summaries from the comparison
            contexts = comparison.get("contexts", {})
            masked_summary = contexts.get("masked", {}).get("summary")
            full_summary = contexts.get("full", {}).get("summary")
            
            if not masked_summary or not full_summary:
                logger.warning(f"Missing summaries for dialogue {comparison.get('dialogue_id', 'unknown')}")
                return None
            
            # Run masked vs full precision/recall evaluation
            masked_pr_eval = await self.evaluate_masked_precision_recall(masked_summary, full_summary)
            
            if masked_pr_eval:
                result = {
                    "dialogue_id": comparison.get("dialogue_id"),
                    "summaries": {
                        "masked": masked_summary,
                        "full": full_summary
                    },
                    "masked_precision_recall": masked_pr_eval
                }
                
                logger.info(f"Processed masked precision/recall for dialogue {comparison.get('dialogue_id', 'unknown')}")
                return result
            else:
                logger.warning(f"Failed to get masked precision/recall for dialogue {comparison.get('dialogue_id', 'unknown')}")
                return None
            
        except Exception as e:
            logger.error(f"Error processing comparison {comparison.get('dialogue_id', 'unknown')}: {e}")
            return None

    def calculate_masked_precision_recall_metrics(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate aggregate metrics for masked precision/recall results"""
        logger = logging.getLogger(__name__)  # Add this line
        if not results:
            return {
                "num_dialogues": 0,
                "message": "No results to process"
            }
        
        try:
            precision_scores = []
            recall_scores = []
            
            for r in results:
                if (r.get("masked_precision_recall") and
                    r["masked_precision_recall"].get("detail_extraction")):
                    
                    detail_extraction = r["masked_precision_recall"]["detail_extraction"]
                    if "precision_fraction" in detail_extraction:
                        precision_scores.append(detail_extraction["precision_fraction"])
                    if "recall_fraction" in detail_extraction:
                        recall_scores.append(detail_extraction["recall_fraction"])
            
            logger.info(f"Found {len(precision_scores)} precision scores and {len(recall_scores)} recall scores")
            
            metrics = {
                "num_dialogues": len(results),
                "masked_precision": self.calc_stats(precision_scores),
                "masked_recall": self.calc_stats(recall_scores)
            }
            
            return metrics
            
        except (KeyError, TypeError) as e:
            logger.warning(f"Error in masked precision/recall metrics calculation: {e}")
            return {
                "num_dialogues": len(results),
                "masked_precision": {"mean": 0, "stdev": 0},
                "masked_recall": {"mean": 0, "stdev": 0}
            }

    def log_masked_precision_recall_summary(self, aggregate_metrics: Dict):
        """Log summary of masked precision/recall metrics"""
        logger = logging.getLogger(__name__)
        
        logger.info("\n=== Masked vs Full Precision & Recall Results ===")
        logger.info(f"Processed Dialogues: {aggregate_metrics.get('num_dialogues', 0)}")
        
        if "masked_precision" in aggregate_metrics and "masked_recall" in aggregate_metrics:
            precision_stats = aggregate_metrics["masked_precision"]
            recall_stats = aggregate_metrics["masked_recall"]
            
            logger.info(f"Masked Precision: {precision_stats.get('mean', 0):.3f} ± {precision_stats.get('stdev', 0):.3f}")
            logger.info(f"Masked Recall: {recall_stats.get('mean', 0):.3f} ± {recall_stats.get('stdev', 0):.3f}")
            
            # Calculate F1 score
            precision_mean = precision_stats.get('mean', 0)
            recall_mean = recall_stats.get('mean', 0)
            if precision_mean > 0 and recall_mean > 0:
                f1_score = 2 * (precision_mean * recall_mean) / (precision_mean + recall_mean)
                logger.info(f"F1 Score: {f1_score:.3f}")
        else:
            logger.warning("No masked precision/recall metrics found in aggregate results")

    



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_file", type=str, required=False,
                        help="Path to the summaries JSONL file")
    parser.add_argument("--output_file", type=str, required=False,
                        help="Path to the output evaluation JSONL file. if not provided, will save to <summaries_file>_evaluated.jsonl")
    parser.add_argument("--evaluation_file", type=str, required=False,
                        help="Path to an existing evaluation JSONL file to summarize")
    parser.add_argument("--summarize", action='store_true',
                        help="If set, will only summarize existing evaluation results instead of running new evaluation")

    args = parser.parse_args()

    if not args.summaries_file and not args.evaluation_file:
        print("Please provide either --summaries_file to run evaluation or --evaluation_file to summarize existing results.")
        sys.exit(1)
    if args.summarize and not args.evaluation_file:
        print("Please provide --evaluation_file to summarize existing results.")
        sys.exit(1)
    if args.summarize:
        print(f"[INFO] Summarizing existing evaluation results from {args.evaluation_file}...")
        evaluations = args.evaluation_file
        if not os.path.exists(evaluations):
            print(f"[ERROR] Evaluation file {evaluations} does not exist.")
            sys.exit(1)
        with open(evaluations, 'r', encoding='utf-8') as f:
            eval_data = [json.loads(line) for line in f if line.strip()]
        evaluator = PredictionEvaluator(openai_api_key=OPENAI_API_KEY)
        aggregate_metrics = evaluator.calculate_aggregate_metrics(eval_data)
        evaluator.log_summary_metrics(aggregate_metrics)
        sys.exit(0)

    summaries_file = args.summaries_file
    output_file = args.output_file if args.output_file else summaries_file.replace(".jsonl", "_evaluated.jsonl")
    output_path = Path(output_file)
    if output_path.exists():
        print(f"Output file {output_path} already exists. Please remove it or specify a different file.")
        sys.exit(1)

    if not os.path.exists(summaries_file):
        print(f"[ERROR] summaries file {summaries_file} does not exist.")
        sys.exit(1)

    

    # Load summaries
    with open(summaries_file, 'r', encoding='utf-8') as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    predictions = predictions[:10] # For testing, limit to first 10 entries

    evaluator = PredictionEvaluator(openai_api_key=OPENAI_API_KEY)

    # Run evaluation
    asyncio.run(evaluator.run_comparison(predictions, output_file))
    print(f"[INFO] Evaluation results saved to {output_file}")

        


