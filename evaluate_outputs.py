import asyncio
import json
import logging
from pathlib import Path
from anthropic import Anthropic
import openai
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
from tqdm import tqdm

# Add data root and conversational data loader
DATA_ROOT = os.getenv("DATA_ROOT", "/gscratch/scrubbed/ebertv/onesided/data")
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
    def __init__(self, calls: int, window: int, delay: int):
        self.calls = calls  # Number of calls allowed
        self.window = window  # Time window in seconds
        self.delay = delay  # Delay when rate limit is hit
        self.timestamps = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            # Remove timestamps outside the window (optimized)
            cutoff = now - self.window
            self.timestamps = [ts for ts in self.timestamps if ts > cutoff]
            
            if len(self.timestamps) >= self.calls:
                # Calculate optimal delay based on oldest timestamp
                wait_time = min(self.delay, self.timestamps[0] + self.window - now + 0.1)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                # Remove expired timestamps after waiting
                now = time.time()
                cutoff = now - self.window
                self.timestamps = [ts for ts in self.timestamps if ts > cutoff]
            
            self.timestamps.append(now)

class PredictionEvaluator:
    def __init__(self, openai_api_key: str):
        """Initialize the evaluator with API clients and optimized rate limiting"""
        # Use synchronous OpenAI client - no async issues like Claude
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Separate semaphores for better concurrency control
        self.openai_sem = Semaphore(MAX_CONCURRENT_REQUESTS // 2)  # Split capacity
        
        # Rate limiters OpenAI
        self.openai_rate_limiter = RateLimiter(OPENAI_RATE_LIMIT_CALLS, OPENAI_RATE_LIMIT_WINDOW, OPENAI_RATE_LIMIT_DELAY)
        
        # Create results directory
        Path("results").mkdir(exist_ok=True)

    def format_full_dialogue_context(self, dialogue: Dict[str, Any], turn_n: int, include_future_context: bool = False) -> str:
        """
        Works very specifically for the 3-turn format used in this project. Needs to be more general later
        """
        output = ""
        
        # # Format all turns up to turn_n (the turn we're predicting)
        # for i, (speaker, utterance) in enumerate(zip(dialogue['speakers'][:turn_n], dialogue['utterances'][:turn_n])):
        #     output += f"Turn {i+1} [{speaker}]: {utterance}\n"
        dialogue = dialogue.split("\n")
        output += dialogue[0] + "\n"
        
        output += "Turn 2 [PREDICTING: Speaker_2]: <-- This turn is being predicted\n"
        
        # Add future context if requested (for scenarios 1.2, 2.2)
        if include_future_context:
            output += "\n--- Future Context (available to model during prediction) ---\n"
            output += dialogue[2] + "\n"
        
        return output
    

    async def evaluate_prediction(self, predicted: str, actual: str, context: str = "") -> Dict[str, Any]:
        """Evaluate prediction using ROUGE metrics and GPT-4 for semantic analysis"""
        try:
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(predicted, actual)
            
            metrics = {
                "rouge1": rouge_scores['rouge1'].fmeasure,
                "rouge2": rouge_scores['rouge2'].fmeasure,
                "rougeL": rouge_scores['rougeL'].fmeasure
            }
            
            # Include context in evaluation if provided
            context_section = ""
            if context.strip():
                context_section = f"""
            ## Full Conversation Context (Unmasked)
            The following shows the complete conversation up to the point being predicted, with all actual utterances visible:
            {context}
            
            """
            
            evaluation_prompt = f"""
            You are evaluating two dialogue responses from a task-oriented conversation. Compare how similar they are:
            
            For the predicted and actual responses, provide detailed reasoning for each evaluation criterion FIRST, then assign a **1–5 score for each factor** below.  
            
            ## Evaluation Criteria
            1. **Semantic Similarity** – Do the responses convey the same overall meaning?
            2. **Intent Preservation** – Do they serve the same conversational function (e.g., offer help, confirm, ask)?
            3. **Specific Information Hallucination** – How much did it make up instead of using XXXXXXX? Focus ONLY on concrete details.
            4. **Contextual Appropriateness** – Does the predicted response fit smoothly in the conversation flow?
            5. **Summary Alignment** – If you summarized both responses, would the summaries essentially match?

            ## Details Extraction and Precision/Recall Calculation
            - Extract **actual_details**: list of concrete, specific, verifiable details in the actual response.
            - Extract **predicted_details**: list of concrete, specific, verifiable details in the predicted response. Treat "XXXXXXX" as correct when replacing an unknown specific.
            - Compare the lists:
                - **TP** = number of predicted_details also in actual_details
                - **FP** = number of predicted_details not in actual_details
                - **FN** = number of actual_details not in predicted_details
            - Calculate:
                - **precision_fraction** = TP / max(1, TP + FP)
                - **recall_fraction** = TP / max(1, TP + FN)

            ## XXXXXXX Analysis
            Count specific information by comparing against the conversation context:
            - **actual_specific_info_count**: How many pieces of specific info are in the actual response that are NOT available in the context
            - **xxx_used_count**: How many times does the predicted response use "XXXXXXX"

            ## Here is the full conversation context:
            {context_section}

            ## Responses to Evaluate
            Predicted: {predicted}
            Actual:    {actual}

            ## Scoring Scale
            5 = Excellent, 4 = Good, 3 = Adequate, 2 = Poor, 1 = Very poor
            
            ## Instructions
            - Provide reasoning for each evaluation criterion 
            - Then assign a **1–5 score for each factor** above  
            - Fill in the "Details Extraction and Precision/Recall Calculation" section e.g. "I want to book a train to Stevenage on Friday" = ["book a train", "to Stevenage", "on Friday"]
            - Focus hallucination evaluation ONLY on concrete specific information
            - Count specific information carefully, ensuring it's NOT in the context before counting
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
                "semantic_similarity_reasoning": "brief reasoning for semantic similarity",
                "semantic_similarity": 1-5,
                "intent_preservation_reasoning": "brief reasoning for intent preservation",
                "intent_preservation": 1-5,
                "specific_hallucination_reasoning": "brief reasoning for hallucination",
                "specific_hallucination": 1-5,
                "contextual_appropriateness_reasoning": "brief reasoning for context fit",
                "contextual_appropriateness": 1-5,
                "summary_alignment_reasoning": "brief reasoning for summary alignment",
                "summary_alignment": 1-5
              }},
              "analysis_counts": {{
                "actual_specific_info_count": 0,
                "xxx_used_count": 0
              }}
            }}"""

            # Use simple OpenAI call with delay
            try:
                # Simple delay before API call
                await asyncio.sleep(0.3)
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert in dialogue analysis. Respond with valid JSON only."},
                        {"role": "user", "content": evaluation_prompt}
                    ]
                )
                    
            except Exception as e:
                logging.error(f"Error calling OpenAI API: {e}")
                return None
            
            try:
                if response is None:
                    return None
                
                # Clean up the response content for better JSON parsing
                content = response.choices[0].message.content.strip()
                
                # Handle cases where the response might have extra text around JSON
                if content.startswith('```json'):
                    content = content.split('```json')[1].split('```')[0].strip()
                elif content.startswith('```'):
                    content = content.split('```')[1].split('```')[0].strip()
                
                semantic_eval = json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse GPT-4 response: {e}")
                if response:
                    raw_content = response.choices[0].message.content
                    logging.error(f"Raw response: {raw_content[:500]}{'...' if len(raw_content) > 500 else ''}")
                    
                    # Try to extract JSON from malformed response
                    try:
                        # Look for JSON-like patterns and attempt basic cleanup
                        content = raw_content.strip()
                        if '{' in content and '}' in content:
                            start = content.find('{')
                            end = content.rfind('}') + 1
                            json_part = content[start:end]
                            
                            # Basic cleanup of common JSON issues
                            json_part = json_part.replace('\n', ' ').replace('\r', ' ')
                            
                            # Fix incomplete strings and truncated JSON
                            import re
                            
                            # If JSON ends with incomplete reasoning field, complete it
                            if json_part.endswith('...'):
                                json_part = json_part[:-3] + '"'
                            
                            # Fix incomplete strings by adding closing quotes if needed
                            if json_part.count('"') % 2 == 1:  # Odd number of quotes means incomplete
                                json_part = json_part.rstrip() + '"'
                            
                            # If JSON is truncated mid-field, try to complete it minimally
                            if not json_part.rstrip().endswith('}'):
                                # Find last complete field and add minimal completion
                                if '"detail_extraction"' in json_part and not '"reasoning_and_scores"' in json_part:
                                    # Truncated in detail extraction section, add minimal completion
                                    json_part = json_part.rstrip()
                                    if json_part.endswith(','):
                                        json_part = json_part[:-1]  # Remove trailing comma
                                    json_part += '},"reasoning_and_scores":{"semantic_similarity_reasoning":"truncated","semantic_similarity":3,"intent_preservation_reasoning":"truncated","intent_preservation":3,"specific_hallucination_reasoning":"truncated","specific_hallucination":3,"contextual_appropriateness_reasoning":"truncated","contextual_appropriateness":3,"summary_alignment_reasoning":"truncated","summary_alignment":3},"analysis_counts":{"actual_specific_info_count":0,"xxx_used_count":0}}'
                                elif '"reasoning"' in json_part:
                                    # Truncated in reasoning section, add minimal completion
                                    json_part = json_part.rstrip()
                                    if json_part.endswith(','):
                                        json_part = json_part[:-1]  # Remove trailing comma
                                    if not json_part.rstrip().endswith('}}'):
                                        json_part += '}}'
                                elif not json_part.rstrip().endswith('}}'):
                                    json_part = json_part.rstrip() + '}}'
                            # Remove problematic quotes in explanation and reasoning fields
                            problematic_fields = ['"overall_explanation"', '"semantic_similarity_reasoning"', 
                                                '"intent_preservation_reasoning"', '"specific_hallucination_reasoning"', 
                                                '"contextual_appropriateness_reasoning"', '"summary_alignment_reasoning"']
                            
                            for field in problematic_fields:
                                if field in json_part:
                                    # Split at field and rebuild
                                    parts = json_part.split(f'{field}:')
                                    if len(parts) == 2:
                                        prefix = parts[0]
                                        suffix = parts[1].strip()
                                        if suffix.startswith('"'):
                                            # Find the end of field
                                            quote_count = 0
                                            end_pos = 1
                                            for i, char in enumerate(suffix[1:], 1):
                                                if char == '"' and suffix[i-1] != '\\':
                                                    quote_count += 1
                                                    if quote_count == 1:  # Found closing quote
                                                        end_pos = i + 1
                                                        break
                                            
                                            field_content = suffix[1:end_pos-1]
                                            rest = suffix[end_pos:]
                                            
                                            # Clean field text
                                            field_content = field_content.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
                                            json_part = f'{prefix}{field}: "{field_content}"{rest}'
                            
                            semantic_eval = json.loads(json_part)
                            logging.warning("Recovered from malformed JSON response")
                        else:
                            return None
                    except:
                        logging.error("Could not recover from malformed JSON")
                        return None
                else:
                    return None
                
            # Map to maintain backward compatibility with existing aggregation code
            mapped_eval = {
                "semantic_similarity": semantic_eval["reasoning_and_scores"]["semantic_similarity"],
                "intent_preservation": semantic_eval["reasoning_and_scores"]["intent_preservation"],
                "precision": semantic_eval["detail_extraction"]["precision_fraction"],  # Use calculated precision
                "recall": semantic_eval["detail_extraction"]["recall_fraction"],  # Use calculated recall
                "anti_hallucination_score": semantic_eval["reasoning_and_scores"]["specific_hallucination"],  # Map to expected field name
                "contextual_appropriateness": semantic_eval["reasoning_and_scores"]["contextual_appropriateness"],
                "summary_alignment": semantic_eval["reasoning_and_scores"]["summary_alignment"],
                "actual_specific_info_count": semantic_eval["analysis_counts"]["actual_specific_info_count"],
                "xxx_used_count": semantic_eval["analysis_counts"]["xxx_used_count"],
                "reasoning_details": semantic_eval["reasoning_and_scores"],  # Keep full reasoning for analysis
                "detail_extraction": semantic_eval["detail_extraction"]  # Keep detail extraction for analysis
            }
            
            return {
                "rouge_scores": metrics,
                "semantic_similarity": mapped_eval,
                "evaluation_prompt": evaluation_prompt
            }
                
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return None

    def calculate_scenario_metrics(self, predictions: List[Dict]) -> Dict[str, Dict]:
        """Calculate average and std dev of metrics for each scenario"""
        # Group predictions by scenario
        scenarios = {}
        for pred in predictions:
            scenario = pred["scenario"]
            if scenario not in scenarios:
                scenarios[scenario] = []
            
            # Handle scenario 3 differently since it has all_turns structure
            if scenario == "scenario_3":
                # Flatten scenario 3 all_turns into individual predictions for metrics calculation
                for i in range(len(pred["predictions"])):
                    individual_pred = {
                        "full_dialogue": pred["full_dialogue"],
                        "prediction": pred["predictions"][i],
                        "actual_response": pred["actual_response"][i],
                        "evaluation": pred["evaluation"][i] if i < len(pred["evaluation"]) else None
                    }
                    scenarios[scenario].append(individual_pred)
            else:
                scenarios[scenario].append(pred)
        
        # Calculate metrics for each scenario
        metrics = {}
        for scenario, preds in scenarios.items():
            # Get unique dialogue IDs and total turns
            unique_dialogues = len(set(p["full_dialogue"] for p in preds))
            total_turns = len(preds)
            rouge1_scores = [p["evaluation"]["rouge_scores"]["rouge1"] for p in preds if p.get("evaluation")]
            rouge2_scores = [p["evaluation"]["rouge_scores"]["rouge2"] for p in preds if p.get("evaluation")]
            rougeL_scores = [p["evaluation"]["rouge_scores"]["rougeL"] for p in preds if p.get("evaluation")]
            
            # Extract individual semantic evaluation categories (updated for new structure)
            semantic_similarity_scores = []
            intent_preservation_scores = []
            precision_scores = []
            recall_scores = []
            contextual_appropriateness_scores = []
            anti_hallucination_scores = []
            summary_alignment_scores = []
            actual_specific_info_counts = []
            xxx_used_counts = []
            
            for p in preds:
                if p.get("evaluation") and "semantic_similarity" in p["evaluation"]:
                    sem_eval = p["evaluation"]["semantic_similarity"]
                    if isinstance(sem_eval, dict):
                        semantic_similarity_scores.append(sem_eval.get("semantic_similarity", 0))
                        intent_preservation_scores.append(sem_eval.get("intent_preservation", 0))
                        precision_scores.append(sem_eval.get("precision", 0))
                        recall_scores.append(sem_eval.get("recall", 0))
                        contextual_appropriateness_scores.append(sem_eval.get("contextual_appropriateness", 0))
                        anti_hallucination_scores.append(sem_eval.get("anti_hallucination_score", 0))
                        summary_alignment_scores.append(sem_eval.get("summary_alignment", 0))
                        
                        # Extract XXXXXXX analysis metrics
                        actual_specific_info_counts.append(sem_eval.get("actual_specific_info_count", 0))
                        xxx_used_counts.append(sem_eval.get("xxx_used_count", 0))
            
            # Word difference calculation
            word_diffs = []
            for p in preds:
                if p.get("evaluation") and "prediction" in p and "actual_response" in p:
                    pred_words = len(p["prediction"].split())
                    actual_words = len(p["actual_response"].split())
                    word_diffs.append(abs(pred_words - actual_words))
            
            metrics[scenario] = {
                "rouge1": {
                    "mean": mean(rouge1_scores) if rouge1_scores else 0,
                    "stdev": stdev(rouge1_scores) if len(rouge1_scores) > 1 else 0
                },
                "rouge2": {
                    "mean": mean(rouge2_scores) if rouge2_scores else 0,
                    "stdev": stdev(rouge2_scores) if len(rouge2_scores) > 1 else 0
                },
                "rougeL": {
                    "mean": mean(rougeL_scores) if rougeL_scores else 0,
                    "stdev": stdev(rougeL_scores) if len(rougeL_scores) > 1 else 0
                },
                "semantic_similarity": {
                    "mean": mean(semantic_similarity_scores) if semantic_similarity_scores else 0,
                    "stdev": stdev(semantic_similarity_scores) if len(semantic_similarity_scores) > 1 else 0
                },
                "intent_preservation": {
                    "mean": mean(intent_preservation_scores) if intent_preservation_scores else 0,
                    "stdev": stdev(intent_preservation_scores) if len(intent_preservation_scores) > 1 else 0
                },
                "precision": {
                    "mean": mean(precision_scores) if precision_scores else 0,
                    "stdev": stdev(precision_scores) if len(precision_scores) > 1 else 0
                },
                "recall": {
                    "mean": mean(recall_scores) if recall_scores else 0,
                    "stdev": stdev(recall_scores) if len(recall_scores) > 1 else 0
                },
                "contextual_appropriateness": {
                    "mean": mean(contextual_appropriateness_scores) if contextual_appropriateness_scores else 0,
                    "stdev": stdev(contextual_appropriateness_scores) if len(contextual_appropriateness_scores) > 1 else 0
                },
                "summary_alignment": {
                    "mean": mean(summary_alignment_scores) if summary_alignment_scores else 0,
                    "stdev": stdev(summary_alignment_scores) if len(summary_alignment_scores) > 1 else 0
                },
                "anti_hallucination_score": {
                    "mean": mean(anti_hallucination_scores) if anti_hallucination_scores else 0,
                    "stdev": stdev(anti_hallucination_scores) if len(anti_hallucination_scores) > 1 else 0
                },
                "actual_specific_info_count": {
                    "mean": mean(actual_specific_info_counts) if actual_specific_info_counts else 0,
                    "stdev": stdev(actual_specific_info_counts) if len(actual_specific_info_counts) > 1 else 0
                },
                "xxx_used_count": {
                    "mean": mean(xxx_used_counts) if xxx_used_counts else 0,
                    "stdev": stdev(xxx_used_counts) if len(xxx_used_counts) > 1 else 0
                },
                "word_difference": {
                    "mean": mean(word_diffs) if word_diffs else 0,
                    "stdev": stdev(word_diffs) if len(word_diffs) > 1 else 0
                },
                "num_dialogues": unique_dialogues,
                "num_turns": total_turns
            }
        
        return metrics

    async def run_evaluation(self, predictions: List[Dict], output_file: str) -> Dict:
        """Run evaluation for all or selected scenario"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s.%(msecs)03d %(message)s',
            datefmt='%H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"🚀 OPTIMIZED PERFORMANCE SETTINGS:")
            logger.info(f"   Total concurrent requests: {MAX_CONCURRENT_REQUESTS}")
            logger.info(f"   OpenAI concurrent: {MAX_CONCURRENT_REQUESTS // 2}, rate limit: {OPENAI_RATE_LIMIT_CALLS}/{OPENAI_RATE_LIMIT_WINDOW}s")
            logger.info(f"   Batch size: {BATCH_SIZE}")
            
            # Process all dialogues sequentially for stability
            logger.info(f"Processing {len(predictions)} predictions sequentially...")
            
            for idx, pred in tqdm(enumerate(predictions)):
                scenario = pred["scenario"]
                logger.info(f"[{idx+1}/{len(predictions)}]")
                
                evaluation_result = pred

                context = pred.get("full_dialogue", "")
                
                if scenario == "scenario_3":
                    # Evaluate each turn in scenario 3
                    if len(pred["predictions"]) != len(pred["actual_response"]):
                        pred["predictions"] = pred["predictions"][:len(pred["actual_response"])]
                    temp = []
                    for i in range(len(pred["predictions"])):
                        turn_eval = await self.evaluate_prediction(
                            predicted=pred["predictions"][i],
                            actual=pred["actual_response"][i],
                            context=context
                        )
                        temp.append(turn_eval)
                    evaluation_result["evaluation"] = temp
                else:
                    # Standard evaluation for other scenarios
                    turn_n = 2 #Specific to this project format
                    context = self.format_full_dialogue_context(context, turn_n, 
                                                                   include_future_context=(scenario in ["scenario_1_2", "scenario_2_2"]))
                    evaluation_result["evaluation"] = await self.evaluate_prediction(
                        predicted=pred["predictions"][0],
                        actual=pred["actual_response"],
                        context=context
                    )
                with open(output_file, 'a', encoding='utf-8') as f_out:
                    f_out.write(json.dumps(evaluation_result, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {}


    def log_summary_metrics(self, metrics: Dict):
        """Log summary metrics in a clean format"""
        logger = logging.getLogger(__name__)
        logging.basicConfig(level = logging.INFO)
        logger.info("\n=== Scenario Performance Summary ===")
        
        for scenario, scenario_metrics in metrics.items():
            logger.info(f"\n{scenario}:")
            logger.info(f"Dialogues: {scenario_metrics['num_dialogues']}")
            logger.info(f"Total Turns: {scenario_metrics['num_turns']}")
            logger.info(f"ROUGE-L: {scenario_metrics['rougeL']['mean']:.2f} ± {scenario_metrics['rougeL']['stdev']:.2f}")
            logger.info(f"Word Difference: {scenario_metrics['word_difference']['mean']:.1f} ± {scenario_metrics['word_difference']['stdev']:.1f}")
            logger.info("--- Evaluation Categories ---")
            logger.info(f"Semantic Similarity: {scenario_metrics['semantic_similarity']['mean']:.2f} ± {scenario_metrics['semantic_similarity']['stdev']:.2f}")
            logger.info(f"Intent Preservation: {scenario_metrics['intent_preservation']['mean']:.2f} ± {scenario_metrics['intent_preservation']['stdev']:.2f}")
            logger.info(f"Precision: {scenario_metrics['precision']['mean']:.2f} ± {scenario_metrics['precision']['stdev']:.2f}")
            logger.info(f"Recall: {scenario_metrics['recall']['mean']:.2f} ± {scenario_metrics['recall']['stdev']:.2f}")
            logger.info(f"Contextual Appropriateness: {scenario_metrics['contextual_appropriateness']['mean']:.2f} ± {scenario_metrics['contextual_appropriateness']['stdev']:.2f}")
            logger.info(f"Summary Alignment: {scenario_metrics['summary_alignment']['mean']:.2f} ± {scenario_metrics['summary_alignment']['stdev']:.2f}")
            logger.info(f"Anti-Hallucination Score: {scenario_metrics['anti_hallucination_score']['mean']:.2f} ± {scenario_metrics['anti_hallucination_score']['stdev']:.2f}")
            logger.info("--- XXXXXXX Analysis ---")
            logger.info(f"Actual Specific Info (not in context): {scenario_metrics['actual_specific_info_count']['mean']:.1f} ± {scenario_metrics['actual_specific_info_count']['stdev']:.1f}")
            logger.info(f"XXXXXXX Used: {scenario_metrics['xxx_used_count']['mean']:.1f} ± {scenario_metrics['xxx_used_count']['stdev']:.1f}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str, required=False,
                        help="Path to the predictions JSONL file")
    parser.add_argument("--output_file", type=str, required=False,
                        help="Path to the output evaluation JSONL file. if not provided, will save to <predictions_file>_evaluated.jsonl")
    parser.add_argument("--evaluation_file", type=str, required=False,
                        help="Path to an existing evaluation JSONL file to summarize")
    parser.add_argument("--summarize", action='store_true',
                        help="If set, will only summarize existing evaluation results instead of running new evaluation")

    args = parser.parse_args()

    if not args.predictions_file and not args.evaluation_file:
        print("Please provide either --predictions_file to run evaluation or --evaluation_file to summarize existing results.")
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
        scenario_metrics = evaluator.calculate_scenario_metrics(eval_data)
        evaluator.log_summary_metrics(scenario_metrics)
        sys.exit(0)

    predictions_file = args.predictions_file
    output_file = args.output_file if args.output_file else predictions_file.replace(".jsonl", "_evaluated.jsonl")
    output_path = Path(output_file)
    if output_path.exists():
        print(f"Output file {output_path} already exists. Please remove it or specify a different file.")
        sys.exit(1)

    if not os.path.exists(predictions_file):
        print(f"[ERROR] Predictions file {predictions_file} does not exist.")
        sys.exit(1)

    

    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    evaluator = PredictionEvaluator(openai_api_key=OPENAI_API_KEY)
    
    for pred in predictions:
        if "actual_response" not in pred or not pred["actual_response"]:
            print(f"[WARNING] Missing actual_response for dialogue {pred.get('dialogue_id', 'unknown')}, scenario {pred.get('scenario', 'unknown')}")
            pred["actual_response"] = "XXXXXXX"
        if "full_dialogue" not in pred or not pred["full_dialogue"]:
            print(f"[WARNING] Missing full_dialogue for dialogue {pred.get('dialogue_id', 'unknown')}, scenario {pred.get('scenario', 'unknown')}")
            pred["full_dialogue"] = ""
        if "predictions" not in pred or not pred["predictions"]:
            print(f"[WARNING] Missing predictions for dialogue {pred.get('dialogue_id', 'unknown')}, scenario {pred.get('scenario', 'unknown')}")
            pred["predictions"] = ["XXXXXXX"]

    # Run evaluation
    asyncio.run(evaluator.run_evaluation(predictions, output_file))
    print(f"[INFO] Evaluation results saved to {output_file}")

        


