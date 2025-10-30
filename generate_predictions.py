#take in a jsonl file with prompts
#take in the model as an argument
#generate predictions for each prompt
#save the predictions to a jsonl file

import asyncio
import json
import logging
from pathlib import Path
from anthropic import Anthropic
import openai
from datetime import datetime
from typing import Dict, List, Any, Optional
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
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
try:
    from transformers import BitsAndBytesConfig
    quant_available = True
except ImportError:
    quant_available = False


# Constants - Optimized for maximum speed (can be overridden by environment variables)
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '30'))  # Reduced to avoid event loop issues
CLAUDE_RATE_LIMIT_CALLS = int(os.getenv('CLAUDE_RATE_LIMIT_CALLS', '50'))  # Claude can handle high throughput
CLAUDE_RATE_LIMIT_WINDOW = int(os.getenv('CLAUDE_RATE_LIMIT_WINDOW', '60'))
CLAUDE_RATE_LIMIT_DELAY = float(os.getenv('CLAUDE_RATE_LIMIT_DELAY', '2'))  # Minimal delay for Claude
OPENAI_RATE_LIMIT_CALLS = int(os.getenv('OPENAI_RATE_LIMIT_CALLS', '50'))  # Reduced to avoid event loop issues
OPENAI_RATE_LIMIT_WINDOW = int(os.getenv('OPENAI_RATE_LIMIT_WINDOW', '60'))
OPENAI_RATE_LIMIT_DELAY = float(os.getenv('OPENAI_RATE_LIMIT_DELAY', '2'))  # Increased delay for stability
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '15'))  # Smaller batches for stability

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv('.env')

# Get API keys
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BASE_DIR = "."
LLAMA_CANDIDATES = [
    "models--meta-llama--Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "models--meta-llama--Meta-Llama-3-70B-Instruct",
    "Meta-Llama-3-70B-Instruct-2shards",
]

def _find_snapshot(path: str) -> Optional[str]:
    """Return first directory beneath *path* that contains a config.json."""
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    snap_root = os.path.join(path, "snapshots")
    if os.path.isdir(snap_root):
        for sub in os.listdir(snap_root):
            cand = os.path.join(snap_root, sub)
            if os.path.isfile(os.path.join(cand, "config.json")):
                return cand
    for root, _, files in os.walk(path):
        if "config.json" in files:
            return root
    return None

# MODEL_ID = next(
#     (_find_snapshot(os.path.join(BASE_DIR, p)) for p in LLAMA_CANDIDATES if _find_snapshot(os.path.join(BASE_DIR, p))), 
#     None
# )
# if MODEL_ID is None:
#     raise FileNotFoundError(f"No Llama-3 snapshot found under {BASE_DIR}")
MODEL_ID = 'meta-llama/Llama-3.2-1B-Instruct'  # Using smaller model for testing
if not CLAUDE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set CLAUDE_API_KEY and OPENAI_API_KEY in .env file")

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

class DialoguePredictor:
    def __init__(self, model: str, quant: str = "4bit", temp: str = 0.0, top_p: str = 1.0, return_probs: bool = False):
        """Initialize the evaluator with API clients and optimized rate limiting"""
        self.model = model.lower()
        if self.model == "claude":
            self.claude_client = Anthropic(api_key=CLAUDE_API_KEY)
            self.claude_sem = Semaphore(MAX_CONCURRENT_REQUESTS // 2)  # Split capacity
            self.claude_rate_limiter = RateLimiter(CLAUDE_RATE_LIMIT_CALLS, CLAUDE_RATE_LIMIT_WINDOW, CLAUDE_RATE_LIMIT_DELAY)
        elif self.model == "openai":
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.openai_sem = Semaphore(MAX_CONCURRENT_REQUESTS // 2)  # Split capacity
            self.openai_rate_limiter = RateLimiter(OPENAI_RATE_LIMIT_CALLS, OPENAI_RATE_LIMIT_WINDOW, OPENAI_RATE_LIMIT_DELAY)
        elif self.model == "llama":
            self.temp = float(temp)
            self.top_p = float(top_p)
            self.return_probs = return_probs
            # MODEL_ID = 'meta-llama/Llama-3.2-1B'
            # print(f"[INFO] Using local Llama snapshot: {MODEL_ID}")
            print("[INFO] Loading tokenizer...")
            # self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            print("[INFO] Loading model...")
            
            # Configure quantization based on CLI argument
            quant_cfg = None
            if quant_available and quant != "none":
                if quant == "4bit":
                    quant_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif quant == "8bit":
                    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            
            self.llamamodel = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=False,
                quantization_config=quant_cfg,
            )
            self.llamamodel.eval()
            print("[INFO] Model loaded successfully")

            if temp > 0.0:
                print(f"[INFO] Using temperature={temp}, top_p={top_p}")
            else:
                print("[INFO] Using deterministic generation (temperature=0.0)")
            if return_probs:
                print("[INFO] Probability mode enabled - will calculate entire response probability")
        elif self.model == "gpt2":
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            self.gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
            self.gpt2model.eval()
            self.temp = float(temp)
            self.top_p = float(top_p)
            self.return_probs = return_probs

        
        # Create results directory
        Path("results").mkdir(exist_ok=True)

    def extract_output(self, text: str) -> Dict:
        """Reused function to extract JSON output from response"""
        pattern = r'```json\n(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return json.loads(match.group(1))
        print("Warning: Invalid output format")
        return None

    async def get_claude_prediction(self, prompt: str) -> str:
        """Get prediction from Claude with simple rate limiting"""
        try:
            # Simple delay before API call
            await asyncio.sleep(0.3)
            sys_message = """You are a dialogue predictor for task-oriented conversations.

Your role is to predict direct, helpful system responses that:
- Address the user's needs clearly and stay focused on the task
- Include ONLY concrete information shown in the conversation context
- ANTI-HALLUCINATION: Use 'XXXXXXX' for any specific information not in the context (names, numbers, addresses, etc.) rather than inventing details.
- Match the conversation style and tone naturally

CRITICAL: Respond with ONLY the direct system response. Never include:
- Explanations or reasoning
- Meta-commentary about the conversation  
- Markdown formatting or delimiters
- Prefixes like 'assistant:' or similar"""
            if 'ANTI-HALLUCINATION' not in prompt:
                sys_message = sys_message.replace("- ANTI-HALLUCINATION: Use 'XXXXXXX' for any specific information not in the context (names, numbers, addresses, etc.) rather than inventing details.", "")
            
            message = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0,
                system=sys_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            logging.error(f"Error calling Claude API: {e}")
            return None
        
    def get_openai_prediction(self, prompt: str) -> str:
        """Get prediction from OpenAI with simple rate limiting"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a dialogue predictor for task-oriented conversations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            return None
        
    def get_llama_prediction(self, prompt: str, max_tokens: int = 500) -> str:
        """Get prediction from local Llama model"""
        try:
            messages = [
                {"role": "system", "content": """You are a dialogue predictor for task-oriented conversations.

Your role is to predict direct, helpful system responses that:
- Address the user's needs clearly and stay focused on the task
- Include ONLY concrete information shown in the conversation context
- ANTI-HALLUCINATION: Use 'XXXXXXX' for any specific information not in the context (names, numbers, addresses, etc.) rather than inventing details.
- Match the conversation style and tone naturally

CRITICAL: Respond with ONLY the direct system response. Never include:
- Explanations or reasoning
- Meta-commentary about the conversation  
- Markdown formatting or delimiters
- Prefixes like 'assistant:' or similar"""},
                {"role": "user", "content": prompt}]
            inp = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.llamamodel.device)

            # inp = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            # print(f"[DEBUG] Input length (tokens): {inp.shape[1]}")
            
            do_sample = self.temp > 0.0

            with torch.inference_mode():
                if self.return_probs:
                    # Generate the full response first
                    out = self.llamamodel.generate(
                        inp, 
                        max_new_tokens=max_tokens, 
                        do_sample=do_sample,
                        temperature=self.temp if do_sample else None,
                        top_p=self.top_p if do_sample else None,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    gen = out[0][inp.shape[-1]:]
                    raw_pred = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
                    
                    # Clean up the prediction
                    pred = re.sub(r"^(assistant|user|system)\s*[:\-\n]*", "", raw_pred, flags=re.IGNORECASE).strip()
                    pred = re.sub(r"(?im)^\s*(i'?m ready to predict\.?|here( is|'s) the predicted utterance:?|here is my prediction:?|i will predict the masked speaker'?s responses:?|i'?ll predict the masked speaker'?s responses:?|prediction:?|predicted utterance:?|response:?|output:?|answer:?|the predicted response:?|the masked speaker'?s response:?|the masked speaker will say:?|the masked speaker says:?|the response is:?|the utterance is:?|the following is the predicted utterance:?|the following is my prediction:?|the following is the response:?|the following is the answer:?|the following is the output:?|the following is the utterance:?)[\s\-:]*\n*","", pred, flags=re.IGNORECASE|re.MULTILINE)
                    pred = "\n".join([line for line in pred.splitlines() if line.strip()])
                    
                    # Calculate probability of the entire response
                    if len(gen) > 0:
                        # Get logits for each token in the generated sequence
                        total_log_prob = 0.0
                        current_input = inp.clone()
                        
                        for token_idx in range(len(gen)):
                            # Get logits for current position
                            outputs = self.model(current_input)
                            logits = outputs.logits[0, -1, :]
                            
                            # Apply temperature
                            if self.temp > 0:
                                logits = logits / self.temp
                            
                            # Apply top-p filtering
                            if self.top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > self.top_p
                                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                                sorted_indices_to_remove[0] = 0
                                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                                logits[indices_to_remove] = float('-inf')
                            
                            # Get probability of the actual token that was generated
                            actual_token = gen[token_idx]
                            probs = torch.softmax(logits, dim=-1)
                            token_prob = probs[actual_token].item()
                            
                            # Add to total log probability
                            total_log_prob += torch.log(torch.tensor(token_prob)).item()
                            
                            # Add token to input for next iteration
                            current_input = torch.cat([current_input, actual_token.unsqueeze(0).unsqueeze(0)], dim=1)
                        
                        # Convert log probability to probability
                        total_prob = torch.exp(torch.tensor(total_log_prob)).item()
                        
                        # Normalize to 0-100 scale using log scale transformation
                        # This maps very small probabilities to a more interpretable range
                        if total_prob > 0:
                            # Use log scale to spread out the values
                            log_prob = torch.log(torch.tensor(total_prob)).item()
                            # Normalize: typical log probs range from -20 to -5, map to 0-100
                            normalized_percentage = max(0, min(100, (log_prob + 20) / 15 * 100))
                        else:
                            normalized_percentage = 0.0
                        
                        return f"{pred} [RESPONSE_PROB: {normalized_percentage:.1f}%]"
                    else:
                        return f"{pred} [RESPONSE_PROB: 0.0%]"
                else:
                    # Original generation method
                    out = self.llamamodel.generate(
                        inp, 
                        max_new_tokens=max_tokens, 
                        do_sample=do_sample,
                        temperature=self.temp if do_sample else None,
                        top_p=self.top_p if do_sample else None,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    gen = out[0][inp.shape[-1]:]
                    raw_pred = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
                    # Remove any leading speaker label or extra text (e.g., 'assistant:', 'user:', etc.)
                    pred = re.sub(r"^(assistant|user|system)\s*[:\-\n]*", "", raw_pred, flags=re.IGNORECASE).strip()
                    # Remove meta-commentary lines (e.g., 'I'm ready to predict', 'Here is the predicted utterance', etc.) from anywhere in the output
                    pred = re.sub(r"(?im)^\s*(i'?m ready to predict\.?|here( is|'s) the predicted utterance:?|here is my prediction:?|i will predict the masked speaker'?s responses:?|i'?ll predict the masked speaker'?s responses:?|prediction:?|predicted utterance:?|response:?|output:?|answer:?|the predicted response:?|the masked speaker'?s response:?|the masked speaker will say:?|the masked speaker says:?|the response is:?|the utterance is:?|the following is the predicted utterance:?|the following is my prediction:?|the following is the response:?|the following is the answer:?|the following is the output:?|the following is the utterance:?)[\s\-:]*\n*","", pred, flags=re.IGNORECASE|re.MULTILINE)
                    # Remove any remaining empty lines
                    pred = "\n".join([line for line in pred.splitlines() if line.strip()])
                    return pred
        except Exception as e:
            logging.error(f"Error generating with Llama model: {e}")
            return None
        
    def get_gpt2_prediction(self, prompt: str, max_tokens: int = 500) -> str:
        """Get prediction from GPT-2 model"""
        try:
            inp = self.gpt2tokenizer.encode(prompt, return_tensors="pt").cuda()

            do_sample = self.temp > 0.0

            with torch.inference_mode():
                if self.return_probs:
                    # Generate the full response first
                    out = self.gpt2model.generate(
                        inp, 
                        max_new_tokens=max_tokens, 
                        do_sample=do_sample,
                        temperature=self.temp if do_sample else None,
                        top_p=self.top_p if do_sample else None,
                        eos_token_id=self.gpt2tokenizer.eos_token_id
                    )
                    gen = out[0][inp.shape[-1]:]
                    raw_pred = self.gpt2tokenizer.decode(gen, skip_special_tokens=True).strip()
                    
                    # Clean up the prediction
                    pred = re.sub(r"^(assistant|user|system)\s*[:\-\n]*", "", raw_pred, flags=re.IGNORECASE).strip()
                    pred = re.sub(r"(?im)^\s*(i'?m ready to predict\.?|here( is|'s) the predicted utterance:?|here is my prediction:?|i will predict the masked speaker'?s responses:?|i'?ll predict the masked speaker'?s responses:?|prediction:?|predicted utterance:?|response:?|output:?|answer:?|the predicted response:?|the masked speaker'?s response:?|the masked speaker will say:?|the masked speaker says:?|the response is:?|the utterance is:?|the following is the predicted utterance:?|the following is my prediction:?|the following is the response:?|the following is the answer:?|the following is the output:?|the following is the utterance:?)[\s\-:]*\n*","", pred, flags=re.IGNORECASE|re.MULTILINE)
                    pred = "\n".join([line for line in pred.splitlines() if line.strip()])
                    
                    # Calculate probability of the entire response
                    if len(gen) > 0:
                        # Get logits for each token in the generated sequence
                        total_log_prob = 0.0
                        current_input = inp.clone()
                        
                        for token_idx in range(len(gen)):
                            # Get logits for current position
                            outputs = self.gpt2model(current_input)
                            logits = outputs.logits[0, -1, :]
                            # Apply temperature
                            if self.temp > 0:
                                logits = logits / self.temp
                            # Apply top-p filtering
                            if self.top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > self.top_p
                                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                                sorted_indices_to_remove[0] = 0
                                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                                logits[indices_to_remove] = float('-inf')
                            # Get probability of the actual token that was generated
                            actual_token = gen[token_idx]
                            probs = torch.softmax(logits, dim=-1)
                            token_prob = probs[actual_token].item()
                            # Add to total log probability
                            total_log_prob += torch.log(torch.tensor(token_prob)).item()
                            # Add token to input for next iteration
                            current_input = torch.cat([current_input, actual_token.unsqueeze(0).unsqueeze(0)], dim=1)
                        # Convert log probability to probability
                        total_prob = torch.exp(torch.tensor(total_log_prob)).item()
                        # Normalize to 0-100 scale using log scale transformation
                        # This maps very small probabilities to a more interpretable range
                        if total_prob > 0:
                            # Use log scale to spread out the values
                            log_prob = torch.log(torch.tensor(total_prob)).item()
                            # Normalize: typical log probs range from -20 to -5, map to 0-100
                            normalized_percentage = max(0, min(100, (log_prob + 20) / 15 * 100))
                        else:
                            normalized_percentage = 0.0
                        return f"{pred} [RESPONSE_PROB: {normalized_percentage:.1f}%]"
                    else:
                        return f"{pred} [RESPONSE_PROB: 0.0%]"
                else:
                    # Original generation method
                    out = self.gpt2model.generate(inp)
                    gen = out[0][inp.shape[-1]:]
                    raw_pred = self.gpt2tokenizer.decode(gen, skip_special_tokens=True).strip()
                    inp = self.gpt2tokenizer.decode(inp[0], skip_special_tokens=True).strip()
                    # Remove any leading speaker label or extra text (e.g., 'assistant:', 'user:', etc.)
                    pred = re.sub(r"^(assistant|user|system)\s*[:\-\n]*", "", raw_pred, flags=re.IGNORECASE).strip()
                    # Remove meta-commentary lines (e.g., 'I'm ready to predict', 'Here is the predicted utterance', etc.) from anywhere in the output
                    pred = re.sub(r"(?im)^\s*(i'?m ready to predict\.?|here( is|'s) the predicted utterance:?|here is my prediction:?|i will predict the masked speaker'?s responses:?|i'?ll predict the masked speaker'?s responses:?|prediction:?|predicted utterance:?|response:?|output:?|answer:?|the predicted response:?|the masked speaker'?s response:?|the masked speaker will say:?|the masked speaker says:?|the response is:?|the utterance is:?|the following is the predicted utterance:?|the following is my prediction:?|the following is the response:?|the following is the answer:?|the following is the output:?|the following is the utterance:?)[\s\-:]*\n*","", pred, flags=re.IGNORECASE|re.MULTILINE)
                    # Remove any remaining empty lines
                    pred = "\n".join([line for line in pred.splitlines() if line.strip()])
                    return pred
        except Exception as e:
            logging.error(f"Error generating with GPT-2 model: {e}")
            return None
        
    async def process_prompt(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single prompt and return predictions"""
        prompt = prompt_data['prompt']
        scenario = prompt_data['scenario']
        result = {
            "dialogue_id": prompt_data.get("dialogue_id", ""),
            "turn_id": prompt_data.get("turn_id", ""),
            "scenario": scenario,
            "full_dialogue": prompt_data.get("full_dialogue", ""),
            "prompt": prompt,
            "actual_response": prompt_data.get("actual_response", ""),
            "predictions": []
        }
        
        if self.model == "claude":
            async with self.claude_sem:
                await self.claude_rate_limiter.acquire()
                response = await self.get_claude_prediction(prompt)
                if response:
                    predictions = self.extract_predictions(response)
                    result["predictions"] = predictions
                else:
                    result["predictions"] = []
        elif self.model == "openai":
            async with self.openai_sem:
                await self.openai_rate_limiter.acquire()
                response = self.get_openai_prediction(prompt)
                if response:
                    predictions = self.extract_predictions(response)
                    result["predictions"] = predictions
                else:
                    result["predictions"] = []
        elif self.model == "llama":
            response = self.get_llama_prediction(prompt)
            if response:
                predictions = self.extract_predictions(response)
                result["predictions"] = predictions
            else:
                result["predictions"] = []
        elif self.model == "gpt2":
            response = self.get_gpt2_prediction(prompt)
            if response:
                predictions = self.extract_predictions(response)
                result["predictions"] = predictions
            else:
                result["predictions"] = []
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        
        return result


    def extract_predictions(self, response: str) -> List[str]:
        """Extract predictions from Claude's response"""
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                predictions = json.loads(json_str)
                if not isinstance(predictions, list):
                    raise ValueError("Response is not a JSON array")
            elif response.startswith("[") and response.endswith("]"):
                predictions = json.loads(response)
                if not isinstance(predictions, list):
                    raise ValueError("Response is not a JSON array")
            else:
                predictions = [response.strip()]
                
            return predictions
        except Exception as e:
            logging.error(f"Failed to extract predictions: {e}")
            return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate predictions for one-sided dialogue scenarios.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with prompts.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file to save predictions. If no file is given, input_file is used, with 'prompts' replaced with 'predictions<model>'.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model to use for generation.")
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = args.input_file.replace("prompts", f"predictions_{args.model_name}")
    output_path = Path(args.output_file)
    if output_path.exists():
        print(f"Output file {output_path} already exists. Please remove it or specify a different file.")
        sys.exit(1)

    prompts = args.input_file
    with open(prompts, 'r') as f:
        prompt_data = [json.loads(line) for line in f]

    predictor = DialoguePredictor(args.model_name)
    
    for prompt in tqdm(prompt_data):
        response = asyncio.run(predictor.process_prompt(prompt))
        with open(output_path, 'a') as out_f:
            out_f.write(json.dumps(response) + '\n')




