import json
import asyncio
import sys
from pathlib import Path
from tqdm import tqdm
import os
import time
import logging
from asyncio import Semaphore
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from typing import Dict, List

# Constants - Conservative settings for stability
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '5'))  # Limited to 5 concurrent calls max
CLAUDE_RATE_LIMIT_CALLS = int(os.getenv('CLAUDE_RATE_LIMIT_CALLS', '80'))  # Claude can handle high throughput
CLAUDE_RATE_LIMIT_WINDOW = int(os.getenv('CLAUDE_RATE_LIMIT_WINDOW', '60'))
CLAUDE_RATE_LIMIT_DELAY = float(os.getenv('CLAUDE_RATE_LIMIT_DELAY', '1'))  # Reduced delay for speed
OPENAI_RATE_LIMIT_CALLS = int(os.getenv('OPENAI_RATE_LIMIT_CALLS', '80'))  # Increased for better throughput
OPENAI_RATE_LIMIT_WINDOW = int(os.getenv('OPENAI_RATE_LIMIT_WINDOW', '60'))
OPENAI_RATE_LIMIT_DELAY = float(os.getenv('OPENAI_RATE_LIMIT_DELAY', '1'))  # Reduced delay for speed
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '5'))  # Smaller batches to match concurrent limit

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv('.env')

# Get API keys
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not CLAUDE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set CLAUDE_API_KEY and OPENAI_API_KEY in .env file")

# Global flag for skipping evaluation
SKIP_EVALUATION = False

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

class DialogueSummarizationComparator:
    def __init__(self, claude_api_key: str, openai_api_key: str, token_count_only: bool = False):
        """Initialize the comparator with API clients and optimized rate limiting"""
        self.claude_client = Anthropic(api_key=claude_api_key)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.token_count_only = token_count_only
        
        # API call counters
        self.claude_api_calls = 0
        
        # Separate semaphores for better concurrency control
        self.claude_sem = Semaphore(MAX_CONCURRENT_REQUESTS)  # All requests can use either API
        
        # Separate rate limiters for Claude and OpenAI
        self.claude_rate_limiter = RateLimiter(min_interval=0.4)
    
    
    async def get_claude_summary(self, prompt) -> str:
        """Get summary from Claude with proper rate limiting"""

        system_prompt = prompt['system_prompt']
        user_prompt = prompt['user_prompt']
        
        async with self.claude_sem:
            await self.claude_rate_limiter.acquire()
            logging.info(f"Claude API call starting at {time.time():.3f}")
            try:
                def _call():
                    return self.claude_client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        temperature=0,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                
                message = await asyncio.to_thread(_call)
                self.claude_api_calls += 1  # Increment counter after successful call
                logging.info(f"Claude API call completed at {time.time():.3f} (Total: {self.claude_api_calls})")
                return message.content[0].text.strip()
                            
            except Exception as e:
                logging.error(f"Error calling Claude API: {e}")
                return None
            
    async def process_single_comparison(self, inputs) -> Dict:
        """Process a single dialogue comparison"""
        dialogue_id = inputs.get("dialogue_id")
        try:
            # Prepare all scenarios
            prompts = inputs.get("prompts")
            masked_prompt = prompts.get("masked")
            full_prompt = prompts.get("full")
            predicted_prompt = prompts.get("predicted")

            convos = inputs.get("dialogues")
            masked_convo = convos.get("masked")
            full_convo = convos.get("full")
            predicted_convo = convos.get("predicted")
            
            
            # Get summaries concurrently instead of sequentially
            summary_tasks = [
                self.get_claude_summary(masked_prompt),
                self.get_claude_summary(full_prompt),
                self.get_claude_summary(predicted_prompt)
            ]
            
            # Add predicted summary task if context is available
            masked_summary, full_summary, predicted_summary = await asyncio.gather(*summary_tasks)
            return {
                "dialogue_id": dialogue_id,
                "masked_conversation": masked_convo,
                "masked_summary": masked_summary,
                "full_conversation": full_convo,
                "full_summary": full_summary,
                "predicted_conversation": predicted_convo,
                "predicted_summary": predicted_summary
            }
            
            
        except Exception as e:
            logging.error(f"Error processing dialogue {dialogue_id}: {e}")
        
        return None
    
    async def process_comparison_batch(self, prompts) -> List[Dict]:
        """Process a batch of dialogues concurrently"""
        results, tasks = [], []

        for prompt in prompts:

            async def one_job(prompt):
                try:
                    result = await self.process_single_comparison(prompt)
                    if result:
                        return result
                    else:
                        logging.warning(f"No result for dialogue {prompt.get('dialogue_id')}")
                        return None
                except Exception as e:
                    logging.error(f"Error processing dialogue {prompt.get('dialogue_id')}: {e}")
                    return None

            tasks.append(asyncio.create_task(one_job(prompt)))

        for finished in await asyncio.gather(*tasks, return_exceptions=True):
            if finished and not isinstance(finished, Exception):
                results.append(finished)
            elif isinstance(finished, Exception):
                logging.error(f"Task exception in batch processing: {finished}")
        
        return results
    
    async def run_comparison(self, prompts):
        """Run full comparison evaluation across dialogues"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        results = []

        try:
            logger.info(f"Processing {len(prompts)} dialogues in batches of {BATCH_SIZE}")
            
            # Process dialogues in batches concurrently
            for batch_start in range(0, len(prompts), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(prompts))
                batch = prompts[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(len(prompts)-1)//BATCH_SIZE + 1}")
                batch_results = await self.process_comparison_batch(batch)
                
                # Add batch results to overall results
                results.extend([r for r in batch_results if r is not None])
                
                # No delay between batches - rate limiting handled by RateLimiter class
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Comparison failed: {str(e)}", exc_info=True)
            raise



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

    comparator = DialogueSummarizationComparator(CLAUDE_API_KEY, OPENAI_API_KEY)
    results = asyncio.run(comparator.run_comparison(prompt_data))
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Predictions saved to {output_path}")

    