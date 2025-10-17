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

# Import token counting utilities
try:
    from token_counter_utils import TokenCounterUtils
except ImportError:
    print("Warning: Could not import TokenCounterUtils")
    TokenCounterUtils = None

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

# Constants - Optimized for maximum speed with 100 samples (can be overridden by environment variables)
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '20'))  # Moderate to avoid giant in-memory queues
CLAUDE_RATE_LIMIT_CALLS = int(os.getenv('CLAUDE_RATE_LIMIT_CALLS', '80'))  # Claude can handle high throughput
CLAUDE_RATE_LIMIT_WINDOW = int(os.getenv('CLAUDE_RATE_LIMIT_WINDOW', '60'))
CLAUDE_RATE_LIMIT_DELAY = float(os.getenv('CLAUDE_RATE_LIMIT_DELAY', '1'))  # Reduced delay for speed
OPENAI_RATE_LIMIT_CALLS = int(os.getenv('OPENAI_RATE_LIMIT_CALLS', '80'))  # Increased for better throughput
OPENAI_RATE_LIMIT_WINDOW = int(os.getenv('OPENAI_RATE_LIMIT_WINDOW', '60'))
OPENAI_RATE_LIMIT_DELAY = float(os.getenv('OPENAI_RATE_LIMIT_DELAY', '1'))  # Reduced delay for speed
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '25'))  # Larger batches for 100 samples

# Enhanced few-shot examples for better prompting
ENHANCED_FEW_SHOT_EXAMPLES = {
    "scenario_1_1": """
EXAMPLE 1 - TRAIN BOOKING:
User: I need a train from Cambridge to London
System: [MASKED]
User: I want to leave after 14:00
→ [PREDICTED] System: I have trains departing after 14:00. The XXXXXXX leaves at XXXXXXX and arrives at XXXXXXX. Would you like me to book it?

EXAMPLE 2 - HOTEL BOOKING:
User: I'm looking for a hotel in the city centre
System: [MASKED]
User: I need free parking
→ [PREDICTED] System: I found hotels in the city centre with free parking. The XXXXXXX Hotel is available. Would you like a reservation?

EXAMPLE 3 - RESTAURANT BOOKING:
User: I need a restaurant for tonight
System: [MASKED]
User: Italian food, please
→ [PREDICTED] System: I have Italian restaurants available for tonight. XXXXXXX has a table at XXXXXXX. Shall I book it?

EXAMPLE 4 - ATTRACTION INFO:
User: I want to visit the museum
System: [MASKED]
User: What's the address?
→ [PREDICTED] System: The museum is located at XXXXXXX Street, postcode XXXXXXX. Phone number is XXXXXXX.
""",
    "scenario_1_2": """
EXAMPLE 1 - COMPLEX BOOKING:
User: I need a restaurant for tonight
System: [MASKED]
User: Italian food, please
→ [PREDICTED] System: I have Italian restaurants available for tonight. XXXXXXX has a table at XXXXXXX. Shall I book it?
User: Perfect, book it for 2 people

EXAMPLE 2 - ACCOMMODATION DETAILS:
User: I need accommodation for next week
System: [MASKED]
User: A guesthouse with WiFi
→ [PREDICTED] System: I found guesthouses with WiFi available next week. The XXXXXXX Guesthouse is £XXXXXXX per night.
User: Book it for 3 nights starting Monday

EXAMPLE 3 - TRANSPORT DETAILS:
User: I need a train to Norwich
System: [MASKED]
User: I want to leave after 17:00
→ [PREDICTED] System: I have trains to Norwich after 17:00. The XXXXXXX departs at XXXXXXX. Would you like tickets?
User: Yes, for 2 people

EXAMPLE 4 - SPECIFIC INFORMATION:
User: I'm looking for a cheap hotel
System: [MASKED]
User: In the west area
→ [PREDICTED] System: I found cheap hotels in the west area. The XXXXXXX is £XXXXXXX per night at XXXXXXX. Free WiFi included.
User: What's the phone number?
""",
    "scenario_2_1": """
EXAMPLE 1 - TRAIN WITH DETAILS:
User: I need a train to Norwich
System: [MASKED - 8 words]
User: I want to leave after 17:00
→ [PREDICTED] System: What time would you prefer to depart after 17:00?

EXAMPLE 2 - RESTAURANT WITH LOCATION:
User: I'm looking for a restaurant
System: [MASKED - 6 words]
User: Chinese food in the centre
→ [PREDICTED] System: I found Chinese restaurants in the centre area available.

EXAMPLE 3 - HOTEL WITH AMENITIES:
User: I need a hotel
System: [MASKED - 7 words]
User: With free parking
→ [PREDICTED] System: There are hotels with free parking available. What area?

EXAMPLE 4 - ATTRACTION WITH INFO:
User: I want to see the theatre
System: [MASKED - 5 words]
User: What's the address?
→ [PREDICTED] System: The theatre is located at XXXXXXX Street, postcode XXXXXXX.
""",
    "scenario_2_2": """
EXAMPLE 1 - COMPLEX BOOKING:
User: I need accommodation
System: [MASKED - 7 words]
User: A hotel with parking
→ [PREDICTED] System: I found hotels with parking available. The XXXXXXX Hotel is available.
User: Book it for next weekend

EXAMPLE 2 - RESTAURANT RESERVATION:
User: I want to eat out tonight
System: [MASKED - 5 words]
User: Something expensive in the west
→ [PREDICTED] System: I found expensive restaurants in the west area available tonight.
User: Make a reservation for 8pm

EXAMPLE 3 - TRANSPORT BOOKING:
User: I need a train
System: [MASKED - 6 words]
User: To London on Friday
→ [PREDICTED] System: I have trains to London available on Friday. What time?
User: Book for 2 people

EXAMPLE 4 - ATTRACTION VISIT:
User: I want to visit the museum
System: [MASKED - 8 words]
User: What are the opening hours?
→ [PREDICTED] System: The museum is open XXXXXXX to XXXXXXX. Entrance is £XXXXXXX.
User: How much are tickets?
"""
}

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv('.env')

# Get API keys
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not CLAUDE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set CLAUDE_API_KEY and OPENAI_API_KEY in .env file")

class RateLimiter:
    def __init__(self, min_interval: float = 0.3):
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

class DialoguePredictionEvaluator:
    def __init__(self, claude_api_key: str, openai_api_key: str, use_few_shot: bool = True, token_count_only: bool = False, disable_anti_hallucination: bool = False, max_prediction_context_turns: int = 20, max_evaluation_context_turns: int = 10):
        """Initialize the evaluator with API clients and optimized rate limiting
        
        Args:
            claude_api_key: Claude API key
            openai_api_key: OpenAI API key
            use_few_shot: Whether to use few-shot examples in prompts
            token_count_only: Whether to only count tokens without making API calls
            disable_anti_hallucination: Whether to disable XXXXXXX anti-hallucination instructions
            max_prediction_context_turns: Maximum number of previous turns to include in Claude prediction context (None = no limit)
            max_evaluation_context_turns: Maximum number of turns (before/after) to include in GPT evaluation context (None = no limit)
        """
        self.claude_client = Anthropic(api_key=claude_api_key)
        # Use synchronous OpenAI client - no async issues like Claude
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.use_few_shot = use_few_shot
        self.token_count_only = token_count_only
        self.disable_anti_hallucination = disable_anti_hallucination
        
        # Context limiting parameters
        self.max_prediction_context_turns = max_prediction_context_turns
        self.max_evaluation_context_turns = max_evaluation_context_turns
        
        # Initialize token counter if needed
        self.token_counter = None
        if self.token_count_only and TokenCounterUtils:
            self.token_counter = TokenCounterUtils()
        
        # API call counters
        self.claude_api_calls = 0
        self.openai_api_calls = 0
        
        # Separate semaphores for better concurrency control
        self.claude_sem = Semaphore(MAX_CONCURRENT_REQUESTS // 2)  # Split capacity
        self.openai_sem = Semaphore(MAX_CONCURRENT_REQUESTS // 2)  # Split capacity
        
        # Separate rate limiters for Claude and OpenAI
        self.claude_rate_limiter = RateLimiter(min_interval=0.4)
        self.openai_rate_limiter = RateLimiter(min_interval=0.4)
        
        # Create results directory
        Path("results").mkdir(exist_ok=True)
        
        # Initialize ROUGE scorer once
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def estimate_target_words(self, actual_utterance: str) -> int:
        """Estimate target word count for response, used in enhanced prompts."""
        return len(actual_utterance.split())
    
    def should_mask_speaker(self, speaker: str, dataset: str) -> bool:
        """
        Determine if a speaker should be masked based on dataset conventions.
        Returns True if the speaker should be masked (i.e., is the system/assistant speaker).
        """
        dataset_lower = dataset.lower()
        speaker_lower = speaker.lower()
        
        # MultiWOZ: System/SYSTEM speakers
        if 'multiwoz' in dataset_lower:
            return speaker in ['System', 'SYSTEM', 'Speaker_2']  # Added Speaker_2 for test split
        
        # Kandor: Participant_R (receiving participant)  
        elif 'kandor' in dataset_lower:
            return speaker == 'Participant_R'
        
        # MedDialogue: Doctor speakers
        elif 'meddialogue' in dataset_lower or 'mts-dialog' in dataset_lower:
            return speaker == 'Doctor'
        
        # DailyDialog: Speaker_2 (alternating speakers, predict Speaker_2)
        elif 'dailydialog' in dataset_lower:
            return speaker == 'Speaker_2'
        
        # AMI: For meeting dialogues, we might want to predict specific roles
        # For now, assume we predict responses from 'Speaker_A' or participants in certain roles
        elif 'ami' in dataset_lower:
            # In AMI, we could predict any speaker, but let's assume Speaker_A is the main facilitator
            return speaker == 'Speaker_A'
        
        # Default fallback: look for common patterns
        else:
            return (
                speaker in ['System', 'SYSTEM', 'Assistant', 'Bot', 'Speaker_2'] or
                speaker == 'Doctor' or
                speaker == 'Participant_R' or
                'system' in speaker_lower or
                'assistant' in speaker_lower or
                'bot' in speaker_lower
            )

    def load_test_split(self, test_split_file: str) -> List[Dict[str, Any]]:
        """Load dialogues from a test split JSON file"""
        logging.info(f"Loading test split from {test_split_file}...")
        
        try:
            with open(test_split_file, 'r', encoding='utf-8') as f:
                test_dialogues = json.load(f)
            
            if not test_dialogues:
                logging.warning(f"No dialogues found in test split file: {test_split_file}")
                return []
            
            # Convert to expected format and assign integer dialogue IDs
            formatted_dialogues = []
            for i, dialogue in enumerate(test_dialogues):
                # Extract dialogue ID number from string (e.g., "test_dialogue_1" -> 0)
                dialogue_id = i  # Use 0-based indexing
                
                formatted_dialogue = {
                    'speakers': dialogue['speakers'],
                    'utterances': dialogue['utterances'],
                    'dataset': dialogue.get('dataset', 'multiwoz').lower(),
                    'dialogue_id': dialogue_id,  # Integer ID for prediction matching
                    'original_dialogue_id': dialogue.get('dialogue_id', f'test_dialogue_{i+1}'),  # Keep original string ID
                    'split': dialogue.get('split', 'test')
                }
                
                formatted_dialogues.append(formatted_dialogue)
            
            logging.info(f"Loaded {len(formatted_dialogues)} dialogues from test split file")
            return formatted_dialogues
            
        except Exception as e:
            logging.error(f"Error loading test split file {test_split_file}: {e}")
            return []

    def load_test_split_txt(self, test_split_file_txt: str) -> List[Dict[str, Any]]:
        """Load dialogues from a test split TXT file with 3-turn format"""
        logging.info(f"Loading test split from TXT file: {test_split_file_txt}...")
        
        try:
            with open(test_split_file_txt, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Split by triple newlines to get individual examples
            examples = content.split('\n\n\n')
            
            if not examples:
                logging.warning(f"No examples found in test split TXT file: {test_split_file_txt}")
                return []
            
            formatted_dialogues = []
            for i, example in enumerate(examples):
                example = example.strip()
                if not example:
                    continue
                
                try:
                    # Parse the 3-turn format
                    # Each example should have exactly 3 turns
                    turns = []
                    speakers = []
                    utterances = []
                    
                    # Split by "Turn X [Speaker]:" pattern to extract turns
                    import re
                    turn_pattern = r'Turn \d+ \[([^\]]+)\]: (.+?)(?=Turn \d+ \[|$)'
                    matches = re.findall(turn_pattern, example, re.DOTALL)
                    
                    if len(matches) != 3:
                        logging.warning(f"Example {i+1} does not have exactly 3 turns, skipping: {len(matches)} turns found")
                        continue
                    
                    for speaker, utterance in matches:
                        speakers.append(speaker)
                        utterances.append(utterance.strip())
                    
                    # Create dialogue in expected format
                    formatted_dialogue = {
                        'speakers': speakers,
                        'utterances': utterances,
                        'dataset': 'multiwoz',  # Default to multiwoz since test file appears to be from multiwoz
                        'dialogue_id': i,  # Use 0-based indexing
                        'original_dialogue_id': f'txt_example_{i+1}',
                        'split': 'test'
                    }
                    
                    formatted_dialogues.append(formatted_dialogue)
                    
                except Exception as e:
                    logging.warning(f"Error parsing example {i+1}: {e}")
                    continue
            
            logging.info(f"Loaded {len(formatted_dialogues)} examples from TXT test split file")
            return formatted_dialogues
            
        except Exception as e:
            logging.error(f"Error loading test split TXT file {test_split_file_txt}: {e}")
            return []

    def load_dataset(self, dataset_name: str, max_dialogues: int = None) -> List[Dict[str, Any]]:
        """Load dialogues using the conversational data loader"""
        print(f"Loading {dataset_name} dataset...")
        logging.info(f"Loading {dataset_name} dataset...")
        
        if ConversationalDataLoader is None:
            print("ConversationalDataLoader not available")
            logging.error("ConversationalDataLoader not available")
            return []
        
        try:
            # Initialize the conversational data loader
            loader = ConversationalDataLoader(DATA_ROOT)
            
            # Load dialogues from the specified dataset
            print(f"Starting to load dialogues from {dataset_name}...")
            logging.info(f"Starting to load dialogues from {dataset_name}...")
            raw_dialogues = []
            dialogue_generator = loader.load_dataset(dataset_name.lower())
            
            # Load dialogues with progress logging
            for i, dialogue in enumerate(dialogue_generator):
                raw_dialogues.append(dialogue)
                if (i + 1) % 100 == 0:
                    logging.info(f"Loaded {i + 1} raw dialogues...")
                # Early exit if we have enough dialogues (before filtering)
                if max_dialogues and len(raw_dialogues) >= max_dialogues * 10:  # Load extra for filtering
                    logging.info(f"Loaded {len(raw_dialogues)} raw dialogues (stopping early for efficiency)")
                    break
            
            if not raw_dialogues:
                logging.warning(f"No dialogues found for dataset: {dataset_name}")
                return []
            
            logging.info(f"Finished loading {len(raw_dialogues)} raw dialogues, starting filtering...")
            
            # Filter dialogues to ensure they have sufficient content
            filtered_dialogues = []
            for i, dialogue in enumerate(raw_dialogues):
                if (i + 1) % 500 == 0:
                    logging.info(f"Filtering progress: {i + 1}/{len(raw_dialogues)} dialogues processed...")
                # Ensure we have speakers and utterances
                if not dialogue.get('speakers') or not dialogue.get('utterances'):
                    continue
                
                # Ensure balanced conversation (at least some back-and-forth)
                speakers = dialogue['speakers']
                utterances = dialogue['utterances']
                
                if len(speakers) != len(utterances):
                    continue
                
                # Check for reasonable dialogue length
                if len(utterances) < 4:  # Need at least 2 turns each
                    continue
                
                # Count speaker distribution to ensure it's a real conversation
                speaker_counts = {}
                for speaker in speakers:
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                
                # Ensure we have at least 2 speakers with multiple turns each
                if len(speaker_counts) < 2 or max(speaker_counts.values()) < 2:
                    continue
                
                # Format for compatibility with existing code
                formatted_dialogue = {
                    'speakers': speakers,
                    'utterances': utterances,
                    'dataset': dialogue.get('dataset', dataset_name),
                    'dialogue_id': dialogue.get('dialogue_id', f"{dataset_name}_unknown")
                }
                
                filtered_dialogues.append(formatted_dialogue)
                
                # Apply max_dialogues limit if specified
                if max_dialogues and len(filtered_dialogues) >= max_dialogues:
                    break
            
            logging.info(f"Loaded {len(filtered_dialogues)} dialogues from {dataset_name}")
            return filtered_dialogues
            
        except Exception as e:
            logging.error(f"Error loading {dataset_name} dataset: {e}")
            return []

    def extract_output(self, text: str) -> Dict:
        """Reused function to extract JSON output from Claude's response"""
        pattern = r'```json\n(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return json.loads(match.group(1))
        print("Warning: Invalid Claude output format")
        return None

    def format_dialogue_context(self, dialogue: Dict[str, Any], turn_n: int, include_next_turn: bool = False, show_durations: bool = False) -> str:
        """
        Format dialogue context with masked system turns and optional word counts
        Args:
            dialogue: The dialogue dictionary
            turn_n: Current turn number we're predicting
            include_next_turn: Whether to include next turn
            show_durations: Whether to show word counts for masked turns (renamed for compatibility)
        """
        # Determine context window limits
        if self.max_prediction_context_turns is None:
            # No limit - use all available context (original behavior)
            start_turn = 0
        else:
            # Apply context limit
            start_turn = max(0, turn_n - self.max_prediction_context_turns)
        
        output = ""
        
        # Add truncation indicator if we cut context
        if start_turn > 0:
            truncated_turns = start_turn
            output += f"[... {truncated_turns} earlier turns truncated for context limit ...]\n"
        
        # Format turns within the context window up to turn_n
        for i, (speaker, utterance) in enumerate(zip(dialogue['speakers'][start_turn:turn_n], dialogue['utterances'][start_turn:turn_n]), start=start_turn):
            output += f"Turn {i+1} [{speaker}]: "
            
            # Determine if this turn should be masked based on speaker and dataset
            should_mask = self.should_mask_speaker(speaker, dialogue.get('dataset', ''))
            
            if should_mask:
                if show_durations:  # Using existing parameter name for compatibility
                    word_count = len(utterance.split())
                    output += f"[MASKED - {word_count} words]\n"
                else:
                    output += "[MASKED]\n"
            else:
                output += f"{utterance}\n"
        
        # Add the turn we need to predict with special marker
        if turn_n < len(dialogue['speakers']):
            next_speaker = dialogue['speakers'][turn_n]
            output += f"Turn {turn_n+1} [Predict this turn : {next_speaker}]: \n"
        
        # Add next turn if requested
        if include_next_turn and turn_n+1 < len(dialogue['utterances']):
            next_speaker = dialogue['speakers'][turn_n+1]
            next_utterance = dialogue['utterances'][turn_n+1]
            output += f"Turn {turn_n+2} [{next_speaker}]: {next_utterance}\n"
        
        return output

    def format_full_dialogue_context(self, dialogue: Dict[str, Any], turn_n: int, include_future_context: bool = False) -> str:
        """
        Format full dialogue context WITHOUT masking for evaluation purposes.
        Shows all actual utterances so the evaluator can determine what info was available.
        
        Args:
            dialogue: The dialogue dictionary
            turn_n: The turn number being predicted (0-indexed)
            include_future_context: Whether to include turns after the predicted turn (for scenarios 1.2, 2.2)
        """
        # Determine context window limits
        if self.max_evaluation_context_turns is None:
            # No limit - use all available context (original behavior)
            start_turn = 0
            end_turn = len(dialogue['utterances']) if include_future_context else turn_n + 1
        else:
            # Apply context limits
            start_turn = max(0, turn_n - self.max_evaluation_context_turns)
            if include_future_context:
                end_turn = min(len(dialogue['utterances']), turn_n + 1 + self.max_evaluation_context_turns)
            else:
                end_turn = turn_n + 1
        
        output = ""
        
        # Add truncation indicator if we cut past context
        if start_turn > 0:
            truncated_turns = start_turn
            output += f"[... {truncated_turns} earlier turns truncated for evaluation context limit ...]\n"
        
        # Format turns within the context window up to turn_n (the turn we're predicting)
        for i, (speaker, utterance) in enumerate(zip(dialogue['speakers'][start_turn:turn_n], dialogue['utterances'][start_turn:turn_n]), start=start_turn):
            output += f"Turn {i+1} [{speaker}]: {utterance}\n"
        
        # Add marker for the turn being predicted
        if turn_n < len(dialogue['speakers']):
            predicted_speaker = dialogue['speakers'][turn_n]
            output += f"Turn {turn_n+1} [PREDICTING: {predicted_speaker}]: <-- This turn is being predicted\n"
        
        # Add future context if requested (for scenarios 1.2, 2.2)
        if include_future_context and turn_n + 1 < len(dialogue['utterances']):
            future_start = turn_n + 1
            future_end = end_turn
            
            if future_end > future_start:
                output += "\n--- Future Context (available to model during prediction) ---\n"
                for i in range(future_start, future_end):
                    if i < len(dialogue['utterances']):
                        speaker = dialogue['speakers'][i]
                        utterance = dialogue['utterances'][i]
                        output += f"Turn {i+1} [{speaker}]: {utterance}\n"
                
                # Add truncation indicator if we cut future context
                if self.max_evaluation_context_turns is not None and future_end < len(dialogue['utterances']):
                    remaining_turns = len(dialogue['utterances']) - future_end
                    output += f"[... {remaining_turns} later turns truncated for evaluation context limit ...]\n"
        
        return output

    def prepare_scenario_1_1(self, dialogue: Dict[str, Any], turn_n: int) -> str:
        """Basic next turn prediction without any additional signals
        
        Implementation:
        1. Shows previous user and system turns
        2. Shows masked system turns with [MASKED]
        3. No duration information or future context
        4. Predicts next system response
        
        Example context structure:
        Turn 1 [USER]: What restaurants are available?
        Turn 2 [SYSTEM]: [MASKED]
        Turn 3 [USER]: I want Italian food.
        <predict next system response>
        """
        context = self.format_dialogue_context(dialogue, turn_n, show_durations=False)
        
        # Calculate the turn number being predicted (1-indexed)
        turn_number = turn_n + 1
        
        base_prompt = f"""Given this conversation context which includes:
        1. Previous responses (up to Turn {turn_number-1})
        
        WHAT YOU'RE PREDICTING: Turn {turn_number} (System response)
        
        STRATEGY: Predict the most appropriate system response based on the conversation history available up to Turn {turn_number}

        Context:
        {context}"""
        
        # Use enhanced prompt with examples
        return self.create_enhanced_prompt(base_prompt, "scenario_1_1", use_few_shot=self.use_few_shot)

    def prepare_scenario_1_2(self, dialogue: Dict[str, Any], turn_n: int) -> str:
        """Next turn prediction with future context
        
        Implementation:
        1. Shows previous user and system turns
        2. Shows masked system turns with [MASKED]
        3. Shows next user turn
        4. No duration information
        5. Predicts system response that led to next user turn
        
        Example context structure:
        Turn 1 [USER]: What restaurants are available?
        Turn 2 [SYSTEM]: [MASKED]
        Turn 3 [USER]: I want Italian food.
        Turn 5 [USER]: That sounds perfect!
        <predict system response that led to Turn 5>
        """
        # Include the next user turn in context (same as scenario 2.2 but no word counts)
        context = self.format_dialogue_context(dialogue, turn_n, include_next_turn=True, show_durations=False)
        
        # Get the next user turn number for clarity
        next_turn_num = turn_n + 2  # +1 for next turn, +1 for 1-indexing
        turn_number = turn_n + 1    # The turn we're predicting (1-indexed)
        
        base_prompt = f"""Given this conversation context which includes:
        1. Previous responses (up to Turn {turn_number-1})
        2. The FUTURE user turn (Turn {next_turn_num}) - READ CAREFULLY BELOW
        
        WHAT YOU'RE PREDICTING: Turn {turn_number} (System response)
        
        FUTURE CONTEXT AVAILABLE: Turn {next_turn_num} (Next user response after your prediction)
        
        HOW TO USE THE FUTURE TURN:
        - DO: Infer what type of system response would cause the user's reaction in Turn {next_turn_num}
        - DON'T: Mention any facts, topics, or details that appear only in Turn {next_turn_num}
        
        STRATEGY: Work backwards from Turn {next_turn_num} to predict a system response using only information available up to Turn {turn_number}

        Context:
        {context}"""
        
        # Use enhanced prompt with examples
        return self.create_enhanced_prompt(base_prompt, "scenario_1_2", use_few_shot=self.use_few_shot)

    def prepare_scenario_2_1(self, dialogue: Dict[str, Any], turn_n: int) -> str:
        """Next turn prediction with word count signals
        
        Implementation:
        1. Shows previous user and system turns
        2. Shows masked system turns with word counts: [MASKED - 8 words]
        3. Shows target word count for prediction
        4. Uses word counts to infer response complexity
        
        Example context structure:
        Turn 1 [USER]: What restaurants are available?
        Turn 2 [SYSTEM]: [MASKED - 8 words]
        Turn 3 [USER]: I want Italian food.
        [Target response: 12 words]
        <predict next system response>
        """
        context = self.format_dialogue_context(dialogue, turn_n, show_durations=True)
        
        # Get word count of the turn we're predicting (turn_n)
        if turn_n < len(dialogue['utterances']):
            target_words = self.estimate_target_words(dialogue['utterances'][turn_n])
        else:
            return None  # Can't predict if turn doesn't exist
        
        # Calculate the turn number being predicted (1-indexed)
        turn_number = turn_n + 1
        
        base_prompt = f"""Given this conversation context with response word counts shown,
        predict the system's next response.
        The system's actual next response contains {target_words} words.

        PREDICTING: Turn {turn_number} (System response, target: {target_words} words)

        Context:
        {context}"""
        
        # Use enhanced prompt with examples
        return self.create_enhanced_prompt(base_prompt, "scenario_2_1", use_few_shot=self.use_few_shot)

    def prepare_scenario_2_2(self, dialogue: Dict[str, Any], turn_n: int) -> str:
        """Next turn prediction with both word count signals and future context
        
        Implementation:
        1. Shows previous user and system turns
        2. Shows masked system turns with word counts: [MASKED - 8 words]
        3. Shows next user turn
        4. Shows target word count
        5. Predicts system response that led to next user turn
        
        Example context structure:
        Turn 1 [USER]: What restaurants are available?
        Turn 2 [SYSTEM]: [MASKED - 8 words]
        Turn 3 [USER]: I want Italian food.
        Turn 5 [USER]: That sounds perfect!
        [Target response: 12 words]
        <predict system response that led to Turn 5>
        """
        # Include the next user turn in context
        context = self.format_dialogue_context(dialogue, turn_n, include_next_turn=True, show_durations=True)
        
        # Get word count of the turn we're predicting (turn_n)
        if turn_n < len(dialogue['utterances']):
            target_words = self.estimate_target_words(dialogue['utterances'][turn_n])
        else:
            return None  # Can't predict if turn doesn't exist
        
        # Get the next user turn number for clarity
        next_turn_num = turn_n + 2  # +1 for next turn, +1 for 1-indexing
        turn_number = turn_n + 1    # The turn we're predicting (1-indexed)
        
        base_prompt = f"""Given this conversation context which includes:
        1. Previous responses with word counts (up to Turn {turn_number-1})
        2. The FUTURE user turn (Turn {next_turn_num}) - READ CAREFULLY BELOW
        
        WHAT YOU'RE PREDICTING: Turn {turn_number} (System response, target: {target_words} words)
        
        FUTURE CONTEXT AVAILABLE: Turn {next_turn_num} (Next user response after your prediction)
        
        HOW TO USE THE FUTURE TURN:
        - DO: Infer what type of system response would cause the user's reaction in Turn {next_turn_num}
        - DON'T: Mention any facts, topics, or details that appear only in Turn {next_turn_num}
        
        STRATEGY: Work backwards from Turn {next_turn_num} to predict a close to {target_words} words system response using only information available up to Turn {turn_number}

        Context:
        {context}"""
        
        # Use enhanced prompt with examples
        return self.create_enhanced_prompt(base_prompt, "scenario_2_2", use_few_shot=self.use_few_shot)

    def prepare_scenario_3(self, dialogue: Dict[str, Any]) -> str:
        """Next turn prediction for all system turns at once with word count signals
        
        Implementation:
        1. Shows all user turns in sequence
        2. Shows target word count for each masked system turn
        3. Predicts all system turns in order
        4. Uses word count patterns to inform response complexity
        
        Example context structure:
        Turn 1 [USER]: What restaurants are available?
        [Target response: 8 words]
        Turn 3 [USER]: I want Italian food.
        [Target response: 12 words]
        Turn 5 [USER]: That sounds perfect!
        [Target response: 6 words]
        <predict all system responses in order>
        """
        output = ""
        target_word_counts = []
        system_turns = []
        
        # Show only user turns and target word counts
        for i, (speaker, utterance) in enumerate(zip(dialogue['speakers'], dialogue['utterances'])):
            if i % 2 == 0:  # User turns
                output += f"Turn {i+1} [{speaker}]: {utterance}\n"
                if i+1 < len(dialogue['utterances']):
                    word_count = len(dialogue['utterances'][i+1].split())
                    output += f"[Target response: {word_count} words]\n"
                    target_word_counts.append(f"Turn {i+2}: {word_count} words")
                    system_turns.append(i+2)  # Track which turns we're predicting
        
        system_turns_str = ", ".join([f"Turn {turn}" for turn in system_turns])
        
        prompt = f"""Given this conversation showing only user turns and target response word counts,
        predict ALL of the system's responses in order.
        Use the word count patterns to help infer appropriate response complexity.

        PREDICTING: {system_turns_str} (All system responses)

        Context:
        {output}

        Format your response EXACTLY as shown:
        ```json
        [
            "System response for Turn 2",
            "System response for Turn 4",
            "System response for Turn 6"
        ]
        ```"""
        return prompt

    def create_enhanced_prompt(self, base_prompt: str, scenario: str, use_few_shot: bool = True) -> str:
        """Create enhanced prompt with optional few-shot examples and detailed instructions."""
    
        # Get enhanced few-shot examples for this scenario (if enabled)
        examples = ENHANCED_FEW_SHOT_EXAMPLES.get(scenario, "") if use_few_shot else ""
        
        # Remove XXXXXXX from examples if anti-hallucination is disabled
        if self.disable_anti_hallucination and examples:
            examples = examples.replace('XXXXXXX', '[SPECIFIC_INFO]')

        # Core instructions - modify based on anti-hallucination setting
        if self.disable_anti_hallucination:
            core_instructions = """
CRITICAL INSTRUCTIONS FOR DIALOGUE COMPLETION:
1. PREDICT THE EXACT SYSTEM RESPONSE that would naturally follow in this conversation
2. PRESERVE ALL SPECIFIC DETAILS: times, dates, names, locations, numbers, reference codes, prices, phone numbers
3. Provide specific and realistic information when needed to make the response helpful
4. Maintain the same information density and factual accuracy as expected
5. Match the tone and style of the conversation
6. Focus on providing the most relevant and complete information
7. You may use future turns (after the prediction turn) as background context to improve accuracy, but you must NOT explicitly include, mention, or preempt any new facts, topics, or requests that appear only in those future turns in your actual prediction.

TASK: You are predicting what the system would say next in a natural conversation.
Your response should be informative, specific, and helpful to the user."""
        else:
            core_instructions = """
CRITICAL INSTRUCTIONS FOR DIALOGUE COMPLETION:
1. PREDICT THE EXACT SYSTEM RESPONSE that would naturally follow in this conversation
2. PRESERVE ALL SPECIFIC DETAILS: times, dates, names, locations, numbers, reference codes, prices, phone numbers
3. ANTI-HALLUCINATION: Use 'XXXXXXX' for ALL specific information not available in the context that you need to provide (names, numbers, addresses, phone numbers, prices, times, etc.)
4. Maintain the same information density and factual accuracy as expected
5. Match the tone and style of the conversation
6. Include exact facts and specific information with XXXXXXX when relevant
7. Focus on providing the most relevant and complete information
8. You may use future turns (after the prediction turn) as background context to improve accuracy, but you must NOT explicitly include, mention, or preempt any new facts, topics, or requests that appear only in those future turns in your actual prediction.

TASK: You are predicting what the system would say next in a natural conversation.
Your response should be informative, specific, and helpful to the user."""
        
        # Add scenario-specific word count guidance
        if scenario in {"scenario_2_1", "scenario_2_2"}:
            core_instructions += "\nNOTE: [MASKED - n words] indicates the expected length of the response.\n"
            
        
        # Build the complete enhanced prompt
        examples_section = f"\n{examples}\n" if examples.strip() else "\n[FEW-SHOT EXAMPLES DISABLED]\n"
        # Build final rules section based on anti-hallucination setting
        if self.disable_anti_hallucination:
            rules_section = """RULES:
• Generate the exact system response that would naturally follow
• Preserve all specific details (times, names, locations, numbers, reference codes, prices)
• Provide realistic and helpful specific information when needed
• Maintain factual accuracy and information completeness
• Match the conversational style and tone
• Do NOT add commentary, labels, or extra text
• Do NOT preface with 'assistant:' or similar
• You may use future turns (after the prediction turn) as background context to improve accuracy, but you must NOT explicitly include, mention, or preempt any new facts, topics, or requests that appear only in those future turns in your actual prediction.
• Focus on providing the most relevant and specific information
• Be helpful and informative to the user"""
        else:
            rules_section = """RULES:
• Generate the exact system response that would naturally follow
• Preserve all specific details (times, names, locations, numbers, reference codes, prices)
• ANTI-HALLUCINATION: Use 'XXXXXXX' for any specific information not available in the context
• Maintain factual accuracy and information completeness
• Match the conversational style and tone
• Do NOT add commentary, labels, or extra text
• Do NOT preface with 'assistant:' or similar
• You may use future turns (after the prediction turn) as background context to improve accuracy, but you must NOT explicitly include, mention, or preempt any new facts, topics, or requests that appear only in those future turns in your actual prediction.
• Focus on providing the most relevant and specific information
• Be helpful and informative to the user"""

        enhanced_prompt = f"""DIALOGUE COMPLETION TASK — {self._desc_for(scenario)}

{core_instructions}
{examples_section}
=== BEGIN CONVERSATION ===
{base_prompt}
=== END CONVERSATION ===

{rules_section}"""
        
        return enhanced_prompt

    def _desc_for(self, scenario: str) -> str:
        """Get description for scenario - from generate_predictions_enhanced.py"""
        DESC = {
            "scenario_1_1": "System immediate reply",
            "scenario_1_2": "Reply + future user turn",
            "scenario_2_1": "Hints",
            "scenario_2_2": "Hints + future turn",
            "scenario_3_1": "Full masked",
            "scenario_3_2": "Full masked + lengths",
            "scenario_4_nplus2": "Predict given 2 turns ahead",
            "scenario_4_nplus3": "Predict given 3 turns ahead",
            "scenario_4_nplus4": "Predict given 4 turns ahead",
            "scenario_4_nplus5": "Predict given 5 turns ahead",
            "scenario_5": "Predict current turn, past predictions",
        }
        return DESC.get(scenario, scenario)

    async def get_claude_prediction(self, prompt: str) -> str:
        """Get prediction from Claude with proper rate limiting"""
        # Build system prompt based on disable_anti_hallucination setting
        if self.disable_anti_hallucination:
            system_prompt = """You are a dialogue predictor for task-oriented conversations.

Your role is to predict direct, helpful system responses that:
- Address the user's needs clearly and stay focused on the task
- Include ONLY concrete information shown in the conversation context
- Provide realistic and helpful specific information when needed
- Match the conversation style and tone naturally

CRITICAL: Respond with ONLY the direct system response. Never include:
- Explanations or reasoning
- Meta-commentary about the conversation  
- Markdown formatting or delimiters
- Prefixes like 'assistant:' or similar"""
        else:
            system_prompt = """You are a dialogue predictor for task-oriented conversations.

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
        
        # If token counting only, count tokens and return mock response
        if self.token_count_only and self.token_counter:
            self.token_counter.count_claude_tokens(prompt, system_prompt, 'prediction_generation')
            return "Mock prediction response for token counting"
        
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
                        messages=[{"role": "user", "content": prompt}]
                    )
                message = await asyncio.to_thread(_call)
                self.claude_api_calls += 1  # Increment counter after successful call
                logging.info(f"Claude API call completed at {time.time():.3f} (Total: {self.claude_api_calls})")
                return message.content[0].text
            except Exception as e:
                logging.error(f"Error calling Claude API: {e}")
                return None

    async def get_prediction_and_evaluation(self, prompt: str, actual: str, context: str, skip_evaluation: bool = False) -> Dict[str, Any]:
        """Get Claude prediction and OpenAI evaluation sequentially"""
        prediction = None
        evaluation = None
        
        try:
            # Get the prediction first
            prediction = await self.get_claude_prediction(prompt)
            
            if not prediction:
                logging.warning("Failed to get Claude prediction")
                return {
                    "prediction": None,
                    "evaluation": None
                }
            
            # In token counting mode, always run evaluation to count tokens (but don't skip based on skip_evaluation)
            if skip_evaluation and not self.token_count_only:
                return {
                    "prediction": prediction,
                    "evaluation": None
                }
            
            # Run evaluation sequentially (will count tokens in token_count_only mode)
            try:
                evaluation = await self.evaluate_prediction(prediction, actual, context)
                if evaluation is None:
                    logging.warning("Failed to get OpenAI evaluation")
            except Exception as eval_error:
                logging.error(f"Error in evaluation: {eval_error}")
                evaluation = None
            
            return {
                "prediction": prediction,
                "evaluation": evaluation
            }
            
        except Exception as e:
            logging.error(f"Error in combined prediction and evaluation: {e}")
            return {
                "prediction": prediction,
                "evaluation": evaluation
            }

    async def evaluate_prediction(self, predicted: str, actual: str, context: str = "") -> Dict[str, Any]:
        """Evaluate prediction using ROUGE metrics and GPT-5 for semantic analysis"""
        try:
            # Calculate ROUGE scores
            rouge_scores = self.rouge.score(predicted, actual)
            
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
            3. **XXXXXXX Masking Compliance** – Does the prediction correctly mask all specific details that weren't previously mentioned in the conversation with "XXXXXXX"? 
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

            ## XXXXXXX Masking Analysis
            Count masking behavior by comparing against the conversation context:
            - **actual_specific_info_count**: How many pieces of specific info are in the actual response that are NOT available in the previous conversation context
            - **xxx_used_count**: How many times does the predicted response use "XXXXXXX" to mask unavailable details

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

            # If token counting only, count tokens and return mock evaluation
            if self.token_count_only and self.token_counter:
                system_prompt = "You are an expert in dialogue analysis. Respond with valid JSON only."
                self.token_counter.count_chatgpt_tokens(evaluation_prompt, system_prompt, 'turn_prediction_rubric')
                
                # Return mock evaluation for token counting
                return {
                    "rouge_scores": metrics,
                    "semantic_similarity": {
                        "semantic_similarity": 3,
                        "intent_preservation": 3,
                        "precision": 0.5,
                        "recall": 0.5,
                        "anti_hallucination_score": 3,
                        "contextual_appropriateness": 3,
                        "summary_alignment": 3,
                        "actual_specific_info_count": 0,
                        "xxx_used_count": 0,
                        "reasoning_details": {"mock": "token_counting_mode"},
                        "detail_extraction": {"mock": "token_counting_mode"}
                    },
                    "evaluation_prompt": "Mock evaluation for token counting"
                }

            # Use OpenAI call with proper rate limiting
            async with self.openai_sem:
                await self.openai_rate_limiter.acquire()
                logging.info(f"OpenAI API call starting at {time.time():.3f}")
                try:
                    def _call():
                        return self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are an expert in dialogue analysis. Respond with valid JSON only."},
                                {"role": "user", "content": evaluation_prompt}
                            ]
                        )
                    response = await asyncio.to_thread(_call)
                    self.openai_api_calls += 1  # Increment counter after successful call
                    logging.info(f"OpenAI API call completed at {time.time():.3f} (Total: {self.openai_api_calls})")
                        
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
                                    json_part += '},"reasoning_and_scores":{"semantic_similarity_reasoning":"truncated","semantic_similarity":3,"intent_preservation_reasoning":"truncated","intent_preservation":3,"xxx_masking_compliance_reasoning":"truncated","xxx_masking_compliance":3,"contextual_appropriateness_reasoning":"truncated","contextual_appropriateness":3,"summary_alignment_reasoning":"truncated","summary_alignment":3},"analysis_counts":{"actual_specific_info_count":0,"xxx_used_count":0}}'
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
                                                '"intent_preservation_reasoning"', '"xxx_masking_compliance_reasoning"', 
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
                "anti_hallucination_score": semantic_eval["reasoning_and_scores"]["xxx_masking_compliance"],  # Map XXXXXXX masking compliance to expected field name
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
            if scenario == "scenario_3" and "all_turns" in pred:
                # Flatten scenario 3 all_turns into individual predictions for metrics calculation
                for turn_eval in pred["all_turns"]:
                    individual_pred = {
                        "dialogue_id": pred["dialogue_id"],
                        "turn": turn_eval["turn"],
                        "scenario": scenario,
                        "evaluation": turn_eval["evaluation"]
                    }
                    scenarios[scenario].append(individual_pred)
            else:
                scenarios[scenario].append(pred)
        
        # Calculate metrics for each scenario
        metrics = {}
        for scenario, preds in scenarios.items():
            # Get unique dialogue IDs and total turns
            unique_dialogues = len(set(p["dialogue_id"] for p in preds))
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
                if p.get("evaluation") and "prediction" in p and "actual" in p:
                    pred_words = len(p["prediction"].split())
                    actual_words = len(p["actual"].split())
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

    async def process_dialogue_batch(self, dialogues: List[Dict], start_idx: int, selected_scenario: str = None, skip_evaluation: bool = False) -> List[Dict]:
        """Process a batch of dialogues concurrently"""
        results, tasks = [], []

        for i, dialogue in enumerate(dialogues, start=start_idx):
            dialogue["dialogue_id"] = i
            for turn_n in range(1, len(dialogue['utterances']), 2):
                actual = dialogue['utterances'][turn_n]
                scenarios = [
                    ("scenario_1_1", self.prepare_scenario_1_1),
                    ("scenario_1_2", self.prepare_scenario_1_2),
                    ("scenario_2_1", self.prepare_scenario_2_1),
                    ("scenario_2_2", self.prepare_scenario_2_2),
                ]
                if selected_scenario:
                    scenarios = [(n, f) for n, f in scenarios if n == selected_scenario]

                for scenario_name, prepare_fn in scenarios:
                    async def one_job(dialogue=dialogue, turn_n=turn_n, actual=actual,
                                      scenario_name=scenario_name, prepare_fn=prepare_fn):
                        try:
                            prompt = prepare_fn(dialogue, turn_n)
                            context = self.format_full_dialogue_context(
                                dialogue, turn_n,
                                include_future_context=(scenario_name in ["scenario_1_2", "scenario_2_2"])
                            )
                            combined = await self.get_prediction_and_evaluation(
                                prompt, actual, context, skip_evaluation
                            )
                            if not combined or not combined.get("prediction"):
                                return None

                            item = {
                                "dialogue_id": dialogue["dialogue_id"],
                                "turn": turn_n,
                                "scenario": scenario_name,
                                "full_prompt": prompt,
                                "prediction": combined["prediction"],
                                "actual": actual
                            }
                            if combined.get("evaluation"):
                                pred = combined["prediction"]
                                item["evaluation"] = {
                                    **combined["evaluation"],
                                    "duration_metrics": {
                                        "predicted_words": len(pred.split()),
                                        "actual_words": len(actual.split())
                                    }
                                }
                            elif skip_evaluation:
                                pred = combined["prediction"]
                                item["basic_metrics"] = {
                                    "predicted_words": len(pred.split()),
                                    "actual_words": len(actual.split()),
                                    "word_difference": abs(len(pred.split()) - len(actual.split()))
                                }
                            return item
                        except Exception as e:
                            logging.error(f"Error processing dialogue {dialogue['dialogue_id']}, turn {turn_n}, scenario {scenario_name}: {e}")
                            return None

                    tasks.append(asyncio.create_task(one_job()))

        for finished in await asyncio.gather(*tasks, return_exceptions=False):
            if finished:
                results.append(finished)
        return results

    async def process_single_dialogue(self, dialogue: Dict, dialogue_idx: int, selected_scenario: str = None) -> Dict:
        """Process a single dialogue through selected or all scenarios"""
        try:
            dialogue_results = []
            
            # Process scenarios 1.x and 2.x
            for turn_n in range(1, len(dialogue['utterances']), 2):
                actual = dialogue['utterances'][turn_n]
                
                # Define available scenarios
                scenarios = [
                    ("scenario_1_1", self.prepare_scenario_1_1),
                    ("scenario_1_2", self.prepare_scenario_1_2),
                    ("scenario_2_1", self.prepare_scenario_2_1),
                    ("scenario_2_2", self.prepare_scenario_2_2)
                ]
                
                # Filter scenarios if one is selected
                if selected_scenario:
                    scenarios = [(name, fn) for name, fn in scenarios if name == selected_scenario]
                
                for scenario_name, prepare_fn in scenarios:
                    prompt = prepare_fn(dialogue, turn_n)
                    prediction = await self.get_claude_prediction(prompt)
                    
                    if prediction:
                        # Generate full context for evaluation (no masking)
                        context = self.format_full_dialogue_context(dialogue, turn_n, 
                                                                   include_future_context=(scenario_name in ["scenario_1_2", "scenario_2_2"]))
                        
                        evaluation = await self.evaluate_prediction(prediction, actual, context)
                        if evaluation:
                            duration_metrics = {
                                "predicted_words": len(prediction.split()),
                                "actual_words": len(actual.split())
                            }
                            
                            dialogue_results.append({
                                "dialogue_id": dialogue_idx,
                                "turn": turn_n,
                                "scenario": scenario_name,
                                "full_prompt": prompt,  # Include the complete enhanced prompt
                                "prediction": prediction,
                                "actual": actual,
                                "evaluation": {
                                    **evaluation,
                                    "duration_metrics": duration_metrics
                                }
                            })
            
            # Process scenario 3 only if no specific scenario is selected or it is the selected one
            if not selected_scenario or selected_scenario == "scenario_3":
                prompt = self.prepare_scenario_3(dialogue)
                response = await self.get_claude_prediction(prompt)
                
                if response:
                    try:
                        predictions_list = self.extract_predictions(response)
                        actual_turns = [utt for i, utt in enumerate(dialogue['utterances']) if i % 2 == 1]
                        
                        if predictions_list and len(predictions_list) > 0:
                            scenario_3_evaluations = []
                            
                            for turn_idx, (pred, actual) in enumerate(zip(predictions_list, actual_turns)):
                                if isinstance(pred, str):
                                    turn_number = turn_idx * 2 + 1
                                    
                                    # Generate full context for this specific turn (no masking)
                                    # Note: scenario_3 doesn't use future context
                                    turn_context = self.format_full_dialogue_context(dialogue, turn_number, include_future_context=False)
                                    
                                    evaluation = await self.evaluate_prediction(pred, actual, turn_context)
                                    
                                    if evaluation:
                                        duration_metrics = {
                                            "predicted_words": len(pred.split()),
                                            "actual_words": len(actual.split())
                                        }
                                        
                                        scenario_3_evaluations.append({
                                            "turn": turn_number,
                                            "prediction": pred,
                                            "actual": actual,
                                            "evaluation": {
                                                **evaluation,
                                                "duration_metrics": duration_metrics
                                            }
                                        })
                            
                            if scenario_3_evaluations:
                                dialogue_results.append({
                                    "dialogue_id": dialogue_idx,
                                    "scenario": "scenario_3",
                                    "full_prompt": prompt,  # Include the complete prompt for scenario 3
                                    "all_turns": scenario_3_evaluations,
                                    "summary": {
                                        "total_turns": len(scenario_3_evaluations),
                                        "avg_semantic_score": sum(e["evaluation"]["semantic_similarity"]["semantic_similarity"] for e in scenario_3_evaluations) / len(scenario_3_evaluations),
                                        "avg_rouge_l": sum(e["evaluation"]["rouge_scores"]["rougeL"] for e in scenario_3_evaluations) / len(scenario_3_evaluations)
                                    }
                                })
                    
                    except Exception as e:
                        logging.error(f"Error processing scenario 3 for dialogue {dialogue_idx}: {e}")
            
            return dialogue_results
            
        except Exception as e:
            logging.error(f"Error processing dialogue {dialogue_idx}: {e}")
            return None

    def extract_predictions(self, response: str) -> List[str]:
        """Extract predictions from Claude's response"""
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
                
            predictions = json.loads(json_str)
            if not isinstance(predictions, list):
                raise ValueError("Response is not a JSON array")
            return predictions
        except Exception as e:
            logging.error(f"Failed to extract predictions: {e}")
            return []

    async def run_evaluation(self, dataset_name: str, num_samples: int = 5, selected_scenario: str = None, skip_evaluation: bool = False, test_split_file: str = None, test_split_file_txt: str = None):
        """Run evaluation for all or selected scenario"""
        # Force logging configuration
        import sys
        
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s.%(msecs)03d %(message)s',
            datefmt='%H:%M:%S',
            stream=sys.stdout,
            force=True
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        mode = "predictions_only" if skip_evaluation else "full_evaluation"
        # Determine dataset name for metadata
        if test_split_file_txt:
            dataset_for_metadata = Path(test_split_file_txt).stem
        elif test_split_file:
            dataset_for_metadata = Path(test_split_file).stem
        else:
            dataset_for_metadata = dataset_name
            
        results = {
            "metadata": {
                "dataset": dataset_for_metadata,
                "num_samples": num_samples,
                "timestamp": datetime.now().isoformat(),
                "selected_scenario": selected_scenario,
                "concurrent_requests": MAX_CONCURRENT_REQUESTS,
                "batch_size": BATCH_SIZE,
                "mode": mode,
                "skip_evaluation": skip_evaluation,
                "test_split_file": test_split_file,
                "test_split_file_txt": test_split_file_txt,
                "api_call_counts": {
                    "claude_api_calls": 0,  # Will be updated at the end
                    "openai_api_calls": 0   # Will be updated at the end
                }
            },
            "predictions": []
        }
        
        try:
            # Load dialogues from test split file (JSON or TXT) or regular dataset
            if test_split_file_txt:
                print(f"Loading dialogues from test split TXT file: {test_split_file_txt}")
                logger.info(f"Loading dialogues from test split TXT file: {test_split_file_txt}")
                dialogues = self.load_test_split_txt(test_split_file_txt)
                
                # Apply num_samples limit if specified (0 or -1 means process all)
                if num_samples and num_samples > 0 and len(dialogues) > num_samples:
                    dialogues = dialogues[:num_samples]
                    print(f"Limited to first {num_samples} dialogues from TXT test split")
                    logger.info(f"Limited to first {num_samples} dialogues from TXT test split")
                elif num_samples <= 0:
                    print(f"Processing all {len(dialogues)} dialogues from TXT test split")
                    logger.info(f"Processing all {len(dialogues)} dialogues from TXT test split")
            elif test_split_file:
                print(f"Loading dialogues from test split file: {test_split_file}")
                logger.info(f"Loading dialogues from test split file: {test_split_file}")
                dialogues = self.load_test_split(test_split_file)
                
                # Apply num_samples limit if specified (0 or -1 means process all)
                if num_samples and num_samples > 0 and len(dialogues) > num_samples:
                    dialogues = dialogues[:num_samples]
                    print(f"Limited to first {num_samples} dialogues from test split")
                    logger.info(f"Limited to first {num_samples} dialogues from test split")
                elif num_samples <= 0:
                    print(f"Processing all {len(dialogues)} dialogues from test split")
                    logger.info(f"Processing all {len(dialogues)} dialogues from test split")
            else:
                print(f"Loading {dataset_name} dataset...")
                logger.info(f"Loading {dataset_name} dataset...")
                dialogues = self.load_dataset(dataset_name, max_dialogues=num_samples)
            
            if not dialogues:
                if test_split_file_txt:
                    error_msg = f"No dialogues found in TXT test split file: {test_split_file_txt}"
                elif test_split_file:
                    error_msg = f"No dialogues found in test split file: {test_split_file}"
                else:
                    error_msg = f"No dialogues found in {dataset_name} dataset"
                print(error_msg)
                logger.error(error_msg)
                return results
                
            scenario_desc = f"scenario {selected_scenario}" if selected_scenario else "all scenarios"
            mode_desc = "predictions only (fast)" if skip_evaluation else "predictions + evaluation (full)"
            if test_split_file_txt:
                source_desc = f"TXT test split ({test_split_file_txt})"
            elif test_split_file:
                source_desc = f"test split ({test_split_file})"
            else:
                source_desc = f"{dataset_name} dataset"
            print(f"Processing {len(dialogues)} dialogues from {source_desc} for {scenario_desc} - {mode_desc}")
            logger.info(f"Processing {len(dialogues)} dialogues from {source_desc} for {scenario_desc} - {mode_desc}")
            logger.info(f"🚀 OPTIMIZED PERFORMANCE SETTINGS:")
            logger.info(f"   Total concurrent requests: {MAX_CONCURRENT_REQUESTS}")
            logger.info(f"   Claude concurrent: {MAX_CONCURRENT_REQUESTS // 2}, rate limit: {CLAUDE_RATE_LIMIT_CALLS}/{CLAUDE_RATE_LIMIT_WINDOW}s")
            logger.info(f"   OpenAI concurrent: {MAX_CONCURRENT_REQUESTS // 2}, rate limit: {OPENAI_RATE_LIMIT_CALLS}/{OPENAI_RATE_LIMIT_WINDOW}s")
            logger.info(f"   Batch size: {BATCH_SIZE}")
            
            # Process all dialogues sequentially for stability
            print(f"Processing {len(dialogues)} dialogues...")
            logger.info(f"Processing {len(dialogues)} dialogues sequentially...")
            
            for batch_start in range(0, len(dialogues), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(dialogues))
                batch = dialogues[batch_start:batch_end]
                
                print(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(len(dialogues)-1)//BATCH_SIZE + 1}")
                logger.info(f"\nProcessing batch {batch_start//BATCH_SIZE + 1}/{(len(dialogues)-1)//BATCH_SIZE + 1}")
                batch_result = await self.process_dialogue_batch(batch, batch_start, selected_scenario, skip_evaluation)
                
                if batch_result:
                    results["predictions"].extend(batch_result)
                
                # No delay between batches - rate limiting handled by RateLimiter class
            
            # Results are already combined above in the sequential processing
            
            # Calculate aggregate metrics only if evaluation was done
            if not skip_evaluation:
                scenario_metrics = self.calculate_scenario_metrics(results["predictions"])
                results["aggregate_metrics"] = scenario_metrics
                self.log_summary_metrics(scenario_metrics)
            else:
                logger.info(f"\n✅ Generated {len(results['predictions'])} predictions (evaluation skipped)")
            
            # Update API call counts in results metadata
            results["metadata"]["api_call_counts"]["claude_api_calls"] = self.claude_api_calls
            results["metadata"]["api_call_counts"]["openai_api_calls"] = self.openai_api_calls
            
            # Log API call summary
            total_api_calls = self.claude_api_calls + self.openai_api_calls
            logger.info(f"\n📊 API CALL SUMMARY:")
            logger.info(f"   Claude API calls: {self.claude_api_calls}")
            logger.info(f"   OpenAI API calls: {self.openai_api_calls}")
            logger.info(f"   Total API calls: {total_api_calls}")
            
            # Save results
            filename_suffix = "_predictions_only" if skip_evaluation else "_predictions"
            
            if test_split_file_txt:
                # Use TXT test split filename (without extension) for output
                test_split_name = Path(test_split_file_txt).stem  # e.g., "test_finetune" from "test_finetune.txt"
                output_file = Path(f"{test_split_name}{filename_suffix}.json")
            elif test_split_file:
                # Use test split filename (without extension) for output
                test_split_name = Path(test_split_file).stem  # e.g., "test" from "test.json"
                output_file = Path(f"{test_split_name}{filename_suffix}.json")
            else:
                # Use dataset name for regular datasets, ensure results directory exists
                output_file = Path(f"{dataset_name}{filename_suffix}.json")
                
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Output token usage summary if in token counting mode
            if self.token_count_only and self.token_counter:
                self.token_counter.log_summary()
                
                # Save token usage to file
                token_summary = self.token_counter.get_summary()
                token_filename_suffix = "_token_usage"
                
                if test_split_file_txt:
                    test_split_name = Path(test_split_file_txt).stem
                    token_output_file = Path(f"{test_split_name}{token_filename_suffix}.json")
                    prompts_output_file = Path(f"{test_split_name}_prompts_verification.txt")
                elif test_split_file:
                    test_split_name = Path(test_split_file).stem
                    token_output_file = Path(f"{test_split_name}{token_filename_suffix}.json")
                    prompts_output_file = Path(f"{test_split_name}_prompts_verification.txt")
                else:
                    token_output_file = Path(f"{dataset_name}{token_filename_suffix}.json")
                    prompts_output_file = Path(f"{dataset_name}_prompts_verification.txt")
                
                with open(token_output_file, "w") as f:
                    json.dump(token_summary, f, indent=2)
                
                # Save all prompts for verification
                self.token_counter.save_prompts_to_file(str(prompts_output_file))
                
                logger.info(f"\n✅ Token usage saved to {token_output_file}")
                logger.info(f"✅ Prompts saved to {prompts_output_file}")
            
            logger.info(f"\n✅ Results saved to {output_file}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {str(e)}", exc_info=True)
            raise

    async def rerun_evaluation_from_file(self, predictions_file: str):
        """Rerun evaluation on existing predictions file"""
        import json
        from pathlib import Path
        import sys
        
        # Force logging configuration
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s.%(msecs)03d %(message)s',
            datefmt='%H:%M:%S',
            stream=sys.stdout,
            force=True
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        print(f"🔄 Rerunning evaluation from existing predictions file: {predictions_file}")
        logger.info(f"Loading predictions from: {predictions_file}")
        
        # Load existing predictions
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading predictions file: {e}")
            logger.error(f"Error loading predictions file: {e}")
            return
        
        if "predictions" not in existing_data:
            print("❌ Invalid predictions file format - missing 'predictions' field")
            logger.error("Invalid predictions file format")
            return
        
        predictions = existing_data["predictions"]
        if not predictions:
            print("❌ No predictions found in file")
            logger.error("No predictions found in file")
            return
        
        print(f"📊 Found {len(predictions)} predictions to re-evaluate")
        logger.info(f"Found {len(predictions)} predictions to re-evaluate")
        
        # Update metadata
        existing_data["metadata"]["timestamp"] = datetime.now().isoformat()
        existing_data["metadata"]["mode"] = "evaluation_rerun"
        existing_data["metadata"]["api_call_counts"] = {
            "claude_api_calls": 0,
            "openai_api_calls": 0
        }
        
        # Process predictions in batches for evaluation
        batch_size = 10  # Smaller batches for evaluation only
        total_batches = (len(predictions) + batch_size - 1) // batch_size
        
        print(f"🚀 Processing {len(predictions)} predictions in {total_batches} batches of {batch_size}")
        logger.info(f"Processing {len(predictions)} predictions in {total_batches} batches")
        
        # Track API calls
        claude_calls = 0
        openai_calls = 0
        
        try:
            for batch_idx in range(0, len(predictions), batch_size):
                batch_end = min(batch_idx + batch_size, len(predictions))
                batch = predictions[batch_idx:batch_end]
                
                print(f"📊 Processing batch {batch_idx//batch_size + 1}/{total_batches} ({len(batch)} predictions)")
                logger.info(f"Processing batch {batch_idx//batch_size + 1}/{total_batches}")
                
                # Create evaluation tasks
                tasks = []
                for pred in batch:
                    if "prediction" not in pred or "actual" not in pred:
                        print(f"⚠️ Skipping prediction with missing data: {pred.get('dialogue_id', 'unknown')}")
                        continue
                    
                    # Create context from full_prompt if available
                    context = pred.get("full_prompt", "")
                    task = self.evaluate_prediction(pred["prediction"], pred["actual"], context)
                    tasks.append((pred, task))
                
                # Run evaluations concurrently
                if tasks:
                    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                    
                    # Update predictions with new evaluations
                    for (pred, _), result in zip(tasks, results):
                        if isinstance(result, Exception):
                            print(f"❌ Evaluation failed for prediction {pred.get('dialogue_id', 'unknown')}: {result}")
                            logger.error(f"Evaluation failed: {result}")
                        else:
                            pred["evaluation"] = result
                            # Count API calls (evaluation uses OpenAI)
                            openai_calls += 1
                
                # Small delay between batches
                if batch_idx + batch_size < len(predictions):
                    await asyncio.sleep(1)
        
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            logger.error(f"Error during evaluation: {e}")
            return
        
        # Update API call counts
        existing_data["metadata"]["api_call_counts"]["claude_api_calls"] = claude_calls
        existing_data["metadata"]["api_call_counts"]["openai_api_calls"] = openai_calls
        
        # Calculate and add aggregate metrics
        existing_data["aggregate_metrics"] = self.calculate_scenario_metrics(predictions)
        
        # Save updated results
        output_file = f"rerun_evaluation_{Path(predictions_file).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Evaluation complete! Results saved to: {output_file}")
            logger.info(f"Results saved to: {output_file}")
            
            # Log summary metrics
            self.log_summary_metrics(existing_data["aggregate_metrics"])
            
            print(f"\n📊 API Usage Summary:")
            print(f"  - Claude API calls: {claude_calls}")
            print(f"  - OpenAI API calls: {openai_calls}")
            print(f"  - Total predictions re-evaluated: {len([p for p in predictions if 'evaluation' in p])}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            logger.error(f"Error saving results: {e}")

    def log_summary_metrics(self, metrics: Dict):
        """Log summary metrics in a clean format"""
        logger = logging.getLogger(__name__)
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

def test_dialogue_context():
    """Test function to verify context formatting"""
    dialogue = {
        'speakers': ['USER', 'SYSTEM', 'USER', 'SYSTEM', 'USER'],
        'utterances': ['u1', 's1', 'u2', 's2', 'u3']
    }
    
    # Test scenario 1.2/2.2 for turn 1 (first system turn)
    evaluator = DialoguePredictionEvaluator(CLAUDE_API_KEY, OPENAI_API_KEY)
    context = evaluator.format_dialogue_context(dialogue, 1, include_next_turn=True)
    expected = "Turn 1 [USER]: u1\nTurn 3 [USER]: u2\nTurn 2 [Predict this turn : SYSTEM]: \n"
    
    print("Testing scenario 1.2/2.2 context formatting:")
    print("Got:\n", context)
    print("Expected:\n", expected)
    print("Correct:", context == expected)

# Add test call at bottom of file
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run dialogue prediction evaluation on various conversational datasets")
    parser.add_argument("--dataset", default="kandor", 
                        choices=["kandor", "meddialogue", "multiwoz", "dailydialog", "mts-dialog", "ami"], 
                        help="Dataset to use: kandor, meddialogue, multiwoz, dailydialog, mts-dialog, or ami")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of dialogues to process (use 0 or -1 to process all dialogues in test split)")
    parser.add_argument("--scenario", choices=["scenario_1_1", "scenario_1_2", "scenario_2_1", "scenario_2_2", "scenario_3"], 
                        help="Specific scenario to run (optional)")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation and only generate predictions (faster)")
    parser.add_argument("--no_few_shot", action="store_true", help="Disable few-shot examples in prompts")
    parser.add_argument("--test", action="store_true", help="Run test dialogue context formatting")
    parser.add_argument("--test-split-file", type=str, help="Path to test split JSON file (e.g., test.json)")
    parser.add_argument("--test-split-file-txt", type=str, help="Path to test split TXT file with 3-turn format (e.g., test_finetune.txt)")
    parser.add_argument("--token-count-only", action="store_true", help="Count tokens only without making API calls (fastest)")
    parser.add_argument("--rerun-evaluation", type=str, help="Path to existing predictions JSON file to rerun evaluation only (skips prediction generation)")
    parser.add_argument("--disable-anti-hallucination", action="store_true", help="Remove XXXXXXX instructions and examples from Claude prompts to measure natural hallucination (evaluation still measures anti-hallucination)")
    parser.add_argument("--max-prediction-context", type=int, default=20, help="Maximum number of previous turns to include in Claude prediction context (use -1 for no limit, default: 20)")
    parser.add_argument("--max-evaluation-context", type=int, default=10, help="Maximum number of turns (before/after) to include in GPT evaluation context (use -1 for no limit, default: 10)")
    args = parser.parse_args()
    
    # Validate that both test split arguments aren't used together
    if args.test_split_file and args.test_split_file_txt:
        parser.error("Cannot use both --test-split-file and --test-split-file-txt at the same time")
    
    # Validate that rerun-evaluation isn't used with other incompatible options
    if args.rerun_evaluation:
        if args.test_split_file or args.test_split_file_txt:
            parser.error("Cannot use --rerun-evaluation with --test-split-file or --test-split-file-txt")
        if args.skip_evaluation:
            parser.error("Cannot use --rerun-evaluation with --skip_evaluation (evaluation is the whole point)")
        if args.token_count_only:
            parser.error("Cannot use --rerun-evaluation with --token-count-only")
    
    # Convert -1 values to None for unlimited context
    max_prediction_context = None if args.max_prediction_context == -1 else args.max_prediction_context
    max_evaluation_context = None if args.max_evaluation_context == -1 else args.max_evaluation_context
    
    evaluator = DialoguePredictionEvaluator(
        claude_api_key=CLAUDE_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        use_few_shot=not args.no_few_shot,
        token_count_only=args.token_count_only,
        disable_anti_hallucination=args.disable_anti_hallucination,
        max_prediction_context_turns=max_prediction_context,
        max_evaluation_context_turns=max_evaluation_context
    )
    
    if args.test:
        test_dialogue_context()
    elif args.rerun_evaluation:
        # Run evaluation rerun mode
        print(f"[INFO] RERUN EVALUATION MODE:")
        print(f"  - 🔄 Loading existing predictions from: {args.rerun_evaluation}")
        print(f"  - 📊 Re-running evaluation only (no new predictions)")
        print(f"  - 🚀 CONCURRENT EVALUATION CALLS (OpenAI only)")
        print(f"  - ⚡ Much faster than full prediction + evaluation")
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(evaluator.rerun_evaluation_from_file(args.rerun_evaluation))
    else:
        # Run enhanced evaluation with full prompts included in output
        print(f"[INFO] Using ENHANCED Claude dialogue prediction with:")
        print(f"  - 🚀 ULTRA-FAST CONCURRENT API CALLS (up to {MAX_CONCURRENT_REQUESTS} parallel)")
        print(f"  - 📊 SEPARATE RATE LIMITERS: Claude ({CLAUDE_RATE_LIMIT_CALLS}/min) & OpenAI ({OPENAI_RATE_LIMIT_CALLS}/min)")
        print(f"  - 📈 LARGE BATCH PROCESSING ({BATCH_SIZE} dialogues per batch)")
        print(f"  - 📊 Support for: MultiWOZ, Kandor, MedDialogue, DailyDialog, MTS-Dialog, AMI")
        print(f"  - 🎯 Few-shot examples: {'ENABLED' if not args.no_few_shot else 'DISABLED'}")
        if args.test_split_file_txt:
            print(f"  - 📄 Using TXT test split file: {args.test_split_file_txt}")
        elif args.test_split_file:
            print(f"  - 📁 Using test split file: {args.test_split_file}")
        if args.token_count_only:
            print(f"  - 🔢 TOKEN COUNT ONLY MODE: No API calls, just token usage analysis")
        
        # Show context limits
        prediction_limit = "UNLIMITED" if max_prediction_context is None else f"{max_prediction_context} turns"
        evaluation_limit = "UNLIMITED" if max_evaluation_context is None else f"{max_evaluation_context} turns"
        print(f"  - 📝 Claude prediction context limit: {prediction_limit}")
        print(f"  - 📊 GPT evaluation context limit: {evaluation_limit}")
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(evaluator.run_evaluation(
            args.dataset, args.num_samples, args.scenario, args.skip_evaluation, args.test_split_file, args.test_split_file_txt
        ))