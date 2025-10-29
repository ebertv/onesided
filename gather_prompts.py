#create conversations for each scenario
#put it into the spot for the conversation wtihin the prompt
#save the full prompt sets to a file

import os, sys, json
from pathlib import Path
from typing import List, Dict, Any
import logging
from prompt_outline import BASE_INSTR, RULES, SCENARIOS, DESC, ENHANCED_FEW_SHOT_EXAMPLES
from conversational_dataloader import ConversationalDataLoader


DATA_ROOT = os.getenv("DATA_ROOT", "./data")
sys.path.append("./data")

def estimate_target_words(actual_utterance: str) -> int:
    """Estimate target word count for response, used in enhanced prompts."""
    return len(actual_utterance.split())

def should_mask_speaker(speaker: str, dataset: str) -> bool:
    """
    Determine if a speaker should be masked based on dataset conventions.
    Returns True if the speaker should be masked (i.e., is the system/assistant speaker).
    """
    dataset_lower = dataset.lower()
    speaker_lower = speaker.lower()
    
    # MultiWOZ: System/SYSTEM speakers
    if 'multiwoz' in dataset_lower:
        return speaker in ['System', 'SYSTEM', 'Speaker_2']
    
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
            speaker in ['System', 'SYSTEM', 'Assistant', 'Bot'] or
            speaker == 'Doctor' or
            speaker == 'Participant_R' or
            'system' in speaker_lower or
            'assistant' in speaker_lower or
            'bot' in speaker_lower
        )

def load_dataset(dataset_name: str, max_dialogues: int = None) -> List[Dict[str, Any]]:
    """Load dialogues using the conversational data loader"""
    logging.info(f"Loading {dataset_name} dataset...")
    
    if ConversationalDataLoader is None:
        logging.error("ConversationalDataLoader not available")
        return []
    
    try:
        # Initialize the conversational data loader
        loader = ConversationalDataLoader(DATA_ROOT)
        
        # Load dialogues from the specified dataset
        raw_dialogues = list(loader.load_dataset(dataset_name.lower()))
        
        if not raw_dialogues:
            logging.warning(f"No dialogues found for dataset: {dataset_name}")
            return []
        
        # Filter dialogues to ensure they have sufficient content
        filtered_dialogues = []
        for dialogue in raw_dialogues:
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


def format_dialogue_context(dialogue: Dict[str, Any], turn_n: int, include_next_turn: bool = False, show_durations: bool = False) -> str:
    """
    Format dialogue context with masked system turns and optional word counts
    Args:
        dialogue: The dialogue dictionary
        turn_n: Current turn number we're predicting
        include_next_turn: Whether to include next turn
        show_durations: Whether to show word counts for masked turns (renamed for compatibility)
    """
    output = ""
    # Format all turns up to turn_n
    for i, (speaker, utterance) in enumerate(zip(dialogue['speakers'][:turn_n], dialogue['utterances'][:turn_n])):
        output += f"Turn {i+1} [{speaker}]: "
        
        # Determine if this turn should be masked based on speaker and dataset
        should_mask = should_mask_speaker(speaker, dialogue.get('dataset', ''))
        
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

def format_full_dialogue_context(dialogue: Dict[str, Any], turn_n: int, include_future_context: bool = False) -> str:
    """
    Format full dialogue context WITHOUT masking for evaluation purposes.
    Shows all actual utterances so the evaluator can determine what info was available.
    
    Args:
        dialogue: The dialogue dictionary
        turn_n: The turn number being predicted (0-indexed)
        include_future_context: Whether to include turns after the predicted turn (for scenarios 1.2, 2.2)
    """
    output = ""
    
    # Format all turns up to turn_n (the turn we're predicting)
    for i, (speaker, utterance) in enumerate(zip(dialogue['speakers'][:turn_n], dialogue['utterances'][:turn_n])):
        output += f"Turn {i+1} [{speaker}]: {utterance}\n"
    
    # Add marker for the turn being predicted
    if turn_n < len(dialogue['speakers']):
        predicted_speaker = dialogue['speakers'][turn_n]
        output += f"Turn {turn_n+1} [PREDICTING: {predicted_speaker}]: <-- This turn is being predicted\n"
    
    # Add future context if requested (for scenarios 1.2, 2.2)
    if include_future_context and turn_n + 1 < len(dialogue['utterances']):
        output += "\n--- Future Context (available to model during prediction) ---\n"
        for i in range(turn_n + 1, len(dialogue['utterances'])):
            speaker = dialogue['speakers'][i]
            utterance = dialogue['utterances'][i]
            output += f"Turn {i+1} [{speaker}]: {utterance}\n"
    
    return output

def prepare_scenario_1_1(dialogue: Dict[str, Any], turn_n: int, use_few_shot: bool) -> str:
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
    context = format_dialogue_context(dialogue, turn_n, show_durations=False)
    
    # Calculate the turn number being predicted (1-indexed)
    turn_number = turn_n + 1
    
    base_prompt = f"""Given this conversation context with masked system responses, predict the system's next response.
    
Context:
{context}"""
    
    # Use enhanced prompt with examples
    return create_enhanced_prompt(base_prompt, "scenario_1_1", use_few_shot=use_few_shot)

def prepare_scenario_1_2(dialogue: Dict[str, Any], turn_n: int, use_few_shot: bool) -> str:
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
    context = format_dialogue_context(dialogue, turn_n, include_next_turn=True, show_durations=False)
    
    # Get the next user turn number for clarity
    next_turn_num = turn_n + 2  # +1 for next turn, +1 for 1-indexing
    turn_number = turn_n + 1    # The turn we're predicting (1-indexed)
    
    base_prompt = f"""Given this conversation context which includes the next user turn,
predict what the system said that led to that user response.
    
PREDICTING: Turn {turn_number} (System response)
GIVEN: Turn {next_turn_num} (Next user response to help predict backwards)

IMPORTANT: You are given the user's NEXT response (Turn {next_turn_num}) to help you predict 
what the system must have said to cause that specific user reaction.

Work backwards from the user's next response to determine what system response 
would logically lead to it.

Context:
{context}"""
    
    # Use enhanced prompt with examples
    return create_enhanced_prompt(base_prompt, "scenario_1_2", use_few_shot=use_few_shot)

def prepare_scenario_2_1(dialogue: Dict[str, Any], turn_n: int, use_few_shot: bool) -> str:
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
    context = format_dialogue_context(dialogue, turn_n, show_durations=True)
    
    # Get word count of the turn we're predicting (turn_n)
    if turn_n < len(dialogue['utterances']):
        target_words = estimate_target_words(dialogue['utterances'][turn_n])
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
    return create_enhanced_prompt(base_prompt, "scenario_2_1", use_few_shot=use_few_shot)

def prepare_scenario_2_2(dialogue: Dict[str, Any], turn_n: int, use_few_shot: bool) -> str:
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
    context = format_dialogue_context(dialogue, turn_n, include_next_turn=True, show_durations=True)
    
    # Get word count of the turn we're predicting (turn_n)
    if turn_n < len(dialogue['utterances']):
        target_words = estimate_target_words(dialogue['utterances'][turn_n])
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
    return create_enhanced_prompt(base_prompt, "scenario_2_2", use_few_shot=use_few_shot)

def prepare_scenario_3(dialogue: Dict[str, Any]) -> str:
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

def create_enhanced_prompt(base_prompt: str, scenario: str, use_few_shot: bool = True) -> str:
    """Create enhanced prompt with optional few-shot examples and detailed instructions."""
    
    # Get enhanced few-shot examples for this scenario (if enabled)
    examples = ENHANCED_FEW_SHOT_EXAMPLES.get(scenario, "") if use_few_shot else ""
    
    # Add scenario-specific word count guidance
    if scenario in {"scenario_2_1", "scenario_2_2"}:
        base_instr = BASE_INSTR + "\nNOTE: [MASKED - n words] indicates the expected length of the response.\n"
    else:
        base_instr = BASE_INSTR
        
    
    # Build the complete enhanced prompt
    examples_section = f"\n{examples}\n" if examples.strip() else "\n[FEW-SHOT EXAMPLES DISABLED]\n"
    enhanced_prompt = f"DIALOGUE COMPLETION TASK — {_desc_for(scenario)}\n\n{base_instr}\n{examples_section}\n=== BEGIN CONVERSATION ===\n{base_prompt}\n=== END CONVERSATION ===\n\n{RULES}"
    
    return enhanced_prompt

def _desc_for(scenario: str) -> str:
    return DESC.get(scenario, scenario)

def load_test_split(test_split_file: str) -> List[Dict[str, Any]]:
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

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Generate prompts for one-sided dialogue scenarios.")
    parser.add_argument("--datasets", type=str, nargs="+", default=["multiwoz", "kandor", "dailydialog"],
                        help="Datasets to load (e.g., multiwoz, kandor, dailydialog, meddialogue, ami). By default, all three of: multiwoz, kandor, dailydialog")
    parser.add_argument("--max_dialogues", type=int, default=None, help="Maximum number of dialogues to load per dataset. If not set, load all available dialogues.")
    parser.add_argument("--start_turn", type=int, default=0, help="Turn index to start predictions from (0-indexed). Default is 0 (first turn).")
    parser.add_argument("--num_target_turns", type=int, default=1, help="Number of target turns to generate per dialogue. Default is 1.")
    parser.add_argument("--all_turns", action="store_true", help="Generate prompts for all eligible turns in each dialogue. If not set, only the first N turns as specified by --num_target_turns are used.")
    parser.add_argument("--use_few_shot", action="store_true", help="Use few-shot examples in prompts.")
    parser.add_argument("--save_prompts", action="store_true", help="Whether to save prompts to a file (JSONL format). If not set, prompts are not saved.")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path to save prompts (JSONL format). If not specified, the file is named prompts_<ARGS>.jsonl.")
    parser.add_argument("--scenarios", type=str, nargs="+", default=SCENARIOS,
                        help=f"Scenarios to generate prompts for (e.g., scenario_1_1, scenario_1_2, scenario_2_1, scenario_2_2, scenario_3). By default, all scenarios: {SCENARIOS}")
    parser.add_argument("--input_file", type=str, default=None, help="Input file path to load dialogues from (JSON format). If not specified, dialogues are loaded from datasets.")
    
    args = parser.parse_args()

    if args.save_prompts and not args.output_file:
        if not args.input_file:
            args.output_file = f"prompts_{'_'.join(args.datasets)}_start{args.start_turn}_{'allturns' if args.all_turns else f'nturns{args.num_target_turns}'}_{f'{args.max_dialogues}dialogues' if args.max_dialogues else ''}.jsonl"
        else:
            input_base = Path(args.input_file).stem
            input_dataset = str(Path(args.input_file).parent).split('/')[-1]
            args.output_file = f"prompts_{input_base}.jsonl"
            if args.scenarios == ["scenario_3"]:
                args.output_file = f'prompts_{input_dataset}_fullconvo.jsonl'
            args.output_file = f'{input_dataset}_{args.output_file}'
    output_file = Path(args.output_file) if args.output_file else None
    print(f"[INFO] Output file: {output_file}" if output_file else "[INFO] No output file specified, prompts will not be saved.")

    if args.input_file:
        # Load dialogues from the specified input file
        print(f"[INFO] Loading dialogues from input file: {args.input_file}")
        all_dialogues = load_test_split(args.input_file)  
        if args.max_dialogues:
            all_dialogues = all_dialogues[:args.max_dialogues]
    else:
        all_dialogues = []
        for ds in args.datasets:
            dialogues = load_dataset(ds, max_dialogues=args.max_dialogues)
            all_dialogues.extend(dialogues)
    
    if not all_dialogues:
        print("[ERROR] No dialogues loaded. Exiting.")
        sys.exit(1)
    
    # Generate prompts for each dialogue and turn
    if output_file and output_file.exists():
        print(f"[WARN] Output file {output_file} already exists and will be overwritten.")
        proceed = input("Type 'yes' to proceed and overwrite, or anything else to cancel: ")
        if proceed.lower() != 'yes':
            print("Operation cancelled by user.")
            sys.exit(0)
        output_file.unlink()
    prompts = []
    for dlg_idx, dlg in tqdm(enumerate(all_dialogues), total=len(all_dialogues), desc="Generating prompts"):
        full_dialogue = ""
        for i, (speaker, utterance) in enumerate(zip(dlg['speakers'], dlg['utterances'])):
            full_dialogue += f"Turn {i+1} [{speaker}]: {utterance}\n"

        for scen in args.scenarios:
            # Determine candidate turns based on dataset conventions
            if dlg['dataset'].lower() == 'multiwoz':
                candidate_turns = [i for i, spk in enumerate(dlg['speakers'][args.start_turn:], args.start_turn) if spk in ['System', 'SYSTEM', 'Speaker_2']]
            elif dlg['dataset'].lower() == 'kandor':
                candidate_turns = [i for i, spk in enumerate(dlg['speakers'][args.start_turn:], args.start_turn) if spk == 'Participant_R']
            elif dlg['dataset'].lower() == 'dailydialog':
                candidate_turns = [i for i in range(args.start_turn, len(dlg['speakers'])) if i % 2 == 1]
            else:
                candidate_turns = [i for i in range(args.start_turn, len(dlg['speakers'])) if should_mask_speaker(dlg['speakers'][i], dlg.get('dataset', ''))]
            
            if not candidate_turns:
                print(f"[WARN] No candidate turns found for dialogue {dlg.get('dialogue_id', dlg_idx)} in dataset {dlg.get('dataset', 'unknown')}")
                continue
            
            target_turns = candidate_turns if args.all_turns else candidate_turns[:args.num_target_turns]
            
            if scen.startswith("scenario_1") or scen.startswith("scenario_2"):
                for turn_n in target_turns:
                    if scen == "scenario_1_1":
                        prompt_text = prepare_scenario_1_1(dlg, turn_n, use_few_shot=args.use_few_shot)
                    elif scen == "scenario_1_2":
                        prompt_text = prepare_scenario_1_2(dlg, turn_n, use_few_shot=args.use_few_shot)
                    elif scen == "scenario_2_1":
                        prompt_text = prepare_scenario_2_1(dlg, turn_n, use_few_shot=args.use_few_shot)
                    elif scen == "scenario_2_2":
                        prompt_text = prepare_scenario_2_2(dlg, turn_n, use_few_shot=args.use_few_shot)
                    else:
                        print(f"[WARN] Unknown scenario: {scen}")
                        continue
                    actual_response = dlg['utterances'][turn_n] if turn_n < len(dlg['utterances']) else ""
                    if not prompt_text:
                        continue
                    if output_file:
                        with open(output_file, "a+", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "dialogue_id": dlg.get('dialogue_id', f"dlg_{dlg_idx}"),
                                "turn_id": turn_n+1,
                                "scenario": scen,
                                "full_dialogue": full_dialogue,
                                "prompt": prompt_text,
                                "target_turn": turn_n+1,
                                "actual_response": actual_response,
                                "dataset": dlg.get('dataset', 'unknown')
                            }, ensure_ascii=False) + "\n")
                    else:
                        prompts.append({
                            "dialogue_id": dlg.get('dialogue_id', f"dlg_{dlg_idx}"),
                            "turn_id": turn_n+1,
                            "scenario": scen,
                            "full_dialogue": full_dialogue,
                            "prompt": prompt_text,
                            "target_turn": turn_n+1,
                            "actual_response": actual_response,
                            "dataset": dlg.get('dataset', 'unknown')
                        })
            elif scen == "scenario_3":
                prompt_text = prepare_scenario_3(dlg)
                actual_response = []
                for i, (speaker, utterance) in enumerate(zip(dlg['speakers'], dlg['utterances'])):
                    if should_mask_speaker(speaker, dlg.get('dataset', '')):
                        actual_response.append(utterance)
                if not prompt_text:
                    continue
                if output_file:
                    with open(output_file, "a+", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "dialogue_id": dlg.get('dialogue_id', f"dlg_{dlg_idx}"),
                            "turn_id": "all_system_turns",
                            "scenario": scen,
                            "full_dialogue": full_dialogue,
                            "prompt": prompt_text,
                            "actual_response": actual_response,
                            "dataset": dlg.get('dataset', 'unknown')
                        }, ensure_ascii=False) + "\n")
                else:
                    prompts.append({
                        "dialogue_id": dlg.get('dialogue_id', f"dlg_{dlg_idx}"),
                        "turn_id": "all_system_turns",
                        "scenario": scen,
                        "full_dialogue": full_dialogue,
                        "prompt": prompt_text,
                        "actual_response": actual_response,
                        "dataset": dlg.get('dataset', 'unknown')
                    })
            else:
                print(f"[WARN] Unknown scenario: {scen}")   
            
    