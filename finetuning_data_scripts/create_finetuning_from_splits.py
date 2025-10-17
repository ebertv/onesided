#!/usr/bin/env python3
"""
Generate finetuning examples from our existing splits.
Uses the splits we already created and generates 3-turn training examples.

Usage:
    python create_finetuning_from_splits.py --dataset soda
    python create_finetuning_from_splits.py --dataset multiwoz
    python create_finetuning_from_splits.py --dataset kandor
    python create_finetuning_from_splits.py --dataset dailydialog
    python create_finetuning_from_splits.py --all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add paths
sys.path.append(".")
sys.path.append("../")  # Parent directory
sys.path.append(str(Path(__file__).parent))

try:
    from create_finetuning_data import FinetuningDataGenerator
except ImportError as e:
    print(f"Error: Could not import FinetuningDataGenerator: {e}")
    sys.exit(1)


def load_split_data(dataset_name: str, split_name: str) -> List[Dict[str, Any]]:
    """Load dialogues from our existing split files"""
    
    # Determine the correct directory based on dataset
    if dataset_name.lower() in ['multiwoz']:
        # Use existing splits analysis directories
        split_dir = Path(f"{dataset_name}_existing_splits_analysis")
    else:
        # Use deterministic splits directories (including SODA now)
        split_dir = Path(f"{dataset_name}_deterministic_splits")
    
    split_file = split_dir / f"{split_name}.json"
    
    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        return []
    
    with open(split_file, 'r', encoding='utf-8') as f:
        dialogues = json.load(f)
    
    return dialogues


def generate_examples_for_split(dataset_name: str, split_name: str, dialogues: List[Dict[str, Any]]) -> List[str]:
    """Generate 3-turn training examples for a split"""
    
    generator = FinetuningDataGenerator()
    examples = []
    
    print(f"  Processing {len(dialogues)} dialogues in {split_name} split...")
    
    for dialogue in dialogues:
        speakers = dialogue.get('speakers', [])
        utterances = dialogue.get('utterances', [])
        
        if not speakers or not utterances:
            continue
        
        # Generate 3-turn examples with step size 2
        for start_turn_idx in range(0, len(utterances) - 2, 2):
            # Check if we can create a valid 3-turn example
            if start_turn_idx + 2 < len(utterances) and start_turn_idx + 2 < len(speakers):
                
                # Check if middle turn is the one we want to predict
                middle_turn_idx = start_turn_idx + 1
                middle_speaker = speakers[middle_turn_idx]
                
                # Dataset-specific logic for which speaker to predict
                should_predict = False
                if dataset_name.lower() == 'multiwoz':
                    should_predict = middle_speaker in ['System', 'system', 'Speaker_2']
                elif dataset_name.lower() == 'kandor':
                    should_predict = middle_speaker in ['Participant_R', 'Speaker_2']
                elif dataset_name.lower() in ['dailydialog']:
                    should_predict = middle_speaker in ['Speaker_2']
                elif dataset_name.lower() == 'soda':
                    # For SODA, predict alternating speakers (odd positions)
                    should_predict = (middle_turn_idx % 2 == 1)
                else:
                    # Default: predict Speaker_2
                    should_predict = middle_speaker in ['Speaker_2']
                
                if should_predict:
                    # Create the 3-turn example
                    example = create_training_example(
                        dialogue, start_turn_idx, utterances, speakers
                    )
                    if example:
                        examples.append(example)
    
    print(f"  Generated {len(examples)} training examples")
    return examples


def create_training_example(dialogue: Dict[str, Any], start_turn_idx: int, 
                          utterances: List[str], speakers: List[str]) -> Optional[str]:
    """Create a single 3-turn training example"""
    
    # Get the three turns
    turn_1_idx = start_turn_idx
    turn_2_idx = start_turn_idx + 1  # This will be masked
    turn_3_idx = start_turn_idx + 2
    
    # Map speakers to generic names
    speaker_1 = "Speaker_1" if turn_1_idx % 2 == 0 else "Speaker_2"
    speaker_2 = "Speaker_1" if turn_2_idx % 2 == 0 else "Speaker_2"  
    speaker_3 = "Speaker_1" if turn_3_idx % 2 == 0 else "Speaker_2"
    
    # Create the example in the format: Turn N [Speaker] Turn N+1 [Speaker] Turn N+2 [Speaker]
    example_parts = []
    
    # Turn 1
    example_parts.append(f"Turn {turn_1_idx + 1} [{speaker_1}]: {utterances[turn_1_idx]}")
    
    # Turn 2 (the one being predicted)
    example_parts.append(f"Turn {turn_2_idx + 1} [{speaker_2}]: {utterances[turn_2_idx]}")
    
    # Turn 3
    example_parts.append(f"Turn {turn_3_idx + 1} [{speaker_3}]: {utterances[turn_3_idx]}")
    
    return " ".join(example_parts)


def process_dataset(dataset_name: str):
    """Process a single dataset and generate finetuning examples for all splits"""
    print(f"\n=== Processing {dataset_name.upper()} ===")
    
    # Determine splits to process
    if dataset_name.lower() in ['multiwoz']:
        splits = ['train', 'dev', 'test']
        output_dir = Path(f"{dataset_name}_existing_splits_analysis")
    else:
        splits = ['train', 'dev', 'test'] 
        output_dir = Path(f"{dataset_name}_deterministic_splits")
    
    if not output_dir.exists():
        print(f"Error: Split directory not found: {output_dir}")
        return
    
    total_examples = 0
    
    for split_name in splits:
        print(f"\n--- {split_name.upper()} SPLIT ---")
        
        # Load split data
        dialogues = load_split_data(dataset_name, split_name)
        
        if not dialogues:
            print(f"  No dialogues found for {split_name}")
            continue
        
        # Generate examples
        examples = generate_examples_for_split(dataset_name, split_name, dialogues)
        
        if examples:
            # Save examples to file
            output_file = output_dir / f"{split_name}_finetune.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, example in enumerate(examples):
                    f.write(example)
                    if i < len(examples) - 1:  # Don't add extra newlines after last example
                        f.write("\n\n\n")  # 3 newlines between examples
                    else:
                        f.write("\n")  # Single newline at end of file
            
            print(f"  Saved {len(examples)} examples to {output_file}")
            total_examples += len(examples)
        else:
            print(f"  No examples generated for {split_name}")
    
    print(f"\nTotal examples for {dataset_name}: {total_examples}")


def main():
    parser = argparse.ArgumentParser(description='Generate finetuning examples from existing splits')
    parser.add_argument('--dataset', choices=['soda', 'multiwoz', 'kandor', 'dailydialog', 'all'],
                        help='Dataset to process')
    parser.add_argument('--all', action='store_true', help='Process all datasets')
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        print("Please specify a dataset with --dataset or use --all")
        parser.print_help()
        return
    
    print("=== FINETUNING DATA GENERATION FROM EXISTING SPLITS ===")
    print("Generating 3-turn training examples from our split files...")
    print()
    
    # Determine datasets to process
    if args.all or args.dataset == 'all':
        datasets = ['soda', 'multiwoz', 'kandor', 'dailydialog']
    else:
        datasets = [args.dataset]
    
    # Process each dataset
    for dataset in datasets:
        process_dataset(dataset)
    
    print("\n" + "=" * 50)
    print("Finetuning data generation complete!")
    print("\nOutput files created:")
    for dataset in datasets:
        if dataset in ['multiwoz']:
            base_dir = f"{dataset}_existing_splits_analysis"
        else:
            base_dir = f"{dataset}_deterministic_splits"
        
        for split in ['train', 'dev', 'test']:
            output_file = Path(base_dir) / f"{split}_finetune.txt"
            if output_file.exists():
                print(f"  {output_file}")


if __name__ == "__main__":
    main() 