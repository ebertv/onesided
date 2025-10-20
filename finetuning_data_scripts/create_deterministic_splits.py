#!/usr/bin/env python3
"""
Analyze existing splits or create deterministic 80/10/10 splits with ZERO randomization.
For datasets that already have splits (SODA, MultiWOZ), just analyze existing splits.
For others (Kandor, DailyDialog), create deterministic splits.

Usage:
    python create_deterministic_splits.py --dataset soda
    python create_deterministic_splits.py --dataset multiwoz  
    python create_deterministic_splits.py --dataset kandor
    python create_deterministic_splits.py --dataset dailydialog
    python create_deterministic_splits.py --all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Add data root and conversational data loader
DATA_ROOT = os.getenv("DATA_ROOT", "../data")
sys.path.append(".")
sys.path.append("../")  # Parent directory
sys.path.append("../data")
sys.path.append(str(Path(__file__).parent))

try:
    from conversational_dataloader import ConversationalDataLoader
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    sys.exit(1)


def combine_consecutive_turns_soda(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Special function for SODA: combine consecutive turns from the same speaker.
    Appends the utterances together with a space.
    """
    speakers = dialogue.get('speakers', [])
    utterances = dialogue.get('utterances', [])
    
    if not speakers or not utterances or len(speakers) != len(utterances):
        return dialogue
    
    # Combine consecutive turns from same speaker
    combined_speakers = []
    combined_utterances = []
    
    i = 0
    while i < len(speakers):
        current_speaker = speakers[i]
        current_utterance = utterances[i]
        
        # Look ahead to see if next turns are from same speaker
        j = i + 1
        while j < len(speakers) and speakers[j] == current_speaker:
            # Append the next utterance to current one
            current_utterance += " " + utterances[j]
            j += 1
        
        # Add the combined turn
        combined_speakers.append(current_speaker)
        combined_utterances.append(current_utterance)
        
        # Move to next different speaker
        i = j
    
    # Create new dialogue with combined turns
    combined_dialogue = dialogue.copy()
    combined_dialogue['speakers'] = combined_speakers
    combined_dialogue['utterances'] = combined_utterances
    combined_dialogue['num_turns'] = len(combined_utterances)
    
    return combined_dialogue


def load_soda_split(split_name: str) -> List[Dict[str, Any]]:
    """Load a specific SODA split directly from JSON file"""
    soda_dir = Path(DATA_ROOT) / "SODA"
    split_file = soda_dir / f"{split_name}.json"
    
    if not split_file.exists():
        print(f"SODA split file not found: {split_file}")
        return []
    
    dialogues = []
    with open(split_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for idx, conversation_data in enumerate(data):
            # Extract dialogue and speakers
            utterances = conversation_data.get('dialogue', [])
            speakers = conversation_data.get('speakers', [])
            
            # Skip if no dialogue or speakers
            if not utterances or not speakers or len(utterances) != len(speakers):
                continue
            
            # Create dialogue dictionary
            dialogue = {
                'dataset': 'SODA',
                'dialogue_id': f"soda_{split_name}_{idx}",
                'utterances': utterances,
                'speakers': speakers,
                'num_turns': len(utterances),
                'split': split_name
            }
            
            # Special processing for SODA: combine consecutive turns from same speaker
            dialogue = combine_consecutive_turns_soda(dialogue)
            
            dialogues.append(dialogue)
    
    return dialogues


def load_multiwoz_split(split_name: str) -> List[Dict[str, Any]]:
    """Load a specific MultiWOZ split directly from JSON files"""
    multiwoz_dir = Path(DATA_ROOT) / "MultiWOZ_2.2" / split_name
    
    if not multiwoz_dir.exists():
        print(f"MultiWOZ split directory not found: {multiwoz_dir}")
        return []
    
    dialogues = []
    
    # Load all JSON files in the split directory
    for json_file in multiwoz_dir.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # MultiWOZ 2.2 format: list of dialogue objects
            for dialogue_data in data:
                dialogue_id = dialogue_data.get('dialogue_id', '')
                turns = dialogue_data.get('turns', [])
                
                if not turns:
                    continue
                
                utterances = []
                speakers = []
                
                # Process turns - they alternate between user and system
                for i, turn in enumerate(turns):
                    utterance = turn.get('utterance', '').strip()
                    if utterance:
                        utterances.append(utterance)
                        # Even indices (0, 2, 4...) are typically user, odd are system
                        if i % 2 == 0:
                            speakers.append('Speaker_1')  # User
                        else:
                            speakers.append('Speaker_2')  # System
                
                if utterances and speakers:
                    dialogue = {
                        'dataset': 'MultiWOZ',
                        'dialogue_id': f"multiwoz_{split_name}_{dialogue_id}",
                        'utterances': utterances,
                        'speakers': speakers,
                        'num_turns': len(utterances),
                        'split': split_name
                    }
                    dialogues.append(dialogue)
    
    return dialogues


def create_deterministic_splits(dialogues: List[Dict[str, Any]], train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    Split dialogues deterministically with NO randomization.
    Takes first 80% for train, next 10% for dev, final 10% for test.
    """
    total = len(dialogues)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    # Split deterministically - no shuffling
    train_dialogues = dialogues[:train_end]
    dev_dialogues = dialogues[train_end:dev_end]
    test_dialogues = dialogues[dev_end:]
    
    # Save split indices
    split_indices = {
        'total_dialogues': total,
        'train_start': 0,
        'train_end': train_end,
        'dev_start': train_end,
        'dev_end': dev_end,
        'test_start': dev_end,
        'test_end': total,
        'train_size': len(train_dialogues),
        'dev_size': len(dev_dialogues),
        'test_size': len(test_dialogues)
    }
    
    return train_dialogues, dev_dialogues, test_dialogues, split_indices


def count_training_examples(dialogues: List[Dict[str, Any]], dataset_name: str) -> int:
    """
    Count how many 3-turn training examples would be generated from dialogues.
    Uses the same logic as FinetuningDataGenerator.create_training_example()
    """
    total_examples = 0
    
    for dialogue in dialogues:
        speakers = dialogue.get('speakers', [])
        utterances = dialogue.get('utterances', [])
        
        if not speakers or not utterances:
            continue
            
        # Count valid 3-turn examples with step size 2
        # Same logic as in create_finetuning_data.py
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
                    # For SODA, predict alternating speakers (odd positions: 1, 3, 5...)
                    # Since middle_turn_idx = start_turn_idx + 1, we want odd middle_turn_idx
                    should_predict = (middle_turn_idx % 2 == 1)
                else:
                    # Default: predict Speaker_2
                    should_predict = middle_speaker in ['Speaker_2']
                
                if should_predict:
                    total_examples += 1
    
    return total_examples


def analyze_existing_splits(dataset_name: str, output_base_dir: str = None):
    """Analyze existing splits for datasets that already have them (SODA, MultiWOZ)"""
    print(f"\n=== Analyzing Existing {dataset_name.upper()} Splits ===")
    
    # Set output directory
    if output_base_dir is None:
        output_dir = Path(f"{dataset_name}_existing_splits_analysis")
    else:
        output_dir = Path(output_base_dir) / f"{dataset_name}_existing_splits_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_data = {}
    total_dialogues = 0
    total_examples = 0
    
    if dataset_name.lower() == 'soda':
        # SODA has train, test, validation splits
        split_names = ['train', 'test', 'validation']
        split_mapping = {'validation': 'dev'}  # Map validation to dev
    elif dataset_name.lower() == 'multiwoz':
        # MultiWOZ has train, dev, test splits
        split_names = ['train', 'dev', 'test']
        split_mapping = {}
    else:
        print(f"Dataset {dataset_name} doesn't have existing splits")
        return None
    
    print(f"Loading existing splits: {split_names}")
    
    for split_name in split_names:
        try:
            # Load specific split
            if dataset_name.lower() == 'soda':
                dialogues = load_soda_split(split_name)
            elif dataset_name.lower() == 'multiwoz':
                dialogues = load_multiwoz_split(split_name)
            
            mapped_split = split_mapping.get(split_name, split_name)
            splits_data[mapped_split] = {
                'dialogues': len(dialogues),
                'original_split_name': split_name
            }
            
            print(f"  {split_name} ({mapped_split}): {len(dialogues)} dialogues")
            
            # Count training examples
            examples = count_training_examples(dialogues, dataset_name)
            splits_data[mapped_split]['examples'] = examples
            print(f"    → {examples} training examples")
            
            total_dialogues += len(dialogues)
            total_examples += examples
            
        except Exception as e:
            print(f"Error loading {split_name} split: {e}")
            continue
    
    # Create comprehensive statistics
    stats = {
        'dataset': dataset_name,
        'split_method': 'existing_predefined_splits',
        'dialogue_counts': {
            'total': total_dialogues,
            'train': splits_data.get('train', {}).get('dialogues', 0),
            'dev': splits_data.get('dev', {}).get('dialogues', 0),
            'test': splits_data.get('test', {}).get('dialogues', 0)
        },
        'example_counts': {
            'total': total_examples,
            'train': splits_data.get('train', {}).get('examples', 0),
            'dev': splits_data.get('dev', {}).get('examples', 0),
            'test': splits_data.get('test', {}).get('examples', 0)
        },
        'split_percentages': {
            'train_dialogues': (splits_data.get('train', {}).get('dialogues', 0) / total_dialogues * 100) if total_dialogues > 0 else 0,
            'dev_dialogues': (splits_data.get('dev', {}).get('dialogues', 0) / total_dialogues * 100) if total_dialogues > 0 else 0,
            'test_dialogues': (splits_data.get('test', {}).get('dialogues', 0) / total_dialogues * 100) if total_dialogues > 0 else 0,
            'train_examples': (splits_data.get('train', {}).get('examples', 0) / total_examples * 100) if total_examples > 0 else 0,
            'dev_examples': (splits_data.get('dev', {}).get('examples', 0) / total_examples * 100) if total_examples > 0 else 0,
            'test_examples': (splits_data.get('test', {}).get('examples', 0) / total_examples * 100) if total_examples > 0 else 0
        },
        'splits_details': splits_data
    }
    
    # Save split files in standardized format
    if splits_data:
        # Load all splits again and save in standardized format
        all_splits = {}
        for split_name in split_names:
            if dataset_name.lower() == 'soda':
                dialogues = load_soda_split(split_name)
            elif dataset_name.lower() == 'multiwoz':
                dialogues = load_multiwoz_split(split_name)
            
            mapped_split = split_mapping.get(split_name, split_name)
            all_splits[mapped_split] = dialogues
        
        # Save standardized split files
        if 'train' in all_splits:
            save_split_files(dataset_name, 
                           all_splits.get('train', []), 
                           all_splits.get('dev', []), 
                           all_splits.get('test', []), 
                           output_dir)
    
    # Save statistics
    stats_file = output_dir / f"{dataset_name}_existing_splits_analysis.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Analysis saved to {stats_file}")
    
    return stats


def create_soda_deterministic_splits(output_base_dir: str = None):
    """Create deterministic splits for SODA with consecutive turn combining"""
    print(f"\n=== Creating Deterministic SODA Splits (with turn combining) ===")
    
    # Set output directory
    if output_base_dir is None:
        output_dir = Path("soda_deterministic_splits")
    else:
        output_dir = Path(output_base_dir) / "soda_deterministic_splits"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all SODA data and combine
    soda_dir = Path(DATA_ROOT) / "SODA"
    all_dialogues = []
    
    split_files = ['train.json', 'test.json', 'validation.json']
    
    for split_file in split_files:
        soda_file = soda_dir / split_file
        if soda_file.exists():
            print(f"Loading SODA {split_file}...")
            with open(soda_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for idx, conversation_data in enumerate(data):
                    utterances = conversation_data.get('dialogue', [])
                    speakers = conversation_data.get('speakers', [])
                    
                    if not utterances or not speakers or len(utterances) != len(speakers):
                        continue
                    
                    dialogue = {
                        'dataset': 'SODA',
                        'dialogue_id': f"soda_combined_{len(all_dialogues)}",
                        'utterances': utterances,
                        'speakers': speakers,
                        'num_turns': len(utterances),
                        'original_split': split_file.replace('.json', '')
                    }
                    
                    # Apply consecutive turn combining
                    dialogue = combine_consecutive_turns_soda(dialogue)
                    all_dialogues.append(dialogue)
    
    print(f"Loaded {len(all_dialogues)} total SODA dialogues")
    
    # Create deterministic splits
    train_dialogues, dev_dialogues, test_dialogues, split_indices = create_deterministic_splits(all_dialogues)
    
    print(f"Deterministic split sizes (NO randomization, with turn combining):")
    print(f"  Train: {len(train_dialogues)} dialogues (indices {split_indices['train_start']}-{split_indices['train_end']-1})")
    print(f"  Dev:   {len(dev_dialogues)} dialogues (indices {split_indices['dev_start']}-{split_indices['dev_end']-1})")  
    print(f"  Test:  {len(test_dialogues)} dialogues (indices {split_indices['test_start']}-{split_indices['test_end']-1})")
    
    # Count training examples for each split
    print(f"\nCounting 3-turn training examples...")
    train_examples = count_training_examples(train_dialogues, 'soda')
    dev_examples = count_training_examples(dev_dialogues, 'soda')
    test_examples = count_training_examples(test_dialogues, 'soda')
    total_examples = train_examples + dev_examples + test_examples
    
    print(f"Training examples per split:")
    print(f"  Train: {train_examples} examples")
    print(f"  Dev:   {dev_examples} examples")
    print(f"  Test:  {test_examples} examples")
    print(f"  Total: {total_examples} examples")
    
    # Save split files
    save_split_files('soda', train_dialogues, dev_dialogues, test_dialogues, output_dir)
    
    # Save comprehensive statistics
    stats = {
        'dataset': 'soda',
        'split_method': 'deterministic_sequential_no_randomization_with_turn_combining',
        'split_ratios': {'train': 0.8, 'dev': 0.1, 'test': 0.1},
        'dialogue_counts': {
            'total': len(all_dialogues),
            'train': len(train_dialogues),
            'dev': len(dev_dialogues),
            'test': len(test_dialogues)
        },
        'dialogue_indices': split_indices,
        'example_counts': {
            'train': train_examples,
            'dev': dev_examples,
            'test': test_examples,
            'total': total_examples
        },
        'split_percentages': {
            'train_dialogues': len(train_dialogues) / len(all_dialogues) * 100,
            'dev_dialogues': len(dev_dialogues) / len(all_dialogues) * 100,
            'test_dialogues': len(test_dialogues) / len(all_dialogues) * 100,
            'train_examples': train_examples / total_examples * 100 if total_examples > 0 else 0,
            'dev_examples': dev_examples / total_examples * 100 if total_examples > 0 else 0,
            'test_examples': test_examples / total_examples * 100 if total_examples > 0 else 0
        }
    }
    
    stats_file = output_dir / "soda_deterministic_split_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_file}")
    
    return stats


def save_split_files(dataset_name: str, train_dialogues: List[Dict], dev_dialogues: List[Dict], 
                    test_dialogues: List[Dict], output_dir: Path):
    """Save split files in standardized JSON format with integer dialogue_ids"""
    
    splits = {
        'train': train_dialogues,
        'dev': dev_dialogues,
        'test': test_dialogues
    }
    
    for split_name, dialogues in splits.items():
        # Convert to standardized format with integer dialogue_ids
        standardized_dialogues = []
        
        for idx, dialogue in enumerate(dialogues):
            standardized_dialogue = {
                'dataset': dialogue.get('dataset', dataset_name.upper()),
                'dialogue_id': idx,  # Integer ID starting from 0
                'split': split_name,
                'speakers': dialogue.get('speakers', []),
                'utterances': dialogue.get('utterances', []),
                'num_turns': len(dialogue.get('utterances', []))
            }
            standardized_dialogues.append(standardized_dialogue)
        
        # Save as JSON file for all datasets
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(standardized_dialogues, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(standardized_dialogues)} dialogues to {output_file}")


def process_dataset(dataset_name: str, output_base_dir: str = None):
    """Process a single dataset - either analyze existing splits or create new ones"""
    
    # Special handling for SODA deterministic splits
    if dataset_name.lower() == 'soda':
        return create_soda_deterministic_splits(output_base_dir)
    
    # Datasets with existing splits (MultiWOZ only now)
    if dataset_name.lower() in ['multiwoz']:
        return analyze_existing_splits(dataset_name, output_base_dir)
    
    # Datasets needing deterministic splits
    print(f"\n=== Creating Deterministic Splits for {dataset_name.upper()} ===")
    
    # Set output directory
    if output_base_dir is None:
        output_dir = Path(f"{dataset_name}_deterministic_splits")
    else:
        output_dir = Path(output_base_dir) / f"{dataset_name}_deterministic_splits"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset using ConversationalDataLoader for datasets without existing splits
    loader = ConversationalDataLoader(DATA_ROOT)
    try:
        dialogues = list(loader.load_dataset(dataset_name))
        print(f"Loaded {len(dialogues)} dialogues from {dataset_name}")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None
    
    if not dialogues:
        print(f"No dialogues found in {dataset_name} dataset")
        return None
    
    # Create deterministic splits
    train_dialogues, dev_dialogues, test_dialogues, split_indices = create_deterministic_splits(dialogues)
    
    print(f"Deterministic split sizes (NO randomization):")
    print(f"  Train: {len(train_dialogues)} dialogues (indices {split_indices['train_start']}-{split_indices['train_end']-1})")
    print(f"  Dev:   {len(dev_dialogues)} dialogues (indices {split_indices['dev_start']}-{split_indices['dev_end']-1})")  
    print(f"  Test:  {len(test_dialogues)} dialogues (indices {split_indices['test_start']}-{split_indices['test_end']-1})")
    
    # Count training examples for each split
    print(f"\nCounting 3-turn training examples...")
    train_examples = count_training_examples(train_dialogues, dataset_name)
    dev_examples = count_training_examples(dev_dialogues, dataset_name)
    test_examples = count_training_examples(test_dialogues, dataset_name)
    total_examples = train_examples + dev_examples + test_examples
    
    print(f"Training examples per split:")
    print(f"  Train: {train_examples} examples")
    print(f"  Dev:   {dev_examples} examples")
    print(f"  Test:  {test_examples} examples")
    print(f"  Total: {total_examples} examples")
    
    # Save split files
    save_split_files(dataset_name, train_dialogues, dev_dialogues, test_dialogues, output_dir)
    
    # Save comprehensive statistics
    stats = {
        'dataset': dataset_name,
        'split_method': 'deterministic_sequential_no_randomization',
        'split_ratios': {'train': 0.8, 'dev': 0.1, 'test': 0.1},
        'dialogue_counts': {
            'total': len(dialogues),
            'train': len(train_dialogues),
            'dev': len(dev_dialogues),
            'test': len(test_dialogues)
        },
        'dialogue_indices': split_indices,
        'example_counts': {
            'train': train_examples,
            'dev': dev_examples,
            'test': test_examples,
            'total': total_examples
        },
        'split_percentages': {
            'train_dialogues': len(train_dialogues) / len(dialogues) * 100,
            'dev_dialogues': len(dev_dialogues) / len(dialogues) * 100,
            'test_dialogues': len(test_dialogues) / len(dialogues) * 100,
            'train_examples': train_examples / total_examples * 100 if total_examples > 0 else 0,
            'dev_examples': dev_examples / total_examples * 100 if total_examples > 0 else 0,
            'test_examples': test_examples / total_examples * 100 if total_examples > 0 else 0
        }
    }
    
    stats_file = output_dir / f"{dataset_name}_deterministic_split_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze existing splits or create deterministic 80/10/10 splits')
    parser.add_argument('--dataset', choices=['soda', 'multiwoz', 'kandor', 'dailydialog', 'all'], 
                        help='Dataset to process')
    parser.add_argument('--all', action='store_true', help='Process all datasets')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Base output directory (default: current directory)')
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        print("Please specify a dataset with --dataset or use --all")
        parser.print_help()
        return
    
    print("=== DATASET ANALYSIS AND SPLITTING ===")
    print("SODA: Create deterministic splits with consecutive turn combining")
    print("MultiWOZ: Analyze existing splits")
    print("Kandor & DailyDialog: Create deterministic splits (80-10-10, no randomization)")
    print()
    
    datasets_to_process = []
    if args.all or args.dataset == 'all':
        datasets_to_process = ['soda', 'multiwoz', 'kandor', 'dailydialog']
    else:
        datasets_to_process = [args.dataset]
    
    all_stats = []
    
    for dataset in datasets_to_process:
        stats = process_dataset(dataset, args.output_dir)
        if stats:
            all_stats.append(stats)
    
    # Save summary statistics
    if all_stats:
        summary_file = Path(args.output_dir or ".") / "dataset_splits_summary.json"
        summary = {
            'datasets_processed': len(all_stats),
            'total_dialogues': sum(s['dialogue_counts']['total'] for s in all_stats),
            'total_examples': sum(s['example_counts']['total'] for s in all_stats),
            'dataset_details': all_stats
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== OVERALL SUMMARY ===")
        print(f"Datasets processed: {len(all_stats)}")
        print(f"Total dialogues: {summary['total_dialogues']:,}")
        print(f"Total training examples: {summary['total_examples']:,}")
        print(f"Summary saved to: {summary_file}")
        
        # Print per-dataset summary table
        print(f"\nPer-dataset breakdown:")
        print(f"{'Dataset':<12} {'Method':<20} {'Dialogues':<10} {'Examples':<10} {'Train':<8} {'Dev':<8} {'Test':<8}")
        print("-" * 80)
        
        for stats in all_stats:
            dataset = stats['dataset']
            method = 'Existing' if stats['split_method'] == 'existing_predefined_splits' else 'Deterministic'
            total_d = stats['dialogue_counts']['total']
            total_e = stats['example_counts']['total']
            train_e = stats['example_counts']['train']
            dev_e = stats['example_counts']['dev']
            test_e = stats['example_counts']['test']
            
            print(f"{dataset:<12} {method:<20} {total_d:<10,} {total_e:<10,} {train_e:<8,} {dev_e:<8,} {test_e:<8,}")


if __name__ == "__main__":
    main() 