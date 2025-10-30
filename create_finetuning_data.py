#!/usr/bin/env python3
"""
Fine-tuning Data Generator for Conversational Datasets

This script generates training data for fine-tuning language models on dialogue completion tasks.
It uses the ConversationalDataLoader to load dialogues and formats them as training examples
where the model learns to predict masked system/assistant responses.

Usage:
    python create_finetuning_data.py --dataset kandor --num_samples 100 --output_file kandor_finetune.txt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import random

# Add data root and conversational data loader
DATA_ROOT = os.getenv("DATA_ROOT", "./data")
sys.path.append(".")
sys.path.append("./data")
sys.path.append(str(Path(__file__).parent))

try:
    from conversational_dataloader import ConversationalDataLoader
except ImportError:
    print("Error: Could not import ConversationalDataLoader")
    print("Make sure conversational_dataloader.py is in the current directory or Python path")
    sys.exit(1)


class FinetuningDataGenerator:
    def __init__(self, 
                 data_dir: str = None,
                 bos_token: str = "[BOS]",
                 eos_token: str = "[EOS]", 
                 sep_token: str = "[sep]",
                 answer_token: str = "[answer]",
                 blank_token: str = "[blank]"):
        """
        Initialize the fine-tuning data generator
        
        Args:
            data_dir: Directory containing the datasets
            bos_token: Beginning of sequence token
            eos_token: End of sequence token  
            sep_token: Separator token between context and target
            answer_token: Answer token (optional, for compatibility)
            blank_token: Token used to mask the target response
        """
        self.data_dir = data_dir or DATA_ROOT
        self.loader = ConversationalDataLoader(self.data_dir)
        
        # Special tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.answer_token = answer_token
        self.blank_token = blank_token
        
    def should_mask_speaker(self, speaker: str, dataset: str) -> bool:
        """
        Determine if a speaker should be masked based on dataset conventions.
        Returns True if the speaker should be masked (i.e., is the system/assistant speaker).
        """
        dataset_lower = dataset.lower()
        speaker_lower = speaker.lower()
        
        # MultiWOZ: System/SYSTEM speakers
        if 'multiwoz' in dataset_lower:
            return speaker in ['System', 'SYSTEM']
        
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
        elif 'ami' in dataset_lower:
            return speaker == 'Speaker_A'
        
        # SODA: For SODA dataset, we'll use alternating pattern (predict Speaker_2)
        elif 'soda' in dataset_lower:
            return speaker == 'Speaker_2'
        
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
    
    def map_speaker_names(self, speaker: str, dataset: str, generic_names: bool = False) -> str:
        """
        Map speaker names to more generic or consistent names if requested
        
        Args:
            speaker: Original speaker name
            dataset: Dataset name
            generic_names: Whether to use generic names like User/Assistant
        """
        if not generic_names:
            return speaker
            
        dataset_lower = dataset.lower()
        
        # For medical datasets, use Patient/Doctor
        if 'meddialogue' in dataset_lower or 'mts-dialog' in dataset_lower:
            if speaker == 'Doctor':
                return 'Doctor'
            else:
                return 'Patient'
        
        # For other datasets, use User/Assistant pattern
        if self.should_mask_speaker(speaker, dataset):
            return 'Assistant'
        else:
            return 'User'
    
    def create_training_example(self, 
                              dialogue: Dict[str, Any], 
                              start_turn_idx: int,
                              generic_names: bool = False,
                              include_answer_token: bool = True) -> Optional[str]:
        """
        Create a single training example from a dialogue
        Creates pattern: Turn N [Speaker_1], Turn N+1 [Speaker_2] (masked), Turn N+2 [Speaker_1]
        
        Args:
            dialogue: Dialogue dictionary from the dataloader
            start_turn_idx: Index of the first turn in the 3-turn window (0-based)
            generic_names: Whether to use generic speaker names (ignored, always uses Speaker_1/Speaker_2)
            include_answer_token: Whether to include the [answer] token (ignored, no tokens used)
            
        Returns:
            Formatted training example string or None if invalid
        """
        speakers = dialogue['speakers']
        utterances = dialogue['utterances']
        dataset = dialogue.get('dataset', '')
        
        # We need 3 consecutive turns starting from start_turn_idx
        if start_turn_idx + 2 >= len(utterances) or start_turn_idx + 2 >= len(speakers):
            return None
        
        context_parts = []
        
        # Turn N (first turn)
        first_speaker = "Speaker_1" if start_turn_idx % 2 == 0 else "Speaker_2"
        first_utterance = utterances[start_turn_idx]
        context_parts.append(f"Turn {start_turn_idx + 1} [{first_speaker}]: {first_utterance}")
        
        # Turn N+1 (middle turn - this will be masked in training)
        middle_turn_idx = start_turn_idx + 1
        middle_speaker = "Speaker_1" if middle_turn_idx % 2 == 0 else "Speaker_2"
        middle_utterance = utterances[middle_turn_idx]
        context_parts.append(f"Turn {middle_turn_idx + 1} [{middle_speaker}]: {middle_utterance}")
        
        # Turn N+2 (third turn)
        third_turn_idx = start_turn_idx + 2
        third_speaker = "Speaker_1" if third_turn_idx % 2 == 0 else "Speaker_2"
        third_utterance = utterances[third_turn_idx]
        context_parts.append(f"Turn {third_turn_idx + 1} [{third_speaker}]: {third_utterance}")
        
        # Format the complete example (no special tokens)
        example = " ".join(context_parts)
        return example
    
    def load_train_only_dialogues(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Load only training dialogues for datasets that have train/dev/test splits
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            List of training dialogues only
        """
        print(f"Loading {dataset_name} dataset (train split only)...")
        
        try:
            all_dialogues = list(self.loader.load_dataset(dataset_name.lower()))
            
            # Filter to only training dialogues for datasets that have splits
            if dataset_name.lower() == 'multiwoz':
                train_dialogues = [d for d in all_dialogues if d.get('split') == 'train']
                print(f"MultiWOZ: Filtered to {len(train_dialogues)} training dialogues out of {len(all_dialogues)} total")
                return train_dialogues
            elif dataset_name.lower() == 'soda':
                train_dialogues = [d for d in all_dialogues if d.get('split') == 'train']
                print(f"SODA: Filtered to {len(train_dialogues)} training dialogues out of {len(all_dialogues)} total")
                return train_dialogues
            else:
                # For datasets without explicit splits (kandor, dailydialog), use all data
                print(f"{dataset_name}: No train/test split, using all {len(all_dialogues)} dialogues")
                return all_dialogues
                
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []

    def load_soda_split_dialogues(self, split_name: str) -> List[Dict[str, Any]]:
        """
        Load specific split dialogues for SODA dataset
        
        Args:
            split_name: Name of the split ('train', 'test', 'validation')
            
        Returns:
            List of dialogues from the specified split
        """
        print(f"Loading SODA dataset ({split_name} split only)...")
        
        try:
            all_dialogues = list(self.loader.load_dataset('soda'))
            
            # Filter to only the specified split
            split_dialogues = [d for d in all_dialogues if d.get('split') == split_name]
            print(f"SODA: Filtered to {len(split_dialogues)} {split_name} dialogues out of {len(all_dialogues)} total")
            return split_dialogues
                
        except Exception as e:
            print(f"Error loading SODA {split_name} split: {e}")
            return []

    def generate_dataset_examples(self, 
                                dataset_name: str,
                                max_dialogues: int = None,
                                generic_names: bool = False,
                                include_answer_token: bool = True,
                                train_only: bool = False) -> List[str]:
        """
        Generate training examples from a dataset
        
        Args:
            dataset_name: Name of the dataset to load
            max_dialogues: Maximum number of dialogues to process
            generic_names: Whether to use generic speaker names  
            include_answer_token: Whether to include [answer] token
            train_only: Whether to use only training data (for datasets with splits)
            
        Returns:
            List of formatted training examples
        """
        if train_only:
            dialogues = self.load_train_only_dialogues(dataset_name)
        else:
            print(f"Loading {dataset_name} dataset...")
            try:
                dialogues = list(self.loader.load_dataset(dataset_name.lower()))
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                return []
        
        if not dialogues:
            print(f"No dialogues found for dataset: {dataset_name}")
            return []
        
        print(f"Loaded {len(dialogues)} dialogues from {dataset_name}")
        
        # Apply max limit if specified
        if max_dialogues:
            dialogues = dialogues[:max_dialogues]
        
        # Count dialogues with enough turns for training examples (3+ turns)
        valid_dialogues = sum(1 for d in dialogues if len(d.get('utterances', [])) >= 3)
        
        print(f"Processing {len(dialogues)} dialogues")
        print(f"Dialogues with 3+ turns (can generate examples): {valid_dialogues}/{len(dialogues)}")
        
        examples = []
        total_examples = 0
        
        for dialogue_idx, dialogue in enumerate(dialogues):
            if (dialogue_idx + 1) % 100 == 0:
                print(f"Processed {dialogue_idx + 1}/{len(dialogues)} dialogues...")
            
            speakers = dialogue.get('speakers', [])
            utterances = dialogue.get('utterances', [])
            
            if len(speakers) != len(utterances):
                continue
            
            # Generate examples with step size 2: turns (0,1,2), (2,3,4), (4,5,6), etc.
            for start_turn_idx in range(0, len(utterances) - 2, 2):
                if start_turn_idx + 2 < len(speakers):
                        example = self.create_training_example(
                            dialogue, 
                        start_turn_idx,
                            generic_names=generic_names,
                            include_answer_token=include_answer_token
                        )
                        
                        if example:
                            examples.append(example)
                            total_examples += 1
        
        print(f"Generated {total_examples} training examples from {len(dialogues)} dialogues")
        return examples
    
    def save_examples(self, examples: List[str], output_file: str, format: str = 'txt'):
        """
        Save training examples to file
        
        Args:
            examples: List of training examples
            output_file: Output file path
            format: Output format ('txt', 'jsonl')
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    json.dump({"text": example}, f)
                    f.write('\n')
        else:
            # Default to plain text format with three newlines between examples
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, example in enumerate(examples):
                    f.write(example)
                    if i < len(examples) - 1:  # Don't add separator after last example
                        f.write('\n\n\n')
                    else:
                        f.write('\n')  # Just one newline at the end
        
        print(f"Saved {len(examples)} examples to {output_path}")

    def create_splits_and_generate(self,
                                 dataset_name: str,
                                 output_dir: str,
                                 train_ratio: float = 0.8,
                                 dev_ratio: float = 0.1,
                                 test_ratio: float = 0.1,
                                 max_dialogues: int = None,
                                 generic_names: bool = False,
                                 include_answer_token: bool = True,
                                 train_only: bool = False,
                                 seed: int = 42) -> Dict[str, Any]:
        """
        Create train/dev/test splits at dialogue level and generate examples for each
        
        Args:
            dataset_name: Name of the dataset to load
            output_dir: Directory to save the split files
            train_ratio: Proportion for training set
            dev_ratio: Proportion for development set  
            test_ratio: Proportion for test set
            max_dialogues: Maximum number of dialogues to process
            generic_names: Whether to use generic speaker names
            include_answer_token: Whether to include [answer] token
            train_only: Whether to use only training data (for datasets with splits)
            seed: Random seed for reproducible splits
            
        Returns:
            Dictionary with split statistics
        """
        # Set random seed for reproducible splits
        random.seed(seed)
        
        # Load dialogues
        if train_only:
            dialogues = self.load_train_only_dialogues(dataset_name)
        else:
            print(f"Loading {dataset_name} dataset...")
            try:
                dialogues = list(self.loader.load_dataset(dataset_name.lower()))
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                return {}
        
        if not dialogues:
            print(f"No dialogues found for dataset: {dataset_name}")
            return {}
        
        print(f"Loaded {len(dialogues)} dialogues from {dataset_name}")
        
        # Apply max limit if specified
        if max_dialogues:
            dialogues = dialogues[:max_dialogues]
            print(f"Limited to {len(dialogues)} dialogues")
        
        # Filter dialogues that can generate examples (3+ turns)
        valid_dialogues = [d for d in dialogues if len(d.get('utterances', [])) >= 3]
        print(f"Dialogues with 3+ turns (can generate examples): {len(valid_dialogues)}/{len(dialogues)}")
        
        # Shuffle dialogues for random split
        random.shuffle(valid_dialogues)
        
        # Calculate split sizes
        total_dialogues = len(valid_dialogues)
        train_size = int(total_dialogues * train_ratio)
        dev_size = int(total_dialogues * dev_ratio)
        test_size = total_dialogues - train_size - dev_size  # Remaining goes to test
        
        # Split dialogues
        train_dialogues = valid_dialogues[:train_size]
        dev_dialogues = valid_dialogues[train_size:train_size + dev_size]
        test_dialogues = valid_dialogues[train_size + dev_size:]
        
        print(f"\n=== DIALOGUE SPLITS ===")
        print(f"Train dialogues: {len(train_dialogues)} ({len(train_dialogues)/total_dialogues*100:.1f}%)")
        print(f"Dev dialogues: {len(dev_dialogues)} ({len(dev_dialogues)/total_dialogues*100:.1f}%)")
        print(f"Test dialogues: {len(test_dialogues)} ({len(test_dialogues)/total_dialogues*100:.1f}%)")
        print(f"Total: {len(train_dialogues) + len(dev_dialogues) + len(test_dialogues)}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate examples for each split
        splits_info = {}
        
        for split_name, split_dialogues in [
            ('train', train_dialogues),
            ('dev', dev_dialogues), 
            ('test', test_dialogues)
        ]:
            if not split_dialogues:
                print(f"\nSkipping {split_name} split (no dialogues)")
                continue
                
            print(f"\nGenerating examples for {split_name} split...")
            examples = []
            
            for dialogue in split_dialogues:
                speakers = dialogue.get('speakers', [])
                utterances = dialogue.get('utterances', [])
                
                if len(speakers) != len(utterances):
                    continue
                
                # Generate examples with step size 2: turns (0,1,2), (2,3,4), (4,5,6), etc.
                for start_turn_idx in range(0, len(utterances) - 2, 2):
                    if start_turn_idx + 2 < len(speakers):
                        example = self.create_training_example(
                            dialogue, 
                            start_turn_idx,
                            generic_names=generic_names,
                            include_answer_token=include_answer_token
                        )
                        
                        if example:
                            examples.append(example)
            
            # Save split file
            split_file = output_path / f"{split_name}.txt"
            self.save_examples(examples, str(split_file), 'txt')
            
            # Store split info
            splits_info[split_name] = {
                'dialogues': len(split_dialogues),
                'examples': len(examples),
                'file': str(split_file)
            }
            
            print(f"{split_name.capitalize()} split: {len(split_dialogues)} dialogues → {len(examples)} examples")
        
        return splits_info

    def create_soda_splits_and_generate(self,
                                       output_dir: str,
                                       max_dialogues: int = None,
                                       generic_names: bool = False,
                                       include_answer_token: bool = True,
                                       seed: int = 42) -> Dict[str, Any]:
        """
        Create train/dev/test splits for SODA dataset using existing train/test/validation splits
        and generate examples for each with triple newline formatting
        
        Args:
            output_dir: Directory to save the split files
            max_dialogues: Maximum number of dialogues to process per split
            generic_names: Whether to use generic speaker names
            include_answer_token: Whether to include [answer] token
            seed: Random seed for reproducible sampling
            
        Returns:
            Dictionary with split statistics
        """
        # Set random seed for reproducible sampling
        random.seed(seed)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate examples for each split
        splits_info = {}
        
        # Map SODA splits to our naming convention
        soda_splits = {
            'train': 'train',
            'dev': 'validation',  # SODA uses 'validation' instead of 'dev'
            'test': 'test'
        }
        
        for split_name, soda_split in soda_splits.items():
            print(f"\nProcessing {split_name} split (from SODA {soda_split})...")
            
            # Load SODA split directly from JSON file
            soda_dir = Path(self.data_dir) / "SODA"
            soda_file = soda_dir / f"{soda_split}.json"
            
            if not soda_file.exists():
                print(f"SODA file not found: {soda_file}")
                continue
            
            print(f"Loading SODA {soda_split}.json...")
            try:
                with open(soda_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"Loaded {len(data)} conversations from {soda_split}.json")
                
                # Apply max limit if specified
                if max_dialogues and len(data) > max_dialogues:
                    data = random.sample(data, max_dialogues)
                    print(f"Sampled {len(data)} conversations")
                
                examples = []
                valid_dialogues = 0
                
                for conversation_data in data:
                    # Extract dialogue and speakers
                    utterances = conversation_data.get('dialogue', [])
                    speakers = conversation_data.get('speakers', [])
                    
                    # Skip if no dialogue or speakers, or mismatched lengths
                    if not utterances or not speakers or len(utterances) != len(speakers):
                        continue
                    
                    # Skip if less than 3 turns (can't generate examples)
                    if len(utterances) < 3:
                        continue
                    
                    valid_dialogues += 1
                    
                    # Create a dialogue object for the create_training_example method
                    dialogue = {
                        'dataset': 'SODA',
                        'dialogue_id': f"soda_{soda_split}_{conversation_data.get('original_index', 0)}",
                        'utterances': utterances,
                        'speakers': speakers,
                        'num_turns': len(utterances),
                        'split': soda_split
                    }
                    
                    # Generate examples with step size 2: turns (0,1,2), (2,3,4), (4,5,6), etc.
                    for start_turn_idx in range(0, len(utterances) - 2, 2):
                        if start_turn_idx + 2 < len(speakers):
                            example = self.create_training_example(
                                dialogue, 
                                start_turn_idx,
                                generic_names=generic_names,
                                include_answer_token=include_answer_token
                            )
                            
                            if example:
                                examples.append(example)
                
                print(f"Valid dialogues (3+ turns): {valid_dialogues}/{len(data)}")
                print(f"Generated {len(examples)} training examples")
                
                # Save split file with triple newline formatting
                split_file = output_path / f"{split_name}.txt"
                self.save_examples(examples, str(split_file), 'txt')
                
                # Store split info
                splits_info[split_name] = {
                    'dialogues': valid_dialogues,
                    'examples': len(examples),
                    'file': str(split_file)
                }
                
                print(f"{split_name.capitalize()} split: {valid_dialogues} dialogues → {len(examples)} examples")
                
            except Exception as e:
                print(f"Error processing SODA {soda_split}: {e}")
        
        return splits_info


def main():
    parser = argparse.ArgumentParser(description='Generate fine-tuning data from conversational datasets')
    parser.add_argument('--dataset', required=True,
                        choices=['kandor', 'meddialogue', 'multiwoz', 'dailydialog', 'mts-dialog', 'ami', 'soda'],
                        help='Dataset to use')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Maximum number of dialogues to process (default: all)')
    parser.add_argument('--output_file', 
                        help='Output file path (default: {dataset}-finetune.txt)')

    parser.add_argument('--format', choices=['txt', 'jsonl'], default='txt',
                        help='Output format (default: txt)')
    parser.add_argument('--generic_names', action='store_true',
                        help='Use generic speaker names (User/Assistant or Patient/Doctor)')
    parser.add_argument('--no_answer_token', action='store_true',
                        help='Don\'t include [answer] token in examples')

    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: uses DATA_ROOT environment variable)')
    parser.add_argument('--train_only', action='store_true',
                        help='Use only training data for datasets with train/dev/test splits (e.g., MultiWOZ)')
    parser.add_argument('--full_dataset', action='store_true',
                        help='Process the entire dataset (overrides --num_samples)')
    
    # Split options
    parser.add_argument('--create_splits', action='store_true',
                        help='Create train/dev/test splits at dialogue level (80-10-10)')
    parser.add_argument('--output_dir', type=str, 
                        help='Output directory for split files (required with --create_splits)')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for reproducible splits (default: 42)')
    
    # Special token customization
    parser.add_argument('--bos_token', default='[BOS]', help='Beginning of sequence token')
    parser.add_argument('--eos_token', default='[EOS]', help='End of sequence token')
    parser.add_argument('--sep_token', default='[sep]', help='Separator token')
    parser.add_argument('--answer_token', default='[answer]', help='Answer token')
    parser.add_argument('--blank_token', default='[blank]', help='Blank/mask token')
    
    args = parser.parse_args()
    
    # Generate default output filename if not specified
    if not args.output_file:
        extension = 'jsonl' if args.format == 'jsonl' else 'txt'
        args.output_file = f"{args.dataset}-finetune.{extension}"
    
    # Initialize generator
    generator = FinetuningDataGenerator(
        data_dir=args.data_dir,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        sep_token=args.sep_token,
        answer_token=args.answer_token,
        blank_token=args.blank_token
    )
    
    # Handle full dataset option
    max_dialogues = None if args.full_dataset else args.num_samples
    
    # Handle create_splits option
    if args.create_splits:
        if not args.output_dir:
            print("Error: --output_dir is required when using --create_splits")
            sys.exit(1)
        
        # Use specialized SODA method if dataset is SODA
        if args.dataset.lower() == 'soda':
            splits_info = generator.create_soda_splits_and_generate(
                output_dir=args.output_dir,
                max_dialogues=max_dialogues,
                generic_names=args.generic_names,
                include_answer_token=not args.no_answer_token,
                seed=args.split_seed
            )
        else:
            splits_info = generator.create_splits_and_generate(
                dataset_name=args.dataset,
                output_dir=args.output_dir,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
                max_dialogues=max_dialogues,
                generic_names=args.generic_names,
                include_answer_token=not args.no_answer_token,
                train_only=args.train_only,
                seed=args.split_seed
            )
        print("\n=== SPLIT GENERATION SUMMARY ===")
        total_examples = sum(info['examples'] for info in splits_info.values())
        for split_name, info in splits_info.items():
            example_pct = info['examples'] / total_examples * 100 if total_examples > 0 else 0
            print(f"{split_name.capitalize()} split: {info['dialogues']} dialogues → {info['examples']} examples ({example_pct:.1f}%)")
        print(f"Total examples across all splits: {total_examples}")
        print(f"Output files saved to: {args.output_dir}")
    else:
        # Generate examples
        examples = generator.generate_dataset_examples(
            dataset_name=args.dataset,
            max_dialogues=max_dialogues,
            generic_names=args.generic_names,
            include_answer_token=not args.no_answer_token,
            train_only=args.train_only
        )
        
        if examples:
            # Save examples
            generator.save_examples(examples, args.output_file, args.format)
            
            # Print sample examples
            print(f"\n=== SAMPLE EXAMPLES ===")
            for i, example in enumerate(examples[:3]):
                print(f"\nExample {i+1}:")
                print(example)
                
            print(f"\n=== SUMMARY ===")
            print(f"Dataset: {args.dataset}")
            print(f"Total examples: {len(examples)}")
            print(f"Output file: {args.output_file}")
            print(f"Format: {args.format}")
            print(f"Generic names: {args.generic_names}")
            print(f"Train only: {args.train_only}")
            print(f"Full dataset: {args.full_dataset}")
            print(f"Pattern: Turn N [Speaker_1] -> Turn N+1 [Speaker_2] (to be masked) -> Turn N+2 [Speaker_1] with step size 2")
        else:
            print("No examples generated. Check dataset availability and parameters.")


if __name__ == "__main__":
    main() 