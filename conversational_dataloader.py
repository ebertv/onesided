#!/usr/bin/env python3
"""
Conversational Dataset Loader

This script loads and iterates through multiple conversational datasets:
- DailyDialog
- MTS-Dialog 
- MultiWOZ
- MedDialogue

Usage:
    python conversational_dataloader.py [--dataset DATASET_NAME] [--max_dialogues N]
"""

import os
import json
import csv
import zipfile
import argparse
from pathlib import Path
from typing import Iterator, Dict, List, Any, Optional
import re


class ConversationalDataLoader:
    """
    A unified dataloader for multiple conversational datasets
    """
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize the dataloader with the data directory
        
        Args:
            data_dir: Directory containing the dataset folders
        """
        self.data_dir = Path(data_dir)
        self.datasets = {
            'dailydialog': self._load_dailydialog,
            'mts-dialog': self._load_mts_dialog,
            'multiwoz': self._load_multiwoz,
            'meddialogue': self._load_meddialogue,
            'ami': self._load_ami,
            'kandor': self._load_kandor,
            'soda': self._load_soda
        }
    
    def _load_dailydialog(self) -> Iterator[Dict[str, Any]]:
        """Load DailyDialog dataset"""
        dailydialog_dir = self.data_dir / "DailyDialog" / "ijcnlp_dailydialog"
        
        if not dailydialog_dir.exists():
            print(f"DailyDialog directory not found: {dailydialog_dir}")
            return
        
        # Load dialogues
        dialogues_file = dailydialog_dir / "dialogues_text.txt"
        if dialogues_file.exists():
            with open(dialogues_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        # Split by __eou__ (end of utterance)
                        utterances = [u.strip() for u in line.split('__eou__') if u.strip()]
                        
                        dialogue = {
                            'dataset': 'DailyDialog',
                            'dialogue_id': f'dd_{i+1}',
                            'utterances': utterances,
                            'num_turns': len(utterances),
                            'speakers': [f'Speaker_{j%2 + 1}' for j in range(len(utterances))]
                        }
                        yield dialogue
    
    def _load_mts_dialog(self) -> Iterator[Dict[str, Any]]:
        """Load MTS-Dialog dataset"""
        mts_dir = self.data_dir / "MTS-Dialog" / "MTS-Dialog-main" / "Main-Dataset"
        
        if not mts_dir.exists():
            print(f"MTS-Dialog directory not found: {mts_dir}")
            return
        
        # Load training set
        training_file = mts_dir / "MTS-Dialog-TrainingSet.csv"
        if training_file.exists():
            with open(training_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    dialogue_text = row.get('dialogue', '')
                    if dialogue_text:
                        # Parse dialogue - split by speaker turns
                        turns = []
                        speakers = []
                        
                        # Simple parsing - look for "Doctor:" and "Patient:" patterns
                        current_speaker = None
                        current_text = ""
                        
                        for line in dialogue_text.split('\n'):
                            line = line.strip()
                            if line.startswith('Doctor:'):
                                if current_text:
                                    turns.append(current_text.strip())
                                    speakers.append(current_speaker)
                                current_speaker = 'Doctor'
                                current_text = line[7:].strip()  # Remove "Doctor:"
                            elif line.startswith('Patient:'):
                                if current_text:
                                    turns.append(current_text.strip())
                                    speakers.append(current_speaker)
                                current_speaker = 'Patient'
                                current_text = line[8:].strip()  # Remove "Patient:"
                            else:
                                current_text += " " + line
                        
                        # Add last turn
                        if current_text and current_speaker:
                            turns.append(current_text.strip())
                            speakers.append(current_speaker)
                        
                        if turns:
                            dialogue = {
                                'dataset': 'MTS-Dialog',
                                'dialogue_id': f'mts_{i+1}',
                                'utterances': turns,
                                'speakers': speakers,
                                'num_turns': len(turns),
                                'section_header': row.get('section_header', ''),
                                'section_text': row.get('section_text', '')
                            }
                            yield dialogue
    
    def _load_multiwoz(self) -> Iterator[Dict[str, Any]]:
        """Load MultiWOZ dataset"""
        multiwoz_dir = self.data_dir / "MultiWOZ_2.2"
        
        if not multiwoz_dir.exists():
            print(f"MultiWOZ directory not found: {multiwoz_dir}")
            return
        
        # Look for JSON files in train, dev, and test directories
        for split_dir in ['train', 'dev', 'test']:
            split_path = multiwoz_dir / split_dir
            if not split_path.exists():
                continue
                
            # Process each dialogue file in the split directory
            for json_file in split_path.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # MultiWOZ 2.2 format: list of dialogues
                        if isinstance(data, list):
                            for dialogue_data in data:
                                if 'turns' in dialogue_data and 'dialogue_id' in dialogue_data:
                                    # Extract turns from the dialogue
                                    turns = []
                                    speakers = []
                                    
                                    for turn in dialogue_data['turns']:
                                        if 'utterance' in turn and 'speaker' in turn:
                                            turns.append(turn['utterance'])
                                            # Map USER/SYSTEM to more readable names
                                            speaker = 'User' if turn['speaker'] == 'USER' else 'System'
                                            speakers.append(speaker)
                                    
                                    if turns:
                                        dialogue = {
                                            'dataset': 'MultiWOZ',
                                            'dialogue_id': dialogue_data['dialogue_id'],
                                            'utterances': turns,
                                            'speakers': speakers,
                                            'num_turns': len(turns),
                                            'services': dialogue_data.get('services', []),
                                            'split': split_dir
                                        }
                                        yield dialogue
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error loading {json_file}: {e}")
                    continue
    
    def _load_meddialogue(self, use_real_corpus: bool = False) -> Iterator[Dict[str, Any]]:
        """Load MedDialogue dataset"""
        meddialogue_dir = self.data_dir / "MedDialogue"
        
        if not meddialogue_dir.exists():
            print(f"MedDialogue directory not found: {meddialogue_dir}")
            return
        
        # Choose which dataset to load
        if use_real_corpus:
            meddialogue_file = meddialogue_dir / "real_meddialogue_corpus.json"
            dataset_type = "Real MedDialogue Corpus"
        else:
            meddialogue_file = meddialogue_dir / "meddialogue_dialogues.json"
            dataset_type = "Enhanced MedDialogue"
        
        if meddialogue_file.exists():
            try:
                with open(meddialogue_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle list of dialogues
                    for dialogue_data in data:
                        dialogue = {
                            'dataset': dataset_type,
                            'dialogue_id': dialogue_data.get('id', 'meddialogue_unknown'),
                            'utterances': dialogue_data.get('dialogue', []),
                            'speakers': dialogue_data.get('speakers', []),
                            'num_turns': dialogue_data.get('num_turns', len(dialogue_data.get('dialogue', []))),
                            'description': dialogue_data.get('description', ''),
                            'medical_entities': dialogue_data.get('medical_entities', []),
                            'specialty': dialogue_data.get('specialty', 'general'),
                            'severity': dialogue_data.get('severity', 'unknown'),
                            'source': dialogue_data.get('source', 'generated')
                        }
                        yield dialogue
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading MedDialogue file {meddialogue_file}: {e}")
        else:
            print(f"{dataset_type} file not found: {meddialogue_file}")
            if not use_real_corpus:
                print("Creating placeholder data...")
                # Create placeholder MedDialogue data if file doesn't exist
                for i in range(3):  # Create 3 placeholder conversations
                    dialogue = {
                        'dataset': 'MedDialogue',
                        'dialogue_id': f'meddialogue_placeholder_{i+1}',
                        'utterances': [
                            f"I have been experiencing some symptoms for the past few days.",
                            "Can you describe your symptoms in more detail?",
                            "I have a headache and feel tired.",
                            "How long have you had these symptoms?",
                            "About 3 days now.",
                            "I recommend getting some rest and monitoring your symptoms."
                        ],
                        'speakers': ['Patient', 'Doctor', 'Patient', 'Doctor', 'Patient', 'Doctor'],
                        'num_turns': 6,
                        'description': f'Placeholder medical consultation {i+1}',
                        'medical_entities': ['headache', 'fatigue'],
                        'specialty': 'general',
                        'severity': 'mild',
                        'source': 'placeholder'
                    }
                    yield dialogue
    
    def _load_ami(self) -> Iterator[Dict[str, Any]]:
        """Load AMI dataset"""
        ami_dir = self.data_dir / "AMI"
        
        if not ami_dir.exists():
            print(f"AMI directory not found: {ami_dir}")
            return
        
        # Load the processed AMI corpus
        ami_file = ami_dir / "ami_meetings_corpus.json"
        
        if ami_file.exists():
            try:
                with open(ami_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle list of meetings
                    for meeting_data in data:
                        dialogue = {
                            'dataset': 'AMI',
                            'dialogue_id': meeting_data.get('id', 'ami_unknown'),
                            'utterances': meeting_data.get('dialogue', []),
                            'speakers': meeting_data.get('speakers', []),
                            'num_turns': meeting_data.get('num_turns', len(meeting_data.get('dialogue', []))),
                            'description': meeting_data.get('description', ''),
                            'meeting_id': meeting_data.get('meeting_id', ''),
                            'participants': meeting_data.get('participants', []),
                            'duration_seconds': meeting_data.get('duration_seconds', 0),
                            'meeting_keywords': meeting_data.get('meeting_keywords', []),
                            'source': meeting_data.get('source', 'ami_corpus'),
                            'type': meeting_data.get('type', 'meeting')
                        }
                        yield dialogue
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading AMI file {ami_file}: {e}")
        else:
            print(f"AMI corpus file not found: {ami_file}")
            print("Creating placeholder data...")
            # Create placeholder AMI data if file doesn't exist
            for i in range(3):  # Create 3 placeholder conversations
                dialogue = {
                    'dataset': 'AMI',
                    'dialogue_id': f'ami_placeholder_{i+1}',
                    'utterances': [
                        f"Good morning everyone, let's start today's meeting.",
                        "Thank you for gathering. What's on our agenda?",
                        "We need to discuss the project timeline and budget.",
                        "I think we should focus on the key milestones first.",
                        "Agreed. Let's review what we've accomplished so far.",
                        "We should also consider potential risks and mitigation strategies."
                    ],
                    'speakers': ['Speaker_A', 'Speaker_B', 'Speaker_A', 'Speaker_C', 'Speaker_B', 'Speaker_A'],
                    'num_turns': 6,
                    'description': f'Placeholder meeting conversation {i+1}',
                    'meeting_keywords': ['meeting', 'project', 'budget', 'timeline'],
                    'participants': ['A', 'B', 'C'],
                    'duration_seconds': 300,
                    'source': 'placeholder',
                    'type': 'meeting'
                }
                yield dialogue
    
    def _load_kandor(self) -> Iterator[Dict[str, Any]]:
        """Load Kandor dataset"""
        kandor_dir = self.data_dir / "kandor"
        
        if not kandor_dir.exists():
            print(f"Kandor directory not found: {kandor_dir}")
            return
        
        # Load the processed kandor corpus
        kandor_file = kandor_dir / "kandor_conversations_corpus.json"
        
        if kandor_file.exists():
            try:
                with open(kandor_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle list of conversations
                    for conversation_data in data:
                        # Extract utterances and speakers from turns
                        utterances = []
                        speakers = []
                        turns_data = conversation_data.get('dialogue', [])
                        
                        for turn in turns_data:
                            if 'utterance' in turn and 'speaker' in turn:
                                utterances.append(turn['utterance'])
                                speakers.append(turn['speaker'])
                        
                        dialogue = {
                            'dataset': 'Kandor',
                            'dialogue_id': conversation_data.get('id', 'kandor_unknown'),
                            'utterances': utterances,
                            'speakers': speakers,
                            'num_turns': len(utterances),
                            'metadata': {
                                'session_id': conversation_data.get('metadata', {}).get('session_id', ''),
                                'total_duration_ms': conversation_data.get('metadata', {}).get('total_duration_ms', 0),
                                'total_turns': conversation_data.get('metadata', {}).get('total_turns', 0),
                                'total_speakers': conversation_data.get('metadata', {}).get('total_speakers', 0),
                                'transcript_source': conversation_data.get('metadata', {}).get('transcript_source', ''),
                                'created_at': conversation_data.get('metadata', {}).get('created_at', '')
                            }
                        }
                        yield dialogue
                        
            except Exception as e:
                print(f"Error loading Kandor dataset: {e}")
        else:
            print(f"Kandor corpus file not found: {kandor_file}")
            # Generate fallback data
            print("Generating fallback Kandor data...")
            fallback_dialogue = {
                'dataset': 'Kandor',
                'dialogue_id': 'kandor_fallback_001',
                'utterances': [
                    'Hello, how are you today?',
                    'I am doing well, thank you for asking. How about you?',
                    'I am great! What would you like to talk about?',
                    'Let us discuss our weekend plans and favorite activities.'
                ],
                'speakers': ['Participant_1', 'Participant_2', 'Participant_1', 'Participant_2'],
                'num_turns': 4,
                'metadata': {
                    'session_id': 'fallback_session',
                    'total_duration_ms': 120000,
                    'total_turns': 4,
                    'total_speakers': 2,
                    'transcript_source': 'fallback',
                    'created_at': '2024-01-01'
                }
            }
            yield fallback_dialogue
    
    def _load_soda(self) -> Iterator[Dict[str, Any]]:
        """Load SODA dataset from train.json, test.json, and validation.json"""
        soda_dir = self.data_dir / "SODA"
        
        if not soda_dir.exists():
            print(f"SODA directory not found: {soda_dir}")
            return
        
        # Load from all three split files
        split_files = ['train.json', 'test.json', 'validation.json']
        
        for split_file in split_files:
            soda_file = soda_dir / split_file
            
            if soda_file.exists():
                try:
                    print(f"Loading SODA {split_file}...")
                    with open(soda_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Handle list of conversations
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
                                'dialogue_id': f"soda_{split_file.replace('.json', '')}_{idx}",
                                'utterances': utterances,
                                'speakers': speakers,
                                'num_turns': len(utterances),
                                'split': split_file.replace('.json', ''),  # Add split information
                                'metadata': {
                                    'original_index': conversation_data.get('original_index', idx),
                                    'head': conversation_data.get('head', ''),
                                    'relation': conversation_data.get('relation', ''),
                                    'tail': conversation_data.get('tail', ''),
                                    'literal': conversation_data.get('literal', ''),
                                    'narrative': conversation_data.get('narrative', ''),
                                    'PersonX': conversation_data.get('PersonX', ''),
                                    'PersonY': conversation_data.get('PersonY', ''),
                                    'PersonZ': conversation_data.get('PersonZ', '')
                                }
                            }
                            yield dialogue
                            
                except Exception as e:
                    print(f"Error loading SODA {split_file}: {e}")
            else:
                print(f"SODA file not found: {soda_file}")
        
        # If no files found, generate fallback data
        if not any((soda_dir / f).exists() for f in split_files):
            print("No SODA split files found. Generating fallback SODA data...")
            fallback_dialogue = {
                'dataset': 'SODA',
                'dialogue_id': 'soda_fallback_001',
                'utterances': [
                    'Hello, how are you today?',
                    'I am doing well, thank you for asking. How about you?',
                    'I am great! What would you like to talk about?',
                    'Let us discuss our weekend plans and favorite activities.'
                ],
                'speakers': ['Participant_1', 'Participant_2', 'Participant_1', 'Participant_2'],
                'num_turns': 4,
                'split': 'fallback',
                'metadata': {
                    'original_index': 0,
                    'head': '',
                    'relation': '',
                    'tail': '',
                    'literal': '',
                    'narrative': '',
                    'PersonX': '',
                    'PersonY': '',
                    'PersonZ': ''
                }
            }
            yield fallback_dialogue
    
    def load_dataset(self, dataset_name: str, use_real_corpus: bool = False) -> Iterator[Dict[str, Any]]:
        """
        Load a specific dataset by name
        
        Args:
            dataset_name: Name of the dataset to load
            use_real_corpus: For MedDialogue, whether to use the real corpus instead of enhanced data
            
        Returns:
            Iterator over dialogue dictionaries
        """
        dataset_name = dataset_name.lower()
        if dataset_name in self.datasets:
            print(f"Loading {dataset_name} dataset...")
            yield from self.datasets[dataset_name]()
        else:
            print(f"Unknown dataset: {dataset_name}")
            print(f"Available datasets: {list(self.datasets.keys())}")
    
    def load_all_datasets(self) -> Iterator[Dict[str, Any]]:
        """
        Load all available datasets
        
        Returns:
            Iterator over all dialogues from all datasets
        """
        for dataset_name in self.datasets:
            print(f"\n=== Loading {dataset_name} ===")
            yield from self.load_dataset(dataset_name)
    
    def get_dataset_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all datasets
        
        Returns:
            Dictionary with statistics for each dataset
        """
        stats = {}
        
        for dataset_name in self.datasets:
            print(f"Analyzing {dataset_name}...")
            dialogues = list(self.load_dataset(dataset_name))
            
            if dialogues:
                total_dialogues = len(dialogues)
                total_utterances = sum(d['num_turns'] for d in dialogues)
                avg_turns_per_dialogue = total_utterances / total_dialogues if total_dialogues > 0 else 0
                
                stats[dataset_name] = {
                    'total_dialogues': total_dialogues,
                    'total_utterances': total_utterances,
                    'avg_turns_per_dialogue': round(avg_turns_per_dialogue, 2)
                }
            else:
                stats[dataset_name] = {
                    'total_dialogues': 0,
                    'total_utterances': 0,
                    'avg_turns_per_dialogue': 0
                }
        
        return stats


def print_dialogue(dialogue: Dict[str, Any]) -> None:
    """
    Pretty print a dialogue
    
    Args:
        dialogue: Dialogue dictionary to print
    """
    print(f"\n--- {dialogue['dataset']} - {dialogue['dialogue_id']} ---")
    print(f"Number of turns: {dialogue['num_turns']}")
    
    for i, (speaker, utterance) in enumerate(zip(dialogue['speakers'], dialogue['utterances'])):
        print(f"Turn {i+1} [{speaker}]: {utterance[:100]}{'...' if len(utterance) > 100 else ''}")
    
    # Print any additional metadata
    for key, value in dialogue.items():
        if key not in ['dataset', 'dialogue_id', 'utterances', 'speakers', 'num_turns']:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Load and explore conversational datasets')
    parser.add_argument('--dataset', type=str, default='all', 
                        help='Dataset to load (dailydialog, mts-dialog, multiwoz, or all)')
    parser.add_argument('--max_dialogues', type=int, default=5,
                        help='Maximum number of dialogues to print (default: 5)')
    parser.add_argument('--stats_only', action='store_true',
                        help='Only show dataset statistics')
    
    args = parser.parse_args()
    
    # Initialize dataloader
    loader = ConversationalDataLoader()
    
    if args.stats_only:
        # Show statistics only
        print("=== DATASET STATISTICS ===")
        stats = loader.get_dataset_statistics()
        
        for dataset_name, dataset_stats in stats.items():
            print(f"\n{dataset_name.upper()}:")
            print(f"  Total Dialogues: {dataset_stats['total_dialogues']}")
            print(f"  Total Utterances: {dataset_stats['total_utterances']}")
            print(f"  Avg Turns per Dialogue: {dataset_stats['avg_turns_per_dialogue']}")
        
        return
    
    # Load and print dialogues
    dialogue_count = 0
    
    if args.dataset.lower() == 'all':
        dialogues = loader.load_all_datasets()
    else:
        dialogues = loader.load_dataset(args.dataset)
    
    print(f"=== PRINTING UP TO {args.max_dialogues} DIALOGUES ===")
    
    for dialogue in dialogues:
        if dialogue_count >= args.max_dialogues:
            break
        
        print_dialogue(dialogue)
        dialogue_count += 1
    
    print(f"\nPrinted {dialogue_count} dialogues.")
    
    if dialogue_count == 0:
        print("No dialogues found. Check if the datasets are properly downloaded.")


if __name__ == "__main__":
    main() 