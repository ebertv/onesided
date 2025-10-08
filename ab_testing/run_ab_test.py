from argparse import ArgumentParser
import os
import json
import random


def get_all_user_comparisons(rater_name, files):
    all_comparisons = []
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                comparison = json.loads(line)
                if comparison['rater'].lower() == rater_name.lower():
                    all_comparisons.append(comparison)
    return all_comparisons


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help='Your first name. (e.g., Victoria)')

    args = parser.parse_args()

    if args.name.lower() not in ['victoria', 'rishabh', 'tuochao', 'shyam', 'noah']:
        raise ValueError('Name not recognized. Please enter one of the following names: Victoria, Rishabh, Tuochao, Shyam, Noah')
    
    print(f'Hello, {args.name}! Welcome to the AB testing script for One-Sided Conversations.')
    print('Please follow the instructions to complete the AB testing task.')
    print('You will be presented with pairs of responses for the same dialogue context.')
    print('For each pair, please choose which response fits better in the context (1 or 2).')
    print('If you feel both responses are equally good, press 0. You will not be penalized for choosing 0, but please use it sparingly.')
    print('Thank you for your participation!')

    files = [
        f'dailydialog_finetune_comparisons.jsonl',
        f'multiwoz_finetune_comparisons.jsonl',
        f'kandor_finetune_comparisons.jsonl',
        f'dailydialog_prompted_comparisons.jsonl',
        f'multiwoz_prompted_comparisons.jsonl',
    ]

    ready = input('Press any key when you are ready to begin: ')
    all_comparisons = get_all_user_comparisons(args.name, files)
    print(f'You have {len(all_comparisons)} comparisons to complete.')
    random.shuffle(all_comparisons)

    for i, comparison in enumerate(all_comparisons):
        print('\n' + '='*50)
        print(f'\nComparison {i+1}/{len(all_comparisons)}:')
        print('Context:')
        print(comparison['full_dialogue'])
        print('\nResponse 1:')
        print(comparison['response_1'])
        print('\nResponse 2:')
        print(comparison['response_2'])
        
        choice = input('Which response fits best in the conversation? (If both fit equally well press 0): ')
        while choice not in ['0', '1', '2']:
            choice = input('Invalid input. Please enter 0 1 or 2: ')

        comparison['answer'] = int(choice)

    output_fp = f'{args.name.lower()}_ab_test_answers.jsonl'
    with open(output_fp, 'w') as f:
        for comparison in all_comparisons:
            f.write(json.dumps(comparison) + '\n')
    print(f'Thank you for completing the AB testing task! Your responses have been saved. Please send the file {output_fp} to Victoria.')
    print('Goodbye!')