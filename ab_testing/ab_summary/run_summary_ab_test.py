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
    print('You will be presented with a full dialogue context and pairs of summaries for that dialogue.')
    print('For each pair, please choose which summary is better (1 or 2).')
    print('You may not pick a tie for summaries, so please choose the better of the two.')
    print('Thank you for your participation!')

    files = [
        'dailydialog_summary_comparisons.jsonl',
        'multiwoz_summary_comparisons.jsonl',
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
        print('\nSummary 1:')
        print(comparison['response_1'])
        print('\nSummary 2:')
        print(comparison['response_2'])
        
        choice = input('Which summary is better? (Enter 1 or 2): ')
        while choice not in ['1', '2']:
            choice = input('Invalid input. Please enter 1 or 2: ')

        comparison['answer'] = int(choice)

    output_fp = f'{args.name.lower()}_summary_ab_test_answers.jsonl'
    with open(output_fp, 'w') as f:
        for comparison in all_comparisons:
            f.write(json.dumps(comparison) + '\n')
    print(f'Thank you for completing the Summary AB testing task! Your responses have been saved. Please send the file {output_fp} to Victoria.')
    print('Goodbye!')