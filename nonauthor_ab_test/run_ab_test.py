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
    
    print(f'Hello {args.name}! Welcome to the AB testing script for One-Sided Conversations.')
    print('Please follow the instructions to complete the AB testing task.')
    print('You will be presented with pairs of responses for the same dialogue context.')
    print('For each pair, please choose which response fits better in the context (1 or 2).')
    print('If you feel both responses are equally good, press 0. You will not be penalized for choosing 0, but please use it sparingly.')
    print('Some responses include XXXXXX rather than specific names, places, or numbers. Please treat these as normal words in the conversation, as if they were names, places or numbers.')
    print('Thank you for your participation!')
    num_to_do = input('How many comparisons would you like to do? (20, 40, 60, 80, 100): ')
    while num_to_do not in ['20', '40', '60', '80', '100']:
        num_to_do = input('Invalid input. Please enter 20, 40, 60, 80, 100): ')
    num_to_do = int(num_to_do)

    files = [
        f'dailydialog_finetune_comparisons.jsonl',
        f'multiwoz_finetune_comparisons.jsonl',
        f'dailydialog_prompted_comparisons.jsonl',
        f'multiwoz_prompted_comparisons.jsonl',
    ]

    num_to_do = num_to_do/20
    num_to_do = int(num_to_do)
    num = random.sample(range(5), num_to_do)
    raters = [f'Rater_{i}' for i in num]
    
    ready = input('Press any key when you are ready to begin: ')
    print(f'You have {num_to_do*20} comparisons to complete.')
    output_fps = []
    for r, rater in enumerate(raters):
        all_comparisons = get_all_user_comparisons(rater, files)
        random.shuffle(all_comparisons)

        for i, comparison in enumerate(all_comparisons):
            print('\n' + '='*50)
            print(f'\nComparison {(r*20)+(i+1)}/{len(raters)*20}:')
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

        output_fp = f'{args.name}_{rater.lower()}_ab_test_answers.jsonl'
        with open(output_fp, 'w') as f:
            for comparison in all_comparisons:
                f.write(json.dumps(comparison) + '\n')
        output_fps.append(output_fp)
    print(f'Thank you for completing the AB testing task! Your responses have been saved')
    print('Please send the following files to Victoria (ebertv@cs.washington.edu):')
    for fp in output_fps:
        print(fp)
    print('Goodbye!')