import json

def mask_infill(full_context):
    turn_prefix = "[Speaker_2]: "
    start = 0
    while True:
      start = full_context.find(turn_prefix, start)
      if start == -1:
        break
      turn_start = start + len(turn_prefix)
      start += len(turn_prefix)
    turn_end = full_context.find(" Turn", turn_start)
    if turn_end == -1:
        turn_end = len(full_context)
    masked_context = full_context[:turn_start] + "<|MASK|>" + full_context[turn_end:]
    return masked_context

def read_finetuned_data(finetune_data_fp):
    '''
    Read in the finetune data, take the pairs of real responses and predicted responses
    Return a list of tuples of the form (full_context (masked), real_response, predicted_response).
    '''
    with open(finetune_data_fp, 'r') as f:
        finetune_data = [json.loads(line) for line in f.readlines()]

    data = []
    for item in finetune_data:
        full_context = mask_infill(item['prompt'])
        real_response = item['actual_response']
        predicted_response = item['predictions'][0]
        data.append((full_context, real_response, predicted_response))
    return data

def get_context_from_prompt(prompt):
    prompt = prompt.split('=== BEGIN CONVERSATION ===')[-1]
    prompt = prompt.split('=== END CONVERSATION ===')[0]
    prompt = prompt.split('Context:')[-1]
    prompt = prompt.strip()
    prompt = prompt.split('\n')
    prompt[1] = prompt[1].replace('[Predict this turn : Speaker_2]: ', 'Speaker_2: <|MASK|> ')
    prompt = ' '.join(prompt)

    return prompt

def read_prompted_data(prompts_fp):
    '''
    Read in the prompted data, take the pairs of real responses and predicted responses
    Return a list of tuples of the form (full_context (masked), real_response, predicted_response).
    '''
    with open(prompts_fp, 'r') as f:
        prompted_data = json.load(f)

    data = []
    for item in prompted_data['predictions']:
        real_response = item['actual']
        predicted_response = item['prediction']
        full_context = get_context_from_prompt(item['full_prompt'])
        data.append((full_context, real_response, predicted_response))
    return data

def read_summary_data(summary_data_fp):
    '''
    Read in the summary data, take the pairs of real responses and predicted responses
    Return a list of tuples of the form (full_conversation, masked_summary, reconstructed summary).
    '''
    with open(summary_data_fp, 'r') as f:
        summary_data = json.load(f)

    data = []
    for item in summary_data['comparisons']:
        full_conversation = item['contexts']['full']['text']
        masked_summary = item['contexts']['masked']['summary']
        reconstructed_summary = item['contexts']['predicted']['summary']
        data.append((full_conversation, masked_summary, reconstructed_summary))
    return data


def create_random_comparisons(pairs, raters, output_dir, compare_type):
    '''
    Given a list of pairs of responses, write to a jsonl file random comparisons.
    Each comparison should be a dictionary with the following keys:
    - dataset: the name of the dataset (e.g. dailydialog, kandor, etc.)
    - index: the index of the pair in the list of pairs
    - full_dialogue: the full context
    - response_1: the first response to compare
    - response_2: the second response to compare
    - rater: the rater assigned to this comparison
    The real response should be randomly assigned to either response_1 or response_2.
    The other response should be the finetuned response.
    Each comparison should be assigned to a random rater from the list of raters.
    Each rater number is paired with the number of comparisons they are assigned to.
    Another file should contain the answers, with the following keys:
    - full_dialogue: the full context
    - response_1: the first response
    - response_2: the second response
    - answer: 1 if response_1 is the real response, 2 if response_2 is the real response
    Return the filepaths of the two files created.
    '''
    import random
    import os
    total_comparisons = sum(raters.values())
    if total_comparisons > len(pairs):
        raise ValueError('Not enough pairs to assign to raters')
    selected_indices = random.sample(range(len(pairs)), total_comparisons)
    selected_pairs = [pairs[i] for i in selected_indices]
    blank_comparison = {
        "dataset": None,
        "model": None,
        "index": None,
        "full_dialogue": None,
        "response_1": None,
        "response_2": None,
        "rater": None
    }
    blank_answer = {
        "dataset": None,
        "model": None,
        "index": None,
        "full_dialogue": None,
        "response_1": None,
        "response_2": None,
    }
    if compare_type == 'summary':
        blank_answer['masked'] = None
        blank_answer['reconstructed'] = None
    if compare_type:
        blank_answer['ground_truth'] = None
        blank_answer['predicted'] = None

    comparisons = []
    answers = []
    rater, num_comparisons = raters.items()
    for _ in range(num_comparisons):
        pair = selected_pairs.pop()
        if compare_type == 'infill':
            full_context, real_response, finetuned_response = pair
        elif compare_type == 'summary':
            full_context, masked_summary, reconstructed_summary = pair
        if random.random() < 0.5:
            response_1 = real_response
            response_2 = finetuned_response
            answer = 1
        else:
            response_1 = finetuned_response
            response_2 = real_response
            answer = 2
        comparison = blank_comparison.copy()
        comparison['dataset'] = dataset_name
        comparison['model'] = model
        comparison['index'] = len(comparisons)
        comparison['full_dialogue'] = full_context
        comparison['response_1'] = response_1
        comparison['response_2'] = response_2
        comparison['rater'] = rater
        comparisons.append(comparison)

        answer_dict = blank_answer.copy()
        answer_dict['dataset'] = dataset_name
        answer_dict['model'] = model
        answer_dict['index'] = len(answers)
        answer_dict['full_dialogue'] = full_context
        answer_dict['response_1'] = response_1
        answer_dict['response_2'] = response_2
        if compare_type == 'infill':
            answer_dict['ground_truth'] = real_response
            answer_dict['predicted'] = finetuned_response
            answer_dict['answer'] = answer
        elif compare_type == 'summary':
            answer_dict['masked'] = masked_summary
            answer_dict['reconstructed'] = reconstructed_summary
            answer_dict['answer'] = answer
        answers.append(answer_dict)
    os.makedirs(output_dir, exist_ok=True)
    comparisons_fp = os.path.join(output_dir, f'{dataset_name}_{compare_type}_comparisons.jsonl')
    answers_fp = os.path.join(output_dir, f'{dataset_name}_{compare_type}_answers.jsonl')
    with open(comparisons_fp, 'w') as f:
        for comparison in comparisons:
            f.write(json.dumps(comparison) + '\n')
    with open(answers_fp, 'w') as f:
        for answer in answers:
            f.write(json.dumps(answer) + '\n')

        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--finetune_data_fp', type=str, required=True, help='Filepath to the finetune data jsonl file.')
    # parser.add_argument('--prompted_data_fp', type=str, required=True, help='Filepath to the prompted data json file.')
    parser.add_argument('type', type=str, choices=['infill', 'summary'], help='Type of prediction to evaluate.')
    parser.add_argument('-n', '--num_comparisons', type=int, default=10, help='Number of comparisons to assign to each rater.')
    parser.add_argument('--output_dir', type=str, required=False, default='ab_test', help='Directory to write the comparison files to.')
    args = parser.parse_args()

    raters = {args.name: args.num_comparisons}

    if args.type == 'infill':
        data = []
        for dataset_name in ['dailydialog', 'multiwoz', 'candor']:
            for model in ['gpt2', 'llama']:
                finetune_data_fp = f'/gscratch/scrubbed/ebertv/onesided/ilm/data/char_masks/custom/{dataset_name}/ilm_{model.removesuffix('2')}_infill_test_withfullconvo.jsonl'
                temp = read_finetuned_data(finetune_data_fp)
                data.extend((temp, dataset_name, model))
            if dataset_name == 'dailydialog':
                dname = 'dd'
                prompts_fp = f'/gscratch/scrubbed/ebertv/onesided/test_predictions/3turn_{dname}_predictions.json'
            elif dataset_name == 'multiwoz':
                dname = 'm'
                prompts_fp = f'/gscratch/scrubbed/ebertv/onesided/test_predictions/3turn_{dname}_predictions.json'
            elif dataset_name == 'candor':
                dname = 'c'
                prompts_fp = f'/gscratch/scrubbed/ebertv/onesided/test_predictions/{dname}_1.2_3turn_predictions.json'
            promp = read_prompted_data(prompts_fp)
            data.extend((promp, dataset_name, 'claude'))
        comparisons_fp, answers_fp = create_random_comparisons(data, raters, args.output_dir)
    elif args.type == 'summary':
        if args.dataset_name == 'dailydialog':
            dataset_name = 'dd'
        elif args.dataset_name == 'multiwoz':
            dataset_name = 'm'
        summary_data_fp = f'/gscratch/scrubbed/ebertv/onesided/test_predictions/{dataset_name}_summary_eval.json'
        summary_data = read_summary_data(summary_data_fp)
        comparisons_fp, answers_fp = create_random_comparisons(summary_data, raters, args.dataset_name, args.output_dir, pred_type='summary')

    print(f'Wrote comparisons to {comparisons_fp}')
    print(f'Wrote answers to {answers_fp}')