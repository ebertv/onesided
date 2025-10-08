import os
import json

def get_real_answer(dataset, full_dialogue):
    if dataset == 'dailydialog':
        with open('dailydialog_summary_answers.jsonl', 'r') as f:
            for line in f:
                item = json.loads(line)
                if item['full_dialogue'] == full_dialogue:
                    return item['answer']
        return None
    elif dataset == 'multiwoz':
        with open('multiwoz_summary_answers.jsonl', 'r') as f:
            for line in f:
                item = json.loads(line)
                if item['full_dialogue'] == full_dialogue:
                    return item['answer']
    else:
        return None

if __name__ == '__main__':
    #collect all responses
    responses = []
    for file in os.listdir('.'):
        if file.endswith('summary_ab_test_answers.jsonl'):
            with open(file, 'r') as f:
                for line in f:
                    responses.append(json.loads(line))
    print(f'Collected {len(responses)} total responses.')

    template = {
        "full_dialogue": None,
        "response_1": None,
        "response_2": None,
        "rater": None,
        "answer": None,
        "dataset": None,
        "pred_type": None,
        'chose_masked': None,
        'chose_reconstructed': None
    }
    data = []
    for response in responses:
        temp = template.copy()
        temp['full_dialogue'] = response['full_dialogue']
        temp['response_1'] = response['response_1']
        temp['response_2'] = response['response_2']
        temp['rater'] = response['rater']
        temp['answer'] = response['answer']
        temp['dataset'] = response.get('dataset', 'unknown')
        full_dialogue = response.get('full_dialogue', None)
        correct_answer = get_real_answer(temp['dataset'], full_dialogue)
        temp['correct_answer'] = correct_answer
        if correct_answer == temp['answer']:
            temp['chose_masked'] = True
            temp['chose_reconstructed'] = False
        else:
            temp['chose_masked'] = False
            temp['chose_reconstructed'] = True
        data.append(temp)
    print(f'Processed {len(data)} total responses.')

    #for each dataset and pred type print how often raters chose the masked vs reconstructed summary
    summary = {}
    for item in data:
        dataset = item['dataset']
        if dataset not in summary:
            summary[dataset] = {
                'total': 0,
                'masked': 0,
                'reconstructed': 0
            }
        summary[dataset]['total'] += 1
        if item['chose_masked']:
            summary[dataset]['masked'] += 1
        elif item['chose_reconstructed']:
            summary[dataset]['reconstructed'] += 1
    print('Summary of results:')
    for dataset, results in summary.items():
        print(f'Dataset: {dataset}')
        print(f'Total comparisons: {results["total"]}')
        print(f'Chose masked summary: {results["masked"]} ({results["masked"]/results["total"]*100:.2f}%)')
        print(f'Chose reconstructed summary: {results["reconstructed"]} ({results["reconstructed"]/results["total"]*100:.2f}%)')
        print('---')