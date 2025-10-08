import os
import json

def get_real_answer_and_pred_type(dataset, full_dialogue):
    if dataset == 'kandor':
        with open('kandor_finetune_answers.jsonl', 'r') as f:
            for line in f:
                item = json.loads(line)
                if item['full_dialogue'] == full_dialogue:
                    return item['answer'], 'finetune'
        return None, 'finetune'
    elif dataset == 'dailydialog':
        with open('dailydialog_finetune_answers.jsonl', 'r') as f:
            for line in f:
                item = json.loads(line)
                if item['full_dialogue'] == full_dialogue:
                    return item['answer'], 'finetune'
        with open('dailydialog_prompted_answers.jsonl', 'r') as f:
            for line in f:
                item = json.loads(line)
                if item['full_dialogue'] == full_dialogue:
                    return item['answer'], 'prompted'
        return None, 'unknown'
    elif dataset == 'multiwoz':
        with open('multiwoz_finetune_answers.jsonl', 'r') as f:
            for line in f:
                item = json.loads(line)
                if item['full_dialogue'] == full_dialogue:
                    return item['answer'], 'finetune'
        with open('multiwoz_prompted_answers.jsonl', 'r') as f:
            for line in f:
                item = json.loads(line)
                if item['full_dialogue'] == full_dialogue:
                    return item['answer'], 'prompted'
        return None, 'unknown'
    else:
        return None, 'unknown'

if __name__ == '__main__':
    #collect all responses
    responses = []
    for file in os.listdir('.'):
        if file.endswith('_ab_test_answers.jsonl'):
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
        "correct_answer": None,
        "dataset": None,
        "pred_type": None,
        'chose_ground_truth': None,
        'chose_model': None,
        'tied': None
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
        correct_answer, pred_type = get_real_answer_and_pred_type(temp['dataset'], full_dialogue)
        temp['correct_answer'] = correct_answer
        temp['pred_type'] = pred_type
        if correct_answer == temp['answer']:
            temp['chose_ground_truth'] = True
            temp['chose_model'] = False
            temp['tied'] = False
        elif temp['answer'] == 0:
            temp['chose_ground_truth'] = False
            temp['chose_model'] = False
            temp['tied'] = True
        else:
            temp['chose_ground_truth'] = False
            temp['chose_model'] = True
            temp['tied'] = False
        data.append(temp)
    print(f'Processed {len(data)} total responses.')

    #for each dataset and pred type print how many chose ground truth, model, tied
    summary = {}
    for item in data:
        key = (item['dataset'], item['pred_type'])
        if key not in summary:
            summary[key] = {'total': 0, 'chose_ground_truth': 0, 'chose_model': 0, 'tied': 0}
        summary[key]['total'] += 1
        if item['chose_ground_truth']:
            summary[key]['chose_ground_truth'] += 1
        elif item['chose_model']:
            summary[key]['chose_model'] += 1
        elif item['tied']:
            summary[key]['tied'] += 1
    print('Summary of choices by dataset and prediction type:')
    for key in summary:
        dataset, pred_type = key
        total = summary[key]['total']
        chose_ground_truth = summary[key]['chose_ground_truth']
        chose_model = summary[key]['chose_model']
        tied = summary[key]['tied']
        print(f'Dataset: {dataset}, Prediction Type: {pred_type} => Total: {total}, Chose Ground Truth: {chose_ground_truth} ({chose_ground_truth/total:.2f}), Chose Model: {chose_model} ({chose_model/total:.2f}), Tied: {tied} ({tied/total:.2f})')
    print('-'*50)

    

    #get total correct overall, by dataset, by pred_type
    total = 0
    correct = 0
    by_dataset = {}
    by_pred_type = {}
    for item in data:
        total += 1
        if item['answer'] == item['correct_answer']:
            correct += 1
            if item['dataset'] not in by_dataset:
                by_dataset[item['dataset']] = {'total': 0, 'correct': 0}
            by_dataset[item['dataset']]['total'] += 1
            by_dataset[item['dataset']]['correct'] += 1
            if item['pred_type'] not in by_pred_type:
                by_pred_type[item['pred_type']] = {'total': 0, 'correct': 0}
            by_pred_type[item['pred_type']]['total'] += 1
            by_pred_type[item['pred_type']]['correct'] += 1
        else:
            if item['dataset'] not in by_dataset:
                by_dataset[item['dataset']] = {'total': 0, 'correct': 0}
            by_dataset[item['dataset']]['total'] += 1
            if item['pred_type'] not in by_pred_type:
                by_pred_type[item['pred_type']] = {'total': 0, 'correct': 0}
            by_pred_type[item['pred_type']]['total'] += 1

    print(f'Overall accuracy: {correct}/{total} = {correct/total:.2f}')
    print('By dataset:')
    for dataset in by_dataset:
        d_total = by_dataset[dataset]['total']
        d_correct = by_dataset[dataset]['correct']
        print(f'  {dataset}: {d_correct}/{d_total} = {d_correct/d_total:.2f}')
    print('By prediction type:')
    for pred_type in by_pred_type:
        p_total = by_pred_type[pred_type]['total']
        p_correct = by_pred_type[pred_type]['correct']
        print(f'  {pred_type}: {p_correct}/{p_total} = {p_correct/p_total:.2f}')

    print('-'*50)

    #for each dataset and pred type count number of responses with answer 0
    ties_by_dataset = {}
    ties_by_pred_type = {}
    for item in data:
        if item['answer'] == 0:
            if item['dataset'] not in ties_by_dataset:
                ties_by_dataset[item['dataset']] = 0
            ties_by_dataset[item['dataset']] += 1
            if item['pred_type'] not in ties_by_pred_type:
                ties_by_pred_type[item['pred_type']] = 0
            ties_by_pred_type[item['pred_type']] += 1
    print('Overall percentage of ties (answer=0): {}/{} = {:.2f}'.format(
        sum(ties_by_dataset.values()), total, sum(ties_by_dataset.values())/total))
    print('Number of ties (answer=0) by dataset:')
    for dataset in ties_by_dataset:
        print(f'  {dataset}: {ties_by_dataset[dataset]}/{by_dataset[dataset]["total"]} = {ties_by_dataset[dataset]/by_dataset[dataset]["total"]:.2f}')
    print('Number of ties (answer=0) by prediction type:')
    for pred_type in ties_by_pred_type:
        print(f'  {pred_type}: {ties_by_pred_type[pred_type]}/{by_pred_type[pred_type]["total"]} = {ties_by_pred_type[pred_type]/by_pred_type[pred_type]["total"]:.2f}')

    print('-'*50)

    #get accuracy as above but only for non-ties
    total = 0
    correct = 0
    by_dataset = {}
    by_pred_type = {}
    for item in data:
        if item['answer'] == 0:
            continue
        total += 1
        if item['answer'] == item['correct_answer']:
            correct += 1
            if item['dataset'] not in by_dataset:
                by_dataset[item['dataset']] = {'total': 0, 'correct': 0}
            by_dataset[item['dataset']]['total'] += 1
            by_dataset[item['dataset']]['correct'] += 1
            if item['pred_type'] not in by_pred_type:
                by_pred_type[item['pred_type']] = {'total': 0, 'correct': 0}
            by_pred_type[item['pred_type']]['total'] += 1
            by_pred_type[item['pred_type']]['correct'] += 1
        else:
            if item['dataset'] not in by_dataset:
                by_dataset[item['dataset']] = {'total': 0, 'correct': 0}
            by_dataset[item['dataset']]['total'] += 1
            if item['pred_type'] not in by_pred_type:
                by_pred_type[item['pred_type']] = {'total': 0, 'correct': 0}
            by_pred_type[item['pred_type']]['total'] += 1

    print(f'Overall accuracy (excluding ties): {correct}/{total} = {correct/total:.2f}')
    print('By dataset (excluding ties):')
    for dataset in by_dataset:
        d_total = by_dataset[dataset]['total']
        d_correct = by_dataset[dataset]['correct']
        print(f'  {dataset}: {d_correct}/{d_total} = {d_correct/d_total:.2f}')
    print('By prediction type (excluding ties):')
    for pred_type in by_pred_type:
        p_total = by_pred_type[pred_type]['total']
        p_correct = by_pred_type[pred_type]['correct']
        print(f'  {pred_type}: {p_correct}/{p_total} = {p_correct/p_total:.2f}')

    print('-'*50)

        

