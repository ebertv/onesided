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
    
def get_inter_annotator_agreement(data):
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    df = pd.DataFrame(data)
    annotators = df['rater'].unique()
    if len(annotators) < 2:
        return 1.0  # If there's only one annotator, agreement is perfect by definition
    pairs = [(annotators[i], annotators[j]) for i in range(len(annotators)) for j in range(i+1, len(annotators))]
    kappas = []
    for a1, a2 in pairs:
        df_a1 = df[df['rater'] == a1][['full_dialogue', 'answer']].rename(columns={'answer': 'answer_a1'})
        df_a2 = df[df['rater'] == a2][['full_dialogue', 'answer']].rename(columns={'answer': 'answer_a2'})
        merged = pd.merge(df_a1, df_a2, on='full_dialogue')
        if len(merged) > 0:
            kappa = cohen_kappa_score(merged['answer_a1'], merged['answer_a2'])
            kappas.append(kappa)
    if len(kappas) == 0:
        return 1.0
    return sum(kappas) / len(kappas)

if __name__ == '__main__':
    #collect all responses
    responses = []
    for file in os.listdir('.'):
        if file.endswith('_ab_test_answers.jsonl'):
            with open(file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    annotator = file.split('_')[0]
                    data['rater'] = annotator
                    responses.append(data)
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

    #make dataframe of each dialogue with number of times ground truth chosen, model chosen, tied
    import pandas as pd
    df = pd.DataFrame(data)
    
    dialogue_summary = df.groupby('full_dialogue').agg(
        total_responses=('answer', 'count'),
        chose_ground_truth=('chose_ground_truth', 'sum'),
        chose_model=('chose_model', 'sum'),
        tied=('tied', 'sum'),
        dataset=('dataset', 'first'),
        model_type=('pred_type', 'first')
    ).reset_index()
    #remove the full dialogue text and replace with index
    dialogue_summary['Full Dialogue'] = dialogue_summary.index
    dialogue_summary = dialogue_summary[['Full Dialogue', 'total_responses', 'chose_ground_truth', 'chose_model', 'tied', 'dataset', 'model_type']]
    dialogue_summary = dialogue_summary.rename(columns={'dataset': 'Dataset', 'model_type': 'Model Type'})
    dialogue_summary.to_csv('dialogue_summary.csv', index=False)
    print('Saved dialogue summary to dialogue_summary.csv')
    print('-'*50)

    #count number of dialogues where all responses were the same
    all_ground_truth = dialogue_summary[dialogue_summary['chose_ground_truth'] == dialogue_summary['total_responses']]
    all_model = dialogue_summary[dialogue_summary['chose_model'] == dialogue_summary['total_responses']]
    all_tied = dialogue_summary[dialogue_summary['tied'] == dialogue_summary['total_responses']]
    print(f'Number of dialogues where all responses chose ground truth: {len(all_ground_truth)}')
    print(f'Number of dialogues where all responses chose model: {len(all_model)}')
    print(f'Number of dialogues where all responses tied: {len(all_tied)}')
    print('-'*50)

    #create new dataframe with the model, the dataset, and the majority vote (ground truth, model, tied)
    def majority_vote(row):
        if row['chose_ground_truth'] > row['chose_model'] and row['chose_ground_truth'] > row['tied']:
            return 'ground_truth'
        elif row['chose_model'] > row['chose_ground_truth'] and row['chose_model'] > row['tied']:
            return 'model'
        elif row['tied'] > row['chose_ground_truth'] and row['tied'] > row['chose_model']:
            return 'tied'
        else:
            return 'no_majority'
    dialogue_summary['majority_vote'] = dialogue_summary.apply(majority_vote, axis=1)
    majority_summary = dialogue_summary.groupby(['Dataset', 'Model Type', 'majority_vote']).size().unstack(fill_value=0).reset_index()
    
    #average across model/dataset
    claude_multiwoz = majority_summary[(majority_summary['Dataset'] == 'multiwoz') & (majority_summary['Model Type'] == 'prompted')]
    llama_multiwoz = majority_summary[(majority_summary['Dataset'] == 'multiwoz') & (majority_summary['Model Type'] == 'finetune')]
    claude_dailydialog = majority_summary[(majority_summary['Dataset'] == 'dailydialog') & (majority_summary['Model Type'] == 'prompted')]
    llama_dailydialog = majority_summary[(majority_summary['Dataset'] == 'dailydialog') & (majority_summary['Model Type'] == 'finetune')]
    print('Majority vote summary by dataset and model type:')
    print(f'Claude on MultiWOZ: {claude_multiwoz.to_dict(orient="records")}')
    print(f'LLaMA on MultiWOZ: {llama_multiwoz.to_dict(orient="records")}')
    print(f'Claude on DailyDialog: {claude_dailydialog.to_dict(orient="records")}')
    print(f'LLaMA on DailyDialog: {llama_dailydialog.to_dict(orient="records")}')
    print('-'*50)
    #print majority summary as percentages
    def percentage(part, whole):
        if whole == 0:
            return 0
        return part / whole * 100
    majority_summary['total'] = majority_summary[['ground_truth', 'model', 'tied', 'no_majority']].sum(axis=1)
    majority_summary['pct_ground_truth'] = majority_summary.apply(lambda row: percentage(row['ground_truth'], row['total']), axis=1)
    majority_summary['pct_model'] = majority_summary.apply(lambda row: percentage(row['model'], row['total']), axis=1)
    majority_summary['pct_tied'] = majority_summary.apply(lambda row: percentage(row['tied'], row['total']), axis=1)
    majority_summary['pct_no_majority'] = majority_summary.apply(lambda row: percentage(row['no_majority'], row['total']), axis=1)
    print('Majority vote summary with percentages:')
    print(majority_summary[['Dataset', 'Model Type', 'total', 'ground_truth', 'pct_ground_truth', 'model', 'pct_model', 'tied', 'pct_tied', 'no_majority', 'pct_no_majority']])
    print('-'*50)
    exit()

    #calculate inter-annotator agreement
    iaa = get_inter_annotator_agreement(data)
    print(f'Inter-annotator agreement: {iaa:.2f}')

    

    # #get total correct overall, by dataset, by pred_type
    # total = 0
    # correct = 0
    # by_dataset = {}
    # by_pred_type = {}
    # for item in data:
    #     total += 1
    #     if item['answer'] == item['correct_answer']:
    #         correct += 1
    #         if item['dataset'] not in by_dataset:
    #             by_dataset[item['dataset']] = {'total': 0, 'correct': 0}
    #         by_dataset[item['dataset']]['total'] += 1
    #         by_dataset[item['dataset']]['correct'] += 1
    #         if item['pred_type'] not in by_pred_type:
    #             by_pred_type[item['pred_type']] = {'total': 0, 'correct': 0}
    #         by_pred_type[item['pred_type']]['total'] += 1
    #         by_pred_type[item['pred_type']]['correct'] += 1
    #     else:
    #         if item['dataset'] not in by_dataset:
    #             by_dataset[item['dataset']] = {'total': 0, 'correct': 0}
    #         by_dataset[item['dataset']]['total'] += 1
    #         if item['pred_type'] not in by_pred_type:
    #             by_pred_type[item['pred_type']] = {'total': 0, 'correct': 0}
    #         by_pred_type[item['pred_type']]['total'] += 1

    # print(f'Overall accuracy: {correct}/{total} = {correct/total:.2f}')
    # print('By dataset:')
    # for dataset in by_dataset:
    #     d_total = by_dataset[dataset]['total']
    #     d_correct = by_dataset[dataset]['correct']
    #     print(f'  {dataset}: {d_correct}/{d_total} = {d_correct/d_total:.2f}')
    # print('By prediction type:')
    # for pred_type in by_pred_type:
    #     p_total = by_pred_type[pred_type]['total']
    #     p_correct = by_pred_type[pred_type]['correct']
    #     print(f'  {pred_type}: {p_correct}/{p_total} = {p_correct/p_total:.2f}')

    # print('-'*50)

    # #for each dataset and pred type count number of responses with answer 0
    # ties_by_dataset = {}
    # ties_by_pred_type = {}
    # for item in data:
    #     if item['answer'] == 0:
    #         if item['dataset'] not in ties_by_dataset:
    #             ties_by_dataset[item['dataset']] = 0
    #         ties_by_dataset[item['dataset']] += 1
    #         if item['pred_type'] not in ties_by_pred_type:
    #             ties_by_pred_type[item['pred_type']] = 0
    #         ties_by_pred_type[item['pred_type']] += 1
    # print('Overall percentage of ties (answer=0): {}/{} = {:.2f}'.format(
    #     sum(ties_by_dataset.values()), total, sum(ties_by_dataset.values())/total))
    # print('Number of ties (answer=0) by dataset:')
    # for dataset in ties_by_dataset:
    #     print(f'  {dataset}: {ties_by_dataset[dataset]}/{by_dataset[dataset]["total"]} = {ties_by_dataset[dataset]/by_dataset[dataset]["total"]:.2f}')
    # print('Number of ties (answer=0) by prediction type:')
    # for pred_type in ties_by_pred_type:
    #     print(f'  {pred_type}: {ties_by_pred_type[pred_type]}/{by_pred_type[pred_type]["total"]} = {ties_by_pred_type[pred_type]/by_pred_type[pred_type]["total"]:.2f}')

    # print('-'*50)

    # #get accuracy as above but only for non-ties
    # total = 0
    # correct = 0
    # by_dataset = {}
    # by_pred_type = {}
    # for item in data:
    #     if item['answer'] == 0:
    #         continue
    #     total += 1
    #     if item['answer'] == item['correct_answer']:
    #         correct += 1
    #         if item['dataset'] not in by_dataset:
    #             by_dataset[item['dataset']] = {'total': 0, 'correct': 0}
    #         by_dataset[item['dataset']]['total'] += 1
    #         by_dataset[item['dataset']]['correct'] += 1
    #         if item['pred_type'] not in by_pred_type:
    #             by_pred_type[item['pred_type']] = {'total': 0, 'correct': 0}
    #         by_pred_type[item['pred_type']]['total'] += 1
    #         by_pred_type[item['pred_type']]['correct'] += 1
    #     else:
    #         if item['dataset'] not in by_dataset:
    #             by_dataset[item['dataset']] = {'total': 0, 'correct': 0}
    #         by_dataset[item['dataset']]['total'] += 1
    #         if item['pred_type'] not in by_pred_type:
    #             by_pred_type[item['pred_type']] = {'total': 0, 'correct': 0}
    #         by_pred_type[item['pred_type']]['total'] += 1

    # print(f'Overall accuracy (excluding ties): {correct}/{total} = {correct/total:.2f}')
    # print('By dataset (excluding ties):')
    # for dataset in by_dataset:
    #     d_total = by_dataset[dataset]['total']
    #     d_correct = by_dataset[dataset]['correct']
    #     print(f'  {dataset}: {d_correct}/{d_total} = {d_correct/d_total:.2f}')
    # print('By prediction type (excluding ties):')
    # for pred_type in by_pred_type:
    #     p_total = by_pred_type[pred_type]['total']
    #     p_correct = by_pred_type[pred_type]['correct']
    #     print(f'  {pred_type}: {p_correct}/{p_total} = {p_correct/p_total:.2f}')

    # print('-'*50)

        

