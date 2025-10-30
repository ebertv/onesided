import json


def create_full_convos(real_data):
    full_convos = []
    for dialog in real_data:
        speakers = dialog['speakers']
        speaker1 = speakers[0]
        speakers = ["Speaker_1" if s == speaker1 else "Speaker_2" for s in speakers]
        utterances = dialog['utterances']
        convo = []
        for i, utt in enumerate(utterances):
            speaker = speakers[i]
            text = utt
            convo.append(f"Turn {i+1} [{speaker}]: {text}")
        full_convo = " ".join(convo)
        full_convos.append(full_convo)
    return full_convos

def get_real_convo(real_convos, full_context):
    for convo in real_convos:
        if full_context in convo:
            return convo
    return None

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--full_convo_path', type=str)

    args = parser.parse_args()

    blank_pred = {
        "scenario": None,
        "full_dialogue": None,
        "actual_response": None,
        "predictions": [None],
    }
    
    ilm_path = args.data_path
    real_convo_path = args.full_convo_path

    with open(ilm_path, 'r') as f:
        ilm_data = [json.loads(line) for line in f.readlines()]
    with open(real_convo_path, 'r') as f:
        real_data = json.load(f)

    real_convos = create_full_convos(real_data)
    
    real_convos = real_convos[:5]

    predictions = []
    print(f"Generating predictions for {len(ilm_data)} ILM examples")
    for i in range(len(ilm_data)):
        full_context = ilm_data[i]['full_context']
        real_convo = get_real_convo(real_convos, full_context)
        if real_convo is None:
            print(f"Could not find real convo for test example {i}")
            continue
        temp = blank_pred.copy()
        temp['scenario'] = '3turn_dialogue'
        temp['actual_response'] = ilm_data[i]['real_infill']
        temp['predictions'] = [ilm_data[i]['generated_infill']]
        temp['full_dialogue'] = ilm_data[i]['full_context']

        # full_context = ilm_data[i]['full_context']
        # real_convo = get_real_convo(real_convos, full_context)
        # if real_convo is None:
        #     print(f"Could not find real convo for test example {i}")
        #     continue
        # temp['full_dialogue'] = real_convo

        predictions.append(temp)

    print(f"Generated {len(predictions)} predictions")

    output = ilm_path.replace('.jsonl', '_withfullconvo.jsonl')
    with open(output, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print(f"Wrote predictions to {output}")



        


