import json
from pathlib import Path
import os

def get_full_conversations(input_file, file_type):
    #if file_type is 'raw', just combine utterances and speakers
    #if file_type is 'predicted', take "full_dialogue" field from each line
    convos = []
    if file_type == "raw":
        with open(input_file, "r") as f:
            data = json.load(f)
            for dialogue in data:
                dialogue_id = dialogue["dialogue_id"]
                convo = ""
                speakers = dialogue["speakers"]
                utterances = dialogue["utterances"]
                for i, (speaker, utterance) in enumerate(zip(speakers, utterances)):
                    convo += f"Turn {i+1} [{speaker}]: {utterance}\n"
                convos.append((dialogue_id, convo.strip()))
    elif file_type == "predicted":
        with open(input_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                dialogue_id = entry["dialogue_id"]
                convos.append((dialogue_id, entry["full_dialogue"].strip()))
    return convos

def get_masked_conversations(full_conversation):
    convos = []
    masked_speakers = ["Speaker_2", "Participant_R"]
    for convo in full_conversation:
        dialogue_id, convo = convo
        masked_convo = ""
        lines = convo.strip().split("\n")
        for line in lines:
            if any(speaker in line for speaker in masked_speakers):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    masked_convo += f"{parts[0]}: [MASKED]\n"
                else:
                    masked_convo += line + "\n"
            else:
                masked_convo += line + "\n"
        convos.append((dialogue_id, masked_convo.strip()))
    return convos

def get_predicted_conversations(input_file):
    #extract predicted user turns from input_file and combine with system turns
    convos = []
    pred_convo = ""
    idx = None
    predictions = {}
    with open(input_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            if idx is None:
                idx = entry["dialogue_id"]
                full_convo = entry["full_dialogue"]
            if entry["dialogue_id"] != idx:
                turns = full_convo.split("\n")
                for turn_idx in sorted(predictions.keys()):
                    turns[turn_idx-1] = f"Turn {turn_idx} [Speaker_2]: {predictions[turn_idx]}"
                pred_convo = "\n".join(turns)
                convos.append((idx, pred_convo.strip()))
                pred_convo = ""
                idx = entry["dialogue_id"]
                full_convo = entry["full_dialogue"]
                predictions = {}
            turn_idx = entry["turn_id"]
            predictions[turn_idx] = entry["predictions"][0] if len(entry["predictions"]) > 0 else "[NO PREDICTION]"
        turns = full_convo.split("\n")
        for turn_idx in sorted(predictions.keys()):
            turns[turn_idx-1] = f"Turn {turn_idx} [Speaker_2]: {predictions[turn_idx]}"
        pred_convo = "\n".join(turns)
        convos.append((idx, pred_convo.strip()))
    return convos
            
def make_prompts(conversations):
    system_prompt = """You are a dialogue summarization assistant. You will be given a conversation between two speakers.

Your task is to provide a comprehensive summary that focuses on:
- The main purpose and goal of the conversation
- What the speakers are trying to accomplish
- The flow and progression of their requests/statements
- Important details that are relevant to the conversation
- The overall purpose and outcome of the conversation

IMPORTANT: If you encounter 'XXXXXXX' in the dialogue, this represents placeholder information (like specific names, numbers, addresses, times, etc.) that was not available in the original context. 
- Treat XXXXXXX as a specific number or piece of information, and use the placeholder in your summary if it is relevant to the summary.
- DO NOT MENTION that [MASKED] or XXXXXXX is being used at all in your summary.
- DO NOT make up or invent specific details to replace XXXXXXX

These should be in one summary paragraph, not broken up into multiple bullets. 
Provide a clear, concise summary that captures the essential elements of the conversation. """
    outputs = {}
    for idx in conversations:
        outputs[idx] = {}
        outputs[idx]["dialogues"] = {}
        prompts = {}
        for scenario_type in conversations[idx]:
            dialogue_context = conversations[idx][scenario_type]
            user_prompt = f"""Please summarize the following dialogue:

    {dialogue_context}

Summary:"""
            
            if scenario_type == "full":
                prompts["full"] = {"system_prompt": system_prompt, "user_prompt": user_prompt}
            elif scenario_type == "masked":
                masked_system_prompt = system_prompt + "\n\nNote: One speaker's responses are masked with [MASKED] in this conversation, so you'll only have partial information."
                prompts["masked"] = {"system_prompt": masked_system_prompt, "user_prompt": user_prompt}
            elif scenario_type == "predicted":
                predicted_system_prompt = system_prompt + "\n\nNote: Some responses in this conversation are predictions made by an AI system and may contain XXXXXXX placeholders for unknown specific information."
                prompts["predicted"] = {"system_prompt": predicted_system_prompt, "user_prompt": user_prompt}
            outputs[idx]["dialogues"][scenario_type] = dialogue_context
        
        outputs[idx]["prompts"] = prompts



    return outputs

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Generate prompts for one-sided dialogue scenarios.")
    parser.add_argument("--save_prompts", action="store_true", help="Whether to save prompts to a file (JSONL format). If not set, prompts are not saved.")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path to save prompts (JSONL format). If not specified, the file is named prompts_<ARGS>.jsonl.")
    parser.add_argument("--input_file", required=True, type=str, default=None, help="Either a single JSON file for a specific dataset split or a JSONL file with generated predicted user turns.")
    parser.add_argument("--type", type=str, choices=["masked", "predicted", "full", "all"], default=["all"], nargs="+",
                        help="Type of prompts to generate: 'masked' (with masked user turns), 'predicted' (with predicted user turns), 'full' (with full dialogues), or 'all' (all types). By default, all types are generated." \
                        "if --input_file is not a JSONL format containing predictions from a model, only 'masked' and 'full' types are applicable.")
    
    args = parser.parse_args()

    if args.type == ["all"]:
        args.type = ["masked", "predicted", "full"]

    if args.input_file.endswith(".jsonl"):
        dataset = args.input_file.split("/")[-1].split("_")[0]
    elif args.input_file.endswith(".json"):
        if "predicted" in args.type:
            print("WARNING: When --input_file is a JSON file, only 'masked' and 'full' prompt types are applicable. Updating --type accordingly.")
            args.type = [t for t in args.type if t != "predicted"]
            con = input("Continue? (y/n): ")
            if con.lower() != "y":
                exit(0)
        dataset = os.path.dirname(args.input_file).split("/")[-1].split("_")[0]

    if args.save_prompts and not args.output_file:
        args.output_file = f"{dataset}_summary_prompts_{'_'.join(args.type)}.jsonl"
    

    print(f"Generating prompts of type(s): {args.type}")
    print(f"Saving output to: {args.output_file}" if args.save_prompts else "Not saving output to file.")



    full_convos = get_full_conversations(args.input_file, "predicted" if args.input_file.endswith(".jsonl") else "raw")
    masked_convos = get_masked_conversations(full_convos) if "masked" in args.type else None
    predicted_convos = get_predicted_conversations(args.input_file) if "predicted" in args.type else None

    convos = {}
    for i in range(len(full_convos)):
        dialogue_id = full_convos[i][0]
        convos[dialogue_id] = {}
        convos[dialogue_id]["full"] = full_convos[i][1]
    for i in range(len(masked_convos) if masked_convos else 0):
        dialogue_id = masked_convos[i][0]
        if dialogue_id in convos:
            convos[dialogue_id]["masked"] = masked_convos[i][1]
    for i in range(len(predicted_convos) if predicted_convos else 0):
        dialogue_id = predicted_convos[i][0]
        if dialogue_id in convos:
            convos[dialogue_id]["predicted"] = predicted_convos[i][1]

    for dialogue_id in convos:
        if convos[dialogue_id].keys() != set(args.type):
            print(f"WARNING: Dialogue ID {dialogue_id} does not have all requested prompt types. Available types: {list(convos[dialogue_id].keys())}")

    prompts = make_prompts(convos)
    print(f"Generated prompts for {len(prompts)} conversations.")
    

    if args.save_prompts:
        with open(args.output_file, "w") as f:
            for dialogue_id in tqdm(prompts, desc="Saving prompts"):
                entry = {
                    "dialogue_id": dialogue_id,
                    "dialogues": prompts[dialogue_id]["dialogues"],
                    "prompts": prompts[dialogue_id]["prompts"]
                }
                f.write(json.dumps(entry) + "\n")
        print(f"Prompts saved to {args.output_file}.")

    