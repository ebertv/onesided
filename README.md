# Reading Between the Lines: the One-Sided Conversation Problem
Official code base for the paper [Reading Between the Lines: the One-Sided Conversation Problem](tbd)

## Installation

### 1. Create Environment

```
conda create -n onesided
conda activate onesided
```

### 2. Install Requirements

```
pip install -r requirements.txt
```

### 3. Collect Datasets
Download all datasets and place in the data folder for this repo
- [The Candor Corpus](https://betterup-data-requests.herokuapp.com/) 
- [MultiWOZ](https://www.kaggle.com/datasets/taejinwoo/multiwoz-22)
- [DailyDialog](https://www.kaggle.com/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in)
- [Soda](https://huggingface.co/datasets/allenai/soda)

Create the data splits by running `create_deterministic_splits.py --all --output_dir data_splits`. To create splits for only a specific dataset, replace `--all` with `--dataset`. Must be one of the above datasets. 

### 4. Create .env file

Create a .env file that contains:
```python
CLAUDE_API_KEY=<Your API Key>
OPENAI_API_KEY=<Your API Key>
```

## Prompting
### Other Party Reconstruction
#### 1. Create prompts for all data
To create reconstruction prompts for all conversations in a datset, run `python gather_prompts.py` with arguments depending on which setting you would like to run. 

**For full prior context with no anti-hallucination instructions:**
```
python gather_prompts.py \
--input_file <a dataset split created in Installation step 3>
--scenarios scenario_1_1
--use_few_shot
--all_turns
--save_prompts
--disable_anti_hallucination
```
Output will be saved to: `prompts_{dataset}_scenario_1_1_noxxxx.jsonl`

**For full prior context:** 
```
python gather_prompts.py \
--input_file <a dataset split created in Installation step 3>
--scenarios scenario_1_1
--use_few_shot
--all_turns
--save_prompts
```
Output will be saved to: `prompts_{dataset}_scenario_1_1.jsonl`

**For full prior context and one future turn:** 
```
python gather_prompts.py \
--input_file <a dataset split created in Installation step 3>
--scenarios scenario_1_2
--use_few_shot
--all_turns
--save_prompts
```
Output will be saved to: `prompts_{dataset}_scenario_1_2.jsonl`

**For full prior context and turn length markers:** 
```
python gather_prompts.py \
--input_file <a dataset split created in Installation step 3>
--scenarios scenario_2_1
--use_few_shot
--all_turns
--save_prompts
```
Output will be saved to: `prompts_{dataset}_scenario_2_1.jsonl`

**For full prior context and one future turn and turn length markers:**
```
python gather_prompts.py \
--input_file <a dataset split created in Installation step 3>
--scenarios scenario_2_2
--use_few_shot
--all_turns
--save_prompts
```
Output will be saved to: `prompts_{dataset}_scenario_2_1.jsonl`

**For local context:**
Make sure that you have shorter examples created by running:
```
python create_finetuning_from_splits.py --all
```
then:
```
python gather_prompts.py \
--input_file <a text file created from create_finetuning_from_splits.py>
--scenarios scenario_1_2
--use_few_shot
--all_turns
--save_prompts
```
Output will be saved to: `prompts_{input_base}_scenario_1_2_limited.jsonl`


#### 2. Generate predictions
To generate predictions, run:
```
python generate_predictions.py \
--input_file <FILE CREATED IN STEP 1>
--model <either claude or llama>
```
Output will be saved to:

#### 3. Evaluate the predictions
To evaluate predictions, run:
```
python evaluate_outputs \ 
--predictions_file <FILE CREATED IN STEP 2>
```
This will save per-utterance evaluations in a .jsonl file named the same as the prediction file with _evaluated appended at the end.

To view a whole file summary, run:
```
python evaluate_outputs \
--evaluation_file <FILE JUST CREATED>
--summarize
```

### Summarization
#### 1. Create prompts for all data
This can be done in two ways: with generated predicitons from Other Party Reconstruction or without. Inputs not using the generated predictions will therefore only be able to summarize full and masked conversations

**With predictions**
```
python gather_summary_prompts.py \
--input_file <Any file created in step 1 of other party reconstruction>
--save_prompts
```
For this paper we used those created under the setting with full prior context and one future turn and turn length markers.

**Without predictions**
```
python gather_summary_prompts.py \
--input_file <a dataset split created in Installation step 3>
--save_prompts
```

Both methods can adjust the prompts created using `--type` and any combination of `['masked', 'predicted', 'full']`. The output file can also be specified. By default, will be `{dataset}_summary_prompts_{'_'.join(args.type)}.jsonl`.

The format of the output will always be one row of the .jsonl file is equal to a single converstion, and includes the dialogue_id, each of the conversations (masked, predicted, and full) and the corresponding prompt for each.

#### 2. Generate summaries from each prompt
To generate summaries, run:
```
python generate_summary_predictions.py \
--input_file <FILE CREATED IN STEP 1>
```

The output file can also be specified. By default, will be the same as the input file but with "prompts" replaced with "predictions_{model}". Only claude summaries are supported for now.

#### 3. Evaluate the generated summaries
To evaluate the generates summaries, run:
```
python evaluate_summary_outputs.py \ 
--summaries_file <FILE CREATED IN STEP 2>
```

To view the summarized output of the dataset, run:
```
python evaluate_summary_outputs.py \
--evaluation_file <FILE JUST CREATED>
--summarize
```
^code not done yet


## Fine tuning

### 1. Format Data
To format data for fine-tuning, if you haven't already, first run:
```
python create_finetuning_from_splits.py --all
```

### 2. Model Set up
Clone [our fork of Chris Donahue's ILM code base](https://github.com/ebertv/ilm/tree/master). And follow all instructions there for creating a custom dataset and fine tuning LLaMA. 

To train your own model, you will need to add your HuggingFace access token to `train_ilm.py` and `tokenize_util.py`.

Otherwise: download [our trained model](https://drive.google.com/file/d/1G5Zs9255bolXA3oF5S2n7ZYJK5-O-xlR/view?usp=sharing) and unzip and place it in the ilm folder. 

### 3. Get Predictions
Then run
```
python inference.py <split> \
--examples_dir ./data/char_masks/custom/{dataset}
--model_type llama
```
Output will be saved to: `./data/char_masks/custom/{dataset}/ilm_{model}\_infill_{split}.jsonl`

To convert to the same format as the prompted models, run:
```
python find_real_convo.py \
--data_path ./data/char_masks/custom/{dataset}/ilm_{model}\_infill_{split}.jsonl
--full_convo_path <the corresponding {split}.json file from the data directory>
```

Evaluation is done in the same method as above:
```
python evaluate_outputs \ 
--predictions_file <FILE CREATED ABOVE>
```



## Citation
If you use this repository in your work please consider citing our paper:
```
TBD
```