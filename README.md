# Reading Between the Lines: the One-Sided Conversation Problem
Official code base for the paper [Reading Between the Lines: the One-Sided Conversation Problem](tbd)

## Installation

### 1. Create Environment

```sh
conda create -n onesided
conda activate onesided
```

### 2. Install Requirements

```sh
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
### Other Party Recreation
#### 1. Create prompts for all data
To create recreation prompts for all conversations in a datset, run `python gather_prompts.py` with arguments depending on which setting you would like to run. 

For full prior context with no anti-hallucination instructions:
```sh
TBD
```
Output will be saved to:

---
For full prior context:
```sh
TBD
```
Output will be saved to:

---
For full prior context and one future turn:
```sh
TBD
```
Output will be saved to:

---
For full prior context and turn length markers:
```sh
TBD
```
Output will be saved to:

---
For full prior context and one future turn and turn length markers:
```sh
TBD
```
Output will be saved to:

---
For local context:
```sh
TBD
```
Output will be saved to:


#### 2. Generate predictions
To generate predictions, run:
```sh
python generate_predictions.py \
--input_file <FILE CREATED IN STEP 1>
--model <either claude or llama>
```
Output will be saved to:

### Summarization
TBD


## Fine tuning

### 1. Format Data
TBD

### 2. Model Set up
If you wish to fine tune your own LLaMA model for conversational infilling,
Clone [our fork of Chris Donahue's ILM code base](https://github.com/ebertv/ilm/tree/master) and follow all instructions there for creating a custom dataset and fine tuning LLaMA.

Otherwise: TBD with our checkpoints

## Evaluation


## Citation
If you use this repository in your work please consider citing our paper:
```
TBD
```