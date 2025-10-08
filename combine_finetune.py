import os


datasets = ["multiwoz", "dailydialog", "soda"] #, kandor]
splits = ["train", "dev", "test"]



for split in splits:
    if split == "dev":
        ilm_split = "valid"
    else:
        ilm_split = split
    with open(f"data/finetune_no_kandor/{ilm_split}.txt", "w") as outfile:
        for dataset in datasets:
            with open(f"all_finetune_data/{dataset}_deterministic_splits/{split}_finetune.txt") as infile:
                for line in infile:
                    outfile.write(line)
            outfile.write("\n")
    print(f"Created {ilm_split}.txt")