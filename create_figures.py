import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argparse import ArgumentParser

def create_human_summary_ranking_figure():
    rankings = {
        'DailyDialog': {
            'Masked': 38,
            'Reconstructed': 12
        },
        'MultiWOZ': {
            'Masked': 22,
            'Reconstructed': 28
        }
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (dataset, values) in zip(axes, rankings.items()):
        ax.bar(values.keys(), values.values(), color=['tab:orange', 'tab:green'])
        ax.set_title(dataset, fontsize=20)

        ax.set_ylabel("Count", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_ylim(0, max(values.values()) + 10)

    plt.tight_layout()
    plt.savefig('figures/human_summary_ranking_figure.pdf')


def create_summary_ranking_figure():
    rankings = {
        'DailyDialog': {
            'Full': {
                'First': (88.0, 1156),
                'Second': (9.7, 127),
                'Third': (2.3, 30)
            },
            'Masked': {
                'First': (4.3, 57),
                'Second': (69.8, 917),
                'Third': (25.8, 339)
            },
            'Reconstructed': {
                'First': (7.6, 100),
                'Second': (20.5, 269),
                'Third': (71.9, 944)
            }
        },
        'MultiWOZ': {
            'Full': {
                'First': (94.7, 947),
                'Second': (5.0, 50),
                'Third': (0.3, 3)
            },
            'Masked': {
                'First': (1.2, 12),
                'Second': (58.2, 582),
                'Third': (40.6, 406)
            },
            'Reconstructed': {
                'First': (4.1, 41),
                'Second': (36.8, 368),
                'Third': (59.1, 591)
            }
        },
        'Candor': {
            'Full': {
                'First': (40.0, 2),
                'Second': (60.0, 3),
                'Third': (0.0, 0)
            },
            'Masked': {
                'First': (40.0, 2),
                'Second': (40.0, 2),
                'Third': (20.0, 1)
            },
            'Reconstructed': {
                'First': (20.0, 1),
                'Second': (0.0, 0),
                'Third': (80.0, 4)
            }
        }  
    }

    datasets = list(rankings.keys())
    categories = ['Full', 'Masked', 'Reconstructed']
    ranks = ['First', 'Second', 'Third']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 7), sharey=True)
    if len(datasets) == 1:
        axes = [axes]  # Make it iterable
    for ax, dataset in zip(axes, datasets):
        x = np.arange(len(ranks))
        bar_width = 0.2

        for i, category in enumerate(categories):
            heights = [rankings[dataset][category][rank][0] for rank in ranks]
            xpos = x + (i - 1) * bar_width  # Center the bars
            bars = ax.bar(xpos, heights, bar_width, label=category, color=colors[i])
            
            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                # ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%',
                        # ha='center', va='bottom', fontsize=10)
    

        ax.set_title(dataset, fontsize=30)
        ax.set_ylim(0, 110)
        ax.set_xticks(x)
        ax.set_xticklabels(ranks, fontsize=30)
        ax.tick_params(axis='y', labelsize=30)


    # axes[0].set_ylabel('Percentage', fontsize=30)
    axes[0].set_yticks(range(0, 110, 10))
    # axes[0].set_yticklabels([f'{i}%' for i in range(0, 110, 10)], fontsize=30)
    #only set every other ytick label
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%' if y % 50 == 0 else ''))
    # Only one legend — place it in the last subplot
    axes[1].legend(title='Input Type', fontsize=30, title_fontsize=30, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.2))


    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.savefig('figures/summary_ranking_figure.pdf')

def create_precision_recall_figure():
    import matplotlib.gridspec as gridspec
    data = {
        'DailyDialog': {
            'Full Context': {'Precision': 0.09, 'Recall': 0.12},
            'Full Context No xxxx': {'Precision': 0.08, 'Recall': 0.13},
            'Full Context & N+1': {'Precision': 0.17, 'Recall': 0.22},
            'Full Context & Turn Lengths': {'Precision': 0.16, 'Recall': 0.16},
            'Full Context & N+1 & Turn Lengths': {'Precision': 0.24, 'Recall': 0.25},
            'Limited Context': {'Precision': 0.17, 'Recall': 0.20},
            'Limited Context (Llama)': {'Precision': 0.04, 'Recall': 0.06},
            'Limited Context (Finetuned Llama)': {'Precision': 0.09, 'Recall': 0.08},
        },
        'MultiWOZ': {
            'Full Context': {'Precision': 0.19, 'Recall': 0.27},
            'Full Context No xxxx': {'Precision': 0.21, 'Recall': 0.34},
            'Full Context (N+1)': {'Precision': 0.22, 'Recall': 0.30},
            'Full Context (Turn Lengths)': {'Precision': 0.33, 'Recall': 0.34},
            'Full Context (N+1 & Turn Lengths)': {'Precision': 0.31, 'Recall': 0.31},
            'Limited Context': {'Precision': 0.22, 'Recall': 0.27},
            'Limited Context (Llama)': {'Precision': 0.11, 'Recall': 0.13},
            'Limited Context (Finetuned Llama)': {'Precision': 0.22, 'Recall': 0.20},
        },
        'Candor': {
            'Full Context': {'Precision': 0.12, 'Recall': 0.13},
            'Full Context No xxxx': {'Precision': 0.11, 'Recall': 0.12},
            'Full Context (N+1)': {'Precision': 0.14, 'Recall': 0.15},
            'Full Context (Turn Lengths)': {'Precision': 0.39, 'Recall': 0.39},
            'Full Context (N+1 & Turn Lengths)': {'Precision': 0.48, 'Recall': 0.48},
            'Limited Context': {'Precision': 0.15, 'Recall': 0.15},
            'Limited Context (Llama)': {'Precision': 0.04, 'Recall': 0.05},
            'Limited Context (Finetuned Llama)': {'Precision': 0.10, 'Recall': 0.10},
        }
    }
    #Make two scatter plots stacked horizontally, one for each dataset
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    for i, (dataset, models) in enumerate(data.items()):
        for j, (model, scores) in enumerate(models.items()):
            axs[i].scatter(scores['Recall'], scores['Precision'], label=model, s=750, marker=markers[j])
        
            axs[i].set_title(f'{dataset}', fontsize=40)
            axs[i].set_xlabel('Recall', fontsize=40)
            axs[i].tick_params(axis='both', which='major', labelsize=40)

            # # Set different axis limits per subplot as you want
            # # Example (customize per dataset):
            # axs[i].set_xlim(min(scores['Recall']) - 0.1, max(scores['Recall']) + 0.1)
            # axs[i].set_ylim(min(scores['Precision']) - 0.1, max(scores['Precision']) + 0.1)

            # Make axes box square (fixed size on figure)
            axs[i].set_aspect('equal', adjustable='box')
        
    axs[0].set_ylabel('Precision', fontsize=40)
    handles, labels = axs[0].get_legend_handles_labels()  # Get legend info from one subplot

    # fig.legend(
    #     handles,
    #     labels,
    #     loc='center left',
    #     bbox_to_anchor=(0.9, 0.5),
    #     fontsize=40,
    #     borderaxespad=0,
    #     frameon=False,
    #     labelspacing=1,
    # )
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        fontsize=40,
        frameon=False,
        labelspacing=1.0,
        handletextpad=1.2,
        columnspacing=1.0,
        borderaxespad=0.0,
        markerscale=1.5
    )

    plt.savefig('figures/precision_recall_figure.pdf', bbox_inches='tight')

def create_summary_rubric():
    data ={
        "Dailydialog": {
            "Full": {
                "Total Score": {"value": 24.23, "error": 1.86},
                "content_coverage": {"value": 4.83, "error": 0.41},
                "dialogue_flow": {"value": 4.80, "error": 0.45},
                "information_accuracy": {"value": 4.87, "error": 0.41},
                "purpose_outcome": {"value": 4.89, "error": 0.36},
                "detail_balance": {"value": 4.84, "error": 0.42}
            },
            "Masked": {
                "Total Score": {"value": 17.50, "error": 3.13},
                "content_coverage": {"value": 3.44, "error": 0.64},
                "dialogue_flow": {"value": 3.58, "error": 0.68},
                "information_accuracy": {"value": 3.51, "error": 0.69},
                "purpose_outcome": {"value": 3.59, "error": 0.79},
                "detail_balance": {"value": 3.38, "error": 0.70}
            },
            "Predicted": {
                "Total Score": {"value": 14.84, "error": 4.72},
                "content_coverage": {"value": 3.04, "error": 0.94},
                "dialogue_flow": {"value": 3.11, "error": 0.93},
                "information_accuracy": {"value": 2.61, "error": 1.08},
                "purpose_outcome": {"value": 3.09, "error": 1.04},
                "detail_balance": {"value": 2.99, "error": 0.97}
            }
        },
        "Multiwoz": {
            "Full": {
                "Total Score": {"value": 24.77, "error": 0.87},
                "content_coverage": {"value": 4.95, "error": 0.22},
                "dialogue_flow": {"value": 4.91, "error": 0.29},
                "information_accuracy": {"value": 4.95, "error": 0.24},
                "purpose_outcome": {"value": 4.98, "error": 0.15},
                "detail_balance": {"value": 4.97, "error": 0.18}
            },
            "Masked": {
                "Total Score": {"value": 17.58, "error": 2.56},
                "content_coverage": {"value": 3.35, "error": 0.53},
                "dialogue_flow": {"value": 3.69, "error": 0.62},
                "information_accuracy": {"value": 3.40, "error": 0.55},
                "purpose_outcome": {"value": 3.74, "error": 0.67},
                "detail_balance": {"value": 3.41, "error": 0.57}
            },
            "Predicted": {
                "Total Score": {"value": 16.64, "error": 3.66},
                "content_coverage": {"value": 3.26, "error": 0.74},
                "dialogue_flow": {"value": 3.54, "error": 0.73},
                "information_accuracy": {"value": 3.01, "error": 0.89},
                "purpose_outcome": {"value": 3.56, "error": 0.83},
                "detail_balance": {"value": 3.28, "error": 0.80}
            }
        },

        "Candor": {
            "Full": {
                "Total Score": {"value": 22.00, "error": 2.74},
                "content_coverage": {"value": 4.40, "error": 0.55},
                "dialogue_flow": {"value": 4.40, "error": 0.55},
                "information_accuracy": {"value": 4.40, "error": 0.55},
                "purpose_outcome": {"value": 4.40, "error": 0.55},
                "detail_balance": {"value": 4.40, "error": 0.55}
            },
            "Masked": {
                "Total Score": {"value": 22.00, "error": 2.55},
                "content_coverage": {"value": 4.40, "error": 0.55},
                "dialogue_flow": {"value": 4.40, "error": 0.55},
                "information_accuracy": {"value": 4.40, "error": 0.55},
                "purpose_outcome": {"value": 4.60, "error": 0.55},
                "detail_balance": {"value": 4.20, "error": 0.84}
            },
            "Predicted": {
                "Total Score": {"value": 18.80, "error": 3.77},
                "content_coverage": {"value": 3.80, "error": 0.84},
                "dialogue_flow": {"value": 3.80, "error": 0.84},
                "information_accuracy": {"value": 3.80, "error": 0.84},
                "purpose_outcome": {"value": 4.00, "error": 0.71},
                "detail_balance": {"value": 3.40, "error": 0.89}
            }
        }
    }

    score_names = ["content_coverage", "dialogue_flow", "information_accuracy", "purpose_outcome", "detail_balance"]
    settings = ["Full", "Masked", "Predicted"]

    # Create subplots
    n_datasets = len(data)
    fig, axs = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6), sharey=True)

    if n_datasets == 1:
        axs = [axs]  # Make it iterable

    # Plot each dataset
    for ax, (dataset_name, dataset_data) in zip(axs, data.items()):
        x = np.arange(len(score_names))
        width = 0.25

        for i, setting in enumerate(settings):
            values = [dataset_data[setting][score]['value'] for score in score_names]
            errors = [dataset_data[setting][score]['error'] for score in score_names]

            ax.bar(x + i * width, values, width, yerr=errors, label=setting, capsize=5)

        ax.set_title(dataset_name)
        ax.set_xticks(x + width)
        ax.set_xticklabels(score_names, rotation=30, ha='right')

        for y in range(1, 5):
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylim(0, 6)
        ax.set_yticks(range(0, 6))
        ax.grid(axis='y')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    #Set common labels
    axs[0].set_ylabel('Score')
    #put one legend for all subplots at the bottom center
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(settings))
            

    plt.tight_layout()
    plt.savefig('figures/summary_rubric_figure.pdf')


def create_human_reconstruction_figure():
    # with open('/gscratch/scrubbed/ebertv/onesided/nonauthor_ab_test/dialogue_summary.csv') as f:
    #     df = pd.read_csv(f, index_col=0)
    
    # df = df.drop(columns=['total_responses'])

    # #make each cell in the non total-response rows a percentage of the total responses in that row
    # for i in range(len(df)):
    #     total = df.iloc[i].sum()
    #     if total > 0:
    #         df.iloc[i] = df.iloc[i] / total * 100
    

    # df.columns = [1, 2, 3]
    # df.index.name = None

    # #set index to ints
    # df.index = df.index.astype(int)

    # df1 = df[df['Dataset'] == 'DailyDialog'].T
    # df2 = df[df['Dataset'] == 'MultiWOZ'].T

    import seaborn as sns
    # from matplotlib.ticker import FuncFormatter
    
    # plt.figure(figsize=(20, 4))
    # sns.set(font_scale=2)
    # ax = sns.heatmap(df, cmap='Blues', fmt='g', cbar=True, cbar_kws=dict(use_gridspec=False,location="top", pad=0.01, shrink=100), vmin=0, vmax=100, square=True,)
    
    # import matplotlib.patches as mpatches
    # # Define your custom labels
    # column_legend = {
    #     1: 'Ground Truth',
    #     2: 'Reconstructed',
    #     3: 'Tied'
    # }
    # # Create legend handles
    # legend_handles = [mpatches.Patch(color='white', label=f'{k}: {v}') for k, v in column_legend.items()]
    # # Then after your heatmap code:
    # plt.legend(
    #     handles=legend_handles,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.7),
    #     ncol=3,
    #     frameon=False,
    #     fontsize=20,
    # )

    # #add percentage sign to colorbar
    # colorbar = ax.collections[0].colorbar
    # colorbar.set_ticks([0, 20, 40, 60, 80, 100])
    # colorbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    # colorbar.ax.tick_params(labelsize=20)


    # plt.subplots_adjust(left=0.02, right=0.98, top=0.60, bottom=0.3)

    csv_path = '/gscratch/scrubbed/ebertv/onesided/nonauthor_ab_test/dialogue_summary.csv'

    # load dataframe (either from string or path)
    df = pd.read_csv(csv_path)
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib import cm

    # ---- Normalize column names ----
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename for convenience
    df = df.rename(columns={
        "full_dialogue": "dialogue",
        "total_responses": "total",
        "chose_ground_truth": "gt",
        "chose_model": "model",
        "tied": "tied",
        "dataset": "dataset",
        "model_type": "mtype"
    })

    # Compute percentages
    for col in ["gt", "model", "tied"]:
        df[f"pct_{col}"] = (df[col] / df["total"]) * 100

    # ---- Helper plotting function ----
    def plot_dataset(ax, subset, dataset_name, base_colors, preferred_order):
        subset["mtype"] = subset["mtype"].str.lower().str.strip()
        subset["dialogue"] = subset["dialogue"].astype(str)

        # Order columns: finetune first, then prompted
        ordered = [m for m in preferred_order if m in subset["mtype"].unique()]
        col_labels, model_per_col = [], []
        for mt in ordered:
            subm = subset[subset["mtype"] == mt].sort_values("dialogue")
            for _, row in subm.iterrows():
                col_labels.append(f"{row['dialogue']}_{mt}")
                model_per_col.append(mt)

        # Matrix rows = gt, model, tied
        vals = np.zeros((3, len(col_labels)))
        for j, lbl in enumerate(col_labels):
            row = subset.iloc[j]
            vals[:, j] = [row["pct_gt"], row["pct_model"], row["pct_tied"]]

        # Build RGB image
        rgb = np.ones((3, len(col_labels), 3))
        for j, mt in enumerate(model_per_col):
            base = np.array(base_colors[mt])
            for i in range(3):
                intensity = vals[i, j] / 100.0
                rgb[i, j, :] = (1 - intensity) * np.ones(3) + intensity * base

        # Plot
        ax.imshow(rgb, aspect="auto", interpolation="nearest", origin="upper")
        # ax.set_xticks(np.arange(len(col_labels)))
        # ax.set_xticklabels(col_labels, rotation=90, fontsize=6)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["ground_truth", "model", "tied"], fontsize=30)
        ax.set_title(f"{dataset_name} — {len(col_labels)} dialogues", fontsize=30)

    # ---- Colors ----
    base_colors = {
        "finetune": np.array([31, 119, 180]) / 255.0,   # blue
        "prompted": np.array([44, 160, 44]) / 255.0     # green
    }
    preferred_order = ["finetune", "prompted"]

    # ---- Plot both datasets ----
    datasets = ["dailydialog", "multiwoz"]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 7*len(datasets)))

    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"].str.lower() == ds]
        if not sub.empty:
            plot_dataset(ax, sub, ds, base_colors, preferred_order)
        else:
            ax.text(0.5, 0.5, f"No data for {ds}", ha="center", va="center")
            ax.axis("off")

    # ---- Single shared colorbars ----
    fig.subplots_adjust(right=0.85)
    for idx, (mt, color) in enumerate(base_colors.items()):
        cmap = LinearSegmentedColormap.from_list(f"cm_{mt}", [(1,1,1), color])
        norm = Normalize(0, 100)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.88, 0.25 + idx*0.25, 0.015, 0.2])  # position
        cb = fig.colorbar(sm, cax=cax, orientation="vertical")
        cb.set_label(f"{mt} %", fontsize=30)
    plt.savefig('figures/human_reconstruction_figure.pdf')

def summary_precision_recall():
    data = {
        'DailyDialog': {
            'Masked': {'Precision': 0.694, 'Recall': 0.557},
            'Reconstructed': {'Precision': 0.552, 'Recall': 0.544},
        },
        'MultiWOZ': {
            'Masked': {'Precision': 0.770, 'Recall': 0.527},
            'Reconstructed': {'Precision': 0.672, 'Recall': 0.576},
        },
        'Candor': {
            'Masked': {'Precision': 0.603, 'Recall': 0.517},
            'Reconstructed': {'Precision': 0.560, 'Recall': 0.500},
        }
    }
    
    #make three scatter plots all square, one for each dataset
    markers = ['o', 's']
    colors = ['tab:orange', 'tab:green']
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (dataset, models) in zip(axes, data.items()):
        for i, (model, scores) in enumerate(models.items()):
            ax.scatter(scores['Recall'], scores['Precision'], label=model, s=750, marker=markers[i], color=colors[i])
        
            ax.set_title(f'{dataset}', fontsize=40)
            ax.set_xlabel('Recall', fontsize=40)
            ax.tick_params(axis='both', which='major', labelsize=40)

            # Make axes box square (fixed size on figure)
            ax.set_aspect('equal', adjustable='box')
            ax.set_box_aspect(1)

    axes[0].set_ylabel('Precision', fontsize=40)
    handles, labels = axes[0].get_legend_handles_labels()  # Get legend info from one subplot
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        fontsize=40,
        frameon=False,
        labelspacing=1.0,
        handletextpad=1.2,
        columnspacing=1.0,
        borderaxespad=0.0,
        markerscale=1.5
    )
    fig.subplots_adjust(wspace=0.3)
    plt.savefig('figures/summary_precision_recall_figure.pdf', bbox_inches='tight')



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('figure', type=str)
    args = parser.parse_args()

    if args.figure == 'human_summary':
        create_human_summary_ranking_figure()
    elif args.figure == 'summary_ranking':
        create_summary_ranking_figure()
    elif args.figure == 'precision_recall':
        create_precision_recall_figure()
    elif args.figure == 'summary_rubric':
        create_summary_rubric()
    elif args.figure == 'human_reconstruction':
        create_human_reconstruction_figure()
    elif args.figure == 'summary_precision_recall':
        summary_precision_recall()
    else:
        print(f'Unknown figure type: {args.figure}')
        exit(1)