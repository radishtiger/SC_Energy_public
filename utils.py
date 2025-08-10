import yaml, time, random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import pandas as pd

# from pypalettes import get_hex


def merge_dict(dictA, dictB):
    new_dict = dictB

    for key, val in dictA.items():
        new_dict[key] = val

    return new_dict


def parser_add(parser, mode="train"):
    """
    mode = one of 'train', 'eval', 'locate'

    """

    rand_time_to_stop = random.random() * 5
    time.sleep(rand_time_to_stop)

    parser.add_argument("--config", default="config.yaml", type=str)
    parser.add_argument("--repre_model", type=str)
    parser.add_argument("--decomposition", type=str)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--loss_fully_separate", type=bool, default=False)
    parser.add_argument("--loss_subset_ordering", type=bool, default=False)
    parser.add_argument("--loss_con_incon_ordering", type=bool, default=False)
    parser.add_argument("--loss_incon_incon_ordering", type=bool, default=False)
    parser.add_argument("--one_pos_vs_many_neg", type=bool, default=False)
    parser.add_argument("--many_pos_vs_one_neg", type=bool, default=False)
    parser.add_argument("--task", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--scratch", type=bool, default=False)
    parser.add_argument("--time_key", type=str, default="")
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument(
        "--finetune_job_id", type=int, default=0
    )  # use the model from this job ID and save outputs under job_id
    parser.add_argument("--pairwise", type=str, default="arbitrary_pairs")
    parser.add_argument("--type", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--shot_num", type=int, default=0)
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--extra_for_eval", type=str, default="")
    parser.add_argument("--locate_type", type=str, default="")
    parser.add_argument("--stepwise_dataset_train_num", type=int, default=None)
    parser.add_argument("--stepwise_dataset_eval_num", type=int, default=None)
    parser.add_argument("--stepwise_dataset_eval2_num", type=int, default=None)
    parser.add_argument("--stepwise_dataset_test_num", type=int, default=None)

    return parser


def params_add(args, mode="train"):
    """
    mode = one of 'train', 'eval', 'locate'

    """

    config_file = args.config

    # Load the configuration yaml file
    with open(config_file, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # Add the arguments you added at the terminal into the configuration variable.
    replace_args = (
        "loss_type",
        "loss_fully_separate",
        "loss_subset_ordering",
        "loss_con_incon_ordering",
        "loss_incon_incon_ordering",
        "one_pos_vs_many_neg",
        "many_pos_vs_one_neg",
        "repre_model",
        "decomposition",
    )
    for r in replace_args:
        if getattr(args, r) != None:
            params["energynet"][r] = getattr(args, r)

    replace_args = (
        "task",
        "dataset",
        "config",
        "scratch",
        "time_key",
        "pairwise",
        "extra_for_eval",
        "job_id",
        "finetune_job_id",
    )
    for r in replace_args:
        if getattr(args, r) != None:
            params[r] = getattr(args, r)

    replace_args = ("type", "model", "shot_num", "prediction_type")
    for r in replace_args:
        if getattr(args, r) != None:
            params["baseline"][r] = getattr(args, r)

    replace_args = ("locate_type",)
    for r in replace_args:
        if getattr(args, r) != None:
            params["locate"][r] = getattr(args, r)

    # Additional configs
    if params["energynet"]["loss_type"] in {
        "supervised",
    }:
        params["energynet"]["output_form"] = "2dim_vec"
        print(f"loss type: {params['energynet']['loss_type']}")

    elif params["energynet"]["loss_type"] in {"triplet", "NCE"}:
        params["energynet"]["output_form"] = "real_num"
        print(f"loss type: {params['energynet']['loss_type']}")

    else:
        print(
            f"Invalid loss type and output_form.\nCurrent: (loss_type, output_form) = ({params['energynet']['loss_type']}, None)"
        )

    replace_args = (
        "stepwise_dataset_train_num",
        "stepwise_dataset_eval_num",
        "stepwise_dataset_eval2_num",
        "stepwise_dataset_test_num",
    )
    for r in replace_args:
        if getattr(args, r) != None:
            if params["dataset"] not in params:
                params[params["dataset"]] = {}
            params[params["dataset"]][r] = getattr(args, r)

    if params["finetune_job_id"] != 0:
        dataset_list = ["lconvqa", "set_nli"]
        for d in dataset_list:
            for r in replace_args:
                if getattr(args, r) != None:
                    if d not in params:
                        params[d] = {}
                    params[d][r] = getattr(args, r)

    return params


def draw_violin_plot(
    total_info_for_violinplot, vio_plot_column_name, eval_steps_names, save_paths
):

    # Create dataframe from provided data
    df = pd.DataFrame(data=total_info_for_violinplot, columns=vio_plot_column_name)

    # Specify the desired order for the set types
    desired_order = [
        "con",
        "con_con",
        "con_con_con",
        "con_con_con_con",  # S_{C}, S_{CC}, S_{CCC}, S_{CCCC}
        "con_con_con_incon",
        "con_con_incon",
        "con_incon",
        "incon",  # S_{CCCI}, S_{CCI}, S_{CI}, S_{I}
        "con_con_incon_incon",
        "con_incon_incon",
        "incon_incon",  # S_{CCII}, S_{CII}, S_{II}
        "con_incon_incon_incon",
        "incon_incon_incon",
        "incon_incon_incon_incon",  # S_{CIII}, S_{III}, S_{IIII}
    ]
    df["datatype"] = pd.Categorical(
        df["datatype"], categories=desired_order, ordered=True
    )
    df = df.sort_values("datatype")

    # Define background color mapping: if the key contains only "con", use one color; if it includes "incon", use another.
    bg_colors = {
        "con": "#80C5E7",
        "con_con": "#80C5E7",
        "con_con_con": "#80C5E7",
        "con_con_con_con": "#80C5E7",
        "con_con_con_incon": "#E8B97A",
        "con_con_incon": "#E8B97A",
        "con_incon": "#E8B97A",
        "incon": "#E8B97A",
        "con_con_incon_incon": "#E8B97A",
        "con_incon_incon": "#E8B97A",
        "incon_incon": "#E8B97A",
        "con_incon_incon_incon": "#E8B97A",
        "incon_incon_incon": "#E8B97A",
        "incon_incon_incon_incon": "#E8B97A",
    }

    # Define LaTeX labels mapping
    label_mapping = {
        "con": r"$S_{C}$",
        "con_con": r"$S_{CC}$",
        "con_con_con": r"$S_{CCC}$",
        "con_con_con_con": r"$S_{CCCC}$",
        "con_con_con_incon": r"$S_{CCCI}$",
        "con_con_incon": r"$S_{CCI}$",
        "con_incon": r"$S_{CI}$",
        "incon": r"$S_{I}$",
        "con_con_incon_incon": r"$S_{CCII}$",
        "con_incon_incon": r"$S_{CII}$",
        "incon_incon": r"$S_{II}$",
        "con_incon_incon_incon": r"$S_{CIII}$",
        "incon_incon_incon": r"$S_{III}$",
        "incon_incon_incon_incon": r"$S_{IIII}$",
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Draw horizontal separator lines for clarity between each set type
    for i in range(len(desired_order) - 1):
        plt.axhline(i + 0.5, color="gray", linestyle="-", linewidth=1)

    # Add background color spans for each set type row
    for i, key in enumerate(desired_order):
        plt.axhspan(i - 0.5, i + 0.5, facecolor=bg_colors[key], alpha=0.3)

    # Create the boxplot (without filling the boxes, i.e. transparent boxes with black edges)
    sns.boxplot(
        data=df,
        y="datatype",
        x="energy value",
        order=desired_order,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        ax=ax,
    )

    # Set y-axis tick labels to the LaTeX labels
    plt.yticks(
        ticks=range(len(desired_order)),
        labels=[label_mapping[label] for label in desired_order],
        fontsize=12,
    )
    # Set axis labels with increased font size
    ax.set_xlabel("Energy Value", fontsize=18)
    ax.set_ylabel("Set Type", fontsize=18)

    # Draw a double-headed arrow along the x-axis below the boxplot to indicate the meaning of energy values
    xmin, xmax = ax.get_xlim()
    # Set a y-coordinate below the current y-axis (e.g., -1.5, adjust if needed)
    y_arrow = 13.5
    ax.annotate(
        "",
        xy=(xmin, y_arrow),
        xycoords="data",
        xytext=(xmax, y_arrow),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", color="black", lw=2),
    )
    # Add text labels at the left and right ends of the arrow
    ax.text(
        xmin,
        y_arrow + 0.7,
        "Consistent",
        fontsize=14,
        ha="left",
        va="top",
        color="black",
    )
    ax.text(
        xmax,
        y_arrow + 0.7,
        "Inconsistent",
        fontsize=14,
        ha="right",
        va="top",
        color="black",
    )

    # # Draw a vertical arrow along the y-axis to indicate increasing inconsistency
    # y_arrow_vertical =0  # Position for the vertical arrow
    # ax.annotate("",
    #             xy=(xmin + 0.0, y_arrow_vertical-0.2), xycoords='data',
    #             xytext=(xmin + 0.0, y_arrow_vertical + 13), textcoords='data',
    #             arrowprops=dict(arrowstyle="<-", color="black", lw=2))
    # ax.text(xmin - 0.1, y_arrow_vertical - 0.9, "Increasing Inconsistency", fontsize=14, va="center", color="black")

    plt.tight_layout()
    fig.savefig(save_paths[0][:-4] + "_datatype_custom_violinplot" + save_paths[0][-4:])
    plt.close()


def draw_violin_plot_noneface(
    total_info_for_violinplot, vio_plot_column_name, eval_steps_names, save_paths
):

    # Create dataframe from provided data
    df = pd.DataFrame(data=total_info_for_violinplot, columns=vio_plot_column_name)

    # Specify the desired order for the set types
    desired_order = [
        "con",
        "con_con",
        "con_con_con",
        "con_con_con_con",  # S_{C}, S_{CC}, S_{CCC}, S_{CCCC}
        "con_con_con_incon",
        "con_con_incon",
        "con_incon",
        "incon",  # S_{CCCI}, S_{CCI}, S_{CI}, S_{I}
        "con_con_incon_incon",
        "con_incon_incon",
        "incon_incon",  # S_{CCII}, S_{CII}, S_{II}
        "con_incon_incon_incon",
        "incon_incon_incon",
        "incon_incon_incon_incon",  # S_{CIII}, S_{III}, S_{IIII}
    ]
    df["datatype"] = pd.Categorical(
        df["datatype"], categories=desired_order, ordered=True
    )
    df = df.sort_values("datatype")

    # Define background color mapping: if the key contains only "con", use one color; if it includes "incon", use another.
    bg_colors = {
        "con": "#80C5E7",
        "con_con": "#80C5E7",
        "con_con_con": "#80C5E7",
        "con_con_con_con": "#80C5E7",
        "con_con_con_incon": "#E8B97A",
        "con_con_incon": "#E8B97A",
        "con_incon": "#E8B97A",
        "incon": "#E8B97A",
        "con_con_incon_incon": "#E8B97A",
        "con_incon_incon": "#E8B97A",
        "incon_incon": "#E8B97A",
        "con_incon_incon_incon": "#E8B97A",
        "incon_incon_incon": "#E8B97A",
        "incon_incon_incon_incon": "#E8B97A",
    }

    # Define LaTeX labels mapping
    label_mapping = {
        "con": r"$S_{C}$",
        "con_con": r"$S_{CC}$",
        "con_con_con": r"$S_{CCC}$",
        "con_con_con_con": r"$S_{CCCC}$",
        "con_con_con_incon": r"$S_{CCCI}$",
        "con_con_incon": r"$S_{CCI}$",
        "con_incon": r"$S_{CI}$",
        "incon": r"$S_{I}$",
        "con_con_incon_incon": r"$S_{CCII}$",
        "con_incon_incon": r"$S_{CII}$",
        "incon_incon": r"$S_{II}$",
        "con_incon_incon_incon": r"$S_{CIII}$",
        "incon_incon_incon": r"$S_{III}$",
        "incon_incon_incon_incon": r"$S_{IIII}$",
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Draw horizontal separator lines for clarity between each set type
    for i in range(len(desired_order) - 1):
        plt.axhline(i + 0.5, color="gray", linestyle="-", linewidth=1)

    # Add background color spans for each set type row
    for i, key in enumerate(desired_order):
        plt.axhspan(i - 0.5, i + 0.5, facecolor=bg_colors[key], alpha=0.3)

    # Create the boxplot (without filling the boxes, i.e. transparent boxes with black edges)
    sns.boxplot(
        data=df,
        y="datatype",
        x="energy value",
        order=desired_order,
        patch_artist=True,
        boxprops={"facecolor": "none", "edgecolor": "black"},
        ax=ax,
    )

    # Set y-axis tick labels to the LaTeX labels
    plt.yticks(
        ticks=range(len(desired_order)),
        labels=[label_mapping[label] for label in desired_order],
        fontsize=12,
    )
    # Set axis labels with increased font size
    ax.set_xlabel("Energy Value", fontsize=18)
    ax.set_ylabel("Set Type", fontsize=18)

    # Draw a double-headed arrow along the x-axis below the boxplot to indicate the meaning of energy values
    xmin, xmax = ax.get_xlim()
    # Set a y-coordinate below the current y-axis (e.g., -1.5, adjust if needed)
    y_arrow = 13.5
    ax.annotate(
        "",
        xy=(xmin, y_arrow),
        xycoords="data",
        xytext=(xmax, y_arrow),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", color="black", lw=2),
    )
    # Add text labels at the left and right ends of the arrow
    ax.text(
        xmin,
        y_arrow + 0.7,
        "Consistent",
        fontsize=14,
        ha="left",
        va="top",
        color="black",
    )
    ax.text(
        xmax,
        y_arrow + 0.7,
        "Inconsistent",
        fontsize=14,
        ha="right",
        va="top",
        color="black",
    )

    # # Draw a vertical arrow along the y-axis to indicate increasing inconsistency
    # y_arrow_vertical =0  # Position for the vertical arrow
    # ax.annotate("",
    #             xy=(xmin + 0.0, y_arrow_vertical-0.2), xycoords='data',
    #             xytext=(xmin + 0.0, y_arrow_vertical + 13), textcoords='data',
    #             arrowprops=dict(arrowstyle="<-", color="black", lw=2))
    # ax.text(xmin - 0.1, y_arrow_vertical - 0.9, "Increasing Inconsistency", fontsize=14, va="center", color="black")

    plt.tight_layout()
    fig.savefig(save_paths[0][:-4] + "nonface_violinplot" + save_paths[0][-4:])
    plt.close()


def draw_violin_plot_whiteface_noarrow(
    total_info_for_violinplot, vio_plot_column_name, eval_steps_names, save_paths
):

    # Create dataframe from provided data
    df = pd.DataFrame(data=total_info_for_violinplot, columns=vio_plot_column_name)

    # Specify the desired order for the set types
    desired_order = [
        "con",
        "con_con",
        "con_con_con",
        "con_con_con_con",  # S_{C}, S_{CC}, S_{CCC}, S_{CCCC}
        "con_con_con_incon",
        "con_con_incon",
        "con_incon",
        "incon",  # S_{CCCI}, S_{CCI}, S_{CI}, S_{I}
        "con_con_incon_incon",
        "con_incon_incon",
        "incon_incon",  # S_{CCII}, S_{CII}, S_{II}
        "con_incon_incon_incon",
        "incon_incon_incon",
        "incon_incon_incon_incon",  # S_{CIII}, S_{III}, S_{IIII}
    ]
    df["datatype"] = pd.Categorical(
        df["datatype"], categories=desired_order, ordered=True
    )
    df = df.sort_values("datatype")

    # Define background color mapping: if the key contains only "con", use one color; if it includes "incon", use another.
    bg_colors = {
        "con": "#80C5E7",
        "con_con": "#80C5E7",
        "con_con_con": "#80C5E7",
        "con_con_con_con": "#80C5E7",
        "con_con_con_incon": "#E8B97A",
        "con_con_incon": "#E8B97A",
        "con_incon": "#E8B97A",
        "incon": "#E8B97A",
        "con_con_incon_incon": "#E8B97A",
        "con_incon_incon": "#E8B97A",
        "incon_incon": "#E8B97A",
        "con_incon_incon_incon": "#E8B97A",
        "incon_incon_incon": "#E8B97A",
        "incon_incon_incon_incon": "#E8B97A",
    }

    # Define LaTeX labels mapping
    label_mapping = {
        "con": r"$S_{C}$",
        "con_con": r"$S_{CC}$",
        "con_con_con": r"$S_{CCC}$",
        "con_con_con_con": r"$S_{CCCC}$",
        "con_con_con_incon": r"$S_{CCCI}$",
        "con_con_incon": r"$S_{CCI}$",
        "con_incon": r"$S_{CI}$",
        "incon": r"$S_{I}$",
        "con_con_incon_incon": r"$S_{CCII}$",
        "con_incon_incon": r"$S_{CII}$",
        "incon_incon": r"$S_{II}$",
        "con_incon_incon_incon": r"$S_{CIII}$",
        "incon_incon_incon": r"$S_{III}$",
        "incon_incon_incon_incon": r"$S_{IIII}$",
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Draw horizontal separator lines for clarity between each set type
    for i in range(len(desired_order) - 1):
        plt.axhline(i + 0.5, color="gray", linestyle="-", linewidth=1)

    # Add background color spans for each set type row
    for i, key in enumerate(desired_order):
        plt.axhspan(i - 0.5, i + 0.5, facecolor=bg_colors[key], alpha=0.3)

    # Create the boxplot (without filling the boxes, i.e. transparent boxes with black edges)
    sns.boxplot(
        data=df,
        y="datatype",
        x="energy value",
        order=desired_order,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        ax=ax,
    )

    # Set y-axis tick labels to the LaTeX labels
    plt.yticks(
        ticks=range(len(desired_order)),
        labels=[label_mapping[label] for label in desired_order],
        fontsize=12,
    )
    # Set axis labels with increased font size
    ax.set_xlabel("Energy Value", fontsize=18)
    ax.set_ylabel("Set Type", fontsize=18)

    # Draw a double-headed arrow along the x-axis below the boxplot to indicate the meaning of energy values
    # xmin, xmax = ax.get_xlim()
    # # Set a y-coordinate below the current y-axis (e.g., -1.5, adjust if needed)
    # y_arrow = 13.5
    # ax.annotate("",
    #             xy=(xmin, y_arrow), xycoords='data',
    #             xytext=(xmax, y_arrow), textcoords='data',
    #             arrowprops=dict(arrowstyle="<->", color="black", lw=2))
    # # Add text labels at the left and right ends of the arrow
    # ax.text(xmin, y_arrow + 0.7, "Consistent", fontsize=14, ha="left", va="top", color="black")
    # ax.text(xmax, y_arrow + 0.7, "Inconsistent", fontsize=14, ha="right", va="top", color="black")

    # # Draw a vertical arrow along the y-axis to indicate increasing inconsistency
    # y_arrow_vertical =0  # Position for the vertical arrow
    # ax.annotate("",
    #             xy=(xmin + 0.0, y_arrow_vertical-0.2), xycoords='data',
    #             xytext=(xmin + 0.0, y_arrow_vertical + 13), textcoords='data',
    #             arrowprops=dict(arrowstyle="<-", color="black", lw=2))
    # ax.text(xmin - 0.1, y_arrow_vertical - 0.9, "Increasing Inconsistency", fontsize=14, va="center", color="black")

    plt.tight_layout()
    fig.savefig(
        save_paths[0][:-4] + "whiteface_noarrow_violinplot" + save_paths[0][-4:]
    )
    plt.close()


def draw_violin_plot_noneface_noarrow(
    total_info_for_violinplot, vio_plot_column_name, eval_steps_names, save_paths
):

    # Create dataframe from provided data
    df = pd.DataFrame(data=total_info_for_violinplot, columns=vio_plot_column_name)

    # Specify the desired order for the set types
    desired_order = [
        "con",
        "con_con",
        "con_con_con",
        "con_con_con_con",  # S_{C}, S_{CC}, S_{CCC}, S_{CCCC}
        "con_con_con_incon",
        "con_con_incon",
        "con_incon",
        "incon",  # S_{CCCI}, S_{CCI}, S_{CI}, S_{I}
        "con_con_incon_incon",
        "con_incon_incon",
        "incon_incon",  # S_{CCII}, S_{CII}, S_{II}
        "con_incon_incon_incon",
        "incon_incon_incon",
        "incon_incon_incon_incon",  # S_{CIII}, S_{III}, S_{IIII}
    ]
    df["datatype"] = pd.Categorical(
        df["datatype"], categories=desired_order, ordered=True
    )
    df = df.sort_values("datatype")

    # Define background color mapping: if the key contains only "con", use one color; if it includes "incon", use another.
    bg_colors = {
        "con": "#80C5E7",
        "con_con": "#80C5E7",
        "con_con_con": "#80C5E7",
        "con_con_con_con": "#80C5E7",
        "con_con_con_incon": "#E8B97A",
        "con_con_incon": "#E8B97A",
        "con_incon": "#E8B97A",
        "incon": "#E8B97A",
        "con_con_incon_incon": "#E8B97A",
        "con_incon_incon": "#E8B97A",
        "incon_incon": "#E8B97A",
        "con_incon_incon_incon": "#E8B97A",
        "incon_incon_incon": "#E8B97A",
        "incon_incon_incon_incon": "#E8B97A",
    }

    # Define LaTeX labels mapping
    label_mapping = {
        "con": r"$S_{C}$",
        "con_con": r"$S_{CC}$",
        "con_con_con": r"$S_{CCC}$",
        "con_con_con_con": r"$S_{CCCC}$",
        "con_con_con_incon": r"$S_{CCCI}$",
        "con_con_incon": r"$S_{CCI}$",
        "con_incon": r"$S_{CI}$",
        "incon": r"$S_{I}$",
        "con_con_incon_incon": r"$S_{CCII}$",
        "con_incon_incon": r"$S_{CII}$",
        "incon_incon": r"$S_{II}$",
        "con_incon_incon_incon": r"$S_{CIII}$",
        "incon_incon_incon": r"$S_{III}$",
        "incon_incon_incon_incon": r"$S_{IIII}$",
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Draw horizontal separator lines for clarity between each set type
    for i in range(len(desired_order) - 1):
        plt.axhline(i + 0.5, color="gray", linestyle="-", linewidth=1)

    # Add background color spans for each set type row
    for i, key in enumerate(desired_order):
        plt.axhspan(i - 0.5, i + 0.5, facecolor=bg_colors[key], alpha=0.3)

    # Create the boxplot (without filling the boxes, i.e. transparent boxes with black edges)
    sns.boxplot(
        data=df,
        y="datatype",
        x="energy value",
        order=desired_order,
        patch_artist=True,
        boxprops={"facecolor": "none", "edgecolor": "black"},
        ax=ax,
    )

    # Set y-axis tick labels to the LaTeX labels
    plt.yticks(
        ticks=range(len(desired_order)),
        labels=[label_mapping[label] for label in desired_order],
        fontsize=12,
    )
    # Set axis labels with increased font size
    ax.set_xlabel("Energy Value", fontsize=18)
    ax.set_ylabel("Set Type", fontsize=18)

    # # Draw a double-headed arrow along the x-axis below the boxplot to indicate the meaning of energy values
    # xmin, xmax = ax.get_xlim()
    # # Set a y-coordinate below the current y-axis (e.g., -1.5, adjust if needed)
    # y_arrow = 13.5
    # ax.annotate("",
    #             xy=(xmin, y_arrow), xycoords='data',
    #             xytext=(xmax, y_arrow), textcoords='data',
    #             arrowprops=dict(arrowstyle="<->", color="black", lw=2))
    # # Add text labels at the left and right ends of the arrow
    # ax.text(xmin, y_arrow + 0.7, "Consistent", fontsize=14, ha="left", va="top", color="black")
    # ax.text(xmax, y_arrow + 0.7, "Inconsistent", fontsize=14, ha="right", va="top", color="black")

    # # Draw a vertical arrow along the y-axis to indicate increasing inconsistency
    # y_arrow_vertical =0  # Position for the vertical arrow
    # ax.annotate("",
    #             xy=(xmin + 0.0, y_arrow_vertical-0.2), xycoords='data',
    #             xytext=(xmin + 0.0, y_arrow_vertical + 13), textcoords='data',
    #             arrowprops=dict(arrowstyle="<-", color="black", lw=2))
    # ax.text(xmin - 0.1, y_arrow_vertical - 0.9, "Increasing Inconsistency", fontsize=14, va="center", color="black")

    plt.tight_layout()
    fig.savefig(save_paths[0][:-4] + "nonface_noarrow_violinplot" + save_paths[0][-4:])
    plt.close()


def draw_hist(total_info_for_hist, hist_column_name, save_paths):
    """
    hist_column_name must include "datatype", "threshold", and "side"
    """
    df = pd.DataFrame(data=total_info_for_hist, columns=hist_column_name)

    ax = sns.histplot(
        data=df, y="datatype", x="energy value", binwidth=0.01, hue="side"
    )
    unique_steps = df["datatype"].unique()
    for i in range(len(unique_steps) - 1):
        plt.axhline(
            i + 0.5, color="gray", linestyle="-", linewidth=1
        )  # Adjust linewidth and color as needed

    # draw threshold
    sns.pointplot(
        x="threshold",
        y="datatype",
        data=df,
        linestyles="--",
        color="black",
        scale=0.2,
        errwidth=0,
        capsize=0,
    )

    plt.tight_layout()
    fig = ax.get_figure()
    for p in save_paths:
        fig.savefig(p[:-4] + "_datatype_" + p[-4:])
    plt.close()


def draw_violin_plot_supervised(
    total_info_for_violinplot, vio_plot_column_name, eval_steps_names, save_paths
):
    df = pd.DataFrame(data=total_info_for_violinplot, columns=vio_plot_column_name)
    group_colors = {
        "con": "#fa1407",  # red
        "con_and_incon": "#fffb08",  # yellow
        "incon": "#05fc68",  # green
        "incon_and_incon": "#0471de",  # blue
    }
    # grouped_palette = {group_colors[group]:group for group in df['datatype'].unique()}
    # df['color'] = df['datatype'].apply(lambda x: grouped_palette[x])

    # # palette = get_hex("Acadia", keep_first_n=len(eval_steps_names))
    # ax = sns.boxplot(data = df, y = 'stepname', x = 'e_val', hue = 'side')
    # unique_steps = df['stepname'].unique()
    # for i in range(len(unique_steps) - 1):
    #     plt.axhline(i + 0.5, color='gray', linestyle='-', linewidth=1)  # Adjust linewidth and color as needed

    # # draw threshold
    # # sns.pointplot(x = 'threshold', y = 'stepname', data = df, linestyles='--', color = 'black', scale=0.2, errwidth=0, capsize = 0)

    # plt.tight_layout()
    # fig = ax.get_figure()
    # for p in save_paths:
    #     fig.savefig(p[:-4] + "_stepname_" + p[-4:])
    # plt.close()

    ax = sns.boxplot(data=df, y="datatype", x="neg_probs")
    unique_steps = df["datatype"].unique()
    for i in range(len(unique_steps) - 1):
        plt.axhline(
            i + 0.5, color="gray", linestyle="-", linewidth=1
        )  # Adjust linewidth and color as needed

    # draw threshold
    # sns.pointplot(x = 'threshold', y = 'datatype', data = df, linestyles='--', color = 'black', scale=0.2, errwidth=0, capsize = 0)

    plt.tight_layout()
    fig = ax.get_figure()
    for p in save_paths:
        fig.savefig(p[:-4] + "_datatype_" + p[-4:])
    plt.close()
