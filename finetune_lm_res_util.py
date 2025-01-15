import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table

if __name__ == "__main__":
    checkpoint_dir = "checkpoints"
    output_dir = "finetune_results"
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for dataset_shot_folder in os.listdir(checkpoint_dir):
        dataset_shot_path = os.path.join(checkpoint_dir, dataset_shot_folder)

        if os.path.isdir(dataset_shot_path):
            dataset, shot = dataset_shot_folder.split("_")

            if shot == "0":
                shot = "full"

            for model_folder in os.listdir(dataset_shot_path):
                model_path = os.path.join(dataset_shot_path, model_folder)

                if os.path.isdir(model_path):
                    csv_file = os.path.join(model_path, "test_result.csv")

                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)

                        # 確保 eval_accuracy 欄位存在
                        if "eval_accuracy" in df.columns:
                            eval_accuracy = df["eval_accuracy"].iloc[0]

                            results.append(
                                {
                                    "Dataset": dataset,
                                    "Shot": str(shot),
                                    "Model": model_folder,
                                    "Eval_Accuracy": eval_accuracy,
                                }
                            )

    results_df = pd.DataFrame(results)

    results_df["Shot_sort"] = results_df["Shot"].replace("full", 101).astype(int)
    results_df = results_df.sort_values(by=["Dataset", "Shot_sort", "Model"]).drop(
        columns=["Shot_sort"]
    )

    results_csv_path = os.path.join(output_dir, "aggregated_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    for dataset in results_df["Dataset"].unique():
        plt.figure(figsize=(10, 6))

        for model in results_df["Model"].unique():
            subset = results_df[
                (results_df["Dataset"] == dataset) & (results_df["Model"] == model)
            ]
            plt.plot(subset["Shot"], subset["Eval_Accuracy"], label=model, marker="o")

        plt.title(f"Eval Accuracy for Dataset: {dataset}")
        plt.xlabel("Shot")
        plt.ylabel("Eval Accuracy")
        plt.legend()
        plt.grid()
        plot_path = os.path.join(output_dir, f"{dataset}_eval_accuracy_plot.png")
        plt.savefig(plot_path)
        plt.show()

        # 數據表格
        subset = results_df[results_df["Dataset"] == dataset]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_axis_off()

        table = Table(ax, bbox=[0, 0, 1, 1])
        col_labels = ["Dataset", "K-Shot", "Model", "Eval_Accuracy"]
        col_widths = [0.25, 0.25, 0.25, 0.25]

        for i, label in enumerate(col_labels):
            table.add_cell(
                -1,
                i,
                0.25,
                0.05,
                text=label,
                loc="center",
                facecolor="lightgrey",
                edgecolor="black",
            )

        # 數據列
        last_model = last_dataset = ""
        for row_idx, row in enumerate(subset.itertuples(index=False)):
            for col_idx, value in enumerate(row):
                if col_idx == 0:  # Model
                    text = "" if value == last_model else value
                    edgecolor = "black" if value != last_model else "white"
                    last_model = value
                elif col_idx == 1:  # Dataset
                    text = "" if value == last_dataset else value
                    edgecolor = "black" if value != last_dataset else "white"
                    last_dataset = value
                else:
                    text = str(value)
                    edgecolor = "black"

                table.add_cell(
                    row_idx,
                    col_idx,
                    0.25,
                    0.05,
                    text=text,
                    loc="center",
                    edgecolor=edgecolor if text else "black",
                )

        ax.add_table(table)
        table_path = os.path.join(output_dir, f"{dataset}_data_table.png")
        plt.savefig(table_path)
        plt.show()
