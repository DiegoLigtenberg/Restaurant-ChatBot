import pandas as pd
import os
import argparse
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np


def check_folders(path):

    if not os.path.exists(path):
        print(f"The experiment {name_folder} does not exist")
        return
    results_path = os.path.join(os.getcwd(), "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)


def get_value(question, input_sentence, type_input):
    while True:
        print(f"\n\n{question}")
        t = input(f"\n{input_sentence} ")
        return type_input(t)


def combine_results():
    work_folder = os.getcwd()
    extension = "csv"
    path = os.path.join(os.getcwd(), "results")
    check_folders(path)
    os.chdir(path)
    files = glob.glob('*.{}'.format(extension))
    files = [x[:-4].split("_") for x in files]
    files_average = [x for x in files if x[1] == "average"]
    files_total = [x for x in files if x[1] == "total"]
    files_names_average = []
    files_names_total = []
    for f in files_average:
        files_names_average.append(os.path.join(
            os.getcwd(), f"{f[0]}_{f[1]}.csv"))
    for f in files_total:
        files_names_total.append(os.path.join(
            os.getcwd(), f"{f[0]}_{f[1]}.csv"))

    li_total = []
    li_avg = []

    for filename_total, filename_average in zip(files_names_total, files_names_average):
        df_total = pd.read_csv(filename_total)
        for index, _ in df_total.iterrows():
            df_total.loc[index, "Unnamed: 0"] += 1
        li_total.append(df_total)
        df_avg = pd.read_csv(filename_average)
        li_avg.append(df_avg)

    frame_total = pd.concat(li_total, axis=0, ignore_index=True)
    frame_avg = pd.concat(li_avg, axis=0, ignore_index=True)
    frame_total = frame_total.drop(columns=["num experiment"])
    return frame_total.rename(columns={"Unnamed: 0": "experiment_num"}), frame_avg.drop(columns=["Unnamed: 0"])


def get_result_task(tot_res):
    task1 = tot_res["experiment_num"] == 0
    task2 = tot_res["experiment_num"] == 1
    task3 = tot_res["experiment_num"] == 2
    task4 = tot_res["experiment_num"] == 3
    task5 = tot_res["experiment_num"] == 4
    task6 = tot_res["experiment_num"] == 5

    jaro8 = tot_res["jaro-winkler"] == 0.8
    jaro7 = tot_res["jaro-winkler"] == 0.7

    task1_df = tot_res[task1]
    task2_df = tot_res[task2]
    task3_df = tot_res[task3]

    task4_8_df = tot_res[task4 & jaro8]
    task5_8_df = tot_res[task5 & jaro8]
    task6_8_df = tot_res[task6 & jaro8]

    task4_7_df = tot_res[task4 & jaro7]
    task5_7_df = tot_res[task5 & jaro7]
    task6_7_df = tot_res[task6 & jaro7]

    print(task4_7_df)
    print(task5_7_df)

    task4_7_complete_ratio = len(
        task4_7_df[task4_7_df["completed"] == True]) / len(task4_7_df) * 100
    task4_8_complete_ratio = len(
        task4_8_df[task4_8_df["completed"] == True]) / len(task4_8_df) * 100
    print(f"Task 4 with jaro 0.7 complete ratio {task4_7_complete_ratio}%")
    print(f"Task 4 with jaro 0.8 complete ratio {task4_8_complete_ratio}%")

    task5_7_complete_ratio = len(
        task5_7_df[task5_7_df["completed"] == True]) / len(task5_7_df) * 100
    task5_8_complete_ratio = len(
        task5_8_df[task5_8_df["completed"] == True]) / len(task5_8_df) * 100
    print(f"Task 5 with jaro 0.7 complete ratio {task5_7_complete_ratio}%")
    print(f"Task 5 with jaro 0.8 complete ratio {task5_8_complete_ratio}%")

    task6_7_complete_ratio = len(
        task6_7_df[task6_7_df["completed"] == True]) / len(task6_7_df) * 100
    task6_8_complete_ratio = len(
        task6_8_df[task6_8_df["completed"] == True]) / len(task6_8_df) * 100
    print(f"Task 6 with jaro 0.7 complete ratio {task6_7_complete_ratio}%")
    print(f"Task 6 with jaro 0.8 complete ratio {task6_8_complete_ratio}%")


def set_dict_value(d, key, val):
    if key in d:
        d[key].append(val)
    else:
        d[key] = [val]
    return d


def create_bar(mean_values, std_values, name, jaro):
    x_values = np.arange(2)
    jaro_values = ["0.9", str(jaro)]

    fig, ax = plt.subplots()
    ax.bar(x_values, mean_values, yerr=std_values, align='center',
           alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Succes rate in percentile')
    ax.set_xticks(x_values)
    ax.set_xticklabels(jaro_values)
    ax.set_title(f'Group that had {str(jaro)} in the last three tasks.')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    # plt.show()


def create_bar_plot(avg_results):
    jaro8 = avg_results["jaro-winkler setting"] == 0.8
    jaro7 = avg_results["jaro-winkler setting"] == 0.7
    first_half_success_8 = avg_results[jaro8]["success rate first half"]
    first_half_success_7 = avg_results[jaro7]["success rate first half"]

    second_half_success_8 = avg_results[jaro8]["success rate second half"]
    second_half_success_7 = avg_results[jaro7]["success rate second half"]

    fh_7_mean = first_half_success_7.mean()
    fh_8_mean = first_half_success_8.mean()
    fh_7_std = first_half_success_7.std()
    fh_8_std = first_half_success_8.std()

    sh_7_mean = second_half_success_7.mean()
    sh_8_mean = second_half_success_8.mean()
    sh_7_std = second_half_success_7.std()
    sh_8_std = second_half_success_8.std()

    create_bar([fh_7_mean, sh_7_mean], [
               fh_7_std, sh_7_std], "Jaro_07_success", 0.7)
    create_bar([fh_8_mean, sh_8_mean], [
               fh_8_std, sh_8_std], "Jaro_08_success", 0.8)


def change_avg_file(avg, folder):
    avg.to_csv(os.path.join(folder, f"old_results_average.csv"),
               float_format='%.3f')
    d = {}
    for _, row in avg.iterrows():
        for i in avg:
            if i != "name" and i != "particapant" and i != "jaro-winkler":
                d = set_dict_value(d, i + set_avg, row[i])
            elif i == "name":
                set_avg = row[i].split(" ")[1]
                if set_avg == "0.9":
                    set_avg = " first half"
                elif set_avg == "total":
                    set_avg = " total"
                else:
                    set_avg = " second half"
            elif i == "particapant":
                if set_avg == " total":
                    d = set_dict_value(d, "particapant", row[i])
            elif i == "jaro-winkler":
                if row[i] == 0.8 or row[i] == 0.7:
                    d = set_dict_value(d, "jaro-winkler setting", row[i])
    df_new = pd.DataFrame(data=d)
    df_new.to_csv(os.path.join(folder, f"results_average.csv"),
                  float_format='%.3f')
    return df_new


def change_total_file(total, folder):
    total.to_csv(os.path.join(folder, f"old_results_total.csv"))
    participants = total.participant.unique()
    experiments = total.experiment_num.unique()
    d = {}
    for p in participants:
        d = set_dict_value(d, "participant", p)
        for e in experiments:
            find_name = total.participant == p
            find_exp = total.experiment_num == e
            exp_row = total[find_name & find_exp]
            ex = exp_row["experiment"].iloc[0].split("_")
            name_exp = f" {ex[1]}"
            for col in exp_row.columns:
                if col != "experiment_num" and col != "participant" and col != "experiment" and col != "jaro-winkler":
                    d = set_dict_value(d, col + name_exp, exp_row[col].iloc[0])
                elif col == "jaro-winkler" and e == 4:
                    d = set_dict_value(
                        d, "jaro-winkler setting", exp_row[col].iloc[0])

    df_new = pd.DataFrame(data=d)
    df_new.to_csv(os.path.join(folder, f"results_total.csv"),
                  float_format='%.3f')

    return df_new


if __name__ == "__main__":

    folder = os.getcwd()
    total_results, avg_results = combine_results()
    total_results = change_total_file(total_results, folder)
    avg_results = change_avg_file(
        avg_results, folder)
    create_bar_plot(avg_results)
