import pandas as pd
import os
import argparse
import sys
import glob


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


def combine_results(name_folder):
    work_folder = os.getcwd()
    extension = "csv"
    path = os.path.join(os.getcwd(), name_folder)
    check_folders(path)
    os.chdir(path)
    files = glob.glob('*.{}'.format(extension))
    files = [x[:-4].split("_") for x in files]
    files = [x for x in files if x[2] != "test"]
    files.sort(key=lambda x: x[3])
    files_names = []
    for f in files:
        files_names.append(os.path.join(
            os.getcwd(), f"{f[0]}_{f[1]}_{f[2]}_{f[3]}.csv"))
    df_from_each_file = (pd.read_csv(f) for f in files_names)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df = concatenated_df.drop(columns=["Unnamed: 0"])
    jaros = []
    success = []
    name_experiment = []
    for i in range(0, len(concatenated_df)):
        f = open(os.path.join(
            os.getcwd(), f"{files[i][0]}_{files[i][1]}_{files[i][2]}_{files[i][3]}.txt"), 'r')
        file_contents = f.read()
        print(
            f"\n\nThis is experiment {files[i][1]} with jaro {str(int(files[i][2])/10)}")
        print(file_contents)
        f.close()
        price_range_found = int(get_value(
            "At which input index was the first pricerange found. The index start at 1. If not found type in 0", "Type an int in:", int))
        if price_range_found != 0:
            concatenated_df.iloc[i, 4] = price_range_found
        did_succeed = bool(get_value(
            "Did the participant succeed in the task. Type in 1 for yes and 0 for no", "Type an int in:", int))
        success.append(did_succeed)
        jaros.append(int(files[i][2]) / 10)
        name_experiment.append(name_folder)

    concatenated_df["num experiment"] = range(0, len(concatenated_df))
    concatenated_df["participant"] = name_experiment
    concatenated_df["jaro-winkler"] = jaros
    concatenated_df["completed"] = success

    first_three = concatenated_df[concatenated_df["jaro-winkler"] == 0.9]
    last_three = concatenated_df[concatenated_df["jaro-winkler"] != 0.9]

    names_exp = [name_folder] * 3
    names_avg = ["average total", "average 0.9",
                 f"average {str(int(files[4][2])/10)}"]
    jaro_comb = [None, 0.9, int(files[4][2]) / 10]

    num_input = [concatenated_df["number of inputs"].mean(
    ), first_three["number of inputs"].mean(), last_three["number of inputs"].mean()]
    num_input_sd = [concatenated_df["number of inputs"].std(
    ), first_three["number of inputs"].std(), last_three["number of inputs"].std()]

    food_inp = [concatenated_df["food found"].mean(
    ), first_three["food found"].mean(), last_three["food found"].mean()]
    food_inp_sd = [concatenated_df["food found"].std(
    ), first_three["food found"].std(), last_three["food found"].std()]

    area_inp = [concatenated_df["area found"].mean(
    ), first_three["area found"].mean(), last_three["area found"].mean()]
    area_inp_sd = [concatenated_df["area found"].std(
    ), first_three["area found"].std(), last_three["area found"].std()]

    pricerange_inp = [concatenated_df["pricerange found"].mean(
    ), first_three["pricerange found"].mean(), last_three["pricerange found"].mean()]
    pricerange_inp_sd = [concatenated_df["pricerange found"].std(
    ), first_three["pricerange found"].std(), last_three["pricerange found"].std()]

    total_words = [concatenated_df["total words"].mean(
    ), first_three["total words"].mean(), last_three["total words"].mean()]
    total_words_sd = [concatenated_df["total words"].std(
    ), first_three["total words"].std(), last_three["total words"].std()]

    average_sen = [concatenated_df["average of sentences"].mean(
    ), first_three["average of sentences"].mean(), last_three["average of sentences"].mean()]

    average_median = [concatenated_df["median of sentences"].mean(
    ), first_three["median of sentences"].mean(), last_three["median of sentences"].mean()]

    average_sd = [concatenated_df["sd of sentences"].mean(
    ), first_three["sd of sentences"].mean(), last_three["sd of sentences"].mean()]

    did_not_understand = [concatenated_df["did_not_understand"].mean(
    ), first_three["did_not_understand"].mean(), last_three["did_not_understand"].mean()]
    did_not_understand_sd = [concatenated_df["did_not_understand"].std(
    ), first_three["did_not_understand"].std(), last_three["did_not_understand"].std()]

    success_rate = [len(concatenated_df[concatenated_df["completed"] == True]) / len(concatenated_df),
                    len(first_three[first_three["completed"]
                                    == True]) / len(first_three),
                    len(last_three[last_three["completed"] == True]) / len(last_three)]
    d = {
        "name": names_avg,
        "particapant": names_exp,
        "jaro-winkler": jaro_comb,
        "average number inputs": num_input,
        "sd number inputs": num_input_sd,
        "average food found": food_inp,
        "sd food input": food_inp_sd,
        "average area found": area_inp,
        "sd area found": area_inp_sd,
        "average pricerange found": pricerange_inp,
        "sd pricerange found": pricerange_inp_sd,
        "average total words": total_words,
        "sd total words": total_words_sd,
        "average of sentences": average_sen,
        "average median of sentences": average_median,
        "average sd of sentences": average_sd,
        "average did not understand": did_not_understand,
        "sd did not understand": did_not_understand_sd,
        "success rate": success_rate}
    df_average = pd.DataFrame(data=d)
    results_folder = os.path.join(work_folder, "results")
    df_average.to_csv(os.path.join(
        results_folder, f"{name_folder}_average.csv"))
    concatenated_df.to_csv(os.path.join(
        results_folder, f"{name_folder}_total.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='The argument parser for the combining the results')
    parser.add_argument("--name", required=True, type=str,
                        help="Name of the folder for which you want to combine the results")

    args = parser.parse_args()

    combine_results(args.name)
