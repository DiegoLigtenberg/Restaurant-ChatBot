import pandas as pd
import Levenshtein
import glob
import os
import classifiers
from collections import Counter
from collections import defaultdict
from adjustText import adjust_text
from matplotlib import patches
from matplotlib import cm
from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt


def import_terms(file_csv):
    df = pd.read_csv(file_csv)
    pricerange = df.pricerange.unique()
    food = df.food.unique()
    area = df.area.unique()
    inform_terms = list(pricerange) + list(food) + list(area)
    request_terms = ["food", "area", "town", "pricerange",
                     "phone", "number", "address", "postcode"]
    return [x for x in inform_terms if str(x) != 'nan'], request_terms


def check_txt_file(file_txt, inform_terms, request_terms, jaro_07, jaro_08, jaro_09, classifier):
    file_o = open(file_txt, 'r')
    for line in file_o:
        line = line[:-1]
        if line.split(" ")[0] == "User:":
            line = line[6:].lower()
            pre = classifier.predict([line])[0]
            if pre == "inform":
                for w in line.split():
                    for i in inform_terms:
                        dis = Levenshtein.jaro_winkler(w, i)
                        if dis >= 0.9:
                            jaro_09[i][w] += 1
                        if dis >= 0.8:
                            jaro_08[i][w] += 1
                        if dis >= 0.7:
                            jaro_07[i][w] += 1
            elif pre == "request":
                for w in line.split():
                    for i in request_terms:
                        dis = Levenshtein.jaro_winkler(w, i)
                        if dis >= 0.9:
                            jaro_09[i][w] += 1
                        if dis >= 0.8:
                            jaro_08[i][w] += 1
                        if dis >= 0.7:
                            jaro_07[i][w] += 1
    file_o.close()
    return jaro_07, jaro_08, jaro_09


def read_list(keys, main_list, main_list_name, first_list, first_list_name, second_list, second_list_name):
    print("\n\n\n")
    for k in keys:
        print("\n")
        print(f"For the term {k}, the following values are found")
        print(f'With the jaro {main_list_name} used in the task')
        print(main_list[k])
        print(f"With jaro {first_list_name}")
        print(first_list[k])
        print(f"With jaro {second_list_name}")
        print(second_list[k])
        print("\n")
    print("\n\n\n")


def sort_jaro(current_list, word, search_list):
    if len(search_list) == 0:
        return current_list
    else:
        dis = -1
        best_word = None
        for s in search_list:
            a = Levenshtein.jaro_winkler(word, s)
            if a > dis:
                best_word = s
                dis = a
        search_list.remove(best_word)
        return sort_jaro(current_list + [best_word], best_word, search_list[:])


def sort_jaro_worst(word, search_list):
    if len(search_list) == 1 or len(search_list) == 2:
        return [word] + search_list
    else:
        best_score = -1
        best_word = None
        worst_score = 999
        worst_word = None
        for i in search_list:
            a = Levenshtein.jaro_winkler(word, i)
            if a > best_score:
                best_word = i
                best_score = a
            if a < worst_score:
                worst_word = i
                worst_score = a
        new_list = search_list[:]
        new_list.remove(best_word)
        new_list.remove(worst_word)
        return [word] + sort_jaro_worst(worst_word, new_list) + [best_word]


def find_first_word(word_list):
    best_score = -1
    best_word = None
    for a in word_list:
        for b in word_list:
            dis = Levenshtein.jaro_winkler(a, b)
            if dis > best_score:
                best_word = a
                best_score = dis
    return best_word


def create_plots(dict_counter, name, jaro, text_on=False, which_text=None):
    distances = {}
    group = {}

    labels = []
    for k1 in dict_counter.keys():
        for k2 in dict_counter[k1].keys():
            name1 = f"{k2}_{k1}"
            distances[name1] = Levenshtein.jaro_winkler(
                k1, k2) + 0.0000001 * np.random.random()
            group[name1] = k1
            labels.append(k1)
    testa = list(set(group.values()))
    best_word = find_first_word(testa)
    testa.remove(best_word)
    testb = sort_jaro_worst(best_word, testa[:])
    colors = []
    if len(testb) <= 20:
        cm_subsection1 = linspace(0.0, 1.0, len(testb))
    elif len(testb) > 20 and len(testb) <= 40:
        cm_subsection1 = linspace(0.0, 1.0, 20)
        cm_subsection2 = linspace(0.0, 1.0, len(testb) - 20)
    else:
        cm_subsection1 = linspace(0.0, 1.0, 20)
        cm_subsection2 = linspace(0.0, 1.0, 20)
        cm_subsection3 = linspace(0.0, 1.0, len(testb) - 40)
    for i in range(len(testb)):
        if i < 20:
            colors.append(cm.tab20(cm_subsection1[i]))
        elif i >= 20 and i < 39:
            colors.append(cm.tab20b(cm_subsection2[i - 20]))
        else:
            colors.append(cm.tab20c(cm_subsection3[i - 40]))
    colorsdict = dict(zip(testb, colors))
    df = pd.Series(distances)

    c = [colorsdict.get(group[label], 'k') for label in df.index]
    fig, aa = plt.subplots(figsize=(11, 9))
    aa.axes.get_xaxis().set_visible(False)
    aa.set_xlim(-11, 0.1)
    aa.set_ylim(jaro-.005, 1.005)
    scatter_items = []
    legend_items = []
    for t, d, ca, l in zip([0 for _ in df], df, c, df.index):
        scatter = aa.scatter(t, d, c=ca, alpha=0.5,
                             edgecolors='none')

    for l, colo in zip(colorsdict.keys(), colorsdict.values()):

        scatter_items += aa.plot((-100,), (-100,),
                                 ls='none', marker='.', c=colo, label=l)
    aa.legend(handles=scatter_items, title="Terms", fontsize=7,
              loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=10)
    aa.spines['left'].set_visible(False)
    aa.spines['top'].set_visible(False)
    aa.spines['bottom'].set_visible(False)
    aa.yaxis.set_label_position('right')
    aa.yaxis.set_ticks_position('right')
    plt.tight_layout()

    # We add a rectangle to make sure the labels don't move to the right
    patch = patches.Rectangle((-0.1, 0), 0.2, 100, fill=False, alpha=0)
    aa.add_patch(patch)
    texts = []
    np.random.seed(0)
    for label, y in zip(df.index, df):
        texts += [aa.text(-.1+np.random.random()/1000, y, label.split("_")[0],
                          color=colorsdict.get(group[label], 'k'), fontsize=9)]

    adjust_text(texts, [0 for _ in df], df.values,  ha='right', va='center', add_objects=[patch],
                expand_text=(1.1, 1.25),
                force_text=(0.75, 0), force_objects=(1, 0),
                autoalign=False, only_move={'points': 'x', 'text': 'x', 'objects': 'x'})
    plt.savefig(f'{name}_scatter.png')
    print(len(list(set(group.values()))))
    print(len(testb))
    print(testb)


def main_function():
    classifier = classifiers.nn()
    jaro_09_09 = defaultdict(Counter)
    jaro_09_08 = defaultdict(Counter)
    jaro_09_07 = defaultdict(Counter)

    jaro_08_09 = defaultdict(Counter)
    jaro_08_08 = defaultdict(Counter)
    jaro_08_07 = defaultdict(Counter)

    jaro_07_09 = defaultdict(Counter)
    jaro_07_08 = defaultdict(Counter)
    jaro_07_07 = defaultdict(Counter)
    main_folder = os.getcwd()
    inform_terms, request_terms = import_terms(
        os.path.join(main_folder, "restaurant_info.csv"))
    extension = "txt"
    os.chdir("/home/jesse/MAIR/Part1/Text_classification/txt_files")
    files = glob.glob('*.{}'.format(extension))
    files_split = [x.split("_") for x in files]
    files_tuple = []
    for fs, f in zip(files_split, files):
        if fs[2] != "test":
            files_tuple.append((f, float(fs[2]) / 10))
    for txt_file, jaro in files_tuple:
        if jaro == 0.9:
            jaro_09_07, jaro_09_08, jaro_09_09 = check_txt_file(os.path.join(os.getcwd(
            ), txt_file), inform_terms, request_terms, jaro_09_07, jaro_09_08, jaro_09_09, classifier)
        if jaro == 0.8:
            jaro_08_07, jaro_08_08, jaro_08_09 = check_txt_file(os.path.join(os.getcwd(
            ), txt_file), inform_terms, request_terms, jaro_08_07, jaro_08_08, jaro_08_09, classifier)
        if jaro == 0.7:
            jaro_07_07, jaro_07_08, jaro_07_09 = check_txt_file(os.path.join(os.getcwd(
            ), txt_file), inform_terms, request_terms, jaro_07_07, jaro_07_08, jaro_07_09, classifier)

    jaro_09_keys = set(list(jaro_09_09.keys()) +
                       list(jaro_09_08.keys()) + list(jaro_09_07.keys()))
    jaro_08_keys = set(list(jaro_08_09.keys()) +
                       list(jaro_08_08.keys()) + list(jaro_08_07.keys()))
    jaro_07_keys = set(list(jaro_07_09.keys()) +
                       list(jaro_07_08.keys()) + list(jaro_07_07.keys()))

    # Jaro 0.9
    read_list(jaro_09_keys, jaro_09_09, "0.9",
              jaro_09_08, "0.8", jaro_09_07, "0.7")
    # Jaro 0.8
    read_list(jaro_08_keys, jaro_08_08, "0.8",
              jaro_08_09, "0.9", jaro_08_07, "0.7")
    # Jaro 0.7
    read_list(jaro_07_keys, jaro_07_07, "0.7",
              jaro_07_08, "0.8", jaro_07_09, "0.9")
    create_plots(jaro_09_09, "Jaro_09", 0.9)
    create_plots(jaro_08_08, "Jaro_08", 0.8)
    create_plots(jaro_07_07, "Jaro_07", 0.7)


if __name__ == "__main__":
    main_function()
