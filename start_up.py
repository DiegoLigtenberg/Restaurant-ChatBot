import os
import argparse
import sys
import signal

signal.signal(signal.SIGINT, signal.SIG_IGN)
if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)
elif sys.platform.startswith("win32") or sys.platform.startswith("cygwin"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def start_experiment(number_task_txt, person, experiment_txt, experiment_name, jaro):
    while True:
        print(f"\n\n\n\n{number_task_txt}")
        print(experiment_txt)
        userInput = input("Are you ready to start, type yes or no: ")
        if userInput[0] == 'y':
            command = f"python chatbot_experiment.py --jaro {jaro} --name {person} --experiment {experiment_name}"
            print("\n\n\n\n\n")
            os.system(f"{command}")
            print("\n\n\n\n\n")
            print("Did the program succeed?")
            userInput = input("type yes or no: ")

            if userInput[0] == 'y':
                break

            elif userInput == "exit":
                sys.exit()
            else:
                print("The program will restart")
                continue
        elif userInput[0] == 'n':
            continue
        elif userInput == "exit":
            sys.exit()
        else:
            print("You need to type in either yes or no")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='The argument parser for the chatbot')
    parser.add_argument("--jaro_experiment", required=True, type=float, choices=[0.7, 0.8],
                        help="Sets which experiment you want to do 0.7 or 0.8. The first 2 will be test with 0.9, then 3 with 0.9 and afterwards 3 with the setting")
    parser.add_argument("--name", required=True, type=str,
                        help="Indentifier for the person for who does the experiment")

    args = parser.parse_args()

    jaro_txt = str(args.jaro_experiment)[-1]
    jaro = args.jaro_experiment
    while True:
        print("\n\nThank you for helping us with this experiment. In this experiment, you will use a restaurant recommender chatbot " +
              "to find 8 restaurants and ask the chatbot some information about these restaurants. Before chatbot starts up, we " +
              "will tell you what kind of restaurant you need to search and what kind of information you need from the restaurant. " +
              "For example, a task could be: \"Find an American restaurant an ask for their phone number, then say goodbye\"." +
              "\n\nThe first two tasks are for you to get more familiar with the system and you can ask the person, who conducts the " +
              "experiment for help, after the first two you can not ask for help. We will gather some information in the background; " +
              "this information contains everything you sent to the chatbot and what it sent back to you.")
        userInput = input("\nAre you okay with this? type yes or no: ")
        if userInput[0] == 'y':
            break
        elif userInput[0] == 'n':
            continue
        elif userInput == "exit":
            sys.exit()
        else:
            print("You need to type in either yes or no")

    test_task1 = "Ask for Italian food in the west and ask their phone number and say goodbye."
    test_task1_ab = "exp_italian_test_1"
    start_experiment("The first example task:", args.name,
                     test_task1, test_task1_ab, 0.9)

    test_task2 = "Ask for an expensive French restaurant in the centre, ask for their phone number and address and say goodbye."
    test_task2_ab = "exp_french_test_2"
    start_experiment("The second example task:", args.name,
                     test_task2, test_task2_ab, 0.9)

    print("\n\nFrom now the real experiment will start and you can not ask for help!!\n")

    task1 = "Ask for a restaurant that serves food from Portugal in the south and ask for their phone number and say goodbye."
    task1_ab = "exp_portugal_9_1"
    start_experiment("The first task:", args.name,
                     task1, task1_ab, 0.9)

    task2 = "Ask for food from Tuscany and say goodbye."
    task2_ab = "exp_tuscan_9_2"
    start_experiment("The second task:", args.name,
                     task2, task2_ab, 0.9)

    task3 = "Find an expensive British restaurant in the centrum and ask for their phone number and their address and say goodbye."
    task3_ab = "exp_british_9_3"
    start_experiment("The third task:", args.name,
                     task3, task3_ab, 0.9)

    task4 = "Ask for a restaurant that serves food from Turkey and is in the centre, when found ask for the address and say goodbye"
    task4_ab = f"exp_turkey_{jaro_txt}_4"
    start_experiment("The fourth task:", args.name,
                     task4, task4_ab, jaro)

    task5 = "Ask for food from Jamaica and say goodbye."
    task5_ab = f"exp_jamaica_{jaro_txt}_5"
    start_experiment("The fifth task:", args.name,
                     task5, task5_ab, jaro)

    task6 = "Find an Indian restaurant in the west and ask for their phone number and their address and say goodbye."
    task6_ab = f"exp_indian_{jaro_txt}_6"
    start_experiment("The sixth task:", args.name,
                     task6, task6_ab, jaro)

    print("\n\nYou are finished with the experiment, thanks for participating. For more info about the experiment, you can contact J. van Remmerden at j.vanremmerden@students.uu.nl.")
