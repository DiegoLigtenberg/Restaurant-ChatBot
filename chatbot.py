import pandas as pd
import classifiers
import Levenshtein
import pyttsx3
import time
import Settings
import sys
import argparse

# edited instead of split
from nltk import word_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

Settings.Configurations.set_Configurations(
    levenshtein=0.9,  # float      loose 0<-->1 strict
    outputAllCaps=False,
    delayAnswers=1,  # float      delay in seconds
    restartDialog=False,
    textToSpeech=False,
    printPrediction=False,
)


"""
The TextConfigurations function replaces the print function.
"""


def TextConfiguration(textinput):

    if Settings.Configurations.delayAnswers > 0:
        time.sleep(Settings.Configurations.delayAnswers)

    if Settings.Configurations.textToSpeech is False:
        if Settings.Configurations.outputAllCaps is True:
            print("System:", textinput.upper())

        if Settings.Configurations.outputAllCaps is False:
            print("System:", textinput)

    if Settings.Configurations.textToSpeech is True:
        converter = pyttsx3.init()
        converter.setProperty("rate", 140)  # Can be more than 100
        converter.setProperty(
            "volume", Settings.Configurations.levenshtein
        )  # Set volume 0-1
        voices = converter.getProperty("voices")
        converter.setProperty("voice", voices[0].id)
        voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
        converter.say(textinput)
        converter.runAndWait()

    return textinput


def createSet(dfset, word):
    wordset = []
    for row in dfset:
        for element in row:
            if element == word:
                wordset = wordset + row
                break
    return list(dict.fromkeys(wordset))


"""
The chatbot class is the main class of the program. 
It handles all the states and inputs. 
"""


class chatbot:

    """
    The init function loads in the data of the input file and the given classifier.
    """

    def __init__(self, classifier, filename):
        self.classifier = classifier
        self.df = pd.read_csv(filename)
        self.values = dict()
        self.values["pricerange"] = list(set(self.df["pricerange"]))
        self.values["area"] = list(set(self.df["area"]))
        self.values["area"] = [
            x for x in self.values["area"] if str(x) != "nan"]
        self.values["food"] = list(set(self.df["food"]))
        self.values["phone"] = list(set(self.df["phone"]))
        self.values["addr"] = list(set(self.df["addr"]))
        self.values["postcode"] = list(set(self.df["postcode"]))
        self.pricerange = None
        self.area = None
        self.food = None
        self.restaurants = []
        self.num_restaurants = 0
        self.cur_index = 0
        self.previous_state = "START"
        self.find_restaurant_once = False
        self.moreReqs = None
        self.busy = None
        self.longtime = None
        self.children = None
        self.romantic = None
        self.pizza = None
        self.studentfriendly = None
        self.seriousbusiness = None
        self.repeatOutput = None

    """
    The handle_input function handles the input of the user, checks what kind of input it is and checks the words
    of the sentences.
    """

    def handle_input(self):
        # take user input and convert to lower case
        userInput = input("User: ")
        userInput = userInput.lower()

        # restart toggle
        if "restart" in userInput:
            self.restart()
            self.handle_input()

        # predict and state dialog act label of user input
        pred = self.classifier.predict([userInput])
        if Settings.Configurations.printPrediction:
            TextConfiguration(f"Prediction is {pred[0]}")

        if pred[0] == "inform":
            # user informs program about their preferences
            self.inform(userInput)
            self.checkdb()

        if pred[0] == "affirm":
            # user answers with yes
            self.affirm()
            self.checkdb()

        if pred[0] == "negate":
            # user answers with no
            self.negate()
            self.checkdb()

        if pred[0] == "reqalts":
            # user wants alternative recommendations from the program
            self.reqalts(userInput)

        if pred[0] == "request":
            # user requests information about a restaurant
            self.request(userInput)

        if pred[0] == "hello":
            # user says hello
            self.repeatOutput = TextConfiguration(
                "Nice to meet you, how may I help you? ")

        if pred[0] == "null":
            # user input is unidentified
            self.repeatOutput = TextConfiguration(
                "I did not understand that, could you repeat that? ")

        if pred[0] == "thankyou":
            # user thanks program
            self.repeatOutput = TextConfiguration("You're welcome")

        if pred[0] == "restart":
            # user wants to restart the program
            self.restart()

        if pred[0] == "reqmore":
            # user requests more restaurants with the same preferences they specified earlier
            self.reqmore()

        if pred[0] == "ack":
            # user acknowledges something
            self.ack()

        if pred[0] == "repeat":
            # user wants system to repeat last statement
            self.repeat()

        if pred[0] == "bye":
            # user say good bye
            # exit program
            self.repeatOutput = TextConfiguration("Goodbye")
            sys.exit()

        # loop the input handling function
        self.handle_input()

    """
    The ask_missing_info function handles any inform request by asking for information that is missing.
    It checks whether the user has given their food type, price range and area, and it checks for the extra requirements.
    """

    def ask_missing_info(self):
        # ask for standard preferences if user hasn't specified a specific preference
        if self.food == None and (len(self.restaurants) == 0 or len(self.restaurants) > 1):
            if self.previous_state == "inform_food":
                self.repeatOutput = TextConfiguration(
                    "I didn't understand that(food)")
            else:
                self.repeatOutput = TextConfiguration(
                    "What kind of food would you like?")
            self.previous_state = "inform_food"
            return
        if self.pricerange == None and (len(self.restaurants) == 0 or len(self.restaurants) > 1):
            if self.previous_state == "inform_pricerange":
                self.repeatOutput = TextConfiguration(
                    "I didn't understand that(price)")
            else:
                self.repeatOutput = TextConfiguration(
                    "Would you like something in the cheap, moderate, or expensive price range?"
                )
            self.previous_state = "inform_pricerange"
            return
        if self.area == None and (len(self.restaurants) == 0 or len(self.restaurants) > 1):
            if self.previous_state == "inform_area":
                self.repeatOutput = TextConfiguration(
                    "I didn't understand that")
            else:
                self.repeatOutput = TextConfiguration(
                    "What part of town do you have in mind?")
            self.previous_state = "inform_area"
            return

        # Adding extra requirements here
        # check answer of user in inform, make specific if statement that checks the precious state
        # should I check every requirement or only a couple and use inference for in between requirements?
        if self.moreReqs == None:
            if self.previous_state == "inform_reqs":
                self.repeatOutput = TextConfiguration(
                    "Please answer either yes or no")
            else:
                self.repeatOutput = TextConfiguration(
                    "Do you have more extra requirements for the restaurant? Answer 'yes' for follow-up questions"
                )
            self.previous_state = "inform_reqs"
            return

        elif self.moreReqs:
            if self.busy == None:
                if self.previous_state == "inform_busy":
                    self.repeatOutput = TextConfiguration(
                        "Please answer either yes or no")
                else:
                    self.repeatOutput = TextConfiguration(
                        "Would you like to go to a busy restaurant?")
                self.previous_state = "inform_busy"
                return

            if self.longtime == None:
                if self.previous_state == "inform_longtime":
                    self.repeatOutput = TextConfiguration(
                        "Please answer either yes or no")
                else:
                    self.repeatOutput = TextConfiguration(
                        "Would you like to go to a restaurant where you have to wait for a long time?"
                    )
                self.previous_state = "inform_longtime"
                return

            if self.children == None:
                if self.previous_state == "inform_children":
                    self.repeatOutput = TextConfiguration(
                        "Please answer either yes or no")
                else:
                    self.repeatOutput = TextConfiguration(
                        "Would you like to take your children to the restaurant?"
                    )
                self.previous_state = "inform_children"
                return

            if self.romantic == None:
                if self.previous_state == "inform_romantic":
                    self.repeatOutput = TextConfiguration(
                        "Please answer either yes or no")
                else:
                    self.repeatOutput = TextConfiguration(
                        "Would you like to go to a restaurant with a romantic atmosphere?"
                    )
                self.previous_state = "inform_romantic"
                return

            if self.pizza == None:
                if self.previous_state == "inform_pizza":
                    self.repeatOutput = TextConfiguration(
                        "Please answer either yes or no")
                else:
                    self.repeatOutput = TextConfiguration(
                        "Would you like a place that serves pizzas?")
                self.previous_state = "inform_pizza"
                return

            if self.studentfriendly == None:
                if self.previous_state == "inform_studentfriendly":
                    self.repeatOutput = TextConfiguration(
                        "Please answer either yes or no")
                else:
                    self.repeatOutput = TextConfiguration(
                        "Would you like to go to a student-friendly restaurant?"
                    )
                self.previous_state = "inform_studentfriendly"
                return

            if self.seriousbusiness == None:
                if self.previous_state == "inform_seriousbusiness":
                    self.repeatOutput = TextConfiguration(
                        "Please answer either yes or no")
                else:
                    self.repeatOutput = TextConfiguration(
                        "Do you want to have a serious talk at the restaurant?"
                    )
                self.previous_state = "inform_seriousbusiness"
                return

            else:
                self.checkdb()

        elif len(self.restaurants) == 0:
            self.checkdb()

    """
    The get_restaurant function presents a restaurant that meets all the preferences specified by the user.
    """

    def get_restaurant(self):
        completestring = ""
        completestring = completestring + (
            f"{self.restaurants.iloc[self.cur_index][0]} is a restaurant"
        )
        if self.food != "DONTCARE":
            completestring = completestring + (
                f" that serves {self.restaurants.iloc[self.cur_index][3]} food"
            )
        if self.pricerange != "DONTCARE":
            if self.area == "DONTCARE":
                completestring = completestring + (
                    f" and is in the {self.restaurants.iloc[self.cur_index][1]} price range"
                )
            else:
                completestring = completestring + (
                    f", is in the {self.restaurants.iloc[self.cur_index][1]} price range"
                )
        if self.area != "DONTCARE":
            if self.pricerange == "DONTCARE":
                completestring = completestring + (
                    f" and is in the {self.restaurants.iloc[self.cur_index][2]} part of town"
                )
            else:
                completestring = completestring + (
                    f", and is in the {self.restaurants.iloc[self.cur_index][2]} part of town"
                )
        self.repeatOutput = TextConfiguration(completestring)

        if self.moreReqs:
            self.check_implications()

    # Handles the repeat request
    def repeat(self):
        TextConfiguration(self.repeatOutput)

    # Handles the reqmore action.

    def reqmore(self):
        # check if all preferences have been specified, then recommend next restaurant in list
        if self.food != None and self.pricerange != None and self.area != None:
            if self.num_restaurants == 1:
                self.repeatOutput = TextConfiguration(
                    "That was the only restaurant in the list that matched your preferences"
                )
            else:
                self.cur_index = (self.cur_index + 1) % self.num_restaurants
                self.get_restaurant()
                if self.cur_index + 1 == self.num_restaurants:
                    self.repeatOutput = TextConfiguration(
                        "This is the last restaurant in the list that matches your preferences"
                    )
        else:
            self.ask_missing_info()

    # Handles the reqalts action.
    def reqalts(self, userInput):
        # split user input and check for every word in the sentence if a type of food, an area and/or a price range is found
        # if a preference is changed by the user, update the preference and set the variable 'changed' to True
        changed = False
        words = userInput.split(" ")
        for w in words:
            for food in self.values["food"]:
                if (
                    Levenshtein.jaro_winkler(food, w)
                    > Settings.Configurations.levenshtein
                ):
                    changed = True
                    self.food = food
            for area in self.values["area"]:
                if (
                    Levenshtein.jaro_winkler(area, w)
                    > Settings.Configurations.levenshtein
                ):
                    changed = True
                    self.area = area
            for pricerange in self.values["pricerange"]:
                if (
                    Levenshtein.jaro_winkler(pricerange, w)
                    > Settings.Configurations.levenshtein
                ):
                    changed = True
                    self.pricerange = pricerange

        # check if all preferences are known, if not then first ask the user for their preferences
        # if all preferences are known and one of them has changed, find restaurants matching the new preferences
        # if the preferences haven't changed, just recommend a different restaurant with the same preferences
        if self.food == None or self.pricerange == None or self.area == None:
            self.ask_missing_info()
        elif changed == True:
            self.checkdb()
        else:
            self.cur_index = (self.cur_index + 1) % self.num_restaurants
            self.get_restaurant()

    # Handles the request action.
    def request(self, userInput):
        # tokenize the user input and check if any restaurants have been found yet
        words = word_tokenize(userInput)
        self.find_restaurant_once = True
        if self.num_restaurants != 0:
            # check for every word in the sentence if the user requests information about the current restaurant
            # particular information about the restaurant includes food type, area, price range, phone number, address, and post code
            request_answered = False
            for w in words:

                # region Request Food
                if (
                    Levenshtein.jaro_winkler("food", w)
                    > Settings.Configurations.levenshtein
                ):
                    if "nan" not in str(self.restaurants.iloc[self.cur_index][3]):
                        self.repeatOutput = TextConfiguration(
                            f"{self.restaurants.iloc[self.cur_index][0]} serves {self.restaurants.iloc[self.cur_index][3]} food"
                        )
                    else:
                        self.repeatOutput = TextConfiguration(
                            f"It is not known what kind of food {self.restaurants.iloc[self.cur_index][0]} serves"
                        )
                    request_answered = True
                # endregion

                # region Request AREA
                if (
                    Levenshtein.jaro_winkler("area", w)
                    > Settings.Configurations.levenshtein
                ):
                    if "nan" not in str(self.restaurants.iloc[self.cur_index][2]):
                        self.repeatOutput = TextConfiguration(
                            f"{self.restaurants.iloc[self.cur_index][0]} is located in the {self.restaurants.iloc[self.cur_index][2]} part of town"
                        )
                    else:
                        self.repeatOutput = TextConfiguration(
                            f"It is not known in what area {self.restaurants.iloc[self.cur_index][0]} is located"
                        )
                    request_answered = True

                elif (
                    Levenshtein.jaro_winkler("town", w)
                    > Settings.Configurations.levenshtein
                ):
                    if "nan" not in self.restaurants.iloc[self.cur_index][2]:
                        self.repeatOutput = TextConfiguration(
                            f"{self.restaurants.iloc[self.cur_index][0]} is located in the {self.restaurants.iloc[self.cur_index][2]} part of town"
                        )
                    else:
                        self.repeatOutput = TextConfiguration(
                            f"It is not known in what area {self.restaurants.iloc[self.cur_index][0]} is located"
                        )
                    request_answered = True
                # endregion

                # region Request PriceRange
                if (
                    Levenshtein.jaro_winkler("pricerange", w)
                    > Settings.Configurations.levenshtein
                ):
                    if "nan" not in str(self.restaurants.iloc[self.cur_index][1]):
                        self.repeatOutput = TextConfiguration(
                            f"{self.restaurants.iloc[self.cur_index][0]} is in the {self.restaurants.iloc[self.cur_index][1]} pricerange"
                        )
                    else:
                        self.repeatOutput = TextConfiguration(
                            f"It is not known what the price range of {self.restaurants.iloc[self.cur_index][0]} is "
                        )
                    request_answered = True
                # endregion

                # region Request Phone Number
                if (
                    Levenshtein.jaro_winkler("phone", w)
                    > Settings.Configurations.levenshtein
                ):
                    if "nan" not in str(self.restaurants.iloc[self.cur_index][4]):
                        self.repeatOutput = TextConfiguration(
                            f"The phone number of {self.restaurants.iloc[self.cur_index][0]} is {self.restaurants.iloc[self.cur_index][4]}"
                        )
                    else:
                        self.repeatOutput = TextConfiguration(
                            f"The phone number of {self.restaurants.iloc[self.cur_index][0]} is not known"
                        )
                    request_answered = True
                elif (
                    Levenshtein.jaro_winkler("number", w)
                    > Settings.Configurations.levenshtein
                    and request_answered == False
                ):
                    if "nan" not in str(self.restaurants.iloc[self.cur_index][4]):
                        self.repeatOutput = TextConfiguration(
                            f"The phone number of {self.restaurants.iloc[self.cur_index][0]} is {self.restaurants.iloc[self.cur_index][4]}"
                        )
                    else:
                        self.repeatOutput = TextConfiguration(
                            f"The phone number of {self.restaurants.iloc[self.cur_index][0]} is not known"
                        )
                    request_answered = True
                # endregion

                # region Request Address
                if (
                    Levenshtein.jaro_winkler("address", w)
                    > Settings.Configurations.levenshtein
                ):
                    if "nan" not in str(self.restaurants.iloc[self.cur_index][5]):
                        self.repeatOutput = TextConfiguration(
                            f"{self.restaurants.iloc[self.cur_index][0]} is on {self.restaurants.iloc[self.cur_index][5]}"
                        )
                    else:
                        self.repeatOutput = TextConfiguration(
                            f"The address of {self.restaurants.iloc[self.cur_index][0]} is not known"
                        )
                    request_answered = True
                # endregion

                # region Request Post Code
                if (
                    Levenshtein.jaro_winkler("postcode", w)
                    > Settings.Configurations.levenshtein
                ):
                    if "nan" not in str(self.restaurants.iloc[self.cur_index][6]):
                        self.repeatOutput = TextConfiguration(
                            f"The postcode of {self.restaurants.iloc[self.cur_index][0]} is {self.restaurants.iloc[self.cur_index][6]}"
                        )
                    else:
                        self.repeatOutput = TextConfiguration(
                            f"The postcode of {self.restaurants.iloc[self.cur_index][0]} is not known"
                        )
                    request_answered = True
                # endregion

        else:
            self.repeatOutput = TextConfiguration(
                "Before requesting information about the restaurant, please first tell in what kind of restaurant you would like to eat"
            )

    """
    In the find_restaurant function, the restaurant are found in the restaurant_info.csv dataset from the given information.
    If no restaurant are found, it will tell the user.
    """

    def find_restaurant(self):
        if self.food == "DONTCARE" and self.area == "DONTCARE":
            self.restaurants = self.df.loc[(
                self.df["pricerange"] == self.pricerange)]
        elif self.food == "DONTCARE" and self.pricerange == "DONTCARE":
            self.restaurants = self.df.loc[(self.df["area"] == self.area)]
        elif self.pricerange == "DONTCARE" and self.area == "DONTCARE":
            self.restaurants = self.df.loc[(self.df["food"] == self.food)]
        elif self.food == "DONTCARE":
            self.restaurants = self.df.loc[
                (self.df["pricerange"] == self.pricerange)
                & (self.df["area"] == self.area)
            ]
            if self.restaurants.shape[0] == 0:
                self.repeatOutput = TextConfiguration(
                    f"I haven't found any {self.pricerange} restaurant in the {self.area}"
                )
        elif self.area == "DONTCARE":
            self.restaurants = self.df.loc[
                (self.df["pricerange"] == self.pricerange)
                & (self.df["food"] == self.food)
            ]
            if self.restaurants.shape[0] == 0:
                self.repeatOutput = TextConfiguration(
                    f"I haven't found any {self.food} restaurant in the price range of {self.pricerange}"
                )
        elif self.pricerange == "DONTCARE":
            self.restaurants = self.df.loc[
                (self.df["area"] == self.area) & (self.df["food"] == self.food)
            ]
            if self.restaurants.shape[0] == 0:
                self.repeatOutput = TextConfiguration(
                    f"I haven't found any {self.food} restaurant in the {self.area} area"
                )
        else:
            self.restaurants = self.df.loc[
                (self.df["pricerange"] == self.pricerange)
                & (self.df["area"] == self.area)
                & (self.df["food"] == self.food)
            ]
            if self.restaurants.shape[0] == 0:
                self.repeatOutput = TextConfiguration(
                    f"I haven't found any {self.food} restaurant in the {self.area} area"
                )

        # update the number of restaurants and the index
        self.num_restaurants = self.restaurants.shape[0]
        self.cur_index = 0
        self.restaurants = self.restaurants.sample(
            frac=1).reset_index(drop=True)
        if self.num_restaurants > 0:
            self.get_restaurant()

    # Calls the inform request handles the input of it.
    def inform(self, userInput):
        words = userInput.split(" ")
        for w in words:
            for food in self.values["food"]:
                if (
                    Levenshtein.jaro_winkler(food, w)
                    > Settings.Configurations.levenshtein
                ):
                    self.food = food
            for area in self.values["area"]:
                if (
                    Levenshtein.jaro_winkler(area, w)
                    > Settings.Configurations.levenshtein
                ):
                    self.area = area
            for pricerange in self.values["pricerange"]:
                if (
                    Levenshtein.jaro_winkler(pricerange, w)
                    > Settings.Configurations.levenshtein
                ):
                    self.pricerange = pricerange

        # User has the word 'food' in their input, but the system doesn't recognise any food type
        if self.food == None and "food" in words:
            food_item = words[words.index("food") - 1]
            for food in self.values["food"]:
                if (
                    Levenshtein.jaro_winkler(food, food_item)
                    > Settings.Configurations.levenshtein
                ):
                    self.food = food
            if self.food == None:
                self.repeatOutput = TextConfiguration(
                    f"I'm sorry but there is no restaurant serving {food_item} food"
                )

        # check if the user doesn't care about the area or if they used an unknown or misspelled area as input
        if self.area == None and ("area" in words or "part" in words):
            if "area" in words:
                area_item = words[words.index("area") - 1]
            if "part" in words:
                area_item = words[words.index("part") - 1]
            if (
                Levenshtein.jaro_winkler("any", area_item)
                > Settings.Configurations.levenshtein
            ):
                self.area = "DONTCARE"
            else:
                for area in self.values["area"]:
                    if (
                        Levenshtein.jaro_winkler(area, area_item)
                        > Settings.Configurations.levenshtein
                    ):
                        self.area = area
                if self.area == None:
                    self.repeatOutput = TextConfiguration(
                        f"I'm sorry but there is no restaurant in the {area_item} area"
                    )

        # check if the user doesn't care about the price range
        if self.pricerange == None and "price" in words:
            price_item = words[words.index("price") - 1]
            if (
                Levenshtein.jaro_winkler("any", price_item)
                > Settings.Configurations.levenshtein
            ):
                self.pricerange = "DONTCARE"

        # check if the user used an unknown or misspelled area as input
        # not sure if this is necessary after the 'area or part' check?
        if self.area == None and "town" in words:
            if "area" in words:
                area_place = words[words.index("area") - 1]
            for area in self.values["area"]:
                if (
                    Levenshtein.jaro_winkler(area, area_place)
                    > Settings.Configurations.levenshtein
                ):
                    self.area = area
            if self.area == None:
                self.repeatOutput = TextConfiguration(
                    f"I'm sorry but there is no restaurant in {area_place} part of the town"
                )

        # check if the user specified that they don't care about a specific preference
        if self.previous_state == "inform_food" and "dont care" in userInput:
            self.food = "DONTCARE"
        if self.previous_state == "inform_pricerange" and "dont care" in userInput:
            self.pricerange = "DONTCARE"
        if self.previous_state == "inform_area" and "dont care" in userInput:
            self.area = "DONTCARE"

    # Negates previously made statement
    def negate(self):
        # check the user's answers for the extra requirements
        if self.previous_state == "inform_reqs":
            self.moreReqs = False
        elif self.previous_state == "inform_busy":
            self.busy = False
        elif self.previous_state == "inform_longtime":
            self.longtime = False
        elif self.previous_state == "inform_children":
            self.children = False
        elif self.previous_state == "inform_romantic":
            self.romantic = False
        elif self.previous_state == "inform_pizza":
            self.pizza = False
        elif self.previous_state == "inform_studentfriendly":
            self.studentfriendly = False
        elif self.previous_state == "inform_seriousbusiness":
            self.seriousbusiness = False

    # This function restarts the sequence by setting each value to null
    def restart(self):
        if Settings.Configurations.restartDialog:
            # change variables to initial state and start conversation again
            self.pricerange = None
            self.area = None
            self.food = None
            self.restaurants = None
            self.previous_state = "START"
            self.find_restaurant_once = False
            self.repeatOutput = TextConfiguration(
                "Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?"
            )
        else:
            self.repeatOutput = TextConfiguration(
                "A restart is not possible."
            )

    def affirm(self):
        # check the user's answers for the extra requirements
        if self.previous_state == "inform_reqs":
            self.moreReqs = True
        elif self.previous_state == "inform_busy":
            self.busy = True
        elif self.previous_state == "inform_longtime":
            self.longtime = True
        elif self.previous_state == "inform_children":
            self.children = True
        elif self.previous_state == "inform_romantic":
            self.romantic = True
        elif self.previous_state == "inform_pizza":
            self.pizza = True
        elif self.previous_state == "inform_studentfriendly":
            self.studentfriendly = True
        elif self.previous_state == "inform_seriousbusiness":
            self.seriousbusiness = True

    def ack(self):
        # if not all preferences are known, ask about those first
        # if preferences are known and one restaurant if found, be excited about that restaurant? (that's how the program in the dataset reacts)
        # if more restaurants are found, recommend the next one
        if self.food == None or self.pricerange == None or self.area == None:
            self.ask_missing_info()
        elif self.num_restaurants == 1:
            self.repeatOutput = TextConfiguration(
                f"{self.restaurants.iloc[self.cur_index][0]} is a great restaurant!"
            )
        elif self.num_restaurants > 1:
            self.cur_index = (self.cur_index + 1) % self.num_restaurants
            self.get_restaurant()

    """
    The checkdb function checks the database while the user hasn't given all of their preferences.
    It checks whether there are no restaurants satisfying the current preferences and asks the user to restate their preferences if that is the case.
    It also immediately recommends a restaurant to the user if there is only one restaurant satisfying the current preferences.
    """

    def checkdb(self):
        if self.pricerange == None:
            if (self.area == "DONTCARE" and self.food == "DONTCARE") or (self.area == None and self.food == None):
                return
            elif self.area == None or self.area == "DONTCARE":
                self.restaurants = self.df.loc[(self.df["food"] == self.food)]
            elif self.food == None or self.area == "DONTCARE":
                self.restaurants = self.df.loc[(
                    self.df["area"] == self.area)]
            else:
                self.restaurants = self.df.loc[
                    (self.df["area"] == self.area) & (
                        self.df["food"] == self.food)
                ]
        elif self.area == None:
            if (self.pricerange == "DONTCARE" and self.food == "DONTCARE") or (self.pricerange == None and self.food == None):
                return
            elif self.pricerange == "DONTCARE" or self.pricerange == None:
                self.restaurants = self.df.loc[(self.df["food"] == self.food)]
            elif self.food == "DONTCARE" or self.food == None:
                self.restaurants = self.df.loc[
                    (self.df["pricerange"] == self.pricerange)
                ]
            else:
                self.restaurants = self.df.loc[
                    (self.df["pricerange"] == self.pricerange)
                    & (self.df["food"] == self.food)
                ]
        elif self.food == None:
            if (self.pricerange == "DONTCARE" and self.area == "DONTCARE") or (self.pricerange == None and self.area == None):
                return
            elif self.pricerange == "DONTCARE" or self.pricerange == None:
                self.restaurants = self.df.loc[(self.df["area"] == self.area)]
            elif self.area == "DONTCARE" or self.area == None:
                self.restaurants = self.df.loc[
                    (self.df["pricerange"] == self.pricerange)
                ]
            else:
                self.restaurants = self.df.loc[
                    (self.df["pricerange"] == self.pricerange)
                    & (self.df["area"] == self.area)
                ]
        else:
            if (
                self.food == "DONTCARE"
                and self.area == "DONTCARE"
                and self.pricerange == "DONTCARE"
            ):
                self.restaurants = self.df.loc
            elif self.food == "DONTCARE" and self.area == "DONTCARE":
                self.restaurants = self.df.loc[
                    (self.df["pricerange"] == self.pricerange)
                ]
            elif self.food == "DONTCARE" and self.pricerange == "DONTCARE":
                self.restaurants = self.df.loc[(self.df["area"] == self.area)]
            elif self.pricerange == "DONTCARE" and self.area == "DONTCARE":
                self.restaurants = self.df.loc[(self.df["food"] == self.food)]
            elif self.food == "DONTCARE":
                self.restaurants = self.df.loc[
                    (self.df["pricerange"] == self.pricerange)
                    & (self.df["area"] == self.area)
                ]
            elif self.area == "DONTCARE":
                self.restaurants = self.df.loc[
                    (self.df["pricerange"] == self.pricerange)
                    & (self.df["food"] == self.food)
                ]
            elif self.pricerange == "DONTCARE":
                self.restaurants = self.df.loc[
                    (self.df["area"] == self.area) & (
                        self.df["food"] == self.food)
                ]
            else:
                self.restaurants = self.df.loc[
                    (self.df["pricerange"] == self.pricerange)
                    & (self.df["area"] == self.area)
                    & (self.df["food"] == self.food)
                ]

        if len(self.restaurants) == 0:
            # no restaurants satisfy the current preferences, find similar restaurants using the suggest function
            self.suggest()
            if len(self.restaurants) == 0:
                if ((self.pricerange == None or self.pricerange == "DONTCARE") and
                    (self.area != None or self.area != "DONTCARE") and
                        (self.food != None or self.food != "DONTCARE")):
                    self.repeatOutput = TextConfiguration(
                        f"No {self.food} restaurant in the {self.area} area is found"
                    )
                elif ((self.pricerange != None or self.pricerange != "DONTCARE") and
                      (self.area == None or self.area == "DONTCARE") and
                        (self.food != None or self.food != "DONTCARE")):
                    self.repeatOutput = TextConfiguration(
                        f"No {self.food} restaurant in the {self.pricerange} price range is found"
                    )
                elif ((self.pricerange != None or self.pricerange != "DONTCARE") and
                      (self.area != None or self.area != "DONTCARE") and
                        (self.food == None or self.food == "DONTCARE")):
                    self.repeatOutput = TextConfiguration(
                        f"No {self.pricerange} restaurant in the {self.area} area is found"
                    )
                else:
                    self.repeatOutput = TextConfiguration(
                        f"No restaurant found, please restate one or more of your preferences."
                    )
                self.handle_input()
        elif len(self.restaurants) == 1:
            self.area = self.area if self.area != None else "DONTCARE"
            self.pricerange = self.pricerange if self.pricerange != None else "DONTCARE"
            self.food = self.food if self.food != None else "DONTCARE"
            # a single restaurant satisfying the current preferences is found
            # ask for extra requirements and then recommend restaurant
            if self.moreReqs == None:
                self.ask_missing_info()
            elif self.moreReqs and self.seriousbusiness == None:
                self.ask_missing_info()
            else:
                self.num_restaurants = len(self.restaurants)
                self.cur_index = 0
                self.get_restaurant()
        elif self.food == None or self.pricerange == None or self.area == None or self.moreReqs == None or (self.moreReqs is True and self.seriousbusiness is None):
            self.ask_missing_info()
        else:
            self.num_restaurants = len(self.restaurants)
            self.get_restaurant()

    """
    The suggest function is used when no restaurant with the exact preferences is found.
    It finds restaurants that have similar attributes to the given preferences.
    """

    def suggest(self):
        foodset = [
            ["thai", "chinese", "korean", "vietnamese", "asian oriental"],
            [
                "mediterranean",
                "spanish",
                "portuguese",
                "italian",
                "romanian",
                "tuscan",
                "catalan",
            ],
            ["french", "european", "bistro", "swiss", "gastropub", "traditional"],
            ["north american", "steakhouse", "british"],
            ["lebanese", "turkish", "persian"],
            ["international", "modern european", "fusion"],
        ]

        regionset = [
            ["centre", "north", "west"],
            ["centre", "north", "east"],
            ["centre", "south", "west"],
            ["centre", "south", "east"],
        ]

        priceranges = [["cheap", "moderate"], ["moderate, expensive"]]

        cuisineset = createSet(foodset, self.food) if len(
            createSet(foodset, self.food)) != 0 else None
        areaset = createSet(regionset, self.area) if len(
            createSet(regionset, self.area)) != 0 else None
        pricerangeset = createSet(priceranges, self.pricerange) if len(
            createSet(priceranges, self.pricerange)) != 0 else None

        if cuisineset != None and pricerangeset == None and areaset == None:
            self.restaurants = self.df.loc[(self.df["food"].isin(cuisineset))]
        elif cuisineset == None and pricerangeset != None and areaset == None:
            self.restaurants = self.df.loc[(
                self.df["pricerange"].isin(pricerangeset))]
        elif cuisineset == None and pricerangeset == None and areaset != None:
            self.restaurants = self.df.loc[(self.df["area"].isin(areaset))]
        elif cuisineset == None and pricerangeset != None and areaset != None:
            self.restaurants = self.df.loc[(self.df["pricerange"] == self.pricerange) &
                                           (self.df["area"].isin(areaset))]
            if len(self.restaurants) == 0:
                self.restaurants = self.df.loc[(self.df["pricerange"].isin(areaset)) &
                                               (self.df["area"] == self.area)]
        elif areaset == None and cuisineset != None and pricerangeset != None:
            self.restaurants = self.df.loc[(self.df["pricerange"].isin(pricerangeset)) &
                                           (self.df["food"] == self.food)]
            if len(self.restaurants) == 0:
                self.restaurants = self.df.loc[(self.df["pricerange"] == self.pricerange) &
                                               (self.df["food"].isin(cuisineset))]
        elif pricerangeset == None and cuisineset != None and areaset != None:
            self.restaurants = self.df.loc[(self.df["area"].isin(areaset)) &
                                           (self.df["food"] == self.food)]
            if len(self.restaurants) == 0:
                self.restaurants = self.df.loc[(self.df["area"] == self.area) &
                                               (self.df["food"].isin(cuisineset))]
        elif pricerangeset != None and cuisineset != None and areaset != None:
            self.restaurants = self.df.loc[(self.df["pricerange"] == self.pricerange) &
                                           (self.df["food"] == self.food) &
                                           (self.df["area"].isin(areaset))]
            if len(self.restaurants) == 0:
                self.restaurants = self.df.loc[(self.df["pricerange"].isin(pricerangeset)) &
                                               (self.df["food"] == self.food) &
                                               (self.df["area"] == self.area)]
                if len(self.restaurants) == 0:
                    self.restaurants = self.df.loc[(self.df["pricerange"] == self.pricerange) &
                                                   (self.df["food"].isin(cuisineset)) &
                                                   (self.df["area"] == self.area)]

        self.cur_index = 0
        self.restaurants = self.restaurants.sample(
            frac=1).reset_index(drop=True)
        self.num_restaurants = len(self.restaurants)
        if self.num_restaurants > 0:
            self.repeatOutput = TextConfiguration(
                f"No restaurant found with these specific preferences, our systems suggests the following restaurant:"
            )
            self.get_restaurant()

    """
    The print_rule_sequence function prints the inference path of the implications function using recursion.
    """

    def print_rule_sequence(self, ruleInfo, curCon, level):
        if level == 0:
            return
        for rule in ruleInfo:
            if curCon == rule[3]:
                level -= 1
                curCon = rule[2]
                self.print_rule_sequence(ruleInfo, curCon, level)
                TextConfiguration(
                    f"{rule[1]}"
                )
                return

    """
    The check_implications function checks if the current restaurant meets the extra requirements given by the user.
    It will check the truth values of the given preferences using an inference loop.
    It then presents the user with the number of requirements that are met.
    The user can then decide whether they want the current restaurant or a different one.
    """

    def check_implications(self):
        # initialise the dictionary with extra preference variables
        dict = {
            "busy": None,
            "longtime": None,
            "children": None,
            "romantic": None,
            "pizza": None,
            "studentfriendly": None,
            "seriousbusiness": None
        }

        initDict = dict.copy()
        ruleInfo = []
        changed = True

        # this while loop checks which rules are true
        # if a statement is true, the truth value of the consequence is changed
        # the ruleInfo array appends information about the rules that apply
        while changed == True:
            oldDict = dict.copy()

            # level 1
            if self.restaurants.iloc[self.cur_index][1] == "cheap" and self.restaurants.iloc[self.cur_index][7] == "good":
                dict["busy"] = True
                rule1 = "Level 1, Rule 1: [cheap, good food] > busy = True"
                ruleInfo.append([1, rule1, "cheap", "busy"])
            if self.restaurants.iloc[self.cur_index][3] == "spanish":
                dict["longtime"] = True
                rule2 = "Level 1, Rule 2: [spanish] > long time = True"
                ruleInfo.append([1, rule2, "spanish", "longtime"])
            if self.restaurants.iloc[self.cur_index][3] == "italian":
                dict["pizza"] = True
                rule3 = "Level 1, Rule 3: [italian] > pizza = True"
                ruleInfo.append([1, rule3, "italian", "pizza"])
            if self.restaurants.iloc[self.cur_index][8] == "spicy":
                dict["children"] = False
                rule4 = "Level 1, Rule 4: [spicy] > children = False"
                ruleInfo.append([1, rule4, "spicy", "!children"])

            # level 2
            if dict["busy"]:
                dict["longtime"] = True
                rule5 = "Level 2, Rule 5: [busy] > long time = True"
                ruleInfo.append([2, rule5, "busy", "longtime"])
            if dict["longtime"]:
                dict["children"] = False
                rule6 = "Level 2, Rule 6: [long time] >  children = False"
                ruleInfo.append([2, rule6, "longtime", "!children"])
            if dict["busy"]:
                dict["romantic"] = False
                rule7 = "Level 2, Rule 7: [busy] > romantic = False"
                ruleInfo.append([2, rule7, "busy", "!romantic"])
            if dict["longtime"]:
                dict["romantic"] = True
                rule8 = "Level 2: Rule 8: [long time] > romantic = True"
                ruleInfo.append([2, rule8, "longtime", "romantic"])
            if dict["children"] is False:
                dict["romantic"] = True
                rule9 = "Level 2: Rule 9: [!children] > romantic = True"
                ruleInfo.append([2, rule9, "!children", "romantic"])
                dict["seriousbusiness"] = True
                rule10 = "Level 2: Rule 10: [!children] > serious business = True"
                ruleInfo.append([2, rule10, "!children", "seriousbusiness"])
            if (self.restaurants.iloc[self.cur_index][1] == "expensive" or self.restaurants.iloc[self.cur_index][1] == "moderate") and dict["longtime"] is True:
                dict["studentfriendly"] = False
                rule11 = "Level 2: Rule 11: [(expensive or moderate), long time] > student-friendly = False"
                ruleInfo.append([2, rule11, "longtime", "!studentfriendly"])
            if self.restaurants.iloc[self.cur_index][1] != "expensive" and dict["pizza"]:
                dict["studentfriendly"] = True
                rule12 = "Level 2: Rule 12: [!expensive, pizza] > student-friendly = True"
                ruleInfo.append([2, rule12, "pizza", "studentfriendly"])

            # level 3
            if dict["studentfriendly"]:
                dict["seriousbusiness"] = False
                rule13 = "Level 3: Rule 13: [student-friendly] > serious business = False"
                ruleInfo.append(
                    [3, rule13, "studentfriendly", "!seriousbusiness"])

            # check if dictionary is changed
            x = 0
            for item in dict:
                if dict[item] != oldDict[item]:
                    x = 1
            if x == 0:
                changed = False

        # every if-statement compares inferences to user requirements
        # a different rule will be printed depending on the truth value of the user's requirement and the truth value of the consequence
        totReqs = 0
        fulfilledReqs = 0
        print("\n")
        if self.busy:
            totReqs += 1
            if dict["busy"]:
                fulfilledReqs += 1
                self.repeatOutput = TextConfiguration(
                    "This restaurant is recommended because:"
                )
                self.print_rule_sequence(ruleInfo, "busy", 1)
            else:
                if initDict["busy"] == dict["busy"]:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is probably not busy"
                    )
                else:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is not recommended because:"
                    )
                    self.print_rule_sequence(ruleInfo, "!busy", 1)
        if self.longtime:
            totReqs += 1
            if dict["longtime"]:
                fulfilledReqs += 1
                self.repeatOutput = TextConfiguration(
                    "This restaurant is recommended because:"
                )
                self.print_rule_sequence(ruleInfo, "longtime", 1)
            else:
                if initDict["longtime"] == dict["longtime"]:
                    self.repeatOutput = TextConfiguration(
                        "You probably don't have to wait a long time at this restaurant"
                    )
                else:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is not recommended because:"
                    )
                    self.print_rule_sequence(ruleInfo, "!longtime", 1)
        if self.children:
            totReqs += 1
            if dict["children"]:
                fulfilledReqs += 1
                self.repeatOutput = TextConfiguration(
                    "This restaurant is probably recommended for children"
                )
            else:
                if initDict["children"] == dict["children"]:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is probably not recommended for children"
                    )
                else:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is not recommended because:"
                    )
                    self.print_rule_sequence(ruleInfo, "!children", 2)
        if self.romantic:
            totReqs += 1
            if dict["romantic"]:
                fulfilledReqs += 1
                self.repeatOutput = TextConfiguration(
                    "This restaurant is recommended because:"
                )
                self.print_rule_sequence(ruleInfo, "romantic", 2)
            else:
                if initDict["romantic"] == dict["romantic"]:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is probably not romantic"
                    )
                else:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is not recommended because:"
                    )
                    self.print_rule_sequence(ruleInfo, "!romantic", 2)
        if self.pizza:
            totReqs += 1
            if dict["pizza"]:
                fulfilledReqs += 1
                self.repeatOutput = TextConfiguration(
                    "This restaurant is recommended because:"
                )
                self.print_rule_sequence(ruleInfo, "pizza", 2)
            else:
                if initDict["pizza"] == dict["pizza"]:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant probably doesn't serve pizza"
                    )
                else:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is not recommended because:"
                    )
                    self.print_rule_sequence(ruleInfo, "!pizza", 2)
        if self.studentfriendly:
            totReqs += 1
            if dict["studentfriendly"]:
                fulfilledReqs += 1
                self.repeatOutput = TextConfiguration(
                    "This restaurant is recommended because:"
                )
                self.print_rule_sequence(ruleInfo, "studentfriendly", 2)
            else:
                if initDict["studentfriendly"] == dict["studentfriendly"]:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is not particularly fit for students"
                    )
                else:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is not recommended because:"
                    )
                    self.print_rule_sequence(ruleInfo, "!studentfriendly", 2)
        if self.seriousbusiness:
            totReqs += 1
            if dict["seriousbusiness"]:
                fulfilledReqs += 1
                self.repeatOutput = TextConfiguration(
                    "This restaurant is recommended because:"
                )
                self.print_rule_sequence(ruleInfo, "seriousbusiness", 3)
            else:
                if initDict["seriousbusiness"] == dict["seriousbusiness"]:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is not fit to discuss serious business"
                    )
                else:
                    self.repeatOutput = TextConfiguration(
                        "This restaurant is not recommended because:"
                    )
                    self.print_rule_sequence(ruleInfo, "!seriousbusiness", 3)

        # this shows the user the number of fulfilled requirements
        # they can then choose themselves if they want to ask for restaurant details or if they want a different restaurant
        if totReqs != 0:
            self.repeatOutput = TextConfiguration(
                f"\nThis restaurant satisfies {fulfilledReqs} out of the {totReqs} extra requirements")

        return

    """
    The start function starts the input loop.
    """

    def start(self):
        self.repeatOutput = TextConfiguration(
            "Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?"
        )
        self.handle_input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='The argument parser for the chatbot')
    parser.add_argument("--classifier", default="lstm", type=str, choices=["lstm", "logistic", "tree", "most_common", "keyword"],
                        help="Sets the classifier used in the chatbot. lstm is for the LTSM network," +
                             "logistic for the logistic regression, tree for the decision tree, " +
                             "most_common for the most common baseline and keyword for the keyword baseline")
    parser.add_argument("--jaro", default=0.9, type=float,
                        help="Set the value for the Jaro Winkler, needs to between 0 and 1.")
    parser.add_argument("--caps", default=False, type=bool,
                        help="Used to set the output to all caps.")
    parser.add_argument("--delay", default=0, type=float,
                        help="Adds a delay in the chatbot in seconds.")
    parser.add_argument("--restart", default=False, type=bool,
                        help="Restart dialog.")
    parser.add_argument("--speech", default=False, type=bool,
                        help="Text to speech.")
    parser.add_argument("--preference", default=False, type=bool,
                        help="Perference ask order.")
    parser.add_argument("--test", default=False, type=bool,
                        help="If you want to test the model")
    parser.add_argument("--prediction", default=False, type=bool,
                        help="Set to true if you want to print out the prediction (for debugging)")

    args = parser.parse_args()

    if args.classifier == "lstm":
        classifier = classifiers.nn()
    elif args.classifier == "logistic":
        classifier = classifiers.logistic_regression()
    elif args.classifier == "tree":
        classifier = classifiers.decision_tree()
    elif args.classifier == "keyword":
        classifier = classifiers.keyword()
    else:
        classifier = classifiers.most_common()

    Settings.Configurations.levenshtein = args.jaro
    Settings.Configurations.outputAllCaps = args.caps
    Settings.Configurations.delayAnswers = args.delay
    Settings.Configurations.restartDialog = args.restart
    Settings.Configurations.textToSpeech = args.speech
    Settings.Configurations.preferenceAskOrder = args.preference
    Settings.Configurations.printPrediction = args.prediction

    if args.test:
        classifier.test_model()

    chat = chatbot(classifier, "restaurant_info.csv")
    chat.start()
