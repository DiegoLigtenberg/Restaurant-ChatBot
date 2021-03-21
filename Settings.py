'''
static class
class with preferred configurations
The code will then run based on these configurations
'''
import Settings


class Configurations():

    levenshtein = 1
    outputAllCaps = False
    delayAnswers = 0
    restartDialog = False
    textToSpeech = 1
    printPrediction = False

    @staticmethod
    def set_Configurations(levenshtein, outputAllCaps, delayAnswers, restartDialog, textToSpeech, printPrediction):
        '''
        levenshtein         : float      if value < 1, the levenshtein distance is used to allow for misspellings
        outputAllCaps       : bool       if turned on, all output is in CAPS
        delayAnswers        : float      if value > 0, the system has a delay before answering the user's utterance
        restartDialog       : bool       if turned on, the user is allowed to restart the whole dialogue
        textToSpeech        : bool       if turned on, the system will speak to the user
        printPrediction     : bool       if turned on, the system will print out the prediction
        '''
        Configurations.levenshtein = levenshtein
        Configurations.outputAllCaps = outputAllCaps
        Configurations.delayAnswers = delayAnswers
        Configurations.restartDialog = restartDialog
        Configurations.textToSpeech = textToSpeech
        Configurations.printPrediction = printPrediction


'''
place this in main code
for levenshtein we can make a variable for > 0.9 and make it 0 in order to putting it off =)
'''
Settings.Configurations.set_Configurations(
    levenshtein=False,
    outputAllCaps=False,
    delayAnswers=0,  # int    delay in seconds
    restartDialog=False,
    textToSpeech=False,
    printPrediction=False,
)
