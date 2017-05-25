import warnings
from asl_data import SinglesData
import math

def recognize(models: dict, test_set: SinglesData, verbose=False):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # verbose = True
    
    # go through all the tests
    for i in range(0, test_set.num_items):
        
        local_probabilities = dict()
        # go through all available models and see which fits better
        for word, model in models.items():
            if verbose:
                if model:
                    print(f"sample {i}: evaluating with word={word}", end="")
                else:
                    print(f"sample {i}: no valid model for {word}", end="")
            logL = -math.inf
            X, lengths = test_set.get_item_Xlengths(i)
            try:
                if model is not None:
                    logL = model.score(X, lengths)
            except:
                logL = -math.inf
                if verbose:
                    print(f"could not test sample {i} using model {word}")
                # raise

            local_probabilities[word] = logL
            if verbose: 
                print(f" score: {logL}")
        
        probabilities.append(local_probabilities)
        best_guess = max(local_probabilities, key=local_probabilities.get)
        if verbose:
            print(f"best guess for sample {i} is {best_guess}")
        guesses.append(best_guess)
        
    return probabilities, guesses