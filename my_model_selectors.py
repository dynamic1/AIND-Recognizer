import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        # return False
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        word = self.this_word
        word_sequences = self.sequences
        if self.verbose:
            # print(f"My sequences for {word}: {self.sequences}")
            pass
        
        hmm_models = {}
        best_score = None
        best_n = None
        if self.verbose:
            print(f"trying to build models with {self.min_n_components} -> {self.max_n_components} states")
        for n in range(self.min_n_components, self.max_n_components):
            # print(f"{n} components")
            try:
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                hmm_models[n] = model
                # evaluate the model
                logL = model.score(self.X, self.lengths)
                # print(f"means:\n{model.means_}")
                # print(f"covars:\n{model.covars_}") 
                parameters_count = n * n + 2*n*len(self.X[0]) -1
                logN = np.log(len(self.lengths))
                bic = -2 * logL + parameters_count * logN
                if best_score is None or best_score < bic :
                    best_score = bic
                    best_n = n
                if self.verbose:
                    print("model created for {} with {} states. score={}".format(word, n, logL))
            except ValueError:
                hmm_models[n] = None
                if self.verbose:
                    print(f"failed to build model with {n} states for {word}")
               
        # acum cred ca trebuie sa le evaluez si sa aleg pe cel mai bun

        if self.verbose:
            print(f"best model for {word} has {best_n} states")
        if best_n is None:
            return None
        return hmm_models[best_n]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        word = self.this_word
        word_sequences = self.sequences
        if self.verbose:
            print(f"My sequences for {word}: {self.sequences}")
        
        hmm_models = {}
        best_score = None
        best_n = None
        if self.verbose:
            print(f"trying to build models with {self.min_n_components} -> {self.max_n_components} states")
        for n in range(self.min_n_components, self.max_n_components):
            # print(f"{n} components")
            try:
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                hmm_models[n] = model
                # evaluate the model
                logL = model.score(self.X, self.lengths)
                
                sum_logL = 0
                count_logs = 0
                #go through all the other words and calculate logL for each, add it to sum_logL
                for other_word in self.hwords.keys():
                    if other_word == word:
                        continue
                    
                    try:
                        other_X, other_lengths = self.hwords[other_word]
                        logL = model.score(other_X, other_lengths) / len(other_lengths)
                        sum_logL += logL
                        count_logs +=1
                    except:
                        # could not score this model for other_word
                        if self.verbose:
                            print(f"could not score {word} model with {n} states on {other_word}")
                
                dic = logL
                if count_logs:
                    dic -= sum_logL / count_logs
                if best_score is None or best_score < dic :
                    best_score = dic
                    best_n = n
                if self.verbose:
                    print("model created for {} with {} states. score={}".format(word, n, logL))
            except ValueError:
                hmm_models[n] = None
                if self.verbose:
                    print(f"failed to build model with {n} states for {word}")
               
        # acum cred ca trebuie sa le evaluez si sa aleg pe cel mai bun

        if self.verbose:
            print(f"best model for {word} has {best_n} states")
        if best_n is None:
            return None
        return hmm_models[best_n]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        # return True
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        word = self.this_word
        word_sequences = self.sequences
        if self.verbose:
            print(f"My sequences for {word}: {self.sequences}")
        # split_method = KFold()
        split_method = KFold(n_splits=min(3,len(self.lengths)))
        # nu stiu daca am nevoie de toate combinatiile (cv_train_idx, cv_test_idx)
        cv_train_idx, cv_test_idx = next(split_method.split(word_sequences))
        # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
        X, lengths = combine_sequences(cv_train_idx, word_sequences)
        Y, y_lengths = combine_sequences(cv_test_idx, word_sequences)
        # print("X: {}".format(X))
        # print("lengths: {}".format(lengths))
        
        hmm_models = {}
        best_score = None
        best_n = None
        if self.verbose:
            print(f"trying to build models with {self.min_n_components} -> {self.max_n_components} states")
        for n in range(self.min_n_components, self.max_n_components):
            # print(f"{n} components")
            try:
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
                hmm_models[n] = model
                # ToDo
                # evaluez modelul dar nu pe X si lengths ( care sunt Train indices), ci pe Test Indices
                logL = model.score(Y, y_lengths)
                if best_score is None or best_score < logL :
                    best_score = logL
                    best_n = n
                if self.verbose:
                    print("model created for {} with {} states. score={}".format(word, n, logL))
            except ValueError:
                hmm_models[n] = None
                if self.verbose:
                    print(f"failed to build model with {n} states for {word}")
               
        # acum cred ca trebuie sa le evaluez si sa aleg pe cel mai bun

        if self.verbose:
            print(f"best model for {word} has {best_n} states")
        return hmm_models[best_n]
