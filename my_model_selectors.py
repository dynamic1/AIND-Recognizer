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
        best_score = math.inf
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
                # parameters_count = model.startprob_.size + model.transmat_.size \
                                    # + model.means_.size + model.covars_.diagonal().size
                logN = np.log(len(self.lengths))
                bic = -2 * logL + parameters_count * logN
                if best_score is None or best_score > bic :
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
                
                sum_logL = 0
                count_logs = 0
                #go through all the other words and calculate logL for each, add it to sum_logL
                # other_words_penalty = np.mean( [model.score(self.hwords[other_word]) \
                                               # for other_word in self.words if other_word != word ])
                # dic = logL - other_words_penalty
                
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
                # raise
               
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

        """
        Score on all folds for every n, and take the average. 
            The idea of CV is to split the data into K-folds, 
        each fold containing a training split and a testing split.
            For each value of the number of components, 
        you iterate and find the average log likelihood over the folds. 
            This is done by fitting on the training split and scoring 
        on the testing split for each fold, and then averaging over all folds.
            Then when you have an average likelihood for each value of 
        the number of components, you return the best one.
        """
        word = self.this_word
        word_sequences = self.sequences
        if self.verbose:
            # print(f"My sequences for {word}: {self.sequences}")
            pass
        if len(self.lengths)<2:
            if self.verbose:
                print(f"not enough sequences ( {self.sequences} ) to train / crossvalidate {word}") 
            return None
        
        hmm_models = {}
        
        best_score_overall = None
        best_n = None
        fold = 0
        
        n_fold_model_score = []
        for n in range(self.min_n_components, self.max_n_components):
            
            models_for_n = 0
            sum_logL_for_n = 0  
            best_score_for_n = None
            best_model_for_n = None
            
            # generate folds for training / testing
            fold = 0
            split_method = KFold(n_splits=min(3,len(self.lengths)))
            for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                
                fold += 1
                if self.verbose:
                    # print(f"trying to build model with {self.min_n_components} -> {self.max_n_components} states: fold={fold}")
                    pass

                # build training and testing data
                X, lengths = combine_sequences(cv_train_idx, word_sequences)
                Y, y_lengths = combine_sequences(cv_test_idx, word_sequences)
            
                # try to train a model on the training split
                try:
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(X, lengths)
                    models_for_n += 1
                    # evaluate the model on the testing split
                    logL = model.score(Y, y_lengths)
                    sum_logL_for_n += logL
                    if best_score_for_n is None or best_score_for_n < logL :
                        best_score_for_n = logL
                        best_model_for_n = model
                        
                    if self.verbose:
                        print(f"model created for {word} with {n} states. fold={fold}, score={logL}")
                except ValueError:
                     if self.verbose:
                        print(f"failed to build model for {word} with {n} states in fold {fold}")
                                        
            if not models_for_n:
                if self.verbose:
                    print(f"no models generated for {word} with {n} states")
                continue
                    
            hmm_models[n] = best_model_for_n           
            # now average the scores for all the folds of this n
            avg_logL_for_n = sum_logL_for_n / models_for_n
            if self.verbose:
                print(f"best model for {word} whith {n} states scores {best_score_for_n}")
                print(f"average score for {word} with {n} states is {avg_logL_for_n}")
            if best_score_overall is None or best_score_overall < avg_logL_for_n :
                best_score_overall = avg_logL_for_n
                best_n = n   
                
        if self.verbose:
            print(f"best model for {word} has {best_n} states")
        if best_n is None:
            return None
        return hmm_models[best_n]
