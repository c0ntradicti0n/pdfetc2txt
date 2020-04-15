import logging
import pprint
from collections import Counter
from typing import Dict, Tuple

import pandas

from TFU.trial_tools import range_parameter
from helpers.list_tools import nd_fractal


class Updater(object):
    # the following looks stupid, but it's hard to crate an empty df with dtypes

    params = {}
    def __init__(self,
                 params : Dict[str, Tuple[float, float, float]],
                 original_ml_kwargs : Dict,
                 patience = 3,
                 n=3,
                 accuracy = 0.01):
        self.original_ml_kwargs = original_ml_kwargs.copy()
        self.init_choices()
        self.n = n
        self.patience = patience
        self.accuracy = accuracy

        self.sized = Counter()
        for param_name, range_tuple in params.items():
            self.init_generator_and_params(param_name, range_tuple, n=self.n)

    def init_generator_and_params(self, param_name, range_tuple, n):
        self.params[param_name] = list(range_parameter(range_tuple, n))
        setattr(self, param_name, range_parameter(range_tuple, n))

    def init_choices(self):
        self.choices = pandas.DataFrame({'param': ["start"], 'value': [0.0], 'score': [0.0], 'state': [{}]})
        self.choices = self.choices[[False]]

    choices_i = 0
    def notate(self, option, score, ml_kwargs):
        self.choices.loc[self.choices.shape[0]] = [*option, score, ml_kwargs]
        self.choices_i += 1

    def update(self):
        if len(self.choices) == 0:
            return self.best_solution
        self.choices.reset_index(drop=True, inplace=True)

        good_option = self.choices['score'].idxmax()
        choice = self.choices.iloc[good_option]

        param = choice.param
        s_tuple = self.params[param]
        other_solutions_of_this_trial = self.choices.loc[self.choices.param.str.contains(param)]
        as_good_solutions = other_solutions_of_this_trial.loc[other_solutions_of_this_trial.score == choice.score]

        if len(as_good_solutions) > 1:
            if self.sized[param] > self.patience or  as_good_solutions.value.std() < self.accuracy:
                logging.info("No patience anymore, good enough")
                self.choices = self.choices.loc[self.choices.param != param]
                return self.update()

            resolution = len(s_tuple) + 1
            self.init_generator_and_params(param, self.params[param], n=resolution)
            s_tuple = self.params[param]
            self.params[param] = list(nd_fractal(choice.value, s_tuple, n=resolution, lense=2))
            self.sized[param] += 1

        else:
            self.best_solution = choice
            resolution = 3
            s_value = as_good_solutions.value.mean()
            self.init_generator_and_params(param, self.params[param], n=resolution)
            s_tuple = self.params[param]
            self.params[param] = list(nd_fractal(s_value, s_tuple, n=resolution, lense=1))
            self.sized[param] -= 1
        self.init_choices()

    def options(self):
        for param in self.params:
            yield from self.get_following(param)

    def get_following(self, param):
        for value in self.params[param]:
            new_option = self.original_ml_kwargs.copy()
            new_option.update({param:value})
            logging.warning(f"setting {param} to {value}")
            yield param, value, new_option

    def give_feedback(self):
        pprint.pprint(self.choices)
        pprint.pprint(self.sized)

