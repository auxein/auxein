# -*- coding: utf-8 -*-
"""Static playground.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Playground:

    def __init__(self, population):
        self.population = population
    

    def train(self, x_train, y_train, max_generations, validations):
        pass

    
    def predict(self, x):
        return x


class Static(Playground):

    def __init__(self, population, mutation, distribution, selection, recombination, replacement):
        super(Static, self).__init__(population=population)
        self.mutation = mutation
        self.distribution = distribution
        self.selection = selection
        self.recombination = recombination
        self.replacement = replacement
    
    
    def train(self, x_train, y_train, max_generations, validation):
        pass
    
    
    def predict(self, x):
        return 12.4