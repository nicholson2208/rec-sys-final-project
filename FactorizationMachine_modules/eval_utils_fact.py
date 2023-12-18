import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import pickle
import random

from abc import ABC, abstractmethod
from collections import defaultdict
from surprise import SVD, Reader, Dataset, Prediction
from sklearn.metrics import mean_squared_error
from itertools import permutations
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm

class UserRecommendations:
    def __init__(self):
        self.recs = []
        
    def add_entry(self, entry):
        self.recs.append(entry)
        
    def select_top(self, k):
        self.recs = sorted(self.recs, key=lambda entry: entry[2], reverse=True)
        if len(self.recs) > k:
            self.recs = self.recs[0:k]

class TestRecommendations:
    def __init__(self):
        self.test_recs = defaultdict(UserRecommendations)
        
    def setup(self, preds, k):
        for entry in preds:
            user = entry.uid
            self.test_recs[user].add_entry(entry)
                   
        for user in self.test_recs.keys():
            self.test_recs[user].select_top(k)
            
    def add_entry(self, user, entry):
        self.test_recs[user].add_entry(entry)
        
    def select_top(self, user, k):
        self.test_recs[user].select_top(k)
            
    def iter_recs(self):
        for user in self.test_recs.keys():
            yield (user, self.test_recs[user].recs)

class Evaluator(ABC):
    def __init__(self):
        self.results_table = None
        self.score = None
        
    def setup(self, trainset, testset):
        pass
    
    @abstractmethod
    def evaluate_user(self, user, user_recs):
        pass
    
    def evaluate(self, test_recs: TestRecommendations):
        scores = []
        self.results_table = {}
        for user, recs in test_recs.iter_recs():
            score = self.evaluate_user(user, recs)
            scores.append(score)
            self.results_table[user] = score
        self.score = np.mean(scores)
        
class ItemwiseEvaluator(Evaluator):
    
    def __init__(self):
        super().__init__()
    
    def evaluate_user(self, user, user_recs):
        return np.mean([self.evaluate_pred(rec) for rec in user_recs])
        
    @abstractmethod
    def evaluate_pred(self, pred):
        pass
    
class ListwiseEvaluator(Evaluator):
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def evaluate_user(self, user, user_recs):
        pass
    
class NDCGEvaluator(ListwiseEvaluator):
    
    def __init__(self, k):
        super().__init__()
        self.rated_table = defaultdict(set)
        self.idcg_table = {}
        self.log_table = {}
        self.list_len = k
    
    def setup(self, trainset, testset):
        for entry in testset:
            self.rated_table[entry['user']].add(entry['item'])
        idcg = 0
        for i in range(0, self.list_len+1):
            self.idcg_table[i] = idcg
            rank_utility = 1 / np.log(i+2)
            self.log_table[i] = rank_utility
            idcg += rank_utility
            
    
    def evaluate_user(self, user, user_recs): 
        
        dcg = 0.0
        for i, pred in enumerate(user_recs):
            if pred[1] in self.rated_table[user]:
                dcg = self.log_table[i]
        
        idcg = 0
        if len(self.rated_table[user]) >= self.list_len:
               idcg = self.idcg_table[self.list_len]
        else:
               idcg = self.idcg_table[len(self.rated_table[user])]
            
        if idcg == 0:
            return 0
        return dcg/idcg

class PrecisionEvaluator(ItemwiseEvaluator):
    
    def __init__(self):
        super().__init__() # calling parent class
        self.rated_table = defaultdict(set) # table to store the rated items
    
    
    def setup(self,trainset, testset):
        for edge in testset:
                self.rated_table[edge['user']].add(edge['item']) # add the item to the table of the things the user likes
        # now we should have a table from the test set with all the rated items from 
                
    def evaluate_pred(self, pred: Prediction):
        # now we calculate the percision per list per user then we get the average for all the lists
        # number of items that are relevant (in self.rated table) / number of recommended items     
        if pred[1] in self.rated_table[pred[0]]: #check if the item exists in the list
            return 1
        else:
            return 0

random.seed(20231110)

def create_prediction_profiles(test_data, train_items, predict_list_len, frac=1.0):
    train_items_lst = list(train_items) # Can't sample from a set
    user_test_profile = defaultdict(set) # add the test data
    for entry in test_data:
        user_id = entry['user']
        item_id = entry['item']
        user_test_profile[user_id].add(item_id)
    
    test_users = list(user_test_profile.keys()) # sample from the test users
    
    test_users_select = random.sample(test_users, int(frac*len(test_users))) # create a big set and add the test users
                      
    user_predict_profile = {}
    
    for user in test_users_select:
        profile = user_test_profile[user]
        sample_items = list(random.sample(train_items_lst, predict_list_len + len(profile)))
        sample_items = sample_items + list(profile)
        user_predict_profile[user] = sample_items
        
    return user_predict_profile
    
# creating a list of recommendations for a user 
def create_test_recommendations(predict_fn, vectorizer, test_data, list_len, train_items, predict_list_len, frac=1.0):
    user_predict_profile = create_prediction_profiles(test_data, train_items, predict_list_len, frac)
    
    trecs = TestRecommendations()
    
    # for all the usrs and items in the profile
    for user, profile in user_predict_profile.items():
        
        for item in profile:
            x_test = vectorizer.transform({'user': user, 'item': item})
            pred = predict_fn(x_test)[0]
            trecs.add_entry(user, (user, item, pred))
        trecs.select_top(user, list_len)
        
    return trecs # return recommendations list for each user

def loadData(file, sample=1.0):
    data = []
    y = []
    users=set()
    items=set()
    for line in file.values:
        (user,item,rating)= line[0],line[1],line[2]
        if random.random() <= sample:
            data.append({ "user": str(user), "item": str(item)})
            y.append(float(rating))
            users.add(user)
            items.add(item)

    return (data, np.array(y), users, items)

def create_train_test_df(train, test, anti_test):  
    train_data_extended = [(follower, following, 1) for follower, following in train]
    test_data_extended = [(follower, following, 1) for follower, following in test]
    anti_test_data_extended = [(follower, following, 0) for follower, following in anti_test]
    
    train_df = pd.DataFrame(
        train_data_extended, columns=["user", "item", "rating"]
    )
    # train_df.to_csv('train_df.csv')
    
    test_data_extended.extend(anti_test_data_extended)
    test_df = pd.DataFrame(test_data_extended, columns=["user", "item", "rating"])
    # test_df.to_csv('test_df.csv')
    
    return train_df, test_df

def extend_data(train, test, train_users):
        
    X_train_data_extended = [{'user': pair[0], 'item': pair[1]} for pair in permutations(list(train_users), 2)]
    
    y_train_extended_list = []

    # This was previously `comb[1]` instead of `comb`. 
    for comb in X_train_data_extended:
        if comb in train:
            y_train_extended_list.append(1)
        else:
            y_train_extended_list.append(0)
            
    y_train_data_extended = np.array(y_train_extended_list, dtype='double')
    
    return X_train_data_extended, y_train_data_extended