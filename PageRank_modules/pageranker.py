# contains class defs for the page ranker
# and recommendations

from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from collections import defaultdict

class PageRanker:
    def __init__(self, G, alpha=0.85):
        """
        G (nx.DiGraph):
            the graph object for this network
        alpha (float):
            the walk continues probability
        
        """
        
        self.G = G
        self.alpha = alpha
        self.output = {}
    
    def get_overall_pagerank(self):
        
        # turn into a list so my other classes work
        return [(k, v) for k,v in nx.pagerank(self.G, alpha=self.alpha).items()]
    
    def get_personalized_pagerank(self, node):
        """
        node (str):
            the name of the node
        """
        
        # you always go back to yourself
        return [(k, v) for k,v in nx.pagerank(self.G, alpha=self.alpha, personalization={node : 1}).items()] 
    
    def test(self, personalized=True):
        
        if not personalized:
            self.output = {"overall" : self.get_overall_pagerank()}
        else:
            # compute personalized pagerank for every node
            
            for n in self.G.nodes:
                self.output[n] = self.get_personalized_pagerank(n)
                
        return self.output
        

class UserRecommendations:
    def __init__(self):
        self.recs = []
        
    def add_entry(self, entry):
        self.recs.append(entry)
        
    def select_top_k(self, k):
        self.recs = sorted(self.recs, key=lambda x: x[1], reverse=True)
        if len(self.recs) > k:
            self.recs = self.recs[0:k]

            
class TestRecommendations:
    def __init__(self, G):
        self.test_recs = defaultdict(UserRecommendations)
        self.G = G
        
    def setup(self, preds, k):
        """
        pred (dict)
            of {user: [(other_user1, pr_score1), (other_user2, pr_score2)], ...}
        
        """
        
        
        for user, rec_list in preds.items():
            for entry in rec_list:
                try:
                    if entry[0] == user:
                        # don't add yourself to the recs
                        continue

                    if (user, entry[0]) in self.G.out_edges(user):
                        # don't add nodes that already exist
                        continue
                    self.test_recs[user].add_entry(entry)
                except Exception as e:
                    print(e, "-----")
                    print((user, entry[0]))
        """        
        for entry in preds:
            user = entry[0]
            self.test_recs[user].add_entry(entry)
        """
            
        for user in self.test_recs.keys():
            self.test_recs[user].select_top_k(k)
            
    def iter_recs(self):
        for user in self.test_recs.keys():
            yield (user, self.test_recs[user].recs)
                        

class Evaluator(ABC):
    def __init__(self):
        self.results_table = None
        self.score = None
        
    def setup(self, trainset, testset):
        pass
    
    def evaluate(self, test_recs: TestRecommendations):
        # for every user and list of recommendations we produce, 
        # evaluate each user (which varies depending on list or itemwise)
        # and then average that over all users!
        
        # make a "best you can do list"
        
        self.results_table = {}
        scores = []
        for user, recs in test_recs.iter_recs():
            # score, weight = self.evaluate_user(user, recs)
            
            # if weighted
                # append score * weight to scores
                # append weight to "best you can do list"
            
            score = self.evaluate_user(user, recs)
            scores.append(score)
            self.results_table[user] = score
        self.score = np.mean(scores)
        
    @abstractmethod
    def evaluate_user(self, user, user_recs):
        pass

    
class ItemwiseEvaluator(Evaluator):
    def evaluate_user(self, user, user_recs):
        return np.mean([self.evaluate_pred(user, rec) for rec in user_recs[1]])
    
    @abstractmethod
    def evaluate_pred(self, pred):
        pass
    

class ListwiseEvaluator(Evaluator):
    
    @abstractmethod
    def evaluate_user(self, user, user_recs):
        pass

    
class NDCGEvaluator(ListwiseEvaluator):
    
    def __init__(self, k, kappa=20):
        super().__init__()
        self.k = k
        self.kappa = kappa
        self.rated_table = defaultdict(set)
        self.idcg = {}
    
    def setup(self, trainset, testset):
        for user, out_link in testset:
            self.rated_table[user].add(out_link)
                
        idcg = 0
        self.idcg[0] = 0
        for i in range(0, self.k):
            idcg += 1/np.log2(i+2)
            self.idcg[i+1] = idcg
                 
    def evaluate_user(self, user, user_recs, weighted=False):
        """
        user 
            node name
            
        user_recs
            should be a list of (other node, pagerank) tuples
            
        weighted (bool):
            if true, do inverse proponsity weighting w kappa
        
        """
        dcg = 0.0
        for i, rec in enumerate(user_recs):
            # print("user:", user, "i", i, "rec:" , rec)
            
            # the first thing in the tuple is the other edge
            if rec[0] in self.rated_table[user]:
                
                #print(f'Rated: {i} Item {pred.iid}')
                dcg += 1/np.log2(i+2)
                
        count_rated_things = len(self.rated_table[user])
        if count_rated_things > self.k:
            count_rated_things = self.k
        
        idcg = self.idcg[count_rated_things]
        
        if idcg == 0:
            return 0
        #print(dcg)
        #print(idcg)
        
        return dcg/idcg

class PrecisionEvaluator(ItemwiseEvaluator):
    """

    """
    
    def __init__(self):
        # I feel like this should be k?
        # on second thought, no because I think we are looking at one item at a time?
        
        super().__init__()
        self.rated_table = defaultdict(set)
        
    
    def setup(self, trainset, testset):
    
        for user, out_link in testset:
            self.rated_table[user].add(out_link)
    
    def evaluate_pred(self, user, pred):
        
        # print("user: ", user, "pred: ", pred)
        
        # the node that you are linked to is the thing that is actually store
        if pred in self.rated_table[user]:
            return 1
        else:
            return 0

