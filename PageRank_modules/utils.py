import networkx as nx
import pickle


def read_g_obj(file="adj_matrices/G_hci.pkl", min_following=15):
    """
    Reads the stored graph object and computes the min_following_core
    
    Params:
        file: (str)
            the path to the stored pkl file for the full graph objs
        min_following: (int)
            the number of following that all nodes must have in the
            final subgraph (i.e. the k_out_core) 
    
    Returns:
        subgraph_hci: (nx.DiGraph)
            a subgraph
    
    """
    
    with open(file, "rb") as pfile: 
        G = pickle.load(pfile)
    
    assert type(G) == nx.DiGraph
    
    # need to do k_core thing here
    
    # the networkx implementation doesn't work because only does 
    # total degree, not in or out degree
    
    min_user, min_n_followers = min(G.out_degree(), key=lambda x: x[1]) 
    
    print("Min out degree -- person {} : \t followers: {}".format(min_user, min_n_followers))
    
    while min_n_followers < min_following:
    
        follows_at_least = [person for person, out_degree in G.out_degree() if out_degree >= min_following] 

        subgraph_hci = nx.subgraph(G, follows_at_least)
        
        G = subgraph_hci.copy()
        
        min_user, min_n_followers = min(G.out_degree(), key=lambda x: x[1])
        print("Min out degree -- person {} : \t followers: {}".format(min_user, min_n_followers))

    subgraph_hci = nx.subgraph(G, follows_at_least)
    
    return subgraph_hci


def check_predictions(predictions_for_group_members, test):
    """
    a function for checking (1) how many it got right -- this is precision -- and (2) the people you should follow
    
    Params:
        predictions_for_group_members (this is a dict):
            a dictionary with our names as keys, and rec list as the values
            for me, a rec list is (person, score) tuples
        test (list):
            an edge list of our test set
    
    Returns:
        output_dict (dict):
            our names as keys, {"predicted" : [list of preds], "correct" : [list of correctly predicted users], 
            "you should follow" : [list of ppl]}
    """
    
    output_dict = {}
    
    for person, recs in predictions_for_group_members.items():
        predicted = []
        correct_prediction = []
        people_I_should_follow = []
        
        
        for rec in recs:
            # my list of recs is a (person, score) tuples
            predicted.append(rec[0])
            
            if (person, rec[0]) in test:
                correct_prediction.append(rec[0])
            else:
                people_I_should_follow.append(rec[0])
            
        
        output_dict[person] = dict(predicted=predicted, correct=correct_prediction, should_follow=people_I_should_follow)
    
    return output_dict
