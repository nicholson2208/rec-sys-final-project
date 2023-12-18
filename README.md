# INFO 5612 Recommender Systems Final Project

## CORGI-Lite: Implementing People Recommendations on hci.social

CORGI (shorthand for “Community-Oriented Recommendation and Governance Infrastructure”) is an ongoing project within That Recommender Systems Lab (TRSL) at the University of Colorado Boulder that hopes to tackle the tensions that we assume users of federated social media platforms maintain — more specifically, on Mastodon. CORGI attempts to complete three tasks: implement post and people recommendation within Mastodon instances, provide governance infrastructure to enable instance members to participate in the design and deployment of a recommendation system, and provide an algorithmic recourse dashboard for content creators to enable them to make sense of how their content is “affected” by the community’s recommender system. This is a lofty goal. For the sake of this class, we limit our scope to recommendations for new users to follow.

[To read the full report](https://github.com/nicholson2208/rec-sys-final-project/blob/main/INFO%205612%20Final%20Project%20Report%20-%20CORGI.pdf)

## Installing this project
- clone it
- `conda env create --file=env.yml`
- `conda activate rec-sys-final-project`
- to update the environment `` conda env update -f env.yml ``

## What are these files

### Data Prep

- `more_user_profiles.ipynb`
    - Scrapes the local timeline and collects information on users. This seeds the crawling of the social graph
- `FollowersGraphs.ipynb`
    - Constructs a user : list of followers and user : list of following dictionaries
- `BuildDataset.ipynb`
    - Makes a `networkx` graph from the follower and following dictionaries, and then makes edge lists train-test splits for each fold of the data. Saves in `train_test/`
 
### Stored Data

- `adj_matrices`
    - a bunch of pickle `networkx` objects representing the full graph
- `train_test`
    - a collection of pickled edge lists representing folds in the data
- `graph_images`
    - some stored figures for our presentation and final report
- `data_dictionaries`
    - pickled dictionaries of  user : list of followers and user : list of following
- `data_csv`
    - data used to seed the construction of the above dictionaries
 
### Analysis

- `PageRank.ipynb`
    - imports `PageRank_modules` and runs PageRank on our dataset
- `light_gcn.ipynb.ipynb`
    - imports `LightGCN_modules` and runs GCN on our dataset
- `factorization_machine.ipynb`
    - imports `factorization_machine_modules` and runs a Factorization Machine on our dataset
- `ndcg_precision_graph.ipynb`
    - aggregates the results from the above and creates a graph

