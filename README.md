# rec-sys-final-project

## What is this project?

- pass

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
 
