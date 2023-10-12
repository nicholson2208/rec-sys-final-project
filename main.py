import pandas as pd
import requests as re
import time
import os
import sqlite3
import json
import random


def init_db(path="data/"):
    """
    SOME FUNCTION TO INITIALIZING A DB

    This could be a CSV for now, maybe should be SQLlite or MongoDB
    
    """
    pass

def write_to_db(df, path="data/"):
    """
    SOME FUNCTION TO WRITE TO A DB

    This could be a CSV for now, maybe should be SQLlite or MongoDB
    
    """

    df.to_csv(path + "some.csv")


def clean_df(df):
    """
    a function to clean the inputs to the db

    """
    pass


def usercount_bot_scraper(path="data/"):
    """
    Pulling the data from a user count bot as a batch
    
    Might be worth making a stream? 


    If you need to look up the user is, do https://mastodon.social/api/v1/accounts/lookup?acct=mastodonusercount
    
    @mastodonusercount@mastodon.social is the actual bot
    """

    # TODO: CHECK WHETHER THE DB EXISTS, CALL A FUNCTION TO MAKE IT IF NOT!
    
    # TODO: MAKE THIS A DB? MAYBE SQLITE? 
    # load the csv
    # the data frame that will hold all of this 
    all_status_df = pd.read_csv(path + "some.csv", index_col=0)

    print(all_status_df.shape)

    last_id = all_status_df.id.max()
    
    # TODO: THIS SHOULD POINT TO THE TIMELINE
    # max limit is 40 posts at a time I think! 
    req_stem = "https://mastodon.social/api/v1/accounts/471607/statuses?limit=40&min_id="
    req_query = req_stem + str(last_id)
    r = re.get(req_query)
    status_list = r.json() if r.status_code == 200 else []


    # TODO: will need to change this when there gets to be lots of posts
    # doing this loop because I might need to paginate if I get too far behind
    while r.status_code == 200 and len(status_list) > 0:

        new_df = pd.DataFrame(status_list)

        print("Just pulled the max id: {0}, min id: {1}".format(new_df.id.max(), new_df.id.min()))

        # all the garbage to get data type to line up here
        new_df = clean_df(new_df)

        all_status_df = pd.concat([all_status_df, new_df])

        # the new min id is the old max id!
        # assuming that it doesn't include both the max and min!
        # I think I that new_df should be the max of the concacted, so I think you can do either
        r = req_query = req_stem + str(new_df.id.max())
        r = re.get(req_query)
        status_list = r.json()

        # don't get booted off the API lol
        time.sleep(2 * random.random())


    # TODO: Change this to write to the db
    write_to_db(all_status_df)
    

def main():
    print("Starting at {}".format(int(time.time())))
    
    try:
        # Something is wrong becasue we are in an infinite loop, figure this out
        # usercount_bot_scraper()
        pass
    except Exception as e:
        print("usercount_bot_scraper failed, ", e)

        
if __name__ == "__main__":
    main()