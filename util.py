import os
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

import seaborn as sns
from matplotlib import pyplot as plt


from patsy import dmatrices
import bambi as bmb
import arviz as az
from scipy.stats import chi2
import statsmodels.api as sm
import matplotlib.pyplot as plt
from IPython.display import display
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


# input: scraped text list
# output: caption, cleaned comments list (usernames and comments), likes
def clean_comments(comments):
    caption, likes = None, None

    # remove comemnts that are non necessary.
    pattern = r'This reel has [\d,]+\s+comments? from Facebook\.'
    for x in comments:
        match = re.search(pattern, x)
        if match:
            comments.remove(x)
            break
    pattern = r'[\d,]+\s+comments? from Facebook'
    for x in comments:
        match = re.search(pattern, x)
        if match:
            comments.remove(x)
            break
    
    # get number of likes. usually at last but sometimes it shifts.
    pattern = r'\b\d[\d,]*\s+likes?\b'
    for x in comments:
        match = re.search(pattern, x)
        if match:
            likes = x
            comments.remove(x)
            break

    # remove comments about cookies. This happens when there are less no of of comments.
    cookie_texts = ['We use cookies and similar technologies to help provide and improve content on Meta Products. We also use them to provide a safer experience by using information that we receive from cookies on and off Instagram, and to provide and improve Meta Products for people who have an account.',
                    'You have control over the optional cookies that we use. Learn more about cookies and how we use them, and review or change your choices at any time in our Cookies Policy.',
                    'We use cookies from other companies in order to show you ads off our Products, and provide features such as maps, payment services and video.',
                    'As part of laws in your region, you can choose whether you consent to us processing your personal data for personalised ads on Meta Products.',
                    'Sign up for Instagram to stay in the loop.',
                    'Start the conversation.',
                    'Essential cookies: These cookies are required to use Meta Products and are necessary for our sites to work as intended.',
                    'Our cookies on other apps and websites: Other companies use our cookies and similar technologies, such as the Like button or Meta pixel, on their apps and websites. These cookies can be used to personalise your ads. These cookies are optional.',
                    'Cookies from other companies: We use these cookies to show you ads off Meta Products and to provide features such as maps and videos on Meta Products. These cookies are optional.',
                    "We'll remember your cookie choices and apply them anywhere you're logged in to Instagram, and where you use your Instagram to log in to other Meta Products.",
                    'You can review or change your choices at any time in your cookies settings.',
                    '•',
                    'You have control over the optional cookies that we use. Learn more about cookies and how we use them in our Cookies Policy.',
                    'We use cookies on apps and websites owned by other companies that use Meta technologies. These cookies help other companies share information with us about your activity on their apps and websites.',
                    ]
    remove_list = []
    for x in comments:
        for ct in cookie_texts: 
            if x.strip() == ct.strip():
                remove_list.append(x)

    for x in remove_list:
        comments.remove(x)

    for x in comments:
        if x == '':
            comments.remove(x)

    if len(comments) == 0:
        caption, comments = None, []
    else:
    # the first 'comment' is usually the caption
        caption = comments[0]
        # these comments are a mix of usernames and the actual comment.
        comments = comments[1:]

    return caption, comments, likes


# inputs: crawl names to pick the columns, the merged dataframe, all urls dict
def compute_fprops(crawl1, crawl2, merged_df, all_urls):

    if crawl2 == 'chronological':
        raise ValueError('chronological should be the first crawl')

    col1 = f'{crawl1}_comment'
    col2 = f'{crawl2}_comment'

    # ranking pres considers the actual order of comments and the other doesnt
    account_fprops_rank = {}
    account_fprops_no_rank = {}

    for account in all_urls.keys():
        urlids = merged_df[merged_df['account'] == account]['urlid'].unique()

        temp_list_no_rank = []
        temp_list_rank = []
        for urlid in urlids:
            # dropping NaN comments which were here due to the addition of the chronological comments
            s1 = merged_df[(merged_df['account'] == account) & (merged_df['urlid'] == urlid)][col1].dropna().tolist()
            s2 = merged_df[(merged_df['account'] == account) & (merged_df['urlid'] == urlid)][col2].dropna().tolist()

            # need to implement diff structure for chronological comments since they will have comments from all crawls 
            try:
                if crawl1 == 'chronological':
                    s1_no_rank = s1[:len(s2)]
                    s1_rank = [x for x in s1 if x in s2]
                else:
                    s1_no_rank = s1
                    s1_rank = s1

                diff_s_no_rank = set(s1_no_rank).symmetric_difference(set(s2))
                diff_s_rank = [1 for x, y in zip(s1_rank, s2) if x != y]

                denom_no_rank = len(set(s1_no_rank).union(set(s2)))
                # denom_no_rank = max(len(s1_rank), len(s2))
                denom_rank = max(len(s1_rank), len(s2))

                fprop_no_rank = len(diff_s_no_rank) / denom_no_rank
                fprop_rank = sum(diff_s_rank) / denom_rank

                temp_list_no_rank.append(fprop_no_rank)
                temp_list_rank.append(fprop_rank)

                merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'fcount_no_rank'] = len(diff_s_no_rank)
                merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'ftrials_no_rank'] = denom_no_rank

                merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'fcount_rank'] = sum(diff_s_rank)
                merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'ftrials_rank'] = denom_rank

            except Exception as e:
                print(e, account, urlid)
                continue
        account_fprops_no_rank[account] = temp_list_no_rank
        account_fprops_rank[account] = temp_list_rank

    return account_fprops_no_rank, account_fprops_rank, merged_df



def get_vif(formula, data):
    y, X = dmatrices(formula, data=data, return_type="dataframe")

    vif_df = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

    vif_df.sort_values(by='VIF', ascending=True, inplace=True)
    return vif_df


# get chronological order of comments
def get_chronological_order_comments(merged_df, all_urls, all_crawls):
    temp_list = []
    for account in all_urls.keys():
        urlids = merged_df[merged_df['account'] == account]['urlid'].unique()

        for urlid in urlids:

            # get comments for a particular post
            temp = merged_df[(merged_df['account'] == account) & (merged_df['urlid'] == urlid)].copy()
            unique_comments = {}

            # get all unique comments across all crawls
            for col in all_crawls:
                for comment, timestamp in zip(temp[f'{col}_comment'], temp[f'{col}_timestamp']):
                    if pd.isna(comment):
                        continue
                    if comment not in unique_comments:
                        unique_comments[comment] = timestamp
                    else:
                        if timestamp != unique_comments[comment]:
                            print('mismatch timestamp for same comment')

            # convert timestamps to datetime objects
            for comment, timestamp in unique_comments.items():
                unique_comments[comment] = datetime.fromisoformat(timestamp)

            # sort comments by timestamp§
            sorted_comments = sorted(unique_comments.items(), key=lambda x: x[1], reverse=True)
            idx = 0
            for comment, timestamp in sorted_comments:
                temp_list.append({'account': account,
                                'urlid': urlid,
                                'comment_num': idx,
                                'chronological_comment': comment,
                                'chronological_timestamp': timestamp})
                idx += 1
                
    temp_df = pd.DataFrame(temp_list)
    return temp_df


def compute_label_comments(crawl1, crawl2, merged_df, all_urls):
    if crawl2 == 'chronological':
        raise ValueError('chronological should be the first crawl')

    col1 = f'{crawl1}_comment'
    col2 = f'{crawl2}_comment'

    for account in all_urls.keys():
        urlids = merged_df[merged_df['account'] == account]['urlid'].unique()

        temp_list_no_rank = []
        temp_list_rank = []
        for urlid in urlids:
            # dropping NaN comments which were here due to the addition of the chronological comments
            s1 = merged_df[(merged_df['account'] == account) & (merged_df['urlid'] == urlid)][col1].dropna().tolist()
            s2 = merged_df[(merged_df['account'] == account) & (merged_df['urlid'] == urlid)][col2].dropna().tolist()

            # need to implement diff structure for chronological comments since they will have comments from all crawls 
            try:
                if crawl1 == 'chronological':
                    s1_no_rank = s1[:len(s2)]
                    s1_rank = [x for x in s1 if x in s2]
                else:
                    s1_no_rank = s1.copy()
                    s1_rank = s1.copy()
                
                for x in s2:
                    if x in s1_no_rank:
                        label_no_rank = 1
                    else :
                        label_no_rank = 0
                    
                    cidx = merged_df.index[merged_df[col2] == x].values[0]
                    cnum = merged_df.loc[cidx, 'comment_num']
                    merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid) & (merged_df['comment_num'] == cnum), 'fcount_no_rank'] = label_no_rank
                
                for i, x in enumerate(s2):
                    if x == s1_rank[i]:
                        label_rank = 1 
                    else:
                        label_rank = 0
                        
                    cidx = merged_df.index[merged_df[col2] == x].values[0]
                    cnum = merged_df.loc[cidx, 'comment_num']
                    merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid) & (merged_df['comment_num'] == cnum), 'fcount_rank'] = label_rank

            except Exception as e:
                print(e, account, urlid)
                continue
    return merged_df