import os
import re
import ast
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

    pattern = r'[\d,]+\s+comments? from Facebook'
    for x in comments:
        match = re.search(pattern, x)
        if match:
            comments.remove(x)
    
    # get number of likes. usually at last but sometimes it shifts.
    pattern = r'\b\d[\d,]*\s+likes?\b'
    for x in comments:
        match = re.search(pattern, x)
        if match:
            likes = x
            comments.remove(x)

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
                    'See photos, videos and more from NBC Entertainment.',
                    'See photos, videos and more from Peacock.',
                    'See photos, videos and more from Bel-Air on Peacock.',
                    'See photos, videos and more from Spotify.',
                    'See photos, videos and more from NYT Cooking.',
                    'See photos, videos and more from ESPN.',
                    'See photos, videos and more from Cat Lovers Club.',
                    'See photos, videos and more from The Dogist.',
                    'See photos, videos and more from Kayo Sports.',
                    'See photos, videos and more from Access Hollywood.',
                    'See photos, videos and more from LADbible.',
                    'See photos, videos and more from Percy Jackson.',
                    'See photos, videos and more from Spectrum.',
                    'See photos, videos and more from Kumail Nanjiani.',
                    'See photos, videos and more from Hulu.',
                    'See photos, videos and more from ESPN College Football.',
                    'See photos, videos and more from Empire State Building.',
                    'See photos, videos and more from The Grade Cricketer.',
                    'See photos, videos and more from Triple M Cricket.',
                    'See photos, videos and more from PBS Food.',
                    'See photos, videos and more from Freeform.',
                    'See photos, videos and more from SPORTbible.',
                    'See photos, videos and more from PBS SoCal.',
                    'See photos, videos and more from Cat Lovers Club.',
                    ]
    
    remove_list = []
    for x in comments:
        for ct in cookie_texts: 
            if x.strip() == ct.strip():
                remove_list.append(x)
    remove_list = list(set(remove_list))

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
                    s1_rank = [x for x in s1 if x in s2]
                    s2_rank = [x for x in s2 if x in s1]

                if len(s1) == 0 or len(s2) == 0:
                    merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'fcount_no_rank'] = -1
                    merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'ftrials_no_rank'] = -1

                    merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'fcount_rank'] = -1
                    merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'ftrials_rank'] = -1 
                else:
                    # diff_s_no_rank = set(s1_no_rank).intersection(set(s2))
                    # denom_no_rank = len(set(s1_no_rank).union(set(s2)))
                    # denom_no_rank = min(len(s1_rank), len(s2))

                    diff_s_no_rank = set(s1_no_rank).symmetric_difference(set(s2))
                    denom_no_rank = len(s1_no_rank) + len(s2) 

                    diff_s_rank = [1 for x, y in zip(s1_rank, s2_rank) if x != y]
                    denom_rank = max(len(s1_rank), len(s2_rank))

                    fprop_no_rank = len(diff_s_no_rank) / denom_no_rank
                    fprop_rank = sum(diff_s_rank) / denom_rank

                    temp_list_no_rank.append(fprop_no_rank)
                    temp_list_rank.append(fprop_rank)

                    merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'fcount_no_rank'] = len(diff_s_no_rank)
                    merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'ftrials_no_rank'] = denom_no_rank

                    merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'fcount_rank'] = sum(diff_s_rank)
                    merged_df.loc[(merged_df['account'] == account) & (merged_df['urlid'] == urlid), 'ftrials_rank'] = denom_rank

            except Exception as e:
                print(e, account, urlid, denom_no_rank, denom_rank)
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
            common_comments = {}

            # get all unique comments across all crawls
            for col in all_crawls:
                for comment, timestamp in zip(temp[f'{col}_comment'], temp[f'{col}_timestamp']):
                    if pd.isna(comment):
                        continue
                    if comment not in common_comments:
                        common_comments[comment] = timestamp
                    else:
                        if timestamp != common_comments[comment]:
                            print('mismatch timestamp for same comment')

            # convert timestamps to datetime objects
            for comment, timestamp in common_comments.items():
                common_comments[comment] = datetime.fromisoformat(timestamp)

            # sort comments by timestamp§
            sorted_comments = sorted(common_comments.items(), key=lambda x: x[1], reverse=True)
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


def compute_label_reg(crawl1, crawl2):
    gender_labels = ['female_female', 'female_male', 'male_male', 'none_female', 'none_male', 'none_none']
    ideology_labels = ['dem_dem', 'dem_rep', 'rep_rep', 'none_dem', 'none_rep', 'none_none']
    location_labels = ['ny_ny', 'ny_texas', 'texas_texas', 'none_ny', 'none_texas', 'none_none']
    gender, ideology, location = '', '', ''

    try:
        gender1 = re.search(r"(female|male)", crawl1, re.I).group(1).lower() 
    except:
        gender1 = 'none'
    try:
        gender2 = re.search(r"(female|male)", crawl2, re.I).group(1).lower()  
    except:
        gender2 = 'none'

    gender_str = gender1 + '_' + gender2
    gender_str_rev = gender2 + '_' + gender1

    if gender_str in gender_labels:
        gender = gender_str
    elif gender_str_rev in gender_labels:
        gender = gender_str_rev

    try:
        ideology1 = re.search(r"(dem|rep)", crawl1, re.I).group(1).lower() 
    except:
        ideology1 = 'none'
    try:
        ideology2 = re.search(r"(dem|rep)", crawl2, re.I).group(1).lower()  
    except:
        ideology2 = 'none'

    ideology_str = ideology1 + '_' + ideology2
    ideology_str_rev = ideology2 + '_' + ideology1

    if ideology_str in ideology_labels:
        ideology = ideology_str
    elif ideology_str_rev in ideology_labels:
        ideology = ideology_str_rev

    try:
        location1 = re.search(r"(ny|texas)", crawl1, re.I).group(1).lower() 
    except:
        location1 = 'none'
    try:
        location2 = re.search(r"(ny|texas)", crawl2, re.I).group(1).lower()  
    except:
        location2 = 'none'

    location_str = location1 + '_' + location2
    location_str_rev = location2 + '_' + location1

    if location_str in location_labels:
        location = location_str
    elif location_str_rev in location_labels:
        location = location_str_rev

    return gender, ideology, location


def get_df_for_regression(df, all_urls, all_crawls):

    reg_df = pd.DataFrame(columns=['account', 'urlid', 'gender', 'ideology', 'fcount_no_rank', 'ftrials_no_rank', 'fcount_rank', 'ftrials_rank'])

    temp_list = []
    for crawl1 in all_crawls:
        for crawl2 in all_crawls:

            if crawl1 == crawl2:
                continue

            temp_dict = {}

            _, _, mod_df = compute_fprops(crawl1, crawl2, df, all_urls)
            temp_df = mod_df[['account', 'urlid', 'fcount_no_rank', 'ftrials_no_rank', 'fcount_rank', 'ftrials_rank']].drop_duplicates()

            gender, ideology, location = compute_label_reg(crawl1, crawl2)

            temp_dict['location'] = location
            temp_dict['gender'] = gender
            temp_dict['ideology'] = ideology
            
            # var1 = min(crawl1, crawl2)
            # var2 = max(crawl1, crawl2)
            # temp_dict['setup_a'] = var1
            # temp_dict['setup_b'] = var2
            # temp_dict['pair_id'] = f'{var1}_{var2}'

            for index, row in temp_df.iterrows():
                temp_dict['urlid'] = row['urlid']
                temp_dict['account'] = row['account']
                temp_dict['fcount_no_rank'] = row['fcount_no_rank']
                temp_dict['ftrials_no_rank'] = row['ftrials_no_rank']
                temp_dict['fcount_rank'] = row['fcount_rank'] 
                temp_dict['ftrials_rank'] = row['ftrials_rank']
                temp_list.append(temp_dict.copy())


    reg_df = pd.DataFrame(temp_list)       
    return reg_df


def numify_metrics(val):
    if pd.notna(val):
        val = val.lower()
        if 'k' in val:
            val = val.replace('k', '')
            num = float(val)*1000
        else:
            num = float(val)
        return num
    else:
        return val


def compute_presence_comments(crawl1, crawl2, df, labels_df):
    col1 = f'{crawl1}_comment'
    col2 = f'{crawl2}_comment' 

    mod_df = pd.DataFrame(columns=['account', 'urlid', 'comment', 'presence_no_rank', 'presence_rank', 'label'])

    temp_list = []
    for account in df['account'].unique():
        urlids = df[df['account'] == account]['urlid'].unique()

        for urlid in urlids:
            # TODO: if using chronological, need to think here
            temp = df[(df['account'] == account) & (df['urlid'] == urlid)].copy()
            s1 = temp[col1].dropna().tolist()
            s2 = temp[col2].dropna().tolist()

            if len(s1) == 0 or len(s2) == 0:
                continue
            else:
                common_comments = set(s1).intersection(set(s2))

                # TODO: need to include code for rankign if needed
                for comment in set(s1).union(set(s2)):
                    temp_dict = {}
                    if pd.isna(comment):
                        continue
                    else:
                        if comment in common_comments:
                            presence_no_rank = 'Yes'
                        else:
                            presence_no_rank = 'No'

                        label = labels_df[labels_df['comment_combined'] == ast.literal_eval(comment)]['label'].values[0]
                    
                    temp_dict['account'] = account
                    temp_dict['urlid'] = urlid
                    temp_dict['comment'] = comment
                    temp_dict['presence_no_rank'] = presence_no_rank
                    temp_dict['presence_rank'] = None
                    temp_dict['label'] = label 
                    temp_list.append(temp_dict.copy())

    mod_df = pd.DataFrame(temp_list)
    return mod_df


def get_df_for_regression_comments(df, all_crawls, labels_df):

    reg_df = pd.DataFrame(columns=['account', 'urlid', 'comment', 'gender', 'ideology', 'location', 'presence_no_rank', 'presence_rank', 'label'])

    temp_list = []
    for crawl1 in all_crawls:
        for crawl2 in all_crawls:

            if crawl1 == crawl2:
                continue

            temp_dict = {}

            mod_df = compute_presence_comments(crawl1, crawl2, df, labels_df)
            # temp_df = mod_df[['account', 'urlid', 'presence_no_rank', 'presence_rank']].drop_duplicates()
            temp_df = mod_df.copy()

            gender, ideology, location = compute_label_reg(crawl1, crawl2)

            temp_dict['location'] = location
            temp_dict['gender'] = gender
            temp_dict['ideology'] = ideology

            var1 = min(crawl1, crawl2)
            var2 = max(crawl1, crawl2)
            temp_dict['setup_a'] = var1
            temp_dict['setup_b'] = var2
            temp_dict['pair_id'] = f'{var1}_{var2}'

            for index, row in temp_df.iterrows():
                temp_dict['urlid'] = row['urlid']
                temp_dict['account'] = row['account']
                temp_dict['comment'] = row['comment']
                temp_dict['presence_no_rank'] = row['presence_no_rank']
                temp_dict['presence_rank'] = row['presence_rank']
                temp_dict['label'] = row['label']
                temp_list.append(temp_dict.copy())


    reg_df = pd.DataFrame(temp_list)       
    return reg_df


def edit_labels(name):
    rename_map = {
        "Intercept": "Intercept",
        "location_diff": "Location Diff [Yes]",
        "gender_diff": "Gender Diff [Yes]",
        "ideology_diff": "Leaning Diff [Yes]",
        "followers_logz": "Followers (log z)",
        "comments_count_logz": "Comments (log z)",
        "type": "Account Type",
        "main_topic": "Post Type [Non-Political]",
        "location": "Location",
        "gender": "Gender",
        "ideology": "Leaning",
        "female_female": "Female_Female*",
        "male_male": "Male_Male'",
        "female_male": "Female_Male*",
        "dem_rep": "Dem_Rep*",
        "dem_dem": "Dem_Dem*",
        "rep_rep": "Rep_Rep'",
        "ny_ny": "NY_NY*",
        "ny_texas": "NY_TX*",
        "texas_texas": "TX_TX'" ,
        "Non-News": "Non-News'",
        "News": "News*",
        "Supportive": "Supportive*",
        "Agaisnst": "Agaisnt",
        "Neutral": "Neutral'"
    }
    base = re.split(r"(\[|\()", name, maxsplit=1)[0]
    rest = name[len(base):]
    return rename_map.get(base, base) + rest