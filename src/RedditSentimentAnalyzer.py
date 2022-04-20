import string
from typing import Any, Tuple, Dict, List
import numpy as np
import pandas as pd
import praw
import time
from datetime import datetime
from time import sleep

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import emoji  # removes emojis
import re  # removes links
import data.sentiment_data as sentiment_data
from spacy.lang.en import English
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import squarify
import investpy
import os
import mplfinance as mpf
import yfinance as yf


# ----------------------------------------Reddit management------------------------------------------------------


def connect_reddit(redcreds: dict) -> praw.Reddit:
    """
    Connects to Reddit using a dictionary with all the credentials needed to connect
    @param redcreds: dict : dictionary with credentials: username, password, client_id, client_secret and, user_agent.
    These credential are obtained in Reddit
    @return: praw.Reddit
    """
    reddit = praw.Reddit(username=redcreds["username"],
                         password=redcreds["password"],
                         client_id=redcreds["client_id"],
                         client_secret=redcreds["client_secret"],
                         user_agent=redcreds["user_agent"], check_for_async=False)
    return reddit


def data_extractor(reddit: praw.Reddit,
                   subs: list[str],
                   list_of_symbols: list[str],
                   blacklist_words: list[str],
                   post_flairs: set = None,
                   limit: int = 1
                   ) -> tuple[int, int, dict[Any, int], list[Any], dict[Any, list[Any]], int, list[str], int]:
    """
    @param reddit: Reddit                - Reddit object
    @param subs: list[str]               - names of subs to analyse in a list
    @param list_of_symbols: list[str]    - The list of symbols to analyse in the subs included in the list [subs]
    @param blacklist_words: list[str]    -
    @param post_flairs: set              - post flairs
    @param limit: int                    - max number of comments extracted
    @return: posts, c_analyzed, tickers, titles, a_comments, picks, subs, picks_ayz
        posts: int       - # of posts analyzed
        c_analyzed: int  - # of comments analyzed
        tickers: dict    - all the tickers found
        titles: list     - list of the title of posts analyzed
        a_comments: dict - all the comments to analyze
        picks: int       - top picks to analyze
        subs: list[str]  - # of subreddits analyzed
        picks_ayz: int   - top picks to analyze
    """

    # set the program parameters
    # subs = ['wallstreetbets' ]     # sub-reddit to search
    # post_flairs = {'Daily Discussion', 'Weekend Discussion', 'Discussion'}    # posts flairs to search
    good_auth = {'AutoModerator'}  # authors whom comments are allowed more than once
    unique_cmt = True  # allow one comment per author per symbol
    ignore_auth_p = {'example'}  # authors to ignore for posts
    ignore_auth_c = {'example'}  # authors to ignore for comment
    upvote_ratio = 0.70  # upvote ratio for post to be considered, 0.70 = 70%
    ups = 20  # define # of upvotes, post is considered if upvotes exceed this #
    limit = 1  # define the limit, comments 'replace more' limit
    upvotes = 2  # define # of upvotes, comment is considered if upvotes exceed this #
    picks = 25  # define # of picks here, prints as "Top ## picks are:"
    picks_ayz = 25  # define # of picks for sentiment analysis
    '''############################################################################'''

    posts, count, c_analyzed, tickers, titles, a_comments = 0, 0, 0, {}, [], {}
    cmt_auth = {}

    for sub in subs:
        subreddit = reddit.subreddit(sub)
        hot_python = subreddit.hot()  # sorting posts by hot
        # Extracting comments, symbols from subreddit
        for submission in hot_python:
            flair = submission.link_flair_text
            author = submission.author.name
            # checking: post upvote ratio # of upvotes, post flair, and author
            flair_is_ok = post_flairs is None or flair is None or len(
                [f for f in flair.split(" ") if f in post_flairs]) > 0
            if submission.upvote_ratio >= upvote_ratio and submission.ups > ups \
                    and flair_is_ok and author not in ignore_auth_p:
                submission.comment_sort = 'new'
                comments = submission.comments
                titles.append(submission.title)
                posts += 1
                if posts % 10 == 1:
                    print("Number of posts checked {}".format(posts))
                    print("Number of comments analysed {}".format(c_analyzed))
                try:
                    submission.comments.replace_more(limit=limit)
                    auth = None
                    for comment in comments:
                        # try except for deleted account?
                        try:
                            auth = comment.author.name
                        except Exception as e:
                            pass
                        c_analyzed += 1
                        # checking: comment upvotes and author
                        if comment.score > upvotes and auth not in ignore_auth_c:
                            split = comment.body.split(" ")
                            for word in split:
                                word = word.replace("$", "")
                                # upper = ticker, length of ticker <= 5, excluded words,
                                # if word.isupper() and len(word) <= 5 and
                                #    word not in my_data.blacklist and word in my_data.us:
                                if word.isupper() and \
                                        len(word) <= 5 and word not in blacklist_words and word in list_of_symbols:
                                    # unique comments, try/except for key errors
                                    if unique_cmt and auth not in good_auth:
                                        try:
                                            if auth in cmt_auth[word]:
                                                break
                                        except Exception as e:
                                            pass
                                        # counting tickers
                                    if word in tickers:
                                        tickers[word] += 1
                                        a_comments[word].append(comment.body)
                                        cmt_auth[word].append(auth)
                                        count += 1
                                    else:
                                        tickers[word] = 1
                                        cmt_auth[word] = [auth]
                                        a_comments[word] = [comment.body]
                                        count += 1
                except Exception as e:
                    print(e)
    return posts, c_analyzed, tickers, titles, a_comments, picks, subs, picks_ayz


def print_helper(tickers: dict,
                 picks: int,
                 c_analyzed: int,
                 posts: int,
                 subs: list,
                 titles: list,
                 tim: time,
                 start_time: time) -> tuple[dict, list, list]:
    """
        Prints out top tickers, and most mentioned tickers
    @param tickers: dict   - all the tickers found
    @param picks: int      - top picks to analyze
    @param c_analyzed: int - # of comments analyzed
    @param posts: int      - # of posts analyzed
    @param subs: list      - list of subreddits analyzed
    @param titles: list    - list of the title of posts analyzed
    @param tim: time obj  - top picks to analyze
    @param start_time: time obj - prog start time
    @return:
    symbols: dict  - dict of sorted tickers based on mentions
    times: list    - include # of time top tickers is mentioned
    top: list      - list of top tickers
    """

    # sorts the dictionary
    symbols = dict(sorted(tickers.items(), key=lambda item: item[1], reverse=True))
    top_picks = list(symbols.keys())[0:picks]
    ctime = (tim.time() - start_time)

    # print top picks
    print(
        "It took {t:.2f} seconds to analyze {c} comments in {p} posts in {s} subreddits.\n".format(t=ctime,
                                                                                                   c=c_analyzed,
                                                                                                   p=posts,
                                                                                                   s=len(subs)))
    print("Posts analyzed saved in titles")
    # for i in titles: print(i)  # prints the title of the posts analyzed

    print(f"\n{picks} most mentioned tickers: ")
    times = []
    top = []
    for i in top_picks:
        print(f"{i}: {symbols[i]}")
        times.append(symbols[i])
        top.append(f"{i}: {symbols[i]}")

    return symbols, times, top


def sentiment_analysis(picks_ayz: int, a_comments: dict, symbols: dict) -> dict:
    """
       Analyzes sentiment of top tickers
    @param picks_ayz: int     - top picks to analyze
    @param a_comments: dict   - all the comments to analyze
    @param symbols: dict      - dict of sorted tickers based on mentions
    @return: dict             -  dictionary of all the sentiment analysis
    """
    scores = {}
    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(sentiment_data.new_words)  # adding custom words from data.py
    picks_sentiment = list(symbols.keys())[0:picks_ayz]

    score_cmnt = {}
    for symbol in picks_sentiment:
        stock_comments = a_comments[symbol]
        for cmnt in stock_comments:

            emojiless = emoji.get_emoji_regexp().sub(u'', cmnt)  # remove emojis

            # remove punctuation
            text_punc = "".join([char for char in emojiless if char not in string.punctuation])
            text_punc = re.sub('[0-9]+', '', text_punc)

            # tokenizeing and cleaning
            tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
            tokenized_string = tokenizer.tokenize(text_punc)
            lower_tokenized = [word.lower() for word in tokenized_string]  # convert to lower case

            # remove stop words
            nlp = English()
            stopwords = nlp.Defaults.stop_words
            sw_removed = [word for word in lower_tokenized if word not in stopwords]

            # normalize the words using lematization
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = ([lemmatizer.lemmatize(w) for w in sw_removed])

            # calculating sentiment of every word in comments n combining them
            score_cmnt = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

            word_count = 0
            for word in lemmatized_tokens:
                if word.upper() not in sentiment_data.us:
                    score = vader.polarity_scores(word)
                    word_count += 1
                    for key, _ in score.items():
                        score_cmnt[key] += score[key]
                else:
                    score_cmnt['pos'] = 2.0

                    # calculating avg.
            try:  # handles: ZeroDivisionError: float division by zero
                for key in score_cmnt:
                    score_cmnt[key] = score_cmnt[key] / word_count
            except Exception as e:
                pass

            # adding score the the specific symbol
            if symbol in scores:
                for key, _ in score_cmnt.items():
                    scores[symbol][key] += score_cmnt[key]
            else:
                scores[symbol] = score_cmnt

                # calculating avg.
        for key in score_cmnt:
            scores[symbol][key] = scores[symbol][key] / symbols[symbol]
            scores[symbol][key] = "{pol:.3f}".format(pol=scores[symbol][key])

    return scores


def visualization(picks_ayz: int,
                  scores: dict,
                  picks: int,
                  symbol_mentions: list,
                  top: list,
                  tree_view: bool = False) -> None:
    """
       Prints sentiment analysis
       Makes a most mentioned picks chart
       Makes a chart of sentiment analysis of top picks
    @param picks_ayz: int            - top picks to analyze
    @param scores: dict              - dictionary of all the sentiment analysis
    @param picks: int                - most mentioned picks
    @param symbol_mentions: list     - include # of time top tickers is mentioned
    @param top: list                 - list of top tickers
    @param tree_view: boolean        - include the tree view or not in the plot
    """
    # printing sentiment analysis
    print("\nSentiment analysis of top picks at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M")))
    df = pd.DataFrame(scores)
    df = df.T
    df = df.rename(columns={'neg': 'Bearish', 'neu': 'Neutral', 'pos': 'Bullish', 'compound': 'Total/Compound'})
    print(df)

    if tree_view:
        # create a color palette, mapped to these values
        cmap = matplotlib.cm.Reds
        mini = min(symbol_mentions)
        maxi = max(symbol_mentions)
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        colors = [cmap(norm(value)) for value in symbol_mentions]

        figure(figsize=(15, 10), dpi=80)
        # Date Visualization
        # most mentioned picks
        squarify.plot(sizes=symbol_mentions, label=top, alpha=.7, color=colors)
        plt.axis('off')
        plt.title("{} most mentioned picks at {}".format(picks,
                                                         datetime.now().strftime("%d/%m/%Y %H:%M")))
        plt.show()
    # Sentiment analysis
    df = df.astype(float)
    # Reorder the columns to have exactly what we want
    df = df[["Bearish", "Neutral", "Bullish", "Total/Compound"]]
    # Set colors to columns
    color_dict = {"Bearish": 'red', "Neutral": 'springgreen', "Bullish": 'forestgreen', "Total/Compound": 'coral'}
    colors = [color_dict.get(x) for x in df.columns]
    print(df)
    df.plot(kind='bar',
            color=colors,
            title="\nSentiment analysis of top picks at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M")),
            figsize=(15, 10))
    plt.show()
    return None


def chart_of_stocks(symbols: list,
                    from_date: str,
                    to_date: str) -> None:
    for st in symbols:
        df = investpy.get_stock_historical_data(stock=st,
                                                country='United States',
                                                from_date='01/01/2018',
                                                to_date=datetime.now().strftime("%d/%m/%Y"))
        kws = dict(volume=True, tight_layout=True)
        mpf.plot(df, **kws, type='candle')
    return None


def chart_of_crypto(symbols: list,
                    base_path: str = "./cryptos_sentiment/",
                    show_charts: bool = True,
                    savefig: bool = False) -> str:
    """

    @param savefig:
    @param symbols:
    @param base_path:
    @param show_charts:
    @return:
    """
    to_date = datetime.now().strftime("%d%m%y-%H%m")
    name_folder = datetime.now().strftime("%d%m%y-%H%m")
    dir_name = base_path + name_folder
    for sy in symbols:
        try:
            data = yf.download(tickers=sy + "-USD", period="10h",
                               interval="5m")
            # data = investpy.get_crypto_historical_data(crypto=sy, from_date='01/01/2021', to_date='18/10/2021')
            if savefig:
                if not os.path.isdir(dir_name):
                    os.mkdir(base_path + name_folder)
                savefig = dir_name + "/" + sy + "-USD" + to_date + ".jpg"
                mpf.plot(data,
                         type='candle',
                         volume=True,
                         title=sy + "-USD 5m",
                         tight_layout=True,
                         figratio=(15, 10),
                         savefig=savefig)
            else:
                # if savefig is included, even with value "None", mpf.plot is unhappy
                mpf.plot(data,
                         type='candle',
                         volume=True,
                         title=sy + "-USD 5m",
                         tight_layout=True,
                         figratio=(15, 10))
        except Exception as e:
            print(e)
            print("No chart for " + sy)
            continue
    if show_charts:
        display_all_images(dir_name)
    return dir_name


def display_all_images(base_dir):
    """

    @param base_dir:
    """
    from IPython.display import display
    from PIL import Image
    import glob

    for f in glob.glob(base_dir + "/*"):
        print(f)
        display(Image.open(f))


def select_outliers_from_scores(scores) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    @param scores:
    @return:
    """
    df = pd.DataFrame(scores).T
    df = df.apply(pd.to_numeric)
    cols = df.select_dtypes('number').columns  # limits to a (float), b (int) and e (timedelta)
    df_sub = df.loc[:, cols]

    # OPTION 1: z-score filter: z-score < 3
    lim1 = np.abs((df_sub - df_sub.mean()) / df_sub.std(ddof=0)) < 3

    # OPTION 2: quantile filter: discard 1% upper / lower values
    lim2 = np.logical_or(df_sub < df_sub.quantile(0.99, numeric_only=False),
                         df_sub > df_sub.quantile(0.01, numeric_only=False))

    # OPTION 3: iqr filter: within 2.22 IQR (equiv. to z-score < 3)
    iqr = df_sub.quantile(0.75, numeric_only=False) - df_sub.quantile(0.25, numeric_only=False)
    lim3 = np.abs((df_sub - df_sub.median()) / iqr) < 2.22

    return df[(lim1 == False).any(axis=1)], df[(lim2 == False).any(axis=1)], df[(lim3 == False).any(axis=1)]


def select_interesting_symbols(scores) -> pd.DataFrame:
    """

    @param scores:
    @return:
    """
    df = pd.DataFrame(scores).T
    df = df.apply(pd.to_numeric)
    df = df[(df.neg > 0.1) | (df.pos > 0.4) | (df.compound < 0) | (df.compound > 0.1)]
    return df
