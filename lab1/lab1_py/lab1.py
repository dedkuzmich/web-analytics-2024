import pandas as pd
import numpy as np
import random
from re import compile

import maxminddb
from datetime import datetime as dt
from user_agents import parse
from crawlerdetect import CrawlerDetect

import matplotlib.pyplot as plt
from scipy.stats import zscore


# Global settings
pd.set_option("display.max_colwidth", 200)
random.seed(2291)
np.random.seed(2291)
#


# Form dataset from Apache access.log
file_log = "input/access.log"  # Source: https://github.com/elastic/examples/tree/master/Common%20Data%20Formats/apache_logs
file_csv = "input/access.csv"
file_ipinfo = "input/country.mmdb"  # Source: https://ipinfo.io/products/free-ip-database
list_log = []

# Parse access.log with regex
pattern = r'(?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) (?P<client>.*?) (?P<user>.*?) \[(?P<datetime>.*?)\] "(?P<method>\w+) (?P<url>.*?) (?P<protocol>HTTP/1.\d)" (?P<status>\d+) (?P<size>\d+) "(?P<referrer>.*?)" "(?P<agent>.*?)"'
regex = compile(pattern)
reader = maxminddb.open_database(file_ipinfo)
with open(file_log, "r") as fin:
    for line in fin:
        match = regex.match(line)
        if not match:  # If cannot parse the line
            continue

        # Get country by IP
        ip = match.group("ip")
        country = reader.get(ip)["country_name"]

        # Get time/date
        datetime_str = match.group("datetime")
        datetime_obj = dt.strptime(datetime_str, '%d/%b/%Y:%H:%M:%S %z')
        date = datetime_obj.date()
        time = datetime_obj.time()

        # Retrieve OS from User-Agent
        agent = match.group("agent")
        os = parse(agent).os.family

        list_log.append({
            "IP": ip,
            "Country": country,

            # "Client": match.group("client"),
            # "User": match.group("user"),

            "Date": date,
            "Time": time,

            "Method": match.group("method"),
            "URL": match.group("url"),
            # "Protocol": match.group("protocol"),

            "Status": match.group("status"),
            "Size": match.group("size"),

            "Referrer": match.group("referrer"),
            "User-Agent": agent,
            "OS": os
        })

# Save parsed access.log as CSV
df = pd.DataFrame(list_log)
df.to_csv(file_csv, index = False)
print(f"Number of entries: {len(df)}")
#


# Load dataset
df = pd.read_csv(file_csv, encoding = "latin-1")
df.info()
df.head()
#


# A) Count users by day
users_by_day = df.groupby("Date")["IP"].nunique()
users_by_day = users_by_day.reset_index().rename(columns = {"IP": "Unique Users"})
users_by_day.to_csv("output/A_users_by_day.csv", index = False)
display(users_by_day)
#


# B) Sort user agents by users
top_user_agents = df.groupby("User-Agent")["IP"].nunique().sort_values(ascending = False)
top_user_agents = top_user_agents.reset_index().rename(columns = {"IP": "Unique Users"})
top_user_agents.to_csv("output/B_top_user_agents.csv", index = False)
display(top_user_agents)
#


# C) Sort OS by users
top_os = df.groupby("OS")["IP"].nunique().sort_values(ascending = False)
top_os = top_os.reset_index().rename(columns = {"IP": "Unique Users"})
top_os.to_csv("output/C_top_os.csv", index = False)
display(top_os)
#


# D) Sort country by users
top_country = df.groupby("Country")["IP"].nunique().sort_values(ascending = False)
top_country = top_country.reset_index().rename(columns = {"IP": "Unique Users"})
top_country.to_csv("output/D_top_countries.csv", index = False)
display(top_country)
#


# E) Select search bots
def get_bot_name(agent):
    bot_name = ""
    crawler_detect = CrawlerDetect()
    if crawler_detect.isCrawler(agent):  # Check if user agent is a bot
        bot_name = crawler_detect.getMatches()  # Get the name of the bot
        if not bot_name:
            bot_name = "Unknown"
    return bot_name

search_bots = df.copy(deep = True)
search_bots["Bot Name"] = search_bots["User-Agent"].apply(get_bot_name)
search_bots = search_bots[search_bots["Bot Name"] != ""]
search_bots = search_bots.groupby("Bot Name")["IP"].nunique().sort_values(ascending = False)
search_bots = search_bots.reset_index().rename(columns = {"IP": "Unique Users"})
search_bots.to_csv("output/E_search_bots.csv", index = False)
display(search_bots)
#


# F) Detect anomalies
# Get datetime objects from separated date & time
datetime = pd.to_datetime(df["Date"] + " " + df["Time"], format = "%Y-%m-%d %H:%M:%S")

# Calculate Z-score for numerical features
cols = ["Size"]  # Status can be added
for col in cols:
    df_zscore = df[[col]].apply(zscore)
    threshold = 3  # Z-score that won"t be treated as anomaly is in range [-3; 3]
    anomalies = df[(df_zscore > threshold).any(axis = 1)]
    anomalies.to_csv(f"output/F_{col}_anomalies.csv", index = False)
    print(f"Found {len(anomalies)} anomalies")

    # Plot anomalies
    plt.figure(figsize = (10, 5))
    plt.plot(datetime, df[col], label = col, color = "blue", alpha = 0.25)
    plt.scatter(datetime[anomalies.index], anomalies[col], color = "green", label = "Anomalies", marker = "*")
    plt.title(f"{col} with anomalies")
    plt.xlabel("Datetime")
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
#
