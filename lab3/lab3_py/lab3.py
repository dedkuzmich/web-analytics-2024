import pandas as pd
import json
import pytumblr
from datetime import datetime as dt
from bs4 import BeautifulSoup


# Authenticate via Tumblr API
with open("credentials.json") as fin:
    credentials = json.load(fin)

CONSUMER_KEY = credentials["CONSUMER_KEY"]
CONSUMER_SECRET = credentials["CONSUMER_SECRET"]
OAUTH_TOKEN = credentials["OAUTH_TOKEN"]
OAUTH_SECRET = credentials["OAUTH_SECRET"]

client = pytumblr.TumblrRestClient(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_SECRET)
#


# Retrieve posts from Tumblr and create dataset
blog_name = "redglassbird"  # Source:   https://www.tumblr.com/redglassbird
num_posts = 1050  # Total number of posts
limit = 50  # Max number of posts that can be retrieved in a single request
list_posts = []
for i in range(num_posts // limit):
    posts = client.posts(blog_name, limit = limit)
    for post in posts["posts"]:
        # Parse Datetime
        datetime_str = post["date"]
        datetime = dt.strptime(datetime_str, "%Y-%m-%d %H:%M:%S %Z")

        # Parse HTML body as text
        html_body = post["body"]
        soup = BeautifulSoup(html_body, "html.parser")
        body = soup.get_text()

        dict_post = {
            "Blog": post["blog_name"],
            "Datetime": datetime,
            "URL": post["post_url"],
            "Title": post["title"],
            "Body": body
        }
        list_posts.append(dict_post)

# Save dataset as CSV
file_csv = f"tumblr-{blog_name}.csv"
df = pd.DataFrame(list_posts)
df.to_csv(file_csv, index = False)
print(f"Number of entries: {len(df)}\n")
df.info()
display(df)
#
