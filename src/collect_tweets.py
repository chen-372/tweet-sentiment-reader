"""
Collect tweets using Tweepy (Twitter API v2) and save to CSV.

Usage:
  export TWITTER_BEARER_TOKEN="YOUR_TOKEN"
  python src/collect_tweets.py --query "(happy OR sad) lang:en -is:retweet" --max_results 100 --out_csv data/collected.csv
"""
import os
import csv
import argparse
import tweepy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Twitter API v2 search query")
    parser.add_argument("--max_results", type=int, default=100, help="Max results (10..100 per page)")
    parser.add_argument("--out_csv", type=str, default="data/collected.csv")
    args = parser.parse_args()


    #bearer = os.getenv("TWITTER_BEARER_TOKEN", None)
    bearer = "AAAAAAAAAAAAAAAAAAAAAEfJ4AEAAAAAa4oZmsHscDWI5J95ZzBYJJp%2BBiQ%3DvslzYTc3YqD5n8TV8fG898vnieLHItZoAe63hf4WjY19OCR8EV"
    if not bearer:
        raise RuntimeError("TWITTER_BEARER_TOKEN is not set")

    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)

    tweets = client.search_recent_tweets(
        query=args.query,
        tweet_fields=["lang", "created_at"],
        max_results=min(max(args.max_results, 10), 100),
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text"])  # you can add a label column after manual labeling
        if tweets.data:
            for t in tweets.data:
                w.writerow([t.text])

    print(f"Wrote CSV: {args.out_csv}")

if __name__ == "__main__":
    main()
