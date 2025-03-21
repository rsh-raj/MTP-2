import requests
import time
import os
import pandas as pd
from datetime import datetime

LANGUAGES = ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "TypeScript", 
             "Swift", "Kotlin", "Dart", "Shell", "C#", "C", "PHP"]


REPOS_PER_LANGUAGE = 500
PAGE_SIZE = 100  # GitHub GraphQL API max
TOKEN_FILE = "access_token.txt"
OUTPUT_DIR = "Top500"
CREATED_AFTER = "2023-01-01"

def get_access_token():
    with open(TOKEN_FILE, 'r') as f:
        return f.read().strip()

def build_query(language, after_cursor=None):
    after = f', after: "{after_cursor}"' if after_cursor else ''
    query = f"""
    {{
      search(query: "language:{language} stars:>0 created:>{CREATED_AFTER} sort:stars", type: REPOSITORY, first: {PAGE_SIZE}{after}) {{
        pageInfo {{
          hasNextPage
          endCursor
        }}
        edges {{
          node {{
            ... on Repository {{
              name
              url
              stargazerCount
              forkCount
              owner {{ login }}
              description
              pushedAt
              createdAt
              primaryLanguage {{ name }}
              issues(states: OPEN) {{ totalCount }}
            }}
          }}
        }}
      }}
    }}
    """
    return query

def fetch_repos(language):
    print(f"Fetching repos for {language}...")
    access_token = get_access_token()
    headers = {
        'Authorization': f'bearer {access_token}',
        'Content-Type': 'application/json'
    }

    repos = []
    after_cursor = None
    while len(repos) < REPOS_PER_LANGUAGE:
        query = build_query(language, after_cursor)
        response = requests.post(
            "https://api.github.com/graphql",
            json={"query": query},
            headers=headers
        )

        if response.status_code != 200:
            raise Exception(f"GraphQL query failed with status code {response.status_code}: {response.text}")

        data = response.json()["data"]["search"]
        for edge in data["edges"]:
            node = edge["node"]
            repos.append({
                "name": node["name"],
                "url": node["url"],
                "stars": node["stargazerCount"],
                "forks": node["forkCount"],
                "language": node["primaryLanguage"]["name"] if node["primaryLanguage"] else None,
                "owner": node["owner"]["login"],
                "description": node["description"],
                "pushed_at": node["pushedAt"],
                "created_at": node["createdAt"],
                "open_issues": node["issues"]["totalCount"]
            })

        if not data["pageInfo"]["hasNextPage"]:
            break
        after_cursor = data["pageInfo"]["endCursor"]
        time.sleep(2)

    print(f"Fetched {len(repos)} repos for {language}")
    return repos[:REPOS_PER_LANGUAGE]

def save_to_csv(language, repos):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(repos)
    filename = f"{OUTPUT_DIR}/top_500_{language.lower()}.csv"
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Saved to {filename}")

def main():
    for lang in LANGUAGES:
        repos = fetch_repos(lang)
        save_to_csv(lang, repos)

if __name__ == "__main__":
    main()
