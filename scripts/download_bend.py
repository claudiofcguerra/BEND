import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup


base_link = "https://sid.erda.dk/"
link = "https://sid.erda.dk/cgi-sid/ls.py?share_id=f6hdp1zTzh&current_dir=.&flags=f"


def get_soup(link):
    source_code = requests.get(link)
    soup = BeautifulSoup(source_code.content, "lxml")
    f = []
    f.extend(
        soup.find_all(
            "a",
            {
                "class": [
                    "leftpad directoryicon",
                ]
            },
        )
    )
    f.extend(soup.find_all("a", {"title": "open"}))
    return f


# download file in link
def download_file(link, destination):
    r = requests.get(link, stream=True, allow_redirects=True)
    total_length = r.headers.get("content-length")
    if total_length is None:
        with open(destination, "wb") as f:
            f.write(r.content)
    else:
        total_length = int(total_length)
        with open(destination, "wb") as f, tqdm(
            total=total_length, unit="B", unit_scale=True, desc=os.path.basename(destination)
        ) as pbar:
            for data in r.iter_content(chunk_size=8192):
                f.write(data)
                pbar.update(len(data))


def rec(link, destination="./"):
    f = get_soup(link)
    for child in f:
        if child.get("title") == "open":
            link = f'{base_link}{child.get("href")}'
            child_path = child.get("href")[27:]
            os.makedirs(f"{destination}/{os.path.dirname(child_path)}", exist_ok=True)
            print(f"Downloading: {destination}/{child_path}")
            download_file(link, f"{destination}/{child_path}")
        else:
            link = f'{base_link}cgi-sid/{child.get("href")}'
            rec(link, destination)


print("DOWNLOADING")
rec(link, destination=".")
