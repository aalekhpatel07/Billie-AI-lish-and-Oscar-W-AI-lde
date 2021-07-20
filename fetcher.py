import requests
import os
from pathlib import Path


def main():
    # pass
    # response = requests.request("GET", url="https://poetrydb.org/author/Wilde").json()
    #
    # for records in response:
    #     with open(f'./data/Wilde/{records["title"]}.txt', 'w') as fl:
    #         fl.write("\n".join(records["lines"]))
    # result = []
    # for root, file_dirs, files in os.walk('./data/Wilde'):
    #     for name in files:
    #         full_path = Path(os.path.join(root, name))
    #         with open(full_path, 'r') as fl:
    #             result.append("".join(fl.readlines()))
    # with open(f'./data/wilde_combined.txt', 'w') as fl:
    #     fl.write("\n".join(result))
    pass


if __name__ == '__main__':
    main()
