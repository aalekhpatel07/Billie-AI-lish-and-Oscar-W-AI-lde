import requests
import time


def initial_request():
    url = "https://deezerdevs-deezer.p.rapidapi.com/search"
    query_string = {"q": "billie eilish"}

    headers = {
        'x-rapidapi-key': "35af91e30amshff3a17b9608017fp15b2edjsn1d8d3172738a",
        'x-rapidapi-host': "deezerdevs-deezer.p.rapidapi.com"
    }
    return requests.request("GET", url, headers=headers, params=query_string)


def subsequent_requests(url):
    headers = {
        'x-rapidapi-key': "35af91e30amshff3a17b9608017fp15b2edjsn1d8d3172738a",
        'x-rapidapi-host': "deezerdevs-deezer.p.rapidapi.com"
    }
    return requests.request("GET", url, headers=headers)


def make_all_requests():
    total = 0
    consecutive = 0
    queue = []
    response = initial_request()
    consecutive += 1
    total += 1
    json_data = response.json()

    tracks = json_data["data"]

    for track in tracks:
        with open(f'./data/eilish/{str(track["title"])}.txt', 'w') as fl:
            fl.writelines([str(track['id'])])
        pass

    url_to_follow = json_data["next"]
    queue.append(url_to_follow)

    while len(queue):
        current = queue.pop(0)
        if consecutive >= 5:
            time.sleep(62)
            consecutive = 0
        response = subsequent_requests(current).json()
        consecutive += 1
        total += 1
        for track in tracks:
            with open(f'./data/eilish/{str(track["title"])}.txt', 'w') as fl:
                fl.writelines([str(track['id'])])
        if "next" in response:
            queue.append(response["next"])
        print(f"Written {total * 25} files so far!")
    return


def main():
    make_all_requests()
    pass


if __name__ == '__main__':
    main()
