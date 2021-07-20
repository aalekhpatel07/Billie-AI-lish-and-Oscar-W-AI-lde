import pdb

import scrapy
from scrapy import Spider
import chompjs
from pprint import pprint as pp
import os
from pathlib import Path


def get_url_map():
    urls = {}
    for root, file_dirs, files in os.walk('./data/eilish'):
        for name in files:
            full_path = Path(os.path.join(root, name))
            with open(full_path, 'r') as fl:
                urls[fl.read()] = full_path
    return urls


def get_start_urls():
    mp = get_url_map()
    return list(map(lambda x: 'https://www.deezer.com/en/track/' + str(x), mp.keys()))


class DeezerLyricsScraper(Spider):
    name = 'deezerlyricsspider'
    start_urls = ['https://www.deezer.com/en/track/655095942']
    stuff = iter(zip(get_start_urls(), get_url_map().items()))
    results = dict()
    url_map = get_url_map()

    def parse(self, response):
        # pdb.set_trace()
        whole_script = response.css('script::text').getall()[1]
        whole_script = whole_script.replace('window.__DZR_APP_STATE__ =', '')
        data = chompjs.parse_js_object(whole_script)
        # self.logger.info(data)
        if 'LYRICS' in data:
            if 'LYRICS_TEXT' in data['LYRICS']:
                self.logger.info(data['LYRICS']['LYRICS_TEXT'])
                track_id = response.url.split("/")[-1]
                track_file_path = self.url_map[track_id]
                with open(track_file_path, "a") as fl:
                    fl.writelines(["\n\n", data['LYRICS']['LYRICS_TEXT']])
        try:
            track_url, (track_id, track_path) = next(self.stuff)
            self.logger.info(track_id)
            yield scrapy.Request(track_url)
        except StopIteration as e:
            self.logger.info("done!")
        # response.follow('https://www.deezer.com/en/track/877873612', self.parse)


def combine_the_tracks():
    result = []
    for root, file_dirs, files in os.walk('./data/eilish'):
        for name in files:
            full_path = Path(os.path.join(root, name))
            with open(full_path, 'r') as fl:
                result.append("".join(fl.readlines()))
    return "\n\n".join(result)


if __name__ == '__main__':
    pass
