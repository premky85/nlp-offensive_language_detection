import csv
import sys
from facebook_scraper import get_posts

page_name = 'VladaRepublikeSlovenije' # test -> change this to array of pages

csvFile = open('../facebook_posts.csv', 'a')
csvWriter = csv.writer(csvFile)

try:
    for post in get_posts(page_name, pages=10):
        csvWriter.writerow([post['text'].encode('utf-8')])
except AttributeError:
    print("No more posts to get")
