#!/usr/bin/env python3
import os
import urllib.request
import argparse
import multiprocessing
import pickle as pkl
from PIL import Image

image_folder = './images/'

def fetch_image(url):
    filename = os.path.split(url)[-1]
    full_path = os.path.join(image_folder, filename)
    if os.path.exists(full_path):
        return

    print('Downloading', filename)
    try:
        urllib.request.urlretrieve(url, full_path)
        photo = Image.open(full_path)
        photo.verify()  # Check if the image is valid
    except Exception as e:
        print(f'Failed to download {url}: {e}')
        if os.path.exists(full_path):
            os.remove(full_path)  # Remove incomplete file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download the images in the dataset into a specified folder.')
    parser.add_argument(
        '-w', '--workers', type=int, default=-1,
        help="num workers used to download images. -x uses (all - x) cores [-1 default]."
    )
    parser.add_argument('-dir', type=str, default='./images/',
        help='the path to save the images, default="./images/"'
    )
    args = parser.parse_args()
    image_folder = args.dir
    num_workers = args.workers

    if num_workers < 0:
        num_workers = multiprocessing.cpu_count() + num_workers

    if not os.path.exists(image_folder):
        print('Creating folder to download images...[{}]'.format(image_folder))
        os.makedirs(image_folder)

    with open("dataset.pkl", "rb") as f:
        db = pkl.load(f)
    URLs = [db[i]['url'] for i in range(0, len(db), 14)]

    print('Downloading {} images with {} workers...'.format(len(URLs), num_workers))
    pool = multiprocessing.Pool(processes=num_workers)
    pool.map(fetch_image, URLs)
    pool.close()
    pool.join()
