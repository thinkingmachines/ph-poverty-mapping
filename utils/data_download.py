# -*- coding: utf-8 -*-

"""Utility methods for downloading GSM images"""

# Import standard library
import gc
import logging
import os
import urllib
from io import BytesIO

# Import modules
import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image
from staticmaps_signature import StaticMapURLSigner
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_static_google_maps(
    filename,
    center=(0, 0),
    scale=1,
    zoom=17,
    imgsize=(400, 400),
    maptype="satellite",
    show=False,
):
    """Get static google map image given a set of coordinates
    and saves it into the filesystem

    Parameters
    ----------
    filename : str
        the filename to save the image into
    center : tuple of ints
        latitute-longitude combination, default is (0,0)
    zoom : int
        zoom size, default is 16
    imgsize : tuple of ints
        image size, default is (400,400)
    maptype : str
        map type for GSM, default is "satellite"
    show : bool
        show resulting image, default is False

    """
    API_KEY = os.environ['GEO_AI_API_KEY']
    SECRET_KEY = os.environ['GEO_AI_SECRET_KEY']
    
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"
    # fmt: off
    gsm_url = (
        base_url
        + "center=" + str(center[0]) + "," + str(center[1])
        + "&zoom=" + str(zoom)
        + "&size=" + "{}x{}".format(imgsize[0], imgsize[1])
        + "&scale=" + str(scale)
        + "&maptype=" + maptype
        + "&sensor=false"
    )
    # fmt: on
    staticmap_url_signer = StaticMapURLSigner(
        public_key=API_KEY, private_key=SECRET_KEY
    )
    gsm_url = staticmap_url_signer.sign_url(gsm_url)

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    b = BytesIO(urllib.request.urlopen(gsm_url).read())
    image = Image.open(b)

    if show:
        image.show()
        plt.imshow(image)
        plt.show()

    # Save image
    image.convert("RGB").save(filename)
    return gsm_url

        
def get_satellite_images_with_labels(
    csv_path, 
    outdir, 
    report_dir, 
    scale=1, 
    zoom=17, 
    imgsize=(400,400), 
    limit=None
):
    """Get satellite image and saves it into an output path based 
    on its night light intensity class (low, medium, high)

    Parameters
    ----------
    csv_path: str
        directory where the lat lon label csv files are stored
    outdir: str
        output directory to store images and report
    limit : int
        number of entries to download (default is None)
    """
    # Create report.txt file for logging
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    report_file = report_dir + "report.csv"
    
    df = pd.read_csv(csv_path) 
    ids = df['ID'][:limit]
    latitude, longitude = df["ntllat"][:limit], df["ntllon"][:limit]
    clusters = df["DHSCLUST"][:limit]
    labels = df["label"][:limit] + '/'

    for idx, (id_, cluster, lat, lon, label) in tqdm(
        enumerate(zip(ids, clusters, latitude, longitude, labels)), total=len(latitude)
    ):
        filename = (
            outdir + str(lat) + "_" + str(lon) + ".jpg"
        )
        try:
            if not os.path.isfile(filename):
                get_static_google_maps(
                    filename, 
                    center=(lat, lon), 
                    scale=scale,
                    zoom=zoom,
                    imgsize=imgsize,
                    show=False
                )
                
            report = {'id':[], 'lat':[], 'lon':[], 'cluster':[], 'filename':[], 'label':[]}
            report['id'].append(id_)
            report['cluster'].append(cluster)
            report['lat'].append(lat)
            report['lon'].append(lon)
            report['filename'].append(filename)
            report['label'].append(label[:-1])
            report = pd.DataFrame(report)
            report = report[['id', 'cluster', 'lat', 'lon', 'filename', 'label']]
            
            if not os.path.isfile(report_file):
                report.to_csv(report_file, index=False)
            else:
                report.to_csv(report_file, mode='a', header=False, index=False)
            
        except urllib.error.HTTPError:
            logger.warn("No image for {}".format(filename))
            pass

        gc.collect()