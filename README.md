<p align="center">
<b><a href="#setup">Setup</a></b>
|
<b><a href="#code-organization">Code Organization</a></b>
|
<b><a href="#data-sources">Data Sources</a></b>
|
<b><a href="#models">Models</a></b>
|
<b><a href="#key-results">Key Results</a></b>
|
<b><a href="#acknowledgements">Acknowledgements</a></b>
</p>

# Philippine Poverty Mapping

This repository accompanies our research work, *"Mapping Philippine Poverty
using Machine Learning, Satellite Imagery, and Crowd-sourced Geospatial
Information"*, currently published in our
[website](https://stories.thinkingmachin.es/philippines-most-vulnerable-communities/).
In this work, **we developed five wealth prediction models** using
state-of-the-art methods and various geospatial data sources.

![pampanga map](./assets/pampanga-map.jpg)


## Setup

In order to run the notebooks, all dependencies must be installed. We provided
a `Makefile` to accomplish this task:

```s
make venv
make build
```

This creates a virtual environment, `venv`, and installs all dependencies found
in `requirements.txt`. In order to run the notebooks inside `venv`, execute the
following command:

```s
ipython kernel install --user --name=venv
```

Notable dependencies include:
- matplotlib==3.0.2
- seaborn==0.9.0
- numpy==1.16.0
- pandas==0.24.0
- torchsummary==1.5.1
- torchvision==0.2.1
- tqdm==4.30.0

## Code Organization 

This repository is divided into three main parts:
- **notebooks/**: contains all Jupyter notebooks for different wealth
    prediction models.
- **utils/**: contains utility methods for loading datasets, building model, and
   performing training routines. 
- **src/**: contains the transfer learning training script.

It is possible to follow our experiments and reproduce the models we've built
by going through the notebooks one-by-one. For model training, we leveraged a
Google Compute Engine (GCE) instance with 

## Data Sources

- **Demographic and Health Survey (DHS)**: we used the [2017 Philippine
    Demographic and Health Survey](https://dhsprogram.com/) as our measure of
    ground-truth for socioeconomic indicators. It is conducted every 3 to 5
    years, and contains nationally representative information on different
    indicators across the country. 
- **Nighttime Luminosity Data**: we obtained nighttime lights data from the
    [Visible Infrared Imaging Radiometer Suite Day/Night Band (VIIRS
    DNB)](https://ngdc.noaa.gov/eog/viirs/download_dnb_composites.html) for the
    year 2016. It includes a continuous luminosity level from 0 to 122, where 0
    is the darkest pixel.
- **Daytime Satellite Imagery**: we captured 134,540 satellite images from the
    [Google Static Maps
    API](https://developers.google.com/maps/documentation/maps-static/intro).
    Our
    [parameter](https://developers.google.com/maps/documentation/maps-static/dev-guide)
    settings are as follows: zoom level=17, scale=1, and image
    size=400x400pixels. These images match the land area covered by a single
    pixel of night time lights data (0.25-sq.km). 
- **High Resolution Settlement Data (HRSL)**: we used this dataset, provided by
    [Facebook Research, CIESIN Columbia, and World
    Bank](https://www.ciesin.columbia.edu/data/hrsl/), to filter out images
    without human settlements. Their population estimates were based on recent
    census data and high resolutoin satellite imagery (0.5-m) from
    DigitalGlobe.
- **OpenStreetMaps Data (OSM)**: we acquired crowd-sourced geospatial data from
    [OpenStreetMaps (OSM)](https://www.openstreetmap.org) via the
    [Geofabrik](https://www.geofabrik.de/) online repository. This dataset is
    volunteer-curated, and covers almost 83% of the entire Philippine street
    network.

## Models


## Key Results


## Acknowledgments

This work was supported by the [UNICEF Innovation
Fund](https://unicefinnovationfund.org/). We would also like to thank Vedran
Sekara, Do-Hyung Kim of UNICEF and Priscilla Moraes of Google for the
insightful discussions and valuable mentorship.



