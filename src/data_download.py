import os
import sys
sys.path.insert(0, '../utils')
import data_download
import data_utils
import logging

SEED = 42
#os.environ['GEO_AI_API_KEY'] = "INSERT API KEY HERE"
#os.environ['GEO_AI_SECRET_KEY'] = "INSERT SECRET KEY HERE"

def main():
    nightlights_bins_file = '../data/nightlights_bins.csv'
    satellite_images_dir = '../data/images/'
    report_file = '../data/report.csv'

    try:
        data_download.get_satellite_images_with_labels(nightlights_bins_file, satellite_images_dir)
    except:
        logging.debug("Could not download satellite images. Please set your API keys.")
    
    # To see how nightlights_bins_file was generated, please refer to notebooks/01_lights_eda.ipynb
    nightlights = pd.read_csv(nightlights_bins_file)
    report = pd.read_csv(report_file)
    
    # Split dataset into training and validation sets
    nightlights = nightlights.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train, val = data_utils.train_val_split(nightlights, train_size=0.9)
    train_balanced = data_utils.balance_dataset(train, size=30000)

    data_utils.train_val_split_images(val, report, satellite_images_dir, phase='val')
    data_utils.train_val_split_images(train_balanced, report, satellite_images_dir, phase='train')

if __name__ == "__main__":
    main()