**About Project:**

The main purpose of this project is to detect building damage in satellite imagery

**Requirements:**
You can find the full list of necessary dependencies **[here](https://github.com/Khalel2023/buildings-damage-analysis/blob/main/requirements.txt)**
This project also requires Python 3.8 or higher, and it's highly recommended to use Miniconda to manage the Python environment, you can install Miniconda [here](https://www.anaconda.com/docs/getting-started/miniconda/install)

**DATASET:**
xBD is a dataset that contains numerous pre- and post-disaster images. 
To get started, first download the dataset from this link: **[xBD](https://xview2.org/dataset)**.

**Simple Instructions:**  

1. After downloading, you will get a zip file.  
2. Extract the zip file.  
3. Inside, there will be several files consisting of one main directory and two subdirectories containing the images and annotations.

**Next step**
Make sure that you have the same structure as shown below

```
xBD
├── disaster_name_1
│ ├── images
│ │ └── <image_id>.png
│ │ └── ...
│ ├── labels
│ │ └── <image_id>.json
│ │ └── ...
├── disaster_name_2
│ ├── images
│ │ └── <image_id>.png
│ │ └── ...
│ ├── labels
│ │ └── <image_id>.json
│ │ └── ...
└── disaster_name_n
```
**Segmentation part**
Use [mask_polygons.py](https://github.com/Khalel2023/buildings-damage-analysis/blob/main/segmentation_part/mask_polygons.py) to create masks for each image
Then use [data_finalize.py](https://github.com/Khalel2023/buildings-damage-analysis/blob/main/segmentation_part/data_finalize.py) to finalize data part, where we create the .txt files with image and label names.

For segmentation, we use a custom U-Net model to identify building footprints.
Before that, we need to create the dataset with [dataset.py](https://github.com/Khalel2023/buildings-damage-analysis/blob/main/segmentation_part/dataset.py) 
Then we can start training our model using [train_model.py](https://github.com/Khalel2023/buildings-damage-analysis/blob/main/segmentation_part/train_model.py)






