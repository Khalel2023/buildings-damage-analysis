**About Project:**

The main purpose of this project is to detect building damages in satellite imagery

**Requirements:**
You can find the full list of necessary dependencies **[here](https://github.com/Khalel2023/buildings-damage-analysis/blob/main/requirements.txt)**
This project also requires Python 3.8 or higher, and it s highly recommended to use Miniconda to manage the Python environment, you can install Miniconda [here](https://www.anaconda.com/docs/getting-started/miniconda/install)

**DATASET:**
xBD is a dataset that contains numerous pre- and post-disaster images. 
To get started,we should first download the basic data from this link: **[xBD](https://xview2.org/dataset)**.

**Simple Instructions:**  

1. After downloading, you will get a zip file.  
2. Extract the zip file.  
3. Inside, there will be several files consisting of one main directory and two subdirectories containing the images and annotations.

**Next step**
Make sure that you have the same structure as shown below


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


    




