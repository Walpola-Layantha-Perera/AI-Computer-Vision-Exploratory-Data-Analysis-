Waymo Dataset Exploratory Data Analysis (EDA)
-----------------------------------------------
This repository contains a Jupyter Notebook to perform an Exploratory Data Analysis (EDA) on the Waymo Open Dataset. The analysis involves visualizing bounding boxes and class distributions within the dataset's images, providing insights into the data structure, class characteristics, and potential challenges for machine learning tasks.

Table of Contents
Overview
Installation
Dataset Structure
Exploratory Data Analysis
Displaying Images with Bounding Boxes
Displaying Multiple Images
Additional Analysis
Acknowledgments
License
Overview
This project demonstrates the use of TensorFlow and Matplotlib to:

Load and display sample images from the Waymo Open Dataset.
Visualize bounding boxes for detected objects, color-coded by class (e.g., vehicles in red, pedestrians in blue, cyclists in green).
Explore additional characteristics of the dataset, such as object distribution across classes.
The goal is to gain initial insights into the dataset to guide further analysis or model development.

This code relies on TensorFlow for loading and processing the dataset. Install TensorFlow according to your environment specifications (CPU or GPU).

Run Jupyter Notebook:

bash
Copy code
jupyter notebook
Launch the Notebook:

Open Exploratory Data Analysis.ipynb in Jupyter to begin the analysis.

Dataset Structure
The Waymo dataset is provided in TFRecord format, and this project works with processed dataset files stored in the data/ directory:

data/train/*.tfrecord: Training data in TFRecord format.
label_map.pbtxt: Label map for class names and IDs.
Exploratory Data Analysis
The EDA notebook covers the following main steps:

Displaying Images with Bounding Boxes
Define the display_images Function:

The display_images function takes a batch of images and overlays bounding boxes in different colors based on object class:

Vehicles: Red
Pedestrians: Blue
Cyclists: Green
Loading a Sample Frame:

The following code loads and parses a TFRecord file, visualizing bounding boxes for each detected object:

python
Copy code
from waymo_open_dataset import dataset_pb2 as open_dataset 

FILENAME = 'data/train/segment-*.tfrecord'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
dataset_iter = dataset.as_numpy_iterator()

data = next(dataset_iter)
frame = open_dataset.Frame()
frame.ParseFromString(data)
Displaying the Image:

The show_camera_images function uses Matplotlib to display each image frame, showing bounding boxes and class labels.

Displaying Multiple Images
Using the loaded dataset and display_images function, multiple images (e.g., 10 random images) can be sampled and displayed with bounding boxes and color-coded labels.

python
Copy code
plt.figure(figsize=(25, 20))
for index, image in enumerate(frame.images):
    _ = show_camera_images(image, [1-10, index + 1])
Additional Analysis
In the final section of the notebook, we explore additional dataset characteristics, such as:

Data Distribution: Distribution of classes (e.g., counts of vehicles, pedestrians, cyclists).
Bounding Box Characteristics: Average bounding box sizes by class.
This section allows for deeper analysis to guide further research or model development.

Acknowledgments
Waymo Open Dataset: This project utilizes the Waymo Open Dataset, which is publicly available for non-commercial use in research and development projects. For more information on the dataset and its terms of use, visit the Waymo Open Dataset website.
TensorFlow: TensorFlow is used in this project for data processing and dataset loading. Special thanks to the TensorFlow developers and maintainers for their work on this powerful open-source machine learning framework.
The Waymo Open Dataset is licensed for non-commercial use, and we adhere to all usage terms in our project. Any use of this project should comply with the licensing terms and limitations set by Waymo LLC.

License
This project is licensed under the MIT License. Please see the LICENSE.md file for more details.
