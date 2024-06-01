# palm-leaf-character-segmentation
A project did to explore the subjects of cv and Neural Networks 
Project Overview:
This project focuses on the segmentation of characters from ancient palm leaf manuscripts. Using Convolutional Neural Networks (CNNs) and computer vision techniques, the goal is to accurately segment characters from preprocessed palm leaf images. This segmentation is crucial for further processing and analysis, such as character recognition and translation.

Introduction
Ancient palm leaf manuscripts are valuable cultural artifacts that contain historical and literary content. However, due to their age and the manual writing process, these manuscripts often present challenges for digitization and analysis. This project aims to address these challenges by using CNNs for the segmentation of individual characters from the manuscripts.

Project Structure
The project is organized as follows:
palm_leaf_segmentation/
│
├── data/
│   ├── raw/
│   ├── preprocessed/
│   └── segmented/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── segmentation_demo.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── segment.py
│
├── models/
│   └── cnn_model.h5
│
├── README.md
└── requirements.txt
