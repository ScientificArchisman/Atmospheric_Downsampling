# 🌌 Atmospheric Downsampling with Super Resolution 🌌

## 🚀 Overview

Welcome to the **Atmospheric Downsampling** repository! Here, we tackle the challenging problem of downsampling atmospheric data by formulating it as a super-resolution problem using state-of-the-art Machine Learning and Deep Learning techniques. Our innovative approach enhances data resolution, paving the way for more accurate environmental predictions and analyses.

## 📚 Table of Contents

- [Introduction](#introduction)
- [Methodology](#-methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🌍 Introduction

Atmospheric data is crucial for understanding climate patterns, weather forecasting, and environmental monitoring. However, downsampling this data can lead to loss of critical information. This project leverages the power of super-resolution techniques to enhance the resolution of atmospheric data, ensuring detailed and accurate representations.

## 🧠 Methodology
### 01. Preprocessing
The preprocessing pipeline is embedded in the [dataloading script](02-dataloading.py). The preprocessing steps taken were as follows:
- The input data was shaped as a 4 dimensional array. The dimensions were ```(N1 variables, N2 time points, N3 latituude points, N4 longitude points)```. 
- The data was min-max scaled to the range [0, 1]. This was done to ensure that the data was within the range of the activation functions used in the neural network. Standard scaling was not used as the data was not normally distributed.
- The data was broken down into ```P x P ``` chunks in the dataloader class. So, the final shape of the data was ```(P, N1, N2, N3/P, N4/P)```.

The following figure shows a chunked representation of the data. The data is broken down into 16x16 chunks. 
<img src = "Images/chunks.png"  alt="Chunked Data" width="600" height="500">