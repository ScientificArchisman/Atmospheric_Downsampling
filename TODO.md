- [x] **Upsample with Bicubic interpolation:**  
Upsample Low resolution Data to High resolution data shape.
Example: 
Low res (***300 time pts***, 7 variables, ***50 pressure levels, 72 lat, 73 long***) 
High resolution (***696 time points***, 7 variables, ***75 pressure, 315 lat, 423 longitude***)

Upsample to high res shape -> (696, 7, 75, 315, 423)

- [x] **Pass through Model and Check**
Model has been built in `mod_srcnn.py`. After upsamplimg, test with one batch of size (1, 7, 75, 315, 423) and check the output shape.

- [ ] **Create data loaders**
Create tensors from `xr.arrays` and create train and test data loaders.
Train size = 80% of the data
Validation size = 20% of the train data
Test size = 20% of the data

- [ ] **Loss function**
MSE loss function to calculate the loss between the predicted and actual data.

- [ ] **Training Loop**
Has veen defined in `srcnn_train.py`. 
