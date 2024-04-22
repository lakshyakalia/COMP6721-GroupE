# COMP6721 (AAI) - Classification of Plant Leaves using Computer Vision and Deep Learning 


### Description:
In today’s time, plant species classification has become a crucial necessity for advancement in the field of medicine and research. Using Convolution Neural Network (CNN) models to train the leaf data enabled not only in faster classification but also reduced the human effort required to a great extent.

![Leaf Image](https://github.com/lakshyakalia/COMP6721-GroupE/blob/main/100%20Images/Dataset2/Mango%20(P0)/0001_0105.JPG)

With regards to the leaf classification, due to various different features evident from distinct plant data, deep learning algorithms provide an edge for the classification issue.
Here, we used three different CNN models, namely, ResNet, GoogleNet and ShuffleNet. The use of these three different CNN models enabled us to reach
a point of higher accuracy and lower loss functions. Image pre-processing and augmentation has also been done prior to feeding the data to the models to enable more robust training of all the models. Random Affine, resizing, rotation and auto-contrast are some of the techniques used for data augmentation.

We also performed optimization by tweaking the hyper-parameters such as batch size and learning rate and recording observations and subsequently making the
model much more streamlined with our classification problem.

In the final stages, we also implemented Transfer Learning which helped in the better performance of the model along with faster training and the reduction of
the overfitting problem.

T-SNE was also used on the ShuffleNet model to visualize the data along with the detection of deviation from the desired results. This enabled us to get a higher
level of understanding of our CNN model and help in optimization.

### Requirements to run python code:



### Instruction on how to train/validate your model:

 1. To run via Google Colab: Change the device type to CUDA.
   2. To Run on MacBook silicon: change device to ops.
   3. While running through Google Colab, make sure to mount google drive and change the path of directory for datasets loading.


### Instructions on how to run the pre-trained model on the provided sample test dataset:
All the codes can be ran from Google Colaboratory. Simply uploda the ipynb file(s) change the url "dataset_url='/content/drive/MyDrive/plant_leaf_dataset/dataset1/'" and run the code.

To use predictor, just change the location of the image to be predicted in the code. Remember to use respective dataset images for respective model predictors.

<b>Note: </b> Predictors names are included with which dataset to predict.

### Description on how to obtain the Dataset from an available download link:
<b>Dataset 1(MalayaKew DATASET):</b>
MalayaKew (MK) Leaf dataset was collected at the Royal Botanic Gardens, Kew, England.
<b>Dataset 2 (Mendeley DataSet):</b>
Chouhan, Siddharth Singh; Kaul, Ajay; Singh, Uday Pratap; Jain, Sanjeev (2019)
<b>Dataset 3:</b>
J, ARUN PANDIAN; GOPAL, GEETHARAMANI (2019), “Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network

[Download From](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9631212)




### Contribution by:

| <p style="text-align: center;">Name</p>           |   <p style="text-align: center;">GitHub Username</p>      |   <p style="text-align: center;">Student ID</p>       |
| ---------------|   --------------------|   ------------|
| <p style="text-align: center;">Karthik Dammu</p>  | <p style="text-align: center;">[KarthikCU1054](https://github.com/KarthikCU1054)</p> | <p style="text-align: center;">40275326</p>  |
| <p style="text-align: center;">MD Tanveer Alamgir</p>  | <p style="text-align: center;">[mdtanveeralamgir](https://github.com/mdtanveeralamgir)</p> | <p style="text-align: center;">40014877</p>  |
| <p style="text-align: center;">Lakshya Kalia</p>  | <p style="text-align: center;">[lakshyakalia](https://github.com/lakshyakalia)</p> | <p style="text-align: center;">40220721</p>  |
| <p style="text-align: center;">Manish Gautam</p>  | <p style="text-align: center;">[manish198](https://github.com/manish198)</p> | <p style="text-align: center;">40191770</p>  |
