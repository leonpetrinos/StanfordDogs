# Stanford Dogs Machine Learning Project

### Description
This is a machine learning project for both classification and regression tasks. The classification task involves identifying dog breeds based on images, while the regression task involves locating the center of a dog in an image. We employ three techniques: Linear Regression, Logistic Regression, and K-Nearest Neighbors (KNN). All images used in this project come from the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). Here is the [link to the report](https://github.com/leonpetrinos/StanfordDogs/blob/main/report.pdf), which provides an analysis of our results.

### How to Run the Project
To clone the repository, type:
```bash
git clone https://github.com/leonpetrinos/StanfordDogs.git
```
To train the models on the dataset, navigate to the directory where the main.py file is located. Here are examples of commands to run the various methods:
#### Linear Regression
```bash
python main.py --method linear_regression --lambda 0 --task center_locating
```

#### Logistic Regression
```bash
python main.py --method logistic_regression --lr 1e-3 --max_iters 500 --task breed_identifying
```

#### K-Nearest Neighbors (KNN)
```bash
python main.py --method knn --K 17 --task <either center_locating or breed_identifying>
```
