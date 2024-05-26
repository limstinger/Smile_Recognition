# Smile_Recognition

 ◭ 2024.05  **CV Term Project**  ◭
 
<br>

## **Introduction**
This project uses deep learning and computer vision techniques to recognise smiling faces in images. <br>
It was implemented using **TensorFlow** and **OpenCV** and trained on a large dataset of faces.

## **Developer**
임민규(Lim Mingyu)

## **Release**
**The CelebA dataset** is a public dataset consisting of approximately 200,000 face images. <br>
Each image exists at 178x218 resolution and is organised via binary labels for 40 different facial attributes.<br>
<br>
Source : [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

## **Set Up & Prerequisites**
Build environment:
- [x] Make sure you have installed, `Python>=3.8`
- [x] Clone this repository;
- [x] Run `pip install -r requirements.txt`;


## **Examples**
In this section, we demonstrate how the trained model from `models.py` is used to predict the degree of smiling in images from the CelebA dataset using `generate.py`. The model outputs a score between 0 and 1, indicating the likelihood of the subject smiling.

## **License**
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE_FILE_LINK) file for details.

