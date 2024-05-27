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

## **Model Configuration**
For a detailed description of the model, see [MODELS_GUIDE.md](MODELS_GUIDE.md)

* he model is trained with an optimized number of **epochs** and **batch size** to balance between accuracy and training time. <br>
* Larger epochs provide more opportunities for training, which improves the model's performance, but there is a risk of overfitting, **so use a moderate number of epochs.**
* As the number of epochs increases, the training time for the model also extends. **Therefore, choose the number of epochs with your CPU in mind.**

* Model example : [smile_model](https://drive.google.com/file/d/1m47kNbkW6g-_l7nlMVp8WIoAIuQLmPOF/view?usp=drive_link)
## **Examples**
In this section, we demonstrate how the trained model from `models.py` is used to predict the degree of smiling in images from the CelebA dataset using `generate.py`. The model outputs a score between 0 and 1, indicating the likelihood of the subject smiling.

### Here's how it works: ###
1. **Random Selection**: The script randomly selects 4 images from the CelebA dataset.
2. **Model Prediction**: Each image is processed by the model, which predicts how likely it is that the person in the image is smiling.
3. **Results**: The model outputs a smiling probability score between 0 and 1, which is interpreted as follows:
   - **0.0 to 0.2**: Not Smiling
   - **0.2 to 0.4**: Barely Smiling
   - **0.4 to 0.6**: Moderately Smiling
   - **0.6 to 0.8**: Smiling
   - **0.8 to 1.0**: Highly Smiling

Below are some examples of the outputs:<br>
![Example](https://github.com/limstinger/Smile_Recognition/assets/113160281/9609b61e-6a83-42e2-a3d7-01aafe3b9389)

## **Limitations**


## **License**
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE_FILE_LINK) file for details.

