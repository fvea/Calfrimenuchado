# Calfrimenuchado: Kaldereta, Afritada, Menudo and Mechado (Image Classifier)

Calfrimenuchado is a personal project about developing a machine-learning model that classifies tomato-based Filipino dishes (Kaldereta, Afritada, Menudo, and Mechado) that are commonly confused with one another by Filipinos. 

## Problem
Menudo, Afritada, Mechado, and Kaldereta are popular Filipino dishes that are sometimes confused with one another by Filipinos due to their similar appearance and some overlapping ingredients. In some instances, the names may also be used interchangeably, adding to the confusion. The infographic by Monterey below summarizes the key differences between the four dishes. As can be seen, these tomato-based dishes have a too-similar set of ingredients making them hard to distinguish from each other by eye.

![image](https://github.com/fvea/Calfrimenuchado/assets/75005859/f77e1101-3aea-4aab-aa99-ab2fbd003f72)

This puzzling problem has become the inspiration for this project. I have thought of a solution to utilize Deep Computer Vision using CNN to automatically classify the images of these four dishes. With this, I tried to develop a machine-learning model that classifies the four dishes using web-scraped images and pre-trained image classifier models.

## Dataset
I frame the problem as a supervised classification task in which a CNN-based Deep Learning model classifies images of Menudo, Afritada, Mechado, and Kaldereta. With that in mind, I collected images of the four dishes from the internet using image web scraping tools such as [Fatkun](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/efcapamiilmdfbbilogcddbdckjhpajj). The image dataset is available publicly at [Roboflow](https://app.roboflow.com/my-projects-iht7n/calfrimenuchado/1).

![image](https://github.com/fvea/Calfrimenuchado/assets/75005859/9841b5d1-806c-490d-acdb-4b6fd67cf2c1)

## Training and Evaluation
For this project, I utilized pre-trained image classifier models such as Xception, Resnet50, Inception, Mobilenet, and EfficientNet, which provide rich feature representations that can be leveraged through transfer learning, adapting the model to specific tasks with less data (as for the current case). 

The training results are available publicly on [Tensorboard](https://tensorboard.dev/experiment/3bkwjlT5QFOjLyXsXvxODA/#scalars&runSelectionState=eyJydW5fMjAyMl8wOF8xOS0xMV8yOV8xOF9tb2JpbGVuZXR2Mi90cmFpbiI6ZmFsc2UsInJ1bl8yMDIyXzA4XzE5LTExXzI5XzE4X21vYmlsZW5ldHYyL3ZhbGlkYXRpb24iOmZhbHNlLCJydW5fMjAyMl8wOF8xOS0xMV80NF80MV9lZmZpY2llbnRuZXRiMi90cmFpbiI6ZmFsc2UsInJ1bl8yMDIyXzA4XzE5LTExXzQ0XzQxX2VmZmljaWVudG5ldGIyL3ZhbGlkYXRpb24iOmZhbHNlLCJydW5fMjAyMl8wOF8xOS0xMl8wMV80M194Y2VwdGlvbi90cmFpbiI6ZmFsc2UsInJ1bl8yMDIyXzA4XzE5LTEyXzAxXzQzX3hjZXB0aW9uL3ZhbGlkYXRpb24iOmZhbHNlLCJydW5fMjAyMl8wOF8xOS0xMl8zMl80Nl9yZXNuZXQ1MC90cmFpbiI6ZmFsc2UsInJ1bl8yMDIyXzA4XzE5LTEyXzMyXzQ2X3Jlc25ldDUwL3ZhbGlkYXRpb24iOmZhbHNlLCJydW5fMjAyMl8wOF8xOS0xMl81Ml81M19pbmNlcHRpb252My90cmFpbiI6ZmFsc2UsInJ1bl8yMDIyXzA4XzE5LTEyXzUyXzUzX2luY2VwdGlvbnYzL3ZhbGlkYXRpb24iOmZhbHNlLCJydW5fMjAyMl8wOF8xOS0xM181NV80OF9tb2JpbGVuZXR2Ml8xLjAwXzIyNF9maW5lX3R1bmVkL3RyYWluIjpmYWxzZSwicnVuXzIwMjJfMDhfMTktMTNfNTVfNDhfbW9iaWxlbmV0djJfMS4wMF8yMjRfZmluZV90dW5lZC92YWxpZGF0aW9uIjpmYWxzZSwicnVuXzIwMjJfMDhfMTktMTRfMDNfMzFfZWZmaWNpZW50bmV0YjJfZmluZV90dW5lZC90cmFpbiI6ZmFsc2UsInJ1bl8yMDIyXzA4XzE5LTE0XzAzXzMxX2VmZmljaWVudG5ldGIyX2ZpbmVfdHVuZWQvdmFsaWRhdGlvbiI6ZmFsc2UsInJ1bl8yMDIyXzA4XzE5LTE0XzI0XzExX3hjZXB0aW9uX2ZpbmVfdHVuZWQvdHJhaW4iOmZhbHNlLCJydW5fMjAyMl8wOF8xOS0xNF8yNF8xMV94Y2VwdGlvbl9maW5lX3R1bmVkL3ZhbGlkYXRpb24iOnRydWUsInJ1bl8yMDIyXzA4XzE5LTE0XzQ0XzI4X3Jlc25ldDUwX2ZpbmVfdHVuZWQvdHJhaW4iOnRydWUsInJ1bl8yMDIyXzA4XzE5LTE0XzQ0XzI4X3Jlc25ldDUwX2ZpbmVfdHVuZWQvdmFsaWRhdGlvbiI6ZmFsc2UsInJ1bl8yMDIyXzA4XzE5LTE1XzE3XzAwX2luY2VwdGlvbl92M19maW5lX3R1bmVkL3RyYWluIjpmYWxzZSwicnVuXzIwMjJfMDhfMTktMTVfMTdfMDBfaW5jZXB0aW9uX3YzX2ZpbmVfdHVuZWQvdmFsaWRhdGlvbiI6ZmFsc2V9). Among the highest performance model is the Resnet50 model with observed accuracy of 83% in the validation dataset. The model's training curves (with fine-tuning) and performance metrics on the validation data are shown in the table below.

|   ResNet50 Training Curves    |    ResNet50 Performance    |
|--------------|--------------|
|  ![image](https://github.com/fvea/Calfrimenuchado/assets/75005859/b4ce75e6-422f-470d-830b-b1477325a0e4)  | ![image](https://github.com/fvea/Calfrimenuchado/assets/75005859/3cd5da24-48cb-4f67-a037-d894ac5cd537)  |

## Sample Classification

![image](https://github.com/fvea/Calfrimenuchado/assets/75005859/a51bcd55-f243-48fc-9df5-c58158619447)



