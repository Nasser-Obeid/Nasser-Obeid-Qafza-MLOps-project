# Nasser-Obeid-Qafza-MLOps-project
This project was made as an exercise for the Qafza MLOps technical training. the main purpose of this project is to showcase how Deep learning can be used as an every day application.

## Table of Contents
1. [Overview](#Overview)
2. [Requirements](#Requirements)
3. [Setup](#Setup)
4. [Docker](#Docker)
5. [Links](#Links)
6. [Notes](#Notes)



## Overview
this is a basic web application that classifies the images of fruits (more about the dataset in the links section) using CNNs in the backend and using FastAPI as backend. it makes request by allowing the user to upload the image of the fruit, then the backend receives the image and processes it through the neural network and returns the label that it predicts to the user.

## Requirements
- Python 3.10.12 or higher.
- pip 22.0.2 or higher
- the dependencies in the requirements.txt file.

## Setup
To run locally, make sure that the system requirements are available, simply create a virtual enviroment in the root of the project folder and activate is, then simply install all the dependincies in the requirements.txt file by simply running "pip install -r requirements.txt", finally, run the main.py script and access the web page on the port that the app is hosted on.

## Docker
optionally, instead of using this repository, you can find a fully setup Docker image of the project on Docker Hub, for more info, check the links section below. 

## Links
- Dataset: https://www.kaggle.com/datasets/sshikamaru/fruit-recognition
- Docker Hub: https://hub.docker.com/r/nasserobeid/fruitclassifier/

## Notes
when using the docker image, make sure that the container port used when running the image is set to 8000, the reason that the port of the container is hardcoded to 8000 as showcased in this repository( this to be changed later). as an example, if you desired to run the image on port 2333 for example, simply run: "docker run -p 2333:8000 fruitclassifier".

