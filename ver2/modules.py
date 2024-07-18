import cv2 as cv
import numpy as np
import os
import time
import mediapipe as mp
import tensorflow
import customtkinter

from matplotlib import pyplot as plt
from tkinter import *
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score