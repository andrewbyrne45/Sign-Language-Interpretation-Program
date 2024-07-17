import cv2 as cv
import numpy as np
import math
import time
import customtkinter

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from tkinter import *

# if label is 0, it will be A. And if label is 1, it will be B etc..
# maybe try to import 'labels.txt' as an easier means, rather than
# manually entering labels
labels = ["A", "B", "C"]