from datetime import datetime
import cv2
import os

class Person():
    def __init__(self, id, center, foot):
        self.trackId = id;
        self.center = center
        self.foot = foot

        self.status = None

        self.isCounted = False

        self.lastAppearTime = None;

    def appear(self):
        self.lastAppearTime = datetime.now()