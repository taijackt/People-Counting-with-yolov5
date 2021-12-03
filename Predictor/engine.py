import cv2
import yaml
import torch
import time
from .tracker import CustomTracker
from .person import Person
from datetime import datetime

def count_center(x1,y1,x2,y2):
    center_x = x1+int((x2-x1)/2)
    center_y = y1+int((y2-y1)/2)
    return (center_x, center_y)

def read_config(config_path="./config.yaml"):
    with open(config_path) as config:
        configs = yaml.load(config, Loader = yaml.FullLoader)
    return configs

class Predictor():
    def __init__(self, configs):
        self.configs =configs

        self.cam_src = self.configs["camera_src"]

        self.target_ID = self.configs["detection"]["person_classID"]
        self.target_threshold = self.configs["detection"]["person_threshold"]

        self.tracker = CustomTracker(maxlost = self.configs["tracking"]["maxlost"])

        self.model = Predictor.get_model()

        self.cam = cv2.VideoCapture(self.cam_src)

        self.total_error = 0

        self.peopleDict = {}

        self.totalIn = 0
        self.totalOut = 0

        self.line_y = int(700*self.configs["line"]["y_coord"])
        self.foot_line_y = int(700*self.configs["line"]["foot_line"])

    @staticmethod
    def get_model(type="l"):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True, force_reload=False)
        return model

    def __print_log(self, message):
        time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        print(f"({time}) : {message}")

    def __preprocess(self, frame):
        return cv2.resize(frame, (1280,700))


    def __postprocess(self, preds):
        persons  = preds[preds[:,5]==self.target_ID]
        persons = persons[persons[:,4]>self.target_threshold].detach().cpu().numpy()
        return persons

    def __handle_get_frame_error(self):
        self.total_error +=1
        if self.total_error>10000:
            self.__print_log("Reconnecting to the camera")
            self.cam = cv2.VideoCapture(self.cam_src)
            self.total_error = 0

    def __draw(self, frame, personTracks):
        cv2.putText(frame, "Person In: "+str(self.totalIn), (15,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2, cv2.LINE_AA)
        cv2.putText(frame, "Person Out: "+str(self.totalOut), (15,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2, cv2.LINE_AA)
        cv2.line(frame, (0, self.line_y), (1280, self.line_y), (255,255,255),1)
        cv2.line(frame, (0, self.foot_line_y), (1280, self.foot_line_y), (255,255,255),1)

        for person in personTracks:
            cv2.rectangle(frame, (person[0],person[1]),(person[2],person[3]),(0,255,0),1)


    def __clean_person_object(self):
        key2del = []
        for key, person in self.peopleDict.items():
            now = datetime.now()
            if (now - person.lastAppearTime).seconds > 30 and person.isCounted:
                key2del.append(key)
            elif (now - person.lastAppearTime).seconds > 60 and not (person.isCounted):
                key2del.append(key)

        for key in key2del:
            self.print_log(f"{key} is deleted.")
            del self.peopleDict[key]
    
    
    def __update_person(self, personTracks):
        for person in personTracks:
            if not (person[4] in self.peopleDict.keys()):
                self.peopleDict[person[4]] = Person(id=person[4],
                                                center=count_center(person[0],person[1],person[2],person[3]),
                                                foot=person[3]
                                            )

            targetDict = self.peopleDict[person[4]]

            # update appear time
            targetDict.appear()

            # comapare
            now_center_y = count_center(person[0], person[1],person[2],person[3])[1]
            now_foot = person[3]

            if not targetDict.isCounted:
                # up to down -> in
                if now_center_y > self.line_y > targetDict.center[1] and now_foot>=self.foot_line_y:
                    targetDict.isCounted = True
                    targetDict.status = "in"
                    self.totalIn+=1

                # down to up -> out
                elif now_center_y < self.line_y < targetDict.center[1] and targetDict.foot >= self.foot_line_y:
                    targetDict.isCounted = True
                    targetDict.status = "out"
                    self.totalOut+=1
                    
    def run(self):
        while True:
            try:
                ret, frame = self.cam.read()
                if not ret:
                    self.__handle_get_frame_error()
                    time.sleep(3)
                    continue

                frame = self.__preprocess(frame)

                preds = self.model(frame).xyxy[0]

                preds = self.__postprocess(preds)

                personTracks = self.tracker.update(preds)

                self.__update_person(personTracks)

                self.__draw(frame, personTracks)

                cv2.imshow("test", frame)
                cv2.waitKey(1)

                self.__clean_person_object
            except Exception as e:
                print(e)
                continue