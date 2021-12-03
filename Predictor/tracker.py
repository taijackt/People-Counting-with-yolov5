from motrackers import tracker
import numpy as np
import torch

class CustomTracker():
    def __init__(self,maxlost):
        self.tracker = CustomTracker.get_tracker(maxlost)
        

    @staticmethod
    def get_tracker(maxlost):
        return tracker.Tracker(max_lost=maxlost)

    @staticmethod
    def yolobbox2bbox(x,y,w,h):
        x1, y1 = x-w/2, y-h/2
        x2, y2 = x+w/2, y+h/2
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def xyxy2xywh(x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def change_format(preds):
        changed_format = []
        for result in preds:
            x1,y1,x2,y2 = CustomTracker.yolobbox2bbox(result[2],result[3],result[4],result[5])
            id = result[1]
            cls = 0
            changed_format.append([x1,y1,x2,y2,id,cls])
        return changed_format

    def update(self, preds):
        preds_xywhs = CustomTracker.xyxy2xywh(preds[:,:4])
        tracks = self.tracker.update(bboxes=preds_xywhs, detection_scores=preds[:,4],class_ids=preds[:,5])
        tracks = np.asarray(CustomTracker.change_format(tracks))
        return tracks
