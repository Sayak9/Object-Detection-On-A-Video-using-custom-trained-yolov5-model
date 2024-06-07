import pathlib

import torch
import numpy as np
import cv2

from pathlib import WindowsPath as Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a local video using OpenCV.
    """

    def __init__(self, video_path, model_name, out_file):
        """
        Initializes the class with video path and model name.
        :param video_path: Path to the local video file.
        :param model_name: Path to the Yolo5 model file.
        """
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)
        self.cap = cv2.VideoCapture(video_path)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        print(type(Path(model_name)))

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=Path(model_name), force_reload=True)
        pathlib.PosixPath = temp
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        process the frames using the Yolo5 model, and write the output video with bounding boxes.
        :return: void
        """

        # Define video codec and output filename (adjust as needed)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # Use directly provided width and height
        frame_size = (640, 640)
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, frame_size)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            cv2.imshow("img", frame)

            # Write processed frame to output video
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()


detection = ObjectDetection(video_path='model/cam7.avi', model_name='model/botbest.pt',
                            out_file='video.avi')
detection()