import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Bool
import subprocess
import time

class BottleFinderNode(Node):
    def __init__(self):
        super().__init__('bottle_finder_node')
        
        # set up an ROS publisher for when the bottle is found
        self.bottle_detected_pub = self.create_publisher(Bool, 'bottle_detected', 10)

        # set up video capture of camera feed
        self.video_captured = cv2.VideoCapture(0)

        # give notification if didn't work
        if not self.video_captured.isOpened():
            print("Error: Could not find Camera.")
            exit(1)
        print("Camera loaded.")

        # setup yolo using weights from darknet
        print("loading weights and configuration into Yolo...")
        self.yolo_neural_net = cv2.dnn.readNet("yolo/yolov4.weights", "yolo/yolov4.cfg")
        
        #load label names
        self.coco_label_name_classes = []
        with open("yolo/coco.names", "r") as file:
            # remove white spaces while loading names
            self.coco_label_name_classes = [line.strip() for line in file.readlines()] 
        names_of_layers = self.yolo_neural_net.getLayerNames() # load layer names
        output_layers = self.yolo_neural_net.getUnconnectedOutLayers() # get output laters of the yolo neural net
        output_layers_1d = output_layers.flatten() # convert to 1d array
        self.output_layers = [names_of_layers[i - 1] for i in output_layers_1d] # store the names of all layers according to the indecies stored in output layers
        print("Detection model loaded successfully.")

    def detect_objects(self):
        print("Starting detection...")
        while True:
            # read the video capture
            read_successful, frame = self.video_captured.read()
            if not read_successful:
                print("Failed to grab frame")
                break

            # convert the image to a blob 
            # blob is the format the neural network accepts, like a token for a LLM
            # normalisation stats based on findins here https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html 
            # 1/255 is the scale factor used because pixles range from 0 to 255
            # image size of 416 by 416
            # 0,0,0 is used because we dont want to subtract any mean values from RGB
            # swap RB is true because openCV opens images in BGR and we want RGB
            # don't crop
            input_blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)

            # input blob into net
            self.yolo_neural_net.setInput(input_blob)

            # do a forward pass on the neural net
            # thus process the image data and produce a prediction
            outputs = self.yolo_neural_net.forward(self.output_layers)
            for output in outputs:
                for detected_object in output:
                    # get the confidence scores for different predictions
                    confidence_scores = detected_object[5:]
                    # get the index of the most likely prediction 
                    index_of_prediction = np.argmax(confidence_scores)
                    # get the score at this index
                    confidence_score = confidence_scores[index_of_prediction]
                    if confidence_score > 0.5:
                        # print prediction of detection
                        print(f"Detected {self.coco_label_name_classes[index_of_prediction]} with confidence score {confidence_score:.2f}")
                        if self.coco_label_name_classes[index_of_prediction] == "bottle":
                            # if the bottle is detected save the image
                            print("Bottle successfully detected! Saving imalast frame and sending signal to stop mini pupper!")
                            cv2.imwrite("detected_the_bottle.jpg", frame)
                            # stop the video and show the detected bottle
                            self.stop_video()
                            self.open_saved_image("detected_the_bottle.jpg")
                            # publish that the bottle is detected to ROS
                            self.bottle_detected_pub.publish(Bool(data=True))
                            # detection is finished
                            return
            # show the latest frame
            cv2.imshow("Image", frame)
            # can quit frame stream with esc key
            key = cv2.waitKey(1)
            if key == 27:  
                self.stop_video()
                break
        
    def stop_video(self):
        # stop the camera feed
        print("Stopping Camera Live Stream...")
        self.video_captured.release()
        cv2.destroyAllWindows()

    def open_saved_image(self, image_path):
        # open the saved bottle image
        subprocess.run(["xdg-open", image_path], check=True)

def main(args=None):
    # initialise ROS2 (client library)
    rclpy.init(args=args)
    # setup bottle finder node
    bottle_finder_node = BottleFinderNode()
    # run detect objects method
    bottle_finder_node.detect_objects()
    # run ROS2 callbacks
    rclpy.spin_once(bottle_finder_node)  
    # shutdown client library and destroy node
    bottle_finder_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
