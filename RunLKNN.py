"""
Adapted from "The Implementation of Optical Flow in Neural Networks", Nicole Ku'ulei-lani Flett
http://nrs.harvard.edu/urn-3:HUL.InstRepos:39011510
"""

import numpy as np
from keras.models import model_from_yaml
import cv2
import imageio


def loadNN(model_file, weights_file):
    yaml_file = open(model_file, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")
    return loaded_model


def LK(frame1, frame2, model):
    img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    nn_input = np.concatenate((img1, img2), axis=0)
    nn_input.resize((28, 28))
    nn_input = np.reshape(nn_input, (1, 28, 28, -1))
    nn_output = model.predict(nn_input)
    print(nn_output)
    LKvector = [nn_output[0][1]-nn_output[0][0], nn_output[0]
                [3] - nn_output[0][2], nn_output[0][5]-nn_output[0][4]]
    print(LKvector)
    return LKvector


def run_network(frame_list, x, y, w, h):
    recording = []
    mod = loadNN('model.yaml', 'model.h5')
    flow_vector = [0, 0, 0, 0, 0, 0]
    for x in range(len(frame_list) - 1):
        first_frame = frame_list[x]
        second_frame = frame_list[x+1]
        flow_vector = LK((imageio.imread(first_frame)[
            :, :, :3].astype(np.float32) / 255.0)[int(y+flow_vector[2]+flow_vector[3]):int(y+flow_vector[2] +
                                                                                           flow_vector[3]+h), int(x+flow_vector[0]+flow_vector[1]):int(x+flow_vector[0]+flow_vector[1]+w)], (imageio.imread(second_frame)[
                                                                                               :, :, :3].astype(np.float32) / 255.0)[int(y+flow_vector[2]+flow_vector[3]):int(y+flow_vector[2] +
                                                                                                                                                                              flow_vector[3]+h), int(x+flow_vector[0]+flow_vector[1]):int(x+flow_vector[0]+flow_vector[1]+w)], mod)
        recording.append(flow_vector)
    return recording
