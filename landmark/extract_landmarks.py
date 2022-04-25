import numpy as np
import scipy.io
import dlib
import matplotlib.pyplot as plt
import imageio


def get_landmark_array(landmarks_dlib_obj):
    # Gets a DLIB landmarks object and returns a [68,2] numpy array with the 
    # landmark coordinates.
    
    landmark_array = np.zeros([68,2])
    
    for i in range(68):
        landmark_array[i,0] = landmarks_dlib_obj.part(i).x
        landmark_array[i,1] = landmarks_dlib_obj.part(i).y
        
    return landmark_array  # numpy array

plt.close('all')

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat") 
image = imageio.imread('../images/test16.jpg')  # load image

faces = detector(image)  # detect faces

for i, face in enumerate(faces):

    landmarks_raw = predictor(image, face)  # detect landmarks

    landmarks = get_landmark_array(landmarks_raw)

    print(landmarks.shape) # 68 landmarks
    # landmarks_frontal = frontalize_landmarks(landmarks, frontalization_weights)
    scipy.io.savemat('../test.mat', mdict={'arr': landmarks})
    
