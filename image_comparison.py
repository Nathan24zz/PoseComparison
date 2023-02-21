# USAGE : python3 image_comparison.py --image1 image/sit.jpg --image2 image/standup.jpg

import argparse
import cv2
import numpy as np
import tensorflow as tf

from pose import Pose
import posenet
from score import Score


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i1", "--image1", required=True,
        help="image file to be scored")
    ap.add_argument("-i2", "--image2", required=True,
        help="image file to be scored")
    args = vars(ap.parse_args())
    
    a = Pose()
    s = Score()
    image_1_points = []
    image_2_points = []

    with tf.compat.v1.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        
        img1 = cv2.imread(args['image1'], cv2.IMREAD_COLOR)
        img2 = cv2.imread(args['image2'], cv2.IMREAD_COLOR)
       
        input_points = a.getpoints(img1, sess, model_cfg, model_outputs)
        input_new_coords = np.asarray(a.roi(input_points)[0:34]).reshape(17,2)
        image_1_points.append(input_new_coords)
        
        input_points = a.getpoints(img2, sess, model_cfg, model_outputs)
        input_new_coords = np.asarray(a.roi(input_points)[0:34]).reshape(17,2)
        image_2_points.append(input_new_coords)

        final_score, score_list = s.compare(np.asarray(image_1_points),np.asarray(image_2_points),1,1)
        print("Total Score : ",final_score)
        print("Score List : ",score_list)
        print("Body Pose Average: ", sum(score_list[5:])/len(score_list[5:]))

if __name__ == '__main__':
    main()
