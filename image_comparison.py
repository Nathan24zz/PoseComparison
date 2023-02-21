# USAGE : python3 image_comparison.py --image1 image/sit.jpg --image2 image/standup.jpg

import argparse
import cv2
import numpy as np
import tensorflow as tf

from pose import Pose
import posenet
from score import Score


def visualize(image, bboxes=None, keypoints=None):
    # keypoints_classes_ids2names = {0: 'Head', 1: 'Trunk', 2: 'RH', 3: 'LH', 4: 'RF', 5: 'LF'}
    keypoints_classes_ids2names = {0: 'NOSE', 1: 'LEFT_EYE', 2: 'RIGHT_EYE', 3: 'LEFT_EAR', 4: 'RIGHT_EAR', 5: 'LEFT_SHOULDER', 6:'RIGHT_SHOULDER', 7:'LEFT_ELBOW', 8:'RIGHT_ELBOW', 9:'LEFT_WRIST', 10:'RIGHT_WRIST', 11:'LEFT_HIP', 12:'RIGHT_HIP' ,13:'LEFT_KNEE', 14:'RIGHT_KNEE',15:'LEFT_ANKLE', 16:'RIGHT_ANKLE'}
    
    if bboxes is not None:
        for bbox in bboxes:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)

    if keypoints is not None:
        for kps in keypoints:
            for idx, kp in enumerate(kps):
                kp = tuple(int(i) for i in kp)
                
                # cv2.circle(image, center_coordinates, radius, color, thickness)
                image = cv2.circle(image, kp, 4, (255,0,0), -1)
                # cv2.putText(image, text, org, font, fontScale, color, thickness, lineType, bottomLeftOrigin)
                image = cv2.putText(image, " " + keypoints_classes_ids2names[idx], kp, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)


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
        input_for_viz = np.array(input_points[0:34]).reshape(17,2)
        image_1_points.append(input_for_viz)
        visualize(img1, keypoints=np.array(image_1_points))
        cv2.imwrite('image_1.jpg', img1)
        # reinitialize image_1_points
        image_1_points = []
        input_new_coords = np.asarray(a.roi(input_points)[0:34]).reshape(17,2)
        image_1_points.append(input_new_coords)


        input_points = a.getpoints(img2, sess, model_cfg, model_outputs)
        input_for_viz = np.array(input_points[0:34]).reshape(17,2)
        image_2_points.append(input_for_viz)
        visualize(img2, keypoints=np.array(image_2_points))
        cv2.imwrite('image_2.jpg', img2)
        # reinitialize image_2_points
        image_2_points = []
        input_new_coords = np.asarray(a.roi(input_points)[0:34]).reshape(17,2)
        image_2_points.append(input_new_coords)

        final_score, score_list = s.compare(np.asarray(image_1_points),np.asarray(image_2_points),1,1)
        print("Total Score : ",final_score)
        print("Score List : ",score_list)
        print("Body Pose Average: ", sum(score_list[5:])/len(score_list[5:]))

if __name__ == '__main__':
    main()
