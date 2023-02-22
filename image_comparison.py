# USAGE : python3 image_comparison.py --image1 image/stand.jpg --image2 image/1079.jpg

import argparse
import cv2
import numpy as np
import tensorflow as tf

import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator

from pose import Pose
import posenet
from score import Score


def reduce_keypoints_posenet(num_keypoints):
    pass

def load_human_posenet_model():
    with tf.compat.v1.Session() as sess:
        return posenet.load_model(101, sess)
    
def human_posenet_detection(image, image_points, model_cfg, model_outputs):
    a = Pose()

    with tf.compat.v1.Session() as sess:
        # model_cfg, model_outputs = posenet.load_model(101, sess)
        
        input_points = a.getpoints(image, sess, model_cfg, model_outputs)
        input_for_viz = np.array(input_points[0:34]).reshape(17,2)
        image_points.append(input_for_viz)
        # print(np.asarray(image_points).shape)
        # print(np.asarray(image_points))
        visualize(image, keypoints=np.array(image_points))
        cv2.imwrite('image_1.jpg', image)
        # reinitialize image_points
        image_points = []
        input_new_coords = np.asarray(a.roi(input_points)[0:34]).reshape(17,2)
        image_points.append(input_new_coords)
        

def load_robot_rcnn_model():
    device = torch.device('cpu')
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model_ = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                    pretrained_backbone=True,
                                                                    num_keypoints=6,
                                                                    num_classes = 2, # Background is the first class, object is the second class
                                                                    rpn_anchor_generator=anchor_generator)

    # load the model checkpoint
    with torch.no_grad():
        model_.load_state_dict(torch.load('/drive0-storage/robot_pose/keypoint_rcnn_training_pytorch-main/keypointsrcnn_weights_no_rotation.pth'))
        model_.to(device)
        model_.eval()
    return model_

def robot_rcnn_detection(image, model):
    device = torch.device('cpu')
    with torch.no_grad():
        orig_frame = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0).to(device)
        output = model(image)

    scores = output[0]['scores']
    # scores
    high_scores_idxs = np.where(scores > 0.55)[0].tolist() # Indexes of boxes with scores > 0.7
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])

    bboxes = []
    for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))

    visualize(orig_frame, bboxes, keypoints)
    cv2.imwrite('image_2.jpg', orig_frame)
    
        
def visualize(image, bboxes=None, keypoints=None):
    # keypoints_classes_ids2names = {0: 'Head', 1: 'Trunk', 2: 'RH', 3: 'LH', 4: 'RF', 5: 'LF'}
    # keypoints_classes_ids2names = {0: 'NOSE', 1: 'LEFT_EYE', 2: 'RIGHT_EYE', 3: 'LEFT_EAR', 4: 'RIGHT_EAR', 5: 'LEFT_SHOULDER', 6:'RIGHT_SHOULDER', 7:'LEFT_ELBOW', 8:'RIGHT_ELBOW', 9:'LEFT_WRIST', 10:'RIGHT_WRIST', 11:'LEFT_HIP', 12:'RIGHT_HIP' ,13:'LEFT_KNEE', 14:'RIGHT_KNEE',15:'LEFT_ANKLE', 16:'RIGHT_ANKLE'}
    
    if bboxes is not None:
        for bbox in bboxes:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image = cv2.rectangle(image, start_point, end_point, (0,255,0), 2)

    if keypoints is not None:
        for kps in keypoints:
            for idx, kp in enumerate(kps):
                kp = tuple(int(i) for i in kp)
                
                # cv2.circle(image, center_coordinates, radius, color, thickness)
                image = cv2.circle(image, kp, 4, (255,0,0), -1)
                # cv2.putText(image, text, org, font, fontScale, color, thickness, lineType, bottomLeftOrigin)
                # image = cv2.putText(image, " " + keypoints_classes_ids2names[idx], kp, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i1", "--image1", required=True,
        help="image file to be scored")
    ap.add_argument("-i2", "--image2", required=True,
        help="image file to be scored")
    args = vars(ap.parse_args())

    s = Score()
    image_1_points = []
    image_2_points = []
    
    # load model
    model_cfg, model_outputs = load_human_posenet_model()
    robot_model = load_robot_rcnn_model()
    
    # looping section if we want real time
    #################################### 
    img1 = cv2.imread(args['image1'], cv2.IMREAD_COLOR)
    img2 = cv2.imread(args['image2'], cv2.IMREAD_COLOR)

    human_posenet_detection(img1, image_1_points, model_cfg, model_outputs)
    robot_rcnn_detection(img2, robot_model)
#     human_posenet_detection(img2, image_2_points, model_cfg, model_outputs)

#     final_score, score_list = s.compare(np.asarray(image_1_points),np.asarray(image_2_points),1,1)
#     print("Total Score : ",final_score)
#     print("Score List : ",score_list)
#     print("Body Pose Average: ", sum(score_list[5:])/len(score_list[5:]))
    #####################################

if __name__ == '__main__':
    main()
