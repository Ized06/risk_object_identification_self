import os
import os.path as osp
import sys
import time
import cv2
import json
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

sys.path.insert(0, '../../../')
import config as cfg
import utils as utl
from models.gcn import GCN_intention as Model
import pdb

from utils.bounding_box_utils import iou

def to_device(x, device):
    return x.unsqueeze(0).to(device)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default='1', type=str)
    parser.add_argument('--inputs', default='camera', type=str)
    parser.add_argument('--cause', default='crossing_vehicle', type=str)
    parser.add_argument('--model',
                        default='./snapshots/crossing_vehicle/2020-2-13_175803_result_w_dataAug/inputs-camera-epoch-1.pth',
                        type=str)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--partial_conv', default=True, type=bool)
    parser.add_argument('--fusion', default='avg',choices=['avg', 'gcn', 'attn'], type=str)
    parser.add_argument('--hidden_size', default=2000, type=int)
    parser.add_argument('--camera_feature', default='resnet50_mapillary', type=str)
    parser.add_argument('--enc_steps', default=3, type=int)
    parser.add_argument('--dec_steps', default=5, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num-waypoints',type=int,default=13)
    parser.add_argument('--num-topologies',type=int,default=5)
    parser.add_argument('--num-intention_types',type=int,default=4)
    parser.add_argument('--dataset',type=str,default='HDD')

    args = cfg.parse_args(parser)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    dir_name = os.path.dirname(args.model)
    model_basename = os.path.basename(args.model)
    print(dir_name)
    print(model_basename)
    model = Model(args.inputs, partialConv = args.partial_conv,fusion = args.fusion,args=args).to(device)
    state_dict = torch.load(args.model)
    state_dict_copy = {}
    for key in state_dict.keys():
        state_dict_copy[key[7:]] = state_dict[key]
    model.load_state_dict(state_dict_copy)
    model.train(False)
    softmax = nn.Softmax(dim=1).to(device)

    camera_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5],),
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])


    time_steps = 3
    time_sample = 10
    visualize = False

    def plot_vel(pred, target, plot_name):
        t = len(pred)
        timestamp = range(1, t+1)
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(timestamp, pred, marker='o', label="Prediction")
        ax.plot(timestamp, target, marker='o', label="Target")
        # Place a legend to the right of this smaller subplot.
        ax.legend(loc='upper right')
        fig.savefig(plot_name)  # save the figure to file
        plt.close(fig)

    def normalize_box( trackers, width, height):
        normalized_trackers = trackers.copy()
        normalized_trackers[:, :, 3] = normalized_trackers[:, :, 1] + normalized_trackers[:, :, 3]
        normalized_trackers[:, :, 2] = normalized_trackers[:, :, 0] + normalized_trackers[:, :, 2]


        tmp = normalized_trackers[:, :, 0] / width
        normalized_trackers[:, :, 0] = normalized_trackers[:, :, 1] /height
        normalized_trackers[:, :, 1] = tmp
        tmp = trackers[:, :, 2] / width
        normalized_trackers[:, :, 2] = normalized_trackers[:, :, 3] / height
        normalized_trackers[:, :, 3] = tmp

        return normalized_trackers

    def find_tracker(tracking, start, end):

        width = 1280
        height = 720

        # t_array saves timestamps
        t_array = tracking[:, 0]
        tracking_index = tracking[np.where(t_array == end)[0],1]
        num_object = len(tracking_index)

        trackers = np.zeros([int((end-start)/10+1), num_object+1, 4])  # Tx(N+1)x4
        trackers[:, 0, :] = np.array([ 0.0, 0.0, width, height])  # Ego bounding box


        for t in range(start, end+1, time_sample):
            current_tracking = tracking[np.where(t_array == t)[0]]
            for i, object_id in enumerate(tracking_index):
                if object_id in current_tracking[:,1]:
                    bbox = current_tracking[np.where(current_tracking[:, 1] == object_id)[0], 2:6]
                    bbox[:, 0] = np.clip(bbox[:, 0], 0, 1279)
                    bbox[:, 2] = np.clip(bbox[:, 0]+bbox[:, 2], 0, 1279)
                    bbox[:, 1] = np.clip(bbox[:,1], 0, 719)
                    bbox[:, 3] = np.clip(bbox[:, 1]+bbox[:, 3], 0, 719)
                    trackers[ int((t-start)/10), i+1,  :] = bbox

        trackers.astype(np.int32)
        normalized_trackers = normalize_box(trackers, width, height)
        return trackers, normalized_trackers

    def visualize_result(frame_id, tracker, filename, gt):
        width =1280.0
        height = 720.0
        camera_name = 'output{}.png'.format(frame_id)
        camera_path = osp.join('/home/cli/sm120145_data/HDD/ScaleAPI_Cause/Cause_images', folder, camera_name)
        frame = cv2.imread(camera_path)
        box = tracker[-1]#x1,y1,x2,y2

        gt_x1 = (gt[0]-0.5*gt[2])*width
        gt_x2 = (gt[0]+0.5*gt[2])*width
        gt_y1 = (gt[1]-0.5*gt[3])*height
        gt_y2 = (gt[1]+0.5*gt[3])*height

        cv2.rectangle(frame, (int(gt_x1), int(gt_y1)), (int(gt_x2), int(gt_y2)), (0, 0, 255), 8)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3 )
        cv2.imwrite(filename+'_'+frame_id+'.png', frame)


    with open(os.path.join('../../../data','HDD_'+args.cause+'_test.txt')) as f:

        all_test = f.readlines()
    all_test = [x.strip() for x in all_test]
    print("Cause: {}, length: {}".format(args.cause,len(all_test)))
    accumIOU = 0.
    Accs = np.zeros(10)
    threshHolds = np.arange(0.5, 1.0, 0.05)
    vis_save_path = './vis/'+args.cause
    result_dict = {}
    if not os.path.isdir(vis_save_path):
        os.makedirs(vis_save_path)

    for cnt, test_sample in enumerate(all_test):
        with torch.set_grad_enabled(False):
            #test_sample = '201706061309 201706061309_005_cauese_congestion_49965_50099 49991 0.58984375 0.5944444444444444 0.29375 0.3638888888888889 1'
            #test_sample = '201706061536 201706061536_020_cauese_crossing_pedestrian_33411_33582 33497 0.14765625 0.55625 0.0359375 0.1486111111111111 1'
            session, folder, frame_id, center_x, center_y, w, h, _ = test_sample.split(' ')
            center_x = float(center_x)
            center_y = float(center_y)
            w = float(w)
            h = float(h)

            tracking_results = np.loadtxt(os.path.join(args.data_root, 'tracking', args.cause, 'refined_results', folder + '.txt'),delimiter=',')

            start_time = int(folder.split('_')[-2])

            use_mask = True

            pred_metrics = []
            target_metrics = []
            et = int(frame_id)-start_time+1
            st = et - 2*time_sample

            trackers, normalized_trackers = find_tracker(tracking_results, st, et)
            normalized_trackers = torch.from_numpy(normalized_trackers.astype(np.float32)).to(device)
            normalized_trackers = normalized_trackers.unsqueeze(0)
            num_box = len(trackers[0])

            camera_inputs = []
            action_logits = []

            hx = torch.zeros((num_box, 512)).to(device)
            cx = torch.zeros((num_box, 512)).to(device)

            transformed_camera_inputs = []
            # without intervention
            for l in range(st, et, time_sample):

                camera_name = 'output{}.png'.format(str(l-1 + start_time))
                camera_path = osp.join('/home/zxiao/data/ROI/roi_test', folder, camera_name)
                camera_inputs.append(Image.open(camera_path).convert('RGB'))  # save for later usage in intervention

                camera_input = camera_transforms(Image.open(camera_path).convert('RGB'))
                camera_input = np.array(camera_input)
                camera_input = to_device(torch.from_numpy(camera_input.astype(np.float32)), device)
                camera_input_preserve = camera_input.clone()
                transformed_camera_inputs.append(camera_input_preserve)
                mask = torch.ones((1, 3, 299, 299)).to(device)

                # assign index for RoIAlign
                # box_ind : (BxN)
                box_ind = np.array([np.arange(1)] * num_box).transpose(1, 0).reshape(-1)
                box_ind = torch.from_numpy(box_ind.astype(np.int32)).to(device)

                if args.partial_conv:
                    camera_input = model.backbone.features(camera_input, mask)
                else:
                    camera_input = model.backbone.features(camera_input)

                # camera_input: 1xCxHxW
                # normalized_trackers: 1xNx4
                # ROIAlign : BxNx1280
                tracker = normalized_trackers[:, (l - st)//time_sample].contiguous()
                box_ind = box_ind.view(1, num_box)
                feature_input = model.cropFeature(camera_input, tracker, box_ind)
                # check feature_input
                feature_input = feature_input.view(-1, 1280)

                hx, cx = model.step(feature_input, hx, cx)

            # Add another frame
            camera_name = 'output{}.png'.format(str(et-1 + start_time))
            camera_path = osp.join('/home/zxiao/data/ROI/roi_test', folder, camera_name)
            camera_inputs.append(Image.open(camera_path).convert('RGB'))  # save for later usage in intervention

            camera_input = camera_transforms(Image.open(camera_path).convert('RGB'))
            camera_input = np.array(camera_input)
            camera_input = to_device(torch.from_numpy(camera_input.astype(np.float32)), device)
            transformed_camera_inputs.append(camera_input)
            transformed_original_camera_inputs = torch.stack(transformed_camera_inputs,dim=1)

            trn_camera_feature = model.resnet.features(transformed_original_camera_inputs.view(-1,3,299,299))
            trn_camera_feature_shape = trn_camera_feature.shape
            trn_camera_feature = trn_camera_feature.view(-1,time_steps,trn_camera_feature_shape[-3],trn_camera_feature_shape[-2],trn_camera_feature_shape[-1])
            trn_sensor_inputs = to_device(torch.zeros(time_steps,20),device)
            intention_feature,waypoint_enc_scores, intention_enc_scores,waypoint_dec_scores,reconstructed_dec_features = model.trn(trn_camera_feature,trn_sensor_inputs)

            updated_feature = model.message_passing(hx, normalized_trackers)  # BxH
            object_feature = model.drop(updated_feature)
            feature = torch.cat((intention_feature, object_feature),1)
            vel = model.vel_classifier_v2(feature)
            # vel = model.vel_classifier(model.drop(updated_feature))
            confidence_go = softmax(vel).to('cpu').numpy()[0][0]
            #print(session, st, et, 'original', softmax(vel).to('cpu').numpy()[0])

            # with intervention
            for i in range(num_box):
                tracker = trackers[:, i, :]
                if i == 0:
                    action_logits.append([0.0, 1.0])
                    continue
                hx = torch.zeros((num_box, 512)).to(device)
                cx = torch.zeros((num_box, 512)).to(device)
                #  trackers: Tx(N+1)x4 (x1, y1, w, h ) without normalization
                #  normalized_trackers: : Tx(N+1)x4 (y1, x1, y2, x2 ) with normalization
                trackers, normalized_trackers = find_tracker(tracking_results, st, et)
                normalized_trackers = torch.from_numpy(normalized_trackers.astype(np.float32)).to(device)
                normalized_trackers = normalized_trackers.unsqueeze(0)

                for l in range(st, et, time_sample):
                    camera_input = np.array(camera_inputs[(l - st)//time_sample])
                    camera_input[int(trackers[(l - st)//time_sample, i, 1]):int(trackers[(l - st)//time_sample, i, 3]),
                    int(trackers[(l - st)//time_sample, i, 0]):int(trackers[(l - st)//time_sample, i, 2]), :] = 0
                    camera_input = Image.fromarray(np.uint8(camera_input))
                    np_camera_input = np.array(camera_input)
                    camera_input = camera_transforms(camera_input)
                    camera_input = np.array(camera_input)
                    camera_input = to_device(torch.from_numpy(camera_input.astype(np.float32)), device)

                    # assign index for RoIAlign
                    # box_ind : (BxN)
                    box_ind = np.array([np.arange(1)] * num_box).transpose(1, 0).reshape(-1)
                    box_ind = torch.from_numpy(box_ind.astype(np.int32)).to(device)

                    if not use_mask:
                        mask = torch.ones((1, 3, 299, 299)).to(device)

                    else:
                        mask = np.ones((1, 3, 299, 299))
                        x1 = int(trackers[(l - st) // time_sample, i, 1]/1280*299)  #x1
                        x2 = int(trackers[(l - st) // time_sample, i, 3]/1280*299)  #x2
                        y1 = int(trackers[(l - st) // time_sample, i, 0]/720*299)  #y1
                        y2 = int(trackers[(l - st) // time_sample, i, 2]/720*299)  #y2
                        mask[:,:, x1:x2, y1:y2] = 0
                        mask = torch.from_numpy(mask.astype(np.float32)).to(device)

                    if args.partial_conv:
                        camera_input = model.backbone.features(camera_input, mask)
                    else:

                        camera_input = model.backbone.features(camera_input)

                    tracker = normalized_trackers[:, (l - st)//time_sample].contiguous()
                    tracker[:, i, :] = 0
                    box_ind = box_ind.view(1, num_box)
                    feature_input = model.cropFeature(camera_input, tracker, box_ind)
                    # check feature_input
                    feature_input = feature_input.view(-1, 1280)

                    hx, cx = model.step(feature_input, hx, cx)

                intervened_trackers = torch.ones((1, time_steps, num_box, 4)).to(device)
                intervened_trackers[:, :, i, :] = 0.0
                intervened_trackers = intervened_trackers * normalized_trackers

                updated_feature = model.message_passing(hx, intervened_trackers)  # BxH
                object_feature = model.drop(updated_feature)
                feature = torch.cat((intention_feature, object_feature),1)
                vel = model.vel_classifier_v2(feature)
                # vel = model.vel_classifier(model.drop(updated_feature))
                action_logits.append(softmax(vel).to('cpu').numpy()[0])  # Nx2
                # print(session, start, end, i, trackers[:,i ], action_logits[i])
            if action_logits:
                cause_object_id = np.argmax(np.array(action_logits)[:, 0])
                #print(session, st, et, cause_object_id, trackers[:, cause_object_id],
                      #action_logits[cause_object_id])

                if 1.0 > confidence_go:
                    if visualize:
                        filename = vis_save_path+'/'+'_'.join([str(session), frame_id])

                        visualize_result(frame_id, trackers[:, cause_object_id], filename, [center_x, center_y, w, h])
                    result_dict[folder + '/' + frame_id] = list(trackers[-1, cause_object_id])

                    pred = trackers[-1, cause_object_id]  # x1, y1, x2, y2
                    # convert to (center_x, center_y, w, h)
                    xywh = np.empty((1, 4))
                    gt_box = np.empty((1, 4))
                    xywh[0,0] = (pred[0] +(pred[2]-pred[0])*0.5) / 1280.0
                    xywh[0,2] = (pred[2] - pred[0])  / 1280.0
                    xywh[0,1] = (pred[1] +(pred[3]-pred[1])*0.5) / 720.0
                    xywh[0,3] = (pred[3] - pred[1])  / 720.0

                    gt_box[0, 0] = center_x
                    gt_box[0, 1] = center_y
                    gt_box[0, 2] = w
                    gt_box[0, 3] = h

                    print('Sample: {}'.format(cnt))
                    cIOU = iou(xywh, gt_box, coords='centroids', mode='element-wise')
                    #print('Current IOU: {}'.format(cIOU[0]))


                    accumIOU += cIOU[0]
                    mIOU = accumIOU / (cnt + 1)
                    #print('mIOU: {}'.format(mIOU))

                    Accs[np.where(threshHolds < cIOU[0])] += 1.

                    Acc_5 = Accs[0] / (cnt + 1)
                    Acc_75 = Accs[5] / (cnt + 1)
                    mAcc = np.sum(Accs / (cnt + 1)) / 10

                    print('Acc_0.5: {}'.format(Acc_5))
                    print('Acc_0.75: {}'.format(Acc_75))
                    print('mAcc: {}'.format(mAcc))

                    #if cIOU[0] > 0.7:
                        #print('Sample: {}'.format(cnt))
                        #print('Current IOU: {}'.format(cIOU[0]))
    print('Acc_0.5: {}'.format(Acc_5))
    print('Acc_0.75: {}'.format(Acc_75))
    print('mAcc: {}'.format(mAcc))

    result_dict['metrics_all'] = [Acc_5,Acc_75,mAcc]
    json_save_dir = os.path.join(dir_name,'ROI_2frame')
    if not os.path.exists(json_save_dir):
        os.makedirs(json_save_dir)
    metric_save_dir = os.path.join(dir_name,'ROI_metric_2frame')
    if not os.path.exists(metric_save_dir):
        os.makedirs(metric_save_dir)
    json_save_name = model_basename.replace('.pth','.json')
    json_file_path = os.path.join(json_save_dir,args.cause+ '_'+ json_save_name)
    with open(json_file_path, 'w') as f:
        json.dump(result_dict, f,indent= 4)

    metric_save_name = model_basename.replace('.pth','.json')
    metric_file_path = os.path.join(metric_save_dir,args.cause+ '_'+ metric_save_name)
    with open(metric_file_path, 'w') as f:
        json.dump([Acc_5,Acc_75,mAcc], f,indent= 4)

# json_file_path = '/home/cli/sm120145_desktop/driving_model/scripts/json/'+args.cause
# if not os.path.isdir(json_file_path):
#     os.makedirs(json_file_path)
# json_file_path = json_file_path+'/ours.json'
# with open(json_file_path, 'w') as f:
#     json.dump(result_dict, f)
