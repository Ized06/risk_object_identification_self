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
from models.gcn import GCN as Model
import pdb
from tqdm import tqdm

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

    args = cfg.parse_args(parser)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    dir_name = os.path.dirname(args.model)
    model_basename = os.path.basename(args.model)
    print(dir_name)
    print(model_basename)
    with open('/home/zxiao/data/dataset/test_topology.json','r') as f:
        test_info_all = json.load(f)
    test_info_all = [x for x in test_info_all if x[7]==4]
    print("Total length: {}".format(len(test_info_all)))
    model = Model(args.inputs, partialConv = args.partial_conv,fusion = args.fusion).to(device)
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
    stimulus_enc_metrics = []
    stimulus_enc_target_metrics = []
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

        trackers = np.zeros([int((end-start)+1), num_object+1, 4])  # Tx(N+1)x4
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
                    trackers[ int((t-start)), i+1,  :] = bbox

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

    epoch = int(args.model.split('-')[-1].split('.')[0])
    print("evaluatiing epoch : {}".format(epoch))
    test_sessions = args.test_session_set
    for test_session in tqdm(test_sessions):
        session_test_info = [x for x in test_info_all if x[0]==test_session and x[7]==4]
        tracking_results = np.loadtxt(os.path.join(args.data_root, 'tracking','refined_results', test_session + '.txt'),delimiter=',')
        print("Session: {}, length:{}".format(test_session,tracking_results[-1][0]))
        for individual_info in session_test_info:
            session,intention_label,cause,start,end,binary_type,seq_id,topology_type = individual_info
            # if end - start > 20:
            #     start = end -20
            for enc_start in range(start,end-3,3):
                enc_indexes = [enc_start+i for i in range(3)]
                start,mid,end = enc_indexes

                trackers, normalized_trackers = find_tracker(tracking_results, start, end)
                normalized_trackers = torch.from_numpy(normalized_trackers.astype(np.float32)).to(device)
                normalized_trackers = normalized_trackers.unsqueeze(0)
                num_box = len(trackers[0])

                camera_inputs = []
                action_logits = []

                hx = torch.zeros((num_box, 512)).to(device)
                cx = torch.zeros((num_box, 512)).to(device)
                for l in range(start, end + 1):

                    if l ==0:
                        l = 1
                    camera_name = '{:05d}.jpg'.format(l)
                    camera_path = osp.join('/home/zxiao/data/real_test',seq_id, camera_name)
                    camera_inputs.append(Image.open(camera_path).convert('RGB'))  # save for later usage in intervention

                    camera_input = camera_transforms(Image.open(camera_path).convert('RGB'))
                    camera_input = np.array(camera_input)
                    camera_input = to_device(torch.from_numpy(camera_input.astype(np.float32)), device)

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
                    tracker = normalized_trackers[:, (l - start)].contiguous()
                    box_ind = box_ind.view(1, num_box)
                    feature_input = model.cropFeature(camera_input, tracker, box_ind)
                    # check feature_input
                    feature_input = feature_input.view(-1, 1280)

                    hx, cx = model.step(feature_input, hx, cx)

                updated_feature = model.message_passing(hx, normalized_trackers)  # BxH
                vel = model.vel_classifier(model.drop(updated_feature))
                # confidence_go = softmax(vel).to('cpu').numpy()[0][0]
                stimulus_enc_score = softmax(vel).cpu().detach().numpy()
                stimulus_enc_target = int(binary_type=='positive')
                stimulus_enc_metrics.extend(stimulus_enc_score)
                stimulus_enc_target_metrics.append(stimulus_enc_target)     

    intention_enc_mAP = utl.compute_result(
        list(range(2)),
        stimulus_enc_metrics,
        stimulus_enc_target_metrics,
        os.path.join(dir_name,'stop_go_eval'),
        model_basename.replace('.pth','.json'),
        ignore_class = [],
        save=True,
    )
    #     pdb.set_trace()

    # result_dict['metrics_all'] = [Acc_5,Acc_75,mAcc]
    # json_save_dir = os.path.join(dir_name,'ROI')
    # if not os.path.exists(json_save_dir):
    #     os.makedirs(json_save_dir)
    # metric_save_dir = os.path.join(dir_name,'ROI_metric')
    # if not os.path.exists(metric_save_dir):
    #     os.makedirs(metric_save_dir)
    # json_save_name = model_basename.replace('.pth','.json')
    # json_file_path = os.path.join(json_save_dir,args.cause+ '_'+ json_save_name)
    # with open(json_file_path, 'w') as f:
    #     json.dump(result_dict, f,indent= 4)

    # metric_save_name = model_basename.replace('.pth','.json')
    # metric_file_path = os.path.join(metric_save_dir,args.cause+ '_'+ metric_save_name)
    # with open(metric_file_path, 'w') as f:
    #     json.dump([Acc_5,Acc_75,mAcc], f,indent= 4)

# json_file_path = '/home/cli/sm120145_desktop/driving_model/scripts/json/'+args.cause
# if not os.path.isdir(json_file_path):
#     os.makedirs(json_file_path)
# json_file_path = json_file_path+'/ours.json'
# with open(json_file_path, 'w') as f:
#     json.dump(result_dict, f)
