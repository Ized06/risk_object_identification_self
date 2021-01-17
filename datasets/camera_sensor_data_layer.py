import os.path as osp
import cv2
import torch
import torch.utils.data as data
import numpy as np
import PIL.Image as Image
import random
import skimage.measure
import pdb
import json
import os

__all__ = [
    'TRNDataLayer',
    'C3DMultiStreamDataLayer',
    'RNNBaselineDataLayer',
    'BaselineDataLayer',
    'GCNDataLayer',
    'JinkyuDataLayer',
]


class TRNDataLayer(data.Dataset):
    def __init__(self, data_root, sessions, enc_steps, dec_steps, training=True):
        self.data_root = data_root
        self.sessions = sessions
        self.enc_steps = enc_steps
        self.dec_steps = dec_steps
        self.training = training

        self.inputs = []
        for session in self.sessions:
            sensor = np.load(osp.join(self.data_root, 'sensor', session + '.npy'))
            target = np.load(osp.join(self.data_root, 'target', session + '.npy'))
            seed = np.random.randint(self.enc_steps) if self.training else 90
            for start, end in zip(range(seed, target.shape[0] - self.dec_steps, self.enc_steps),
                                  range(seed + self.enc_steps, target.shape[0] - self.dec_steps, self.enc_steps)):
                enc_target = target[start:end]
                dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                self.inputs.append([
                    session, start, end, sensor[start:end],
                    enc_target, dec_target,
                ])

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                target_matrix[i, j] = target_vector[i + j + 1]
        return target_matrix

    def __getitem__(self, index):
        session, start, end, sensor_inputs, enc_target, dec_target = self.inputs[index]

        camera_inputs = np.load(
            osp.join(self.data_root, 'inceptionresnetv2', session + '.npy'), mmap_mode='r')[start:end]
        camera_inputs = torch.from_numpy(camera_inputs.astype(np.float32))
        sensor_inputs = torch.from_numpy(sensor_inputs.astype(np.float32))
        enc_target = torch.from_numpy(enc_target.astype(np.int64))
        dec_target = torch.from_numpy(dec_target.astype(np.int64))

        return camera_inputs, sensor_inputs, enc_target, dec_target.view(-1)

    def __len__(self):
        return len(self.inputs)


class C3DMultiStreamDataLayer(data.Dataset):
    def __init__(self, data_root, sessions, online=True, duration=16):
        self.data_root = data_root
        self.sessions = sessions
        self.online = online
        self.duration = duration

        self.inputs = []
        for session in self.sessions:
            sensor = np.load(osp.join(self.data_root, 'sensor', session + '.npy'))
            target = np.load(osp.join(self.data_root, 'target', session + '.npy'))
            for idx in range(90, target.shape[0] - self.duration):
                if self.online:
                    start, end = idx - self.duration + 1, idx + 1
                else:
                    start, end = idx - self.duration // 2, idx + self.duration // 2
                self.inputs.append([session, idx, sensor[start:end], target[[idx]]])

    def __getitem__(self, index):
        session, idx, sensor_inputs, target = self.inputs[index]

        folder = 'resnet3d50-online' if self.online else 'resnet3d50-offline'
        camera_inputs = np.load(
            osp.join(self.data_root, folder, session + '.npy'), mmap_mode='r')[idx]
        camera_inputs = torch.from_numpy(camera_inputs.astype(np.float32))
        sensor_inputs = torch.from_numpy(sensor_inputs.astype(np.float32))
        target = torch.from_numpy(target.astype(np.int64))

        return camera_inputs, sensor_inputs.contiguous().view(-1), target

    def __len__(self):
        return len(self.inputs)


class RNNBaselineDataLayer(data.Dataset):
    def __init__(self, data_root, sessions, time_steps, training=True):
        self.data_root = data_root
        self.sessions = sessions
        self.time_steps = time_steps
        self.training = training

        self.inputs = []
        for session in self.sessions:
            sensor = np.load(osp.join(self.data_root, 'sensor', session + '.npy'))
            target = np.load(osp.join(self.data_root, 'target', session + '.npy'))
            seed = np.random.randint(self.time_steps) if self.training else 0
            for start, end in zip(range(seed, target.shape[0], self.time_steps),
                                  range(seed + self.time_steps, target.shape[0], self.time_steps)):
                self.inputs.append([session, start, end, sensor[start:end], target[start:end]])

    def __getitem__(self, index):
        session, start, end, sensor_inputs, target = self.inputs[index]

        camera_inputs = np.load(
            osp.join(self.data_root, 'inceptionresnetv2', session + '.npy'), mmap_mode='r')[start:end]
        camera_inputs = camera_inputs.transpose([0,3,1,2])
        camera_inputs = torch.from_numpy(camera_inputs.astype(np.float32))
        sensor_inputs = torch.from_numpy(sensor_inputs.astype(np.float32))
        target = torch.from_numpy(target.astype(np.int64))

        return camera_inputs, sensor_inputs, target

    def __len__(self):
        return len(self.inputs)

class BaselineDataLayer(data.Dataset):
    def __init__(self, data_root, cause, sessions, time_steps, camera_transforms, data_augmentation = False, training=True):
        self.width = 1280
        self.height = 720
        self.data_root = data_root
        self.cause = cause
        self.sessions = sessions
        self.time_steps = time_steps
        self.num_box = 25
        self.training = training
        self.camera_transforms = camera_transforms
        self.data_augmentation = data_augmentation

        self.inputs = []
        for session in self.sessions:

            target = np.load(osp.join(self.data_root, 'sensor', session + '.npy'))
            steer_target = target[:, 1] # steering angle
            vel_target = target[:, 3]

            #load positive and negative samples of this session
            positive_sample = np.load(osp.join(self.data_root, 'binary_training', self.cause, session+'_positive.npy'))
            negative_sample = np.load(osp.join(self.data_root, 'binary_training', self.cause, session+'_negative.npy'))


            for timestamp in negative_sample:
                st, et = timestamp[0], timestamp[1]
                for start, end in zip(range(st, et, self.time_steps),
                                       range(st + self.time_steps, et, self.time_steps)):

                    self.inputs.append([session, start, end, np.array([0])])

            for timestamp in positive_sample:
                st, et = timestamp[0], timestamp[1]
                for start, end in zip(range(st, et, self.time_steps),
                                      range(st + self.time_steps, et, self.time_steps)):

                    self.inputs.append([session, start, end, np.array([1])])

    def normalize_box(self, trackers):
        trackers[:, :, 3] = trackers[:, :, 1] + trackers[:, :, 3]
        trackers[:, :, 2] = trackers[:, :, 0] + trackers[:, :, 2]


        tmp = trackers[:, :, 0] / self.width
        trackers[:, :, 0] = trackers[:, :, 1] /self.height
        trackers[:, :, 1] = tmp
        tmp = trackers[:, :, 2] / self.width
        trackers[:, :, 2] = trackers[:, :, 3] / self.height
        trackers[:, :, 3] = tmp

        return trackers

    def process_tracking(self, session, start, end):

        # load tracking results from txt file
        tracking_results = np.loadtxt(osp.join(self.data_root, 'tracking', 'refined_results', session + '.txt'), delimiter=',')

        t_array = tracking_results[:, 0]
        tracking_index = tracking_results[np.where(t_array == end)[0],1]

        num_object = min(len(tracking_index),self.num_box-1)

        trackers = np.zeros([self.time_steps, self.num_box, 4])   # TxNx4
        trackers[:, 0, :] = np.array([ 0.0, 0.0, self.width, self.height])  # Ego bounding box

        for t in range(start, end):
            current_tracking = tracking_results[np.where(t_array == t+1)[0]]
            for i, object_id in enumerate(tracking_index):
                if i > self.num_box-2: break
                if object_id in current_tracking[:,1]:
                    bbox = current_tracking[np.where(current_tracking[:, 1] == object_id)[0], 2:6]

                    trackers[t-start, i+1 , :] = bbox

        trackers = self.normalize_box(trackers)  # TxNx4 : y1, x1, y2, x2

        return trackers, num_object

    def __getitem__(self, index):

        session, start, end, vel_target = self.inputs[index]

        camera_inputs = []
        for idx in range(start, end):
            camera_name = str(idx+1).zfill(5)+'.jpg'
            camera_path = osp.join(self.data_root, 'camera', session, camera_name)
            img = self.camera_transforms(Image.open(camera_path).convert('RGB'))
            img = np.array(img)
            camera_inputs.append(img)

        camera_inputs = np.stack(camera_inputs)
        camera_inputs = torch.from_numpy(camera_inputs.astype(np.float32))  # (t, c, w, h)

        trackers, num_object = self.process_tracking(session, start, end)
        trackers = torch.from_numpy(trackers.astype(np.float32)) # TxNx4 : y1, x1, y2, x2

        if self.data_augmentation and vel_target[0] == 0: # add data augmentation
            obj_id = random.randint(0,num_object*2)
            if obj_id < num_object:
                obj_id += 1
                mask = np.ones((self.time_steps, 3, 299, 299))
                for t in range(self.time_steps):
                    x1, y1, x2, y2 = trackers[t, obj_id, :]
                    x1 = int(x1*299)  # x1
                    x2 = int(x2*299)  # x2
                    y1 = int(y1*299)  # y1
                    y2 = int(y2* 299)  # y2
                    mask[t, :, x1:x2, y1:y2] = 0
                mask = torch.from_numpy(mask.astype(np.float32))
            else:
                mask = torch.ones((self.time_steps, 3, 299, 299))

        else:
            mask = torch.ones((self.time_steps, 3, 299, 299))

        vel_target = torch.from_numpy(vel_target.astype(np.int64))

        return camera_inputs, mask,  vel_target

    def __len__(self):
        return len(self.inputs)



class GCNDataLayer(data.Dataset):
    def __init__(self, args, data_root,cause, sessions, phase, time_steps, camera_transforms, data_augmentation = False, dist = False, training=True):
        self.width = 1280
        self.height = 720
        self.num_box = 25
        self.data_root = data_root
        self.cause = cause
        self.phase = phase
        self.sessions = sessions
        self.time_steps = time_steps
        self.training = phase='train'
        self.camera_transforms = camera_transforms
        self.data_augmentation = data_augmentation
        self.dist = None
        args.target_topology = 4
        args.stop_go_variation_num = 40
        args.stop_go_tracking_root = '/home/zxiao/data/ROI/tracking_refine_hdd_dataset'
        args.stop_go_info_folder = '/home/zxiao/data/ROI/new_dataset_parallel_10'
        args.enc_steps = 3
        args.dec_steps = 5
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.data_root = os.path.join(args.stop_go_info_folder,'data')
        self.stop_go_tracking_root = args.stop_go_tracking_root

        self.inputs = []
        with open(args.topology_info_file,'r') as f:
            topology_info = json.load(f)

        with open(os.path.join(args.stop_go_info_folder,'stop_go_positive.json'),'r') as f:
            self.positive_info = json.load(f)
        with open(os.path.join(args.stop_go_info_folder,'stop_go_negative.json'),'r') as f:
            self.negative_info = json.load(f)
        self.positive_seqs = [x[0] for x in self.positive_info]
        self.negative_seqs = [x[0] for x in self.negative_info]

        for seq_id in topology_info.keys():
            session = seq_id.split('_')[0]
            original_start = int(seq_id.split('_')[1])
            original_end = int(seq_id.split('_')[2])
            if session not in self.sessions:
                continue

            activity_idx = topology_info[seq_id]['activity_idx']
            topology_type = topology_info[seq_id]['topology_type']
            state = topology_info[seq_id]['state']
            
            # Skip sequences that do not meet the criteria
            if len(state)!=original_end-original_start+1:
                continue
            if topology_type != args.target_topology:
                continue
            if topology_type == 5:
                continue
            
            for variation_idx in range(args.stop_go_variation_num+1):
                new_seq_id = '_'.join([seq_id,str(variation_idx)])
                if variation_idx < args.stop_go_variation_num:
                    # Contains both posotive and negative
                    # Filter seq_without risk objects larger than a certain threshold
                    if new_seq_id not in self.positive_seqs:
                        continue
                    tracking_info_file = os.path.join(args.stop_go_tracking_root,new_seq_id+'.txt')
                    if not os.path.exists(tracking_info_file):
                        continue
                    if os.stat(tracking_info_file).st_size == 0:
                        continue
                    try:
                        tracking_results = np.loadtxt(tracking_info_file, delimiter=',') 
                        t_array = tracking_results[:, 0]
                    except IndexError:
                        continue
                    seq_positive_negative_info = [ x for x in self.positive_info if x[0] == new_seq_id][0]
                    _,positive_start,_ = seq_positive_negative_info
                    negative_start = int(original_start)
                    negative_end = int(positive_start)
                    positive_start = int(positive_start)
                    positive_end = int(original_end)
                    negative_length = negative_end - negative_start
                    positive_length = positive_end - positive_start
                    if negative_end - negative_start > args.dec_steps:
                        negative_states = state[:negative_length]
                        negative_state_indexes =  list(range(len(negative_states)))
                        if self.training:
                            tmp_indexes = sorted(random.sample(negative_state_indexes,self.dec_steps))
                            state_part1_indexes = tmp_indexes[:3]
                            state_part2_indexes = tmp_indexes[3:]
                        else:
                            tmp_indexes = sorted([negative_state_indexes[0], 
                                                negative_state_indexes[len(negative_state_indexes)//4], 
                                                negative_state_indexes[len(negative_state_indexes)//2],
                                                negative_state_indexes[3*len(negative_state_indexes)//4],
                                                negative_state_indexes[-1]])
                            state_part1_indexes = tmp_indexes[:3]
                            state_part2_indexes = tmp_indexes[3:]

                        selected_indexes = state_part1_indexes + state_part2_indexes
                        if len(selected_indexes) != self.dec_steps:
                            print("State length mismatch")

                        stimulus =  np.array([0])

                        # Encoder: pick part one indexes (3)
                        enc_indexes = [int(x) for x in state_part1_indexes]
                        self.inputs.append([
                                session, seq_id, variation_idx,
                                enc_indexes, stimulus
                            ])



                    if positive_end - positive_start > args.dec_steps:
                        positive_states_current = state[negative_length]
                        positive_states_prediction = state[negative_length:]
                        positive_state_indexes =  list(range(len(positive_states_prediction)))
                        
                        if self.training:
                            tmp_indexes = sorted(random.sample(positive_state_indexes,self.dec_steps))
                            state_part1_indexes = tmp_indexes[:3]
                            state_part2_indexes = tmp_indexes[3:]
                        else:
                            tmp_indexes = sorted([positive_state_indexes[0], 
                                                positive_state_indexes[len(positive_states_prediction)//4], 
                                                positive_state_indexes[len(positive_states_prediction)//2],
                                                positive_state_indexes[3*len(positive_states_prediction)//4],
                                                positive_state_indexes[-1]])
                            state_part1_indexes = tmp_indexes[:3]
                            state_part2_indexes = tmp_indexes[3:]

                        selected_indexes = state_part1_indexes + state_part2_indexes
                        if len(selected_indexes) != self.dec_steps:
                            print("State length mismatch")
                        stimulus =  np.array([1])

                        # Encoder: pick part one indexes (3)
                        enc_indexes = [int(x+negative_length) for x in state_part1_indexes]
                        self.inputs.append([
                                session, seq_id, variation_idx,
                                enc_indexes, stimulus
                            ])
        # for session in self.sessions:

        #     #load positive and negative samples of this session
        #     positive_sample = np.load(osp.join(self.data_root, 'binary_training', self.cause,  session+'_positive.npy'))
        #     negative_sample = np.load(osp.join(self.data_root, 'binary_training', self.cause,  session+'_negative.npy'))


        #     for timestamp in negative_sample:
        #         st, et = timestamp[0], timestamp[1]
        #         for start, end in zip(range(st, et, self.time_steps),
        #                                range(st + self.time_steps, et, self.time_steps)):

        #             self.inputs.append([session, start, end, np.array([0])])

        #     for timestamp in positive_sample:
        #         st, et = timestamp[0], timestamp[1]
        #         for start, end in zip(range(st, et, self.time_steps),
        #                               range(st + self.time_steps, et, self.time_steps)):

        #             self.inputs.append([session, start, end, np.array([1])])


    def normalize_box(self, trackers):
        trackers[:, :, 3] = trackers[:, :, 1] + trackers[:, :, 3]
        trackers[:, :, 2] = trackers[:, :, 0] + trackers[:, :, 2]


        tmp = trackers[:, :, 0] / self.width
        trackers[:, :, 0] = trackers[:, :, 1] /self.height
        trackers[:, :, 1] = tmp
        tmp = trackers[:, :, 2] / self.width
        trackers[:, :, 2] = trackers[:, :, 3] / self.height
        trackers[:, :, 3] = tmp

        return trackers

    def cal_pairwise_distance(self,X):
        """
        computes pairwise distance between each element
        Args:
            X: [N,D]
            Y: [M,D]
        Returns:
            dist: [N,M] matrix of euclidean distances
        """

        Y =X
        X = np.expand_dims(X[0], axis=0)
        rx=np.reshape(np.sum(np.power(X,2),axis=1),(-1,1))
        ry=np.reshape(np.sum(np.power(Y,2),axis=1),(-1,1))
        dist=np.clip(rx-2.0*np.matmul(X,np.transpose(Y))+np.transpose(ry),0.0,float('inf'))

        return np.sqrt(dist)

    def compute_dist(self, tracker, depth , num_object):
        ##################################
        # tracker: Nx4 y1,x1,y2,x2
        #
        ##################################
        self._inv_intrinsics = np.linalg.inv(np.array([[936.86, 0.0, 647.48], [0.0, 936.4, 404.14], [0.0, 0.0, 1.0]]))
        threshold = 5

        center_x = np.expand_dims((tracker[:,1]+tracker[:,3])*0.5*self.width, axis=1) #Nx1
        center_y = np.expand_dims((tracker[:,0]+tracker[:,2])*0.5*self.height, axis=1) #Nx1

        center = np.concatenate([center_x, center_y], axis=1) #Nx2
        depth_list = depth[center_y.astype(np.int32), center_x.astype(np.int32)]  # (N,)

        depth_list[0] = 1.0
        depth_list = np.reshape(depth_list, [-1])
        center[0, :] = np.array([640, 719])  # ego position
        center = np.append(center, np.ones([np.shape(center)[0], 1]).astype(np.float32), axis=1)  # N*3
        center_3d = np.multiply(np.matmul(self._inv_intrinsics, np.transpose(center)), depth_list)
        distance_map = self.cal_pairwise_distance(np.transpose(center_3d))
        distance_mask = np.ones(self.num_box)
        zero_index = np.where(distance_map>threshold)
        distance_mask[zero_index[1]] = 0
        distance_mask[num_object+1:] = 0


        return distance_mask


    # def process_tracking(self, session, start, end):

    #     # load tracking results from txt file
    #     tracking_results = np.loadtxt(osp.join(self.data_root, 'tracking', 'refined_results', session + '.txt'), delimiter=',')

    #     t_array = tracking_results[:, 0]
    #     tracking_index = tracking_results[np.where(t_array == end)[0],1]

    #     num_object = len(tracking_index)

    #     trackers = np.zeros([self.time_steps, self.num_box, 4])   # TxNx4
    #     trackers[:, 0, :] = np.array([ 0.0, 0.0, self.width, self.height])  # Ego bounding box

    #     for t in range(start, end):
    #         current_tracking = tracking_results[np.where(t_array == t+1)[0]]
    #         for i, object_id in enumerate(tracking_index):
    #             if i > self.num_box - 2: break
    #             if object_id in current_tracking[:,1]:
    #                 bbox = current_tracking[np.where(current_tracking[:, 1] == object_id)[0], 2:6]

    #                 trackers[t-start, i+1 , :] = bbox
    #     trackers = self.normalize_box(trackers)  # TxNx4 : y1, x1, y2, x2

    #     return trackers, num_object

    def process_tracking(self, seq_id, inquiry_indexes):

        end = inquiry_indexes[-1]
        # load tracking results from txt file
        tracking_results = np.loadtxt(os.path.join(self.stop_go_tracking_root,seq_id+'.txt'), delimiter=',')
        # tracking_results = np.loadtxt(osp.join(self.data_root, 'tracking', 'refined_results', seq_id + '.txt'), delimiter=',')

        t_array = tracking_results[:, 0]
        tracking_index = tracking_results[np.where(t_array == end)[0],1]

        num_object = len(tracking_index)

        trackers = np.zeros([self.time_steps, self.num_box, 4])   # TxNx4
        trackers[:, 0, :] = np.array([ 0.0, 0.0, self.width, self.height])  # Ego bounding box
        # for t in range(start, end):
        for idx,t_idx in enumerate(inquiry_indexes):
            current_tracking = tracking_results[np.where(t_array == t_idx)[0]]
            for i, object_id in enumerate(tracking_index):
                if i > self.num_box - 2: break
                if object_id in current_tracking[:,1]:
                    bbox = current_tracking[np.where(current_tracking[:, 1] == object_id)[0], 2:6]

                    trackers[idx, i+1 , :] = bbox
        trackers = self.normalize_box(trackers)  # TxNx4 : y1, x1, y2, x2
        return trackers, num_object

    def __getitem__(self, index):

        # session, start, end, vel_target = self.inputs[index]
        session, seq_id, variation_idx, enc_indexes, stimulus = self.inputs[index]
        seq_id_with_object = seq_id + '_' + str(variation_idx)

        camera_inputs = []
        image_w_object_folder = os.path.join(self.data_root,seq_id_with_object)
        images = os.listdir(image_w_object_folder)
        images = sorted(images,key=lambda x: int(x[:-4]))
        image_name_dict = dict()
        for idx,image in enumerate(images):
            image_name_dict[idx] = image
        for idx in enc_indexes:
            camera_path = os.path.join(image_w_object_folder,image_name_dict[idx])
            img = self.camera_transforms(Image.open(camera_path).convert('RGB'))
            img = np.array(img)
            camera_inputs.append(img)          
        # for idx in range(start, end):
        #     camera_name = str(idx+1).zfill(5)+'.jpg'
        #     camera_path = osp.join(self.data_root, 'camera', session, camera_name)
        #     img = self.camera_transforms(Image.open(camera_path).convert('RGB'))
        #     img = np.array(img)
        #     camera_inputs.append(img)

        camera_inputs = np.stack(camera_inputs)
        camera_inputs = torch.from_numpy(camera_inputs.astype(np.float32))  # (t, c, w, h)

        trackers, num_object = self.process_tracking(seq_id_with_object, enc_indexes)

        dist_mask = np.ones(self.num_box)
        if self.dist:
            # load depth result
            depth = 1.0 / (cv2.imread(osp.join(self.data_root, 'depth_midas', session, str(end).zfill(5) + '.png'),0) / 255.0 + np.exp(-10))
            dist_mask = self.compute_dist(trackers[-1, :, ], depth,num_object)  # Nx4 : y1,x1,y2,x2


        dist_mask = torch.from_numpy(dist_mask.astype(np.float32))


        if self.data_augmentation and stimulus[0] == 0: # add data augmentation
            obj_id = random.randint(0,num_object*2)
            if obj_id < num_object:
                obj_id += 1
                trackers[:,obj_id,:] = 0
                mask = np.ones((self.time_steps, 3, 299, 299))
                for t in range(self.time_steps):
                    x1, y1, x2, y2 = trackers[t, obj_id, :]
                    x1 = int(x1*299)  # x1
                    x2 = int(x2*299)  # x2
                    y1 = int(y1*299)  # y1
                    y2 = int(y2* 299)  # y2
                    mask[t, :, x1:x2, y1:y2] = 0
                mask = torch.from_numpy(mask.astype(np.float32))
            else:
                mask = torch.ones((self.time_steps, 3, 299, 299))

        else:
            mask = torch.ones((self.time_steps, 3, 299, 299))

        trackers = torch.from_numpy(trackers.astype(np.float32))
        vel_target = torch.from_numpy(stimulus.astype(np.int64))
        return camera_inputs, trackers, mask, dist_mask, vel_target

    def __len__(self):
        return len(self.inputs)


class JinkyuDataLayer(data.Dataset):
    def __init__(self, data_root, cause, sessions, time_steps, camera_transforms, training=True):
        self.width = 1280
        self.height = 720
        self.data_root = data_root
        self.cause = cause
        self.sessions = sessions
        self.time_steps = time_steps
        self.training = training
        self.camera_transforms = camera_transforms

        self.inputs = []
        for session in self.sessions:

            target = np.load(osp.join(self.data_root, 'sensor', session + '.npy'))
            steer_target = target[:, 1] # steering angle
            vel_target = target[:, 3]

            #load positive and negative samples of this session
            positive_sample = np.load(osp.join(self.data_root, 'binary_training', self.cause, session+'_positive.npy'))
            negative_sample = np.load(osp.join(self.data_root, 'binary_training', self.cause, session+'_negative.npy'))


            for timestamp in negative_sample:
                st, et = timestamp[0], timestamp[1]
                for start, end in zip(range(st, et, self.time_steps),
                                       range(st + self.time_steps, et, self.time_steps)):

                    self.inputs.append([session, start, end, steer_target[start:end], vel_target[start:end]])

            for timestamp in positive_sample:
                st, et = timestamp[0], timestamp[1]
                for start, end in zip(range(st, et, self.time_steps),
                                      range(st + self.time_steps, et, self.time_steps)):

                    self.inputs.append([session, start, end, steer_target[start:end], vel_target[start:end]])


    def __getitem__(self, index):

        session, start, end, steer_target, vel_target = self.inputs[index]

        camera_inputs = []
        for idx in range(start, end):
            camera_name = str(idx+1).zfill(5)+'.jpg'
            camera_path = osp.join(self.data_root, 'camera', session, camera_name)
            img = self.camera_transforms(Image.open(camera_path).convert('RGB'))
            img = np.array(img)
            camera_inputs.append(img)

        camera_inputs = np.stack(camera_inputs)
        camera_inputs = torch.from_numpy(camera_inputs.astype(np.float32))  # (t, c, w, h)

        steer_target = torch.from_numpy(steer_target.astype(np.float32))
        vel_target = torch.from_numpy(vel_target.astype(np.float32))

        return camera_inputs,  steer_target, vel_target

    def __len__(self):
        return len(self.inputs)