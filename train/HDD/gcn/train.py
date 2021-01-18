import os
import sys
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data

sys.path.insert(0, '../../../')
import config as cfg
import utils as utl
from datasets import GCNDataLayer as DataLayer
from models.gcn import GCN_intention as Model
import pdb

def to_device(x, device):
    return x.to(device)#.transpose(0,1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--inputs', default='camera', type=str)
    parser.add_argument('--cause', default='crossing_pedestrian', type=str)
    parser.add_argument('--gpu', default='0, 1', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=5e-04, type=float)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--time_steps', default=3, type=int)
    parser.add_argument('--partial_conv', default=True, type=bool)
    parser.add_argument('--data_augmentation', default= True, type=bool)
    parser.add_argument('--dist', default=False, type=bool)
    parser.add_argument('--fusion', default='attn',choices=['avg', 'gcn', 'attn'], type=str)
    parser.add_argument('--topology-info-file',type=str, default='/home/zxiao/data/dataset/topology_info_w_go_straight.json')
    parser.add_argument('--model', default='TRN', type=str)
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
    print(args.cause)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    url = 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth'

    model = Model(args.inputs, args.time_steps, partialConv = args.partial_conv, fusion = args.fusion,args=args)
    model.loadmodel(url)
    model = nn.DataParallel(model).to(device)
    print("Model Parameters:", count_parameters(model))


    softmax = nn.Softmax(dim=1).to(device)
    weights = [0.2, 1.0]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    camera_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5],),
    ])

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object = datetime.fromtimestamp(timestamp)
    if not args.data_augmentation:
        data_augmentation = 'wo_dataAug'
    else:
        data_augmentation = 'w_dataAug'
    result_name = "{}-{}-{}_{:02d}{:02d}{:02d}_{}_{}_result.json".format(dt_object.year, dt_object.month, dt_object.day,
                                                                      dt_object.hour, dt_object.minute,
                                                                      dt_object.second, data_augmentation,args.fusion)
    formated_time = "{}-{}-{}_{:02d}{:02d}{:02d}".format(dt_object.year, dt_object.month, dt_object.day, dt_object.hour,
                                                         dt_object.minute, dt_object.second)

for epoch in range(1, args.epochs+1):
        data_sets = {
            phase: DataLayer(
                args = args,
                data_root=args.data_root,
                cause=args.cause,
                phase=phase,
                sessions=getattr(args, phase+'_session_set'),
                camera_transforms = camera_transforms,
                time_steps=args.time_steps,
                data_augmentation= args.data_augmentation,
                dist=args.dist
            )
            for phase in args.phases
        }
        # sample = data_sets['train'][0]
        # pdb.set_trace()
        data_loaders = {
            phase: data.DataLoader(
                data_sets[phase],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            for phase in args.phases
        }

        losses = {phase: 0.0 for phase in args.phases}
        vel_metrics = []
        target_metrics = []
        mAP = 0.0

        start = time.time()
        for phase in args.phases:
            training = phase=='train'
            if training:
                model.train(True)
            else:
                if epoch%args.test_interval == 0 or epoch > 15:
                    model.train(False)
                else:
                    continue

            with torch.set_grad_enabled(training):
                for batch_idx, (camera_inputs, trackers, mask, dist_mask, vel_target) \
                        in enumerate(data_loaders[phase]):

                    batch_size = camera_inputs.shape[0]

                    camera_inputs = to_device(camera_inputs, device) # (bs, t, c , w, h)
                    mask = mask.to(device)
                    dist_mask = dist_mask.to(device)
                    trackers = to_device(trackers, device) # (bs, t, n, 4)
                    vel_target = to_device(vel_target, device).view(-1) #(bs)

                    vel = model(camera_inputs, trackers, device, dist_mask, mask )
                    vel_loss = criterion(vel, vel_target)
                    loss = vel_loss
                    print('batch idx:', batch_idx, '/', len(data_loaders[phase]),':', loss.item())
                    losses[phase] += loss.item()*batch_size
                    if args.debug:
                        print(loss.item())

                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        vel = softmax(vel).to('cpu').numpy()
                        vel_target = vel_target.to('cpu').numpy()
                        vel_metrics.extend(vel)
                        target_metrics.extend(vel_target)
        end = time.time()

        if epoch%args.test_interval == 0 or epoch > 15:
            result_path = snapshot_path = os.path.join('./results', args.cause)
            if not os.path.isdir(result_path):
                os.makedirs(result_path)
            epoch_result_name = 'inputs-{}-epoch-{}.json'.format(args.inputs, epoch)
            mAP, result = utl.compute_result(args.class_index, vel_metrics, target_metrics,
                                     result_path, epoch_result_name)

            result['Epoch'] = epoch
            with open(os.path.join(result_path, result_name), 'a') as f:
                json.dump(result, f)

            snapshot_path = os.path.join('./snapshots', args.cause, formated_time+'_'+data_augmentation+'_'+args.fusion)

            if not os.path.isdir(snapshot_path):
                os.makedirs(snapshot_path)
            snapshot_name = 'inputs-{}-epoch-{}.pth'.format(args.inputs, epoch)
            torch.save(model.state_dict(), os.path.join(snapshot_path, snapshot_name))

        print('Epoch {:2} | train loss: {:.5f} | test loss: {:.5f} mAP: {:.5f} | '
              'running time: {:.2f} sec'.format(
                  epoch,
                  losses['train']/len(data_loaders['train'].dataset),
                  losses['test']/len(data_loaders['test'].dataset),
                  mAP,
                  end-start,
        ))
