import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

import pdb

def fc_relu(in_features, out_features, inplace=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(inplace=inplace),
    )


class TopoTRN(nn.Module):
    def __init__(self, args):
        super(TopoTRN, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.num_waypoints = args.num_waypoints
        self.num_intention_types = args.num_intention_types
        self.dropout = args.dropout

        self.feature_extractor = build_feature_extractor(args)
        self.future_size = self.feature_extractor.fusion_size
        self.fusion_size = self.feature_extractor.fusion_size * 2

        self.hx_trans = fc_relu(self.hidden_size, self.hidden_size)
        self.cx_trans = fc_relu(self.hidden_size, self.hidden_size)
        self.fusion_linear = fc_relu(self.num_waypoints, self.hidden_size)
        self.future_linear = fc_relu(self.hidden_size, self.future_size)

        self.enc_drop = nn.Dropout(self.dropout)
        self.enc_cell = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.dec_drop = nn.Dropout(self.dropout)
        self.dec_cell = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.waypoint_classifier = nn.Linear(self.hidden_size, self.num_waypoints)
        self.intention_classifier = nn.Linear(self.hidden_size, self.num_intention_types)
        self.dec_linear = nn.Linear(self.hidden_size, 2048)
        self.convtranspose = nn.Conv2d(20,2048,kernel_size=1)
        self.camera_convtranspose = nn.Sequential(
            nn.Conv2d(20, 2048, kernel_size=1),
            nn.ReLU(inplace=True),
        )
    def encoder(self, camera_input, sensor_input, future_input, enc_hx, enc_cx):
        fusion_input = self.feature_extractor(camera_input, sensor_input)
        fusion_input = torch.cat((fusion_input, future_input), 1)
        enc_hx, enc_cx = \
                self.enc_cell(self.enc_drop(fusion_input), (enc_hx, enc_cx))
        waypoint_enc_score = self.waypoint_classifier(self.enc_drop(enc_hx))
        intention_enc_score = self.intention_classifier(self.enc_drop(enc_hx))
        return enc_hx, enc_cx, waypoint_enc_score, intention_enc_score

    def decoder(self, fusion_input, dec_hx, dec_cx):
        dec_hx, dec_cx = \
                self.dec_cell(self.dec_drop(fusion_input), (dec_hx, dec_cx))
        dec_features = self.dec_drop(dec_hx)
        waypoint_dec_score = self.waypoint_classifier(dec_features)
        # reconstructed_dec_feature = dec_features.reshape(dec_features.shape[0],-1,10,10)
        # reconstructed_dec_feature = self.convtranspose(reconstructed_dec_feature)
        reconstructed_dec_feature = self.dec_linear(dec_features)
        return dec_hx, dec_cx, waypoint_dec_score, reconstructed_dec_feature

    def step(self, camera_input, sensor_input, future_input, enc_hx, enc_cx):
        # Encoder -> time t
        enc_hx, enc_cx, waypoint_enc_score,topology_enc_score,intention_enc_score = \
                self.encoder(camera_input, sensor_input, future_input, enc_hx, enc_cx)

        # Decoder -> time t + 1
        waypoint_dec_score_stack = []
        reconstructed_dec_feature_stack = []
        dec_hx = self.hx_trans(enc_hx)
        dec_cx = self.cx_trans(enc_cx)
        fusion_input = camera_input.new_zeros((camera_input.shape[0], self.hidden_size))
        future_input = camera_input.new_zeros((camera_input.shape[0], self.future_size))
        for dec_step in range(self.dec_steps):
            dec_hx, dec_cx, waypoint_dec_score,reconstructed_dec_feature = self.decoder(fusion_input, dec_hx, dec_cx)
            waypoint_dec_score_stack.append(waypoint_dec_score)
            reconstructed_dec_feature_stack.append(reconstructed_dec_feature)
            fusion_input = self.fusion_linear(waypoint_dec_score)
            future_input = future_input + self.future_linear(dec_hx)
        future_input = future_input / self.dec_steps

        return future_input, enc_hx, enc_cx, waypoint_enc_score,topology_enc_score, \
               intention_enc_score, waypoint_dec_score_stack, reconstructed_dec_feature_stack

    def forward_archived(self, camera_inputs, sensor_inputs,camera_inputs_dec=None):
        if camera_inputs_dec is None:
            return self.forward_test(camera_inputs, sensor_inputs)
        else:
            return self.forward_train(camera_inputs, sensor_inputs,camera_inputs_dec)
    
    def forward(self, camera_inputs, sensor_inputs):
        batch_size = camera_inputs.shape[0]
        enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        future_input = camera_inputs.new_zeros((batch_size, self.future_size))
        waypoint_enc_score_stack = []
        intention_enc_score_stack = []

        waypoint_dec_score_stack = []
        reconstructed_dec_feature_stack = []


        # Encoder -> time t
        for enc_step in range(self.enc_steps):
            enc_hx, enc_cx, waypoint_enc_score,intention_enc_score = self.encoder(
                camera_inputs[:, enc_step],
                sensor_inputs[:, enc_step],
                future_input, enc_hx, enc_cx,
            )
            waypoint_enc_score_stack.append(waypoint_enc_score)
            intention_enc_score_stack.append(intention_enc_score)

            # Decoder -> time t + 1
            dec_hx = self.hx_trans(enc_hx)
            dec_cx = self.cx_trans(enc_cx)
            fusion_input = camera_inputs.new_zeros((batch_size, self.hidden_size))
            future_input = camera_inputs.new_zeros((batch_size, self.future_size))
            for dec_step in range(self.dec_steps):
                dec_hx, dec_cx, waypoint_dec_score,reconstructed_dec_feature = self.decoder(fusion_input, dec_hx, dec_cx)
                waypoint_dec_score_stack.append(waypoint_dec_score)
                reconstructed_dec_feature_stack.append(reconstructed_dec_feature)
                fusion_input = self.fusion_linear(waypoint_dec_score)
                future_input = future_input + self.future_linear(dec_hx)
            future_input = future_input / self.dec_steps

        waypoint_enc_scores = torch.stack(waypoint_enc_score_stack, dim=1).view(-1, self.num_waypoints)
        intention_enc_scores = torch.stack(intention_enc_score_stack, dim=1).view(-1, self.num_intention_types)
        waypoint_dec_scores = torch.stack(waypoint_dec_score_stack, dim=1).view(-1, self.num_waypoints)
        reconstructed_dec_features = torch.stack(reconstructed_dec_feature_stack, dim=1)
        intention_feature = self.enc_drop(enc_hx)
        return intention_feature,waypoint_enc_scores, intention_enc_scores,waypoint_dec_scores,reconstructed_dec_features

    def forward_train(self, camera_inputs, sensor_inputs,camera_inputs_dec):
        batch_size = camera_inputs.shape[0]
        enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        future_input = camera_inputs.new_zeros((batch_size, self.future_size))
        waypoint_enc_score_stack = []
        intention_enc_score_stack = []

        waypoint_dec_score_stack = []
        reconstructed_dec_feature_stack = []

        target_feature_stack = []
        for i in range(self.enc_steps*self.dec_steps):
            target_feature_stack.append(self.feature_extractor.camera_linear(camera_inputs_dec[:,i]))
        # target_feature = self.feature_extractor.camera_linear(camera_inputs_dec)
        target_features = torch.stack(target_feature_stack, dim=1)
        # Encoder -> time t
        for enc_step in range(self.enc_steps):
            enc_hx, enc_cx, waypoint_enc_score,intention_enc_score = self.encoder(
                camera_inputs[:, enc_step],
                sensor_inputs[:, enc_step],
                future_input, enc_hx, enc_cx,
            )
            waypoint_enc_score_stack.append(waypoint_enc_score)
            intention_enc_score_stack.append(intention_enc_score)

            # Decoder -> time t + 1
            dec_hx = self.hx_trans(enc_hx)
            dec_cx = self.cx_trans(enc_cx)
            fusion_input = camera_inputs.new_zeros((batch_size, self.hidden_size))
            future_input = camera_inputs.new_zeros((batch_size, self.future_size))
            for dec_step in range(self.dec_steps):
                dec_hx, dec_cx, waypoint_dec_score,reconstructed_dec_feature = self.decoder(fusion_input, dec_hx, dec_cx)
                waypoint_dec_score_stack.append(waypoint_dec_score)
                reconstructed_dec_feature_stack.append(reconstructed_dec_feature)
                fusion_input = self.fusion_linear(waypoint_dec_score)
                future_input = future_input + self.future_linear(dec_hx)
            future_input = future_input / self.dec_steps

        waypoint_enc_scores = torch.stack(waypoint_enc_score_stack, dim=1).view(-1, self.num_waypoints)
        intention_enc_scores = torch.stack(intention_enc_score_stack, dim=1).view(-1, self.num_intention_types)
        waypoint_dec_scores = torch.stack(waypoint_dec_score_stack, dim=1).view(-1, self.num_waypoints)
        reconstructed_dec_features = torch.stack(reconstructed_dec_feature_stack, dim=1)

        a = reconstructed_dec_features.view(-1,self.feature_extractor.fusion_size)
        b = target_features.view(-1,self.feature_extractor.fusion_size)
        reconstructed_distance = torch.sum(((a - b)) ** 2, 1).sqrt()
        return waypoint_enc_scores, intention_enc_scores,waypoint_dec_scores,reconstructed_distance