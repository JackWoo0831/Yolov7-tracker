import numpy as np  
from basetrack import BaseTrack, TrackState, STrack, BaseTracker
from kalman_filter import KalmanFilter, NaiveKalmanFilter
from reid_models.deepsort_reid import Extractor
import matching
import torch 
import torch.nn as nn 
from torchvision.ops import nms

class Munkrs(nn.Module):
    """
    DHN module in paper "How to train your multi-object tracker"
    """
    def __init__(self, element_dim, hidden_dim, target_size, bidirectional, minibatch, is_cuda, is_train=True,
                 sigmoid=True, trainable_delta=False):
        super(Munkrs, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirect = bidirectional
        self.minibatch = minibatch
        self.is_cuda = is_cuda
        self.sigmoid = sigmoid
        if trainable_delta:
            if self.is_cuda:
                self.delta = torch.nn.Parameter(torch.FloatTensor([10]).cuda())
            else:
                self.delta = torch.nn.Parameter(torch.FloatTensor([10]))

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_row = nn.GRU(element_dim, hidden_dim, bidirectional=self.bidirect, num_layers=2, dropout=0.2)
        self.lstm_col = nn.GRU(hidden_dim*2, hidden_dim, bidirectional=self.bidirect, num_layers=2, dropout=0.2)

        # The linear layer that maps from hidden state space to tag space
        if self.bidirect:
            # *2 directions * 2 ways concat
            self.hidden2tag_1 = nn.Linear(hidden_dim * 2, 256)
            self.hidden2tag_2 = nn.Linear(256, 64)
            self.hidden2tag_3 = nn.Linear(64, target_size)
        else:
            # * 2 ways concat
            self.hidden2tag_1 = nn.Linear(hidden_dim, target_size)

        self.hidden_row = self.init_hidden(1)
        self.hidden_col = self.init_hidden(1)

        # init layers
        if is_train:
            for m in self.modules():
                if isinstance(m, nn.GRU):
                    print("weight initialization")
                    torch.nn.init.orthogonal_(m.weight_ih_l0.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l0.data)
                    torch.nn.init.orthogonal_(m.weight_ih_l0_reverse.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l0_reverse.data)

                    # initial gate bias as -1
                    m.bias_ih_l0.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l0.data[0:self.hidden_dim].fill_(-1)
                    m.bias_ih_l0_reverse.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l0_reverse.data[0:self.hidden_dim].fill_(-1)

                    torch.nn.init.orthogonal_(m.weight_ih_l1.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l1.data)
                    torch.nn.init.orthogonal_(m.weight_ih_l1_reverse.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l1_reverse.data)

                    # initial gate bias as one
                    m.bias_ih_l1.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l1.data[0:self.hidden_dim].fill_(-1)
                    m.bias_ih_l1_reverse.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l1_reverse.data[0:self.hidden_dim].fill_(-1)



    def init_hidden(self, batch):
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim),
        # one for hidden, others for memory cell

        if self.bidirect:
            if self.is_cuda:
                hidden = torch.zeros(2*2, batch, self.hidden_dim).cuda()
            else:
                hidden = torch.zeros(2*2, batch, self.hidden_dim)

        else:
            if self.is_cuda:
                hidden = torch.zeros(2, batch, self.hidden_dim).cuda()
            else:
                hidden = torch.zeros(2, batch, self.hidden_dim)
        return hidden

    def forward(self, Dt):
        self.hidden_row = self.init_hidden(Dt.size(0))
        self.hidden_col = self.init_hidden(Dt.size(0))

        # Dt is of shape [batch, h, w]
        # input_row is of shape [h*w, batch, 1], [time steps, mini batch, element dimension]
        # row lstm #

        input_row = Dt.contiguous().view(Dt.size(0), -1, 1).permute(1, 0, 2).contiguous()
        lstm_R_out, self.hidden_row = self.lstm_row(input_row, self.hidden_row)

        # column lstm #
        # lstm_R_out is of shape [seq_len=h*w, batch, hidden_size * num_directions]

        # [h * w*batch, hidden_size * num_directions]
        lstm_R_out = lstm_R_out.view(-1, lstm_R_out.size(2))

        # [h * w*batch, 1]
        # lstm_R_out = self.hidden2tag_1(lstm_R_out).view(-1, Dt.size(0))

        # [h,  w, batch, hidden_size * num_directions]
        lstm_R_out = lstm_R_out.view(Dt.size(1), Dt.size(2), Dt.size(0), -1)

        # col wise vector
        # [w,  h, batch, hidden_size * num_directions]
        input_col = lstm_R_out.permute(1, 0, 2, 3).contiguous()
        # [w*h, batch, hidden_size * num_directions]
        input_col = input_col.view(-1, input_col.size(2), input_col.size(3)).contiguous()
        lstm_C_out, self.hidden_col = self.lstm_col(input_col, self.hidden_col)

        # undo col wise vector
        # lstm_out is of shape [seq_len=time steps=w*h, batch, hidden_size * num_directions]

        # [h, w, batch, hidden_size * num_directions]
        lstm_C_out = lstm_C_out.view(Dt.size(2), Dt.size(1), Dt.size(0), -1).permute(1, 0, 2, 3).contiguous()

        # [h*w*batch, hidden_size * num_directions]
        lstm_C_out = lstm_C_out.view(-1, lstm_C_out.size(3))

        # [h*w, batch, 1]
        tag_space = self.hidden2tag_1(lstm_C_out)
        tag_space = self.hidden2tag_2(tag_space)
        tag_space = self.hidden2tag_3(tag_space).view(-1, Dt.size(0))
        if self.sigmoid:
            tag_scores = torch.sigmoid(tag_space)
        else:
            tag_scores = tag_space
        # tag_scores is of shape [batch, h, w] as Dt
        return tag_scores.view(Dt.size(1), Dt.size(2), -1).permute(2, 0, 1).contiguous()


class DeepMOT(BaseTracker):
    def __init__(self, opts, frame_rate=30, *args, **kwargs) -> None:
        super().__init__(opts, frame_rate, *args, **kwargs)

        self.DHN = Munkrs(element_dim=1, hidden_dim=256, target_size=1,
                 bidirectional=True, minibatch=1, is_cuda=True,
                 is_train=False)  # DHN 

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DHN.to(self.device)
        self.DHN.load_state_dict(torch.load(opts.dhn_path))

        self.filter_small_area = False  # filter area < 50 bboxs, TODO: why some bboxs has 0 area
        self.low_conf_thresh = max(0.15, self.opts.conf_thresh - 0.3)  # low threshold for second matching

        self.use_apperance_model = False


    def update(self, det_results, ori_img):
        """
        this func is called by every time step

        det_results: numpy.ndarray or torch.Tensor, shape(N, 6), 6 includes bbox, conf_score, cls
        ori_img: original image, np.ndarray, shape(H, W, C)
        """
        if isinstance(det_results, torch.Tensor):
            det_results = det_results.cpu().numpy()
        if isinstance(ori_img, torch.Tensor):
            ori_img = ori_img.numpy()

        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        # convert the scale to origin size
        # NOTE: yolo v7 origin out format: [xc, yc, w, h, conf, cls0_conf, cls1_conf, ..., clsn_conf]
        # TODO: check here, if nesscessary use two ratio
        img_h, img_w = ori_img.shape[0], ori_img.shape[1]
        ratio = [img_h / self.model_img_size[0], img_w / self.model_img_size[1]]  # usually > 1
        det_results[:, 0], det_results[:, 2] =  det_results[:, 0]*ratio[1], det_results[:, 2]*ratio[1]
        det_results[:, 1], det_results[:, 3] =  det_results[:, 1]*ratio[0], det_results[:, 3]*ratio[0]

        """step 1. filter results and init tracks"""
               
        # filter small area bboxs
        if self.filter_small_area:  
            small_indicies = det_results[:, 2]*det_results[:, 3] > 50
            det_results = det_results[small_indicies]

        # run NMS
        if self.NMS:
            # NOTE: Note nms need tlbr format
            nms_indices = nms(torch.from_numpy(STrack.xywh2tlbr(det_results[:, :4])), torch.from_numpy(det_results[:, 4]), 
                            self.opts.nms_thresh)
            det_results = det_results[nms_indices.numpy()]

        # cal high and low indicies
        det_high_indicies = det_results[:, 4] >= self.det_thresh
        det_low_indicies = np.logical_and(np.logical_not(det_high_indicies), det_results[:, 4] > self.low_conf_thresh)

        det_high, det_low = det_results[det_high_indicies], det_results[det_low_indicies]
        if det_high.shape[0] > 0:
            D_high = [STrack(cls, STrack.xywh2tlwh(xywh), score, kalman_format=self.opts.kalman_format)
                        for (cls, xywh, score) in zip(det_high[:, -1], det_high[:, :4], det_high[:, 4])]
        else:
            D_high = []

        if det_low.shape[0] > 0:
            D_low = [STrack(cls, STrack.xywh2tlwh(xywh), score, kalman_format=self.opts.kalman_format)
                            for (cls, xywh, score) in zip(det_low[:, -1], det_low[:, :4], det_low[:, 4])]
        else:
            D_low = []

        # Do some updates
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # update track state
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Kalman predict, update every mean and cov of tracks
        STrack.multi_predict(stracks=strack_pool, kalman=self.kalman)

        """Step 2. first match"""
        Dist_mat = matching.ecu_iou_distance(strack_pool, D_high, ori_img.shape[:2])
        
        # pass DHN
        # NOTE: first frame, strack_pool is empty
        if strack_pool and D_high:
            Dist_tensor = torch.tensor(Dist_mat, dtype=torch.float32).unsqueeze(0)  # (1, len(strack_pool), len(D_high))
            # forward
            Dist_tensor = 1.0 - self.DHN(Dist_tensor.to(self.device))
            # convert to ndarray
            Dist_mat = Dist_tensor.cpu().detach().numpy().squeeze(0)  # (len(strack_pool), len(D_high))

        # match
        matched_pair0, u_tracks0_idx, u_dets0_idx = matching.linear_assignment(Dist_mat, thresh=0.9)
        for itrack_match, idet_match in matched_pair0:
            track = strack_pool[itrack_match]
            det = D_high[idet_match]

            if track.state == TrackState.Tracked:  # normal track
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)

        u_tracks0 = [strack_pool[i] for i in u_tracks0_idx if strack_pool[i].state == TrackState.Tracked]
        u_dets0 = [D_high[i] for i in u_dets0_idx]

        """Step 3. second match, only IoU"""

        Dist_mat = matching.iou_distance(atracks=u_tracks0, btracks=D_low)
        matched_pair1, u_tracks1_idx, u_dets1_idx = matching.linear_assignment(Dist_mat, thresh=0.5)
        # u_det1 = [D_low[i] for i in u_dets1_idx]

        for itrack_match, idet_match in matched_pair1:
            track = u_tracks0[itrack_match]
            det = D_low[idet_match]

            if track.state == TrackState.Tracked:  # normal track
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)
        
        """ Step 4. deal with rest tracks and dets"""
        # deal with final unmatched tracks
        for idx in u_tracks1_idx:
            track = strack_pool[idx]
            track.mark_lost()
            lost_stracks.append(track)

        # deal with unconfirmed tracks, match new track of last frame and new high conf det
        # Apperance_dist = matching.embedding_distance(tracks=unconfirmed, detections=u_det1, metric='cosine')
        IoU_dist = matching.iou_distance(atracks=unconfirmed, btracks=u_dets0)
        # Dist_mat = self.gamma * IoU_dist + (1. - self.gamma) * Apperance_dist
        Dist_mat = IoU_dist
        matched_pair2, u_tracks2_idx, u_det2_idx = matching.linear_assignment(Dist_mat, thresh=0.7)

        for itrack_match, idet_match in matched_pair2:
            track = unconfirmed[itrack_match]
            det = u_dets0[idet_match]
            track.update(det, self.frame_id)
            activated_starcks.append(track)

        for u_itrack2_idx in u_tracks2_idx:
            track = unconfirmed[u_itrack2_idx]
            track.mark_removed()
            removed_stracks.append(track)

        # deal with new tracks
        for idx in u_det2_idx:
            det = u_dets0[idx]
            if det.score > self.det_thresh + 0.1:
                det.activate(self.frame_id)
                activated_starcks.append(det)

        """ Step 5. remove long lost tracks"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # update all
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)


        # print
        if self.debug_mode:
            print('===========Frame {}=========='.format(self.frame_id))
            print('Activated: {}'.format([track.track_id for track in activated_starcks]))
            print('Refind: {}'.format([track.track_id for track in refind_stracks]))
            print('Lost: {}'.format([track.track_id for track in lost_stracks]))
            print('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return [track for track in self.tracked_stracks if track.is_activated]

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb