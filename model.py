import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group
import random

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class Pct(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 256))


        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        self.replace_pob = 0.0
        self.mask_ratio = [0.25, 0.45]
    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()

        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(center.shape[:2]) < ratio).bool().to(center.device)

        return bool_masked_pos

    def _random_replace(self, group_input_tokens, bool_masked_pos, noaug=False):
        '''
            group_input_tokens : B G C
            bool_masked_pos : B G
            -----------------
            replaced_group_input_tokens: B G C
        '''
        # skip replace
        if noaug or self.replace_pob == 0:
            return group_input_tokens, bool_masked_pos

        replace_mask = (torch.rand(group_input_tokens.shape[:2]) < self.replace_pob).to(bool_masked_pos.device).bool()
        replace_mask = (replace_mask & ~bool_masked_pos)  # do not replace the mask pos

        overall_mask = (replace_mask + bool_masked_pos).bool().to(bool_masked_pos.device)  # True for flake input

        detached_group_input_tokens = group_input_tokens.detach()
        flatten_group_input_tokens = detached_group_input_tokens.reshape(
            detached_group_input_tokens.size(0) * detached_group_input_tokens.size(1),
            detached_group_input_tokens.size(2))
        idx = torch.randperm(flatten_group_input_tokens.shape[0])
        shuffled_group_input_tokens = flatten_group_input_tokens[idx].reshape(detached_group_input_tokens.size(0),
                                                                              detached_group_input_tokens.size(1),
                                                                              detached_group_input_tokens.size(2))

        replace_mask = replace_mask.unsqueeze(-1).type_as(detached_group_input_tokens)
        replaced_group_input_tokens = group_input_tokens * (
                    1 - replace_mask) + shuffled_group_input_tokens * replace_mask
        return replaced_group_input_tokens, overall_mask



    def forward(self, x, Use_mask=False):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        if Use_mask == True:
            bool_masked_pos = self._mask_center_rand(feature_1.permute(0, 2, 1), noaug = False) # B G

            replaced_group_input_tokens, overall_mask = self._random_replace(feature_1.permute(0, 2, 1),
                                                                             bool_masked_pos.clone(), noaug=False)
            batch_size, seq_len, _ = replaced_group_input_tokens.size()

            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            maksed_group_input_tokens = replaced_group_input_tokens * (1 - w) + mask_token * w
            maksed_group_input_tokens = maksed_group_input_tokens.permute(0, 2, 1)
            x = self.pt_last(maksed_group_input_tokens)
            tokens = x.clone()
        else:
            x = self.pt_last(feature_1)
            tokens = x.detach()

        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x, tokens

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x