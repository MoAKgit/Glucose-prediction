# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 07:39:07 2022

@author: Mohammad
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import random
import torch.nn.init as torch_init

class Transformer(nn.Module):
    def __init__(self,args, n_outputs, dropout = 0.0):
        super().__init__()
        
        self.in_channels = 32
        self.inter_channels = 32
        self.n_outputs = n_outputs
        
        self.g = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        # bn = nn.BatchNorm1d
        bn = nn.InstanceNorm1d
        self.W = nn.Sequential(
                nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                # nn.Dropout(dropout)
                # bn(self.in_channels)
            )
        # nn.init.constant_(self.W[1].weight, 0)
        # nn.init.constant_(self.W[1].bias, 0)
        
        self.head_conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
        self.head_conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels= 1,
                         kernel_size=1, stride=1, padding=0)
        
        self.linear = nn.Linear(in_features= args.input_length, 
                                out_features= args.out_length)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, features_only = False):
        batch_size = x.size(0)
        # print('111111111', x.shape)
        # x = self.dropout(x)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        
        x = W_y + x
        # print('1111111111', x.shape)
        if features_only == True:
          return self.head_conv1(x)
        
        x = self.head_conv2(x).squeeze(1)
        
        x = self.linear(x)
        # print("11111111111",x.shape)
        # if features_only == True:
        #   return x  
        # x = self.sigmoid(x)
        return x


class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        input_features = 3
        num_layers = 2
        hidden_size = 100
        out_length = args.out_length
        
        self.lstm = nn.LSTM(input_size =input_features,
                            num_layers = num_layers,
                            hidden_size = hidden_size,
                            proj_size  = 1,
                            batch_first=True)
        
        self.linear = nn.Linear(in_features= args.input_length, 
                                out_features= args.out_length, bias= True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(args.dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, features_only = False):
        
        # print("1111111111", x.shape)
        x = x.permute(0,2,1)
        x,x_hidden= self.lstm(x)
        # print("2222222222", x_hidden[1].shape)
        x = torch.squeeze(x, 2)
        
        if features_only == True:
          return x 
        x = self.linear(x)
        # x = self.relu(x)
        x = self.sigmoid(x)
        # print("22222222222",x.shape)
        return x

    #############################################
class CNNLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_features = 3
        num_layers = 1
        hidden_size = 100
        
        self.conv1 = nn.Conv1d(in_channels = input_features, 
                               out_channels = 32, 
                               kernel_size = 1)
        self.lstm = nn.LSTM(input_size = 32,
                            num_layers = num_layers,
                            hidden_size = hidden_size,
                            proj_size  = 32,
                            batch_first=True)
        self.linear1 = nn.Linear(in_features= args.input_length, 
                                out_features= int(args.input_length/2))
        
        self.linear2 = nn.Linear(in_features= int(args.input_length/2), 
                                out_features= int(args.input_length/4))
        
        self.linear3 = nn.Linear(in_features= int(args.input_length/4), 
                                out_features= args.out_length)
        self.drop_out1 = nn.Dropout(0.2)
        self.drop_out2 = nn.Dropout(0.15)
        self.sigmoid = nn.Sigmoid()
    def forward (self, x, features_only = False):
        # print("11111111111111", x.shape)
        x = self.conv1(x)
        x = x.permute(0,2,1)
        x,_= self.lstm(x)
        
        if features_only == True:
          x = x.permute(0,2,1)
          return x 
      
        x = torch.squeeze(x, 2)
        
        # print("33333333333", x.shape)
        x = self.linear1(x)
        # x = self.drop_out1(x)
        x = self.linear2(x)
        # x = self.drop_out2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        # print("44444444444", x.shape)
        return x
#################################################
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(input_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        # src : [sen_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        
        # embedded : [sen_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        return hidden, cell
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers, dropout=dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        # input = [batch_size]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        input = input.unsqueeze(0)
        # input : [1, ,batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            'hidden dimensions of encoder and decoder must be equal.'
        assert encoder.n_layers == decoder.n_layers, \
            'n_layers of encoder and decoder must be equal.'
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [sen_len, batch_size]
        # trg = [sen_len, batch_size]
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        # first input to the decoder is the <sos> token.
        input = trg[0, :]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states 
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # replace predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            # update input : use ground_truth when teacher_force 
            input = trg[t] if teacher_force else top1
            
        return outputs
    

###################################################   TCN

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, 
                                           padding=padding, 
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        # self.init_weights()
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    def __init__(self,args,
                 num_channels, 
                 kernel_size=2, 
                 dropout=0.0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        
        
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[0] if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                      stride=1, dilation=dilation_size,
                                      padding=(kernel_size-1) * dilation_size, 
                                      dropout=dropout)]

        self.network = nn.Sequential(*layers)
        
        self.linear = nn.Linear(in_features= args.input_length, 
                                out_features= args.out_length)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x, features_only = False):
        
        # print(out.shape)
        x = self.network(x)
        x = self.dropout(x)
        
        if features_only == True:
          return x 
        
        x = self.linear(x.squeeze(1))
        x = self.sigmoid(x)
        
        return x
    


class MultiTemporalConvNet(nn.Module):
    def __init__(self,args, dropout=0.0):
        super(MultiTemporalConvNet, self).__init__()
        layers = []
        
        
        kernel_size = 2
        self.layer1_1 = TemporalBlock(3, 3, kernel_size, 
                                     stride=1, dilation=1,
                                     padding=(kernel_size-1) * 1, dropout=dropout)
        
        self.layer1_2 = TemporalBlock(3, 3, kernel_size, 
                                     stride=1, dilation=2,
                                     padding=(kernel_size-1) * 2, dropout=dropout)
        self.layer1_3 = TemporalBlock(3, 3, kernel_size, 
                                     stride=1, dilation=4,
                                     padding=(kernel_size-1) * 4, dropout=dropout)
        
        
        self.head1 = TemporalBlock(4*3, 3, kernel_size, 
                                      stride=1, dilation=1,
                                      padding=(kernel_size-1) * 1, dropout=dropout)
        
        
        self.layer2_1 = TemporalBlock(3, 3, kernel_size, 
                                      stride=1, dilation=1,
                                      padding=(kernel_size-1) * 1, dropout=dropout)
        self.layer2_2 = TemporalBlock(3, 3, kernel_size, 
                                      stride=1, dilation=2,
                                      padding=(kernel_size-1) * 2, dropout=dropout)
        self.layer2_3 = TemporalBlock(3, 3, kernel_size, 
                                      stride=1, dilation=4,
                                      padding=(kernel_size-1) * 4, dropout=dropout)
        
        self.head2 = TemporalBlock(4*3, 1, kernel_size, 
                                      stride=1, dilation=1,
                                      padding=(kernel_size-1) * 1, dropout=dropout)
        
   
        self.linear = nn.Linear(in_features= args.input_length, 
                                out_features= args.out_length)
        self.dropout = nn.Dropout(0.0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, features_only = False):
        
        x1 = self.layer1_1(x)
        x2 = self.layer1_2(x)
        x3 = self.layer1_3(x)
        x = torch.cat( (x1, x2, x3, x), 1 )
        x = self.head1(x)
        x1 = self.layer2_1(x)
        x2 = self.layer2_2(x)
        x3 = self.layer2_3(x)
        x = torch.cat( (x1, x2, x3, x), 1 ) 

        x = self.head2(x)
        if features_only == True:
          return x  
        
        x = self.linear(x.squeeze(1))
        x = self.sigmoid(x)
        
        return x
    

class new(nn.Module):
    def __init__(self,args, device ):
        super(new, self).__init__()
        
        
        self.device = device
        
        self.tranformer1 = Transformer(args,  32)
        self.tranformer2 = Transformer(args,  32)
        self.tranformer3 = Transformer(args,  32)
        self.TCN1 = TemporalConvNet(args, [32, 32, 32])
        self.TCN2 = TemporalConvNet(args, [32, 32, 32])
        self.TCN3 = TemporalConvNet(args, [32, 32, 32])
        
        
        self.TCN_head = TemporalConvNet(args, [3,1])
        
        self.MultiTCN = MultiTemporalConvNet(args)
        
        
        
        self.lstm = LSTM(args)
        self.CNN_lstm = CNNLSTM(args)
        
        self.linear_head1 = nn.Linear(in_features= 10, 
                                out_features= args.out_length)
        self.linear_head2 = nn.Linear(in_features= 20, 
                                out_features= args.out_length)
        self.linear_head3 = nn.Linear(in_features= 30, 
                                out_features= args.out_length)
        
        self.linear_head = nn.Linear(in_features= 30, 
                                out_features= args.out_length)
        
        
        self.head = TemporalBlock(2, 1, 2, 
                                      stride=1, dilation=1,
                                      padding= 1)
        
        self.head_conv1_1 = nn.Conv1d(in_channels=1*32 , out_channels=1,
                         kernel_size=3, stride=1, padding=1, bias= True)
        
        # self.head_conv2_1 = nn.Conv1d(in_channels=32, out_channels=32,
        #                  kernel_size=3, stride=1, padding=1, bias= True)
        
        # self.head_conv3_1 = nn.Conv1d(in_channels=32, out_channels=1,
        #                  kernel_size=3, stride=1, padding=1, bias= True)
        
        
        self.head_conv1_2 = nn.Conv1d(in_channels=1*32 , out_channels=1,
                         kernel_size=3, stride=1, padding=1, bias= True)
        
        # self.head_conv2_2 = nn.Conv1d(in_channels=32, out_channels=32,
        #                  kernel_size=3, stride=1, padding=1, bias= True)
        
        # self.head_conv3_2 = nn.Conv1d(in_channels=32, out_channels=1,
        #                  kernel_size=3, stride=1, padding=1, bias= True)
        
        self.head_conv1_3 = nn.Conv1d(in_channels=1*32 , out_channels=1,
                         kernel_size=3, stride=1, padding=1, bias= True)
        
        # self.head_conv2_3 = nn.Conv1d(in_channels=32, out_channels=32,
        #                  kernel_size=3, stride=1, padding=1, bias= True)
        
        # self.head_conv3_3 = nn.Conv1d(in_channels=32, out_channels=1,
        #                  kernel_size=3, stride=1, padding=1, bias= True)
        
        
        self.init_conv1_1 = nn.Conv1d(in_channels=3, out_channels=32,
                         kernel_size=3, stride=1, padding=1, bias= True)
        self.init_conv2_1 = nn.Conv1d(in_channels=32, out_channels=32,
                         kernel_size=3, stride=1, padding=1, bias= True)
        
        self.init_conv1_2 = nn.Conv1d(in_channels=3, out_channels=32,
                         kernel_size=3, stride=1, padding=1, bias= True)
        self.init_conv2_2 = nn.Conv1d(in_channels=32, out_channels=32,
                         kernel_size=3, stride=1, padding=1, bias= True)
        
        self.init_conv1_3 = nn.Conv1d(in_channels=3, out_channels=32,
                         kernel_size=3, stride=1, padding=1, bias= True)
        self.init_conv2_3 = nn.Conv1d(in_channels=32, out_channels=32,
                         kernel_size=3, stride=1, padding=1, bias= True)
        
        self.linear1 = nn.Linear(in_features= args.input_length, 
                                out_features= int(args.input_length/2))
        
        self.linear2 = nn.Linear(in_features= int(args.input_length/2), 
                                out_features= int(args.input_length/4))
        
        self.linear3 = nn.Linear(in_features= int(args.input_length/4), 
                                out_features= args.out_length)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
        
        self.W = torch.nn.Parameter(0.001*torch.randn(3))
        self.W.requires_grad = True
        
    def forward(self, x, features_only = False):
        x_in1 = x[:,:,20:]
        x_in2 = x[:,:,10:]
        x_in3 = x
        
        
        #######################
        x = x_in1
        x1 = self.init_conv1_1(x)
        x2 = self.init_conv2_1(x1)
        x = x2 
        #######################
        x3 = self.tranformer1(x, features_only = True )
        x4 = self.TCN1(x3, features_only = True)
        #######################
        x_head1 = self.head_conv1_1(x4)
        #######################
        
        
        #######################
        x = x_in2
        x1 = self.init_conv1_2(x)
        x2 = self.init_conv2_2(x1)
        x = x2
        #######################
        x3 = self.tranformer2(x, features_only = True )
        x4 = self.TCN2(x3, features_only = True)
        #######################
        x_head2 = self.head_conv1_2(x4)
        #######################       
        
        
        #######################
        x = x_in3
        x1 = self.init_conv1_3(x)
        x2 = self.init_conv2_3(x1)
        x = x2
        #######################
        x3 = self.tranformer3(x, features_only = True )
        x4 = self.TCN3(x3, features_only = True)
        #######################
        x_head3 = self.head_conv1_3(x4)
        #######################
        
        zero_pad = torch.zeros(x_head1.shape[0], 
                               x_head1.shape[1], 30-x_head1.shape[2]  ).to(self.device)
        
        
        x_head1 = torch.cat( (zero_pad, x_head1), 2 )

        zero_pad = torch.zeros(x_head2.shape[0], 
                               x_head2.shape[1], 30-x_head2.shape[2]  ).to(self.device)
        x_head2 = torch.cat( (zero_pad, x_head2), 2 )

        
        xhead_all = torch.cat( (x_head1, x_head2, x_head3), 1 )
        
        
        x = self.TCN_head(xhead_all, features_only = True)

        x = x.squeeze(1)
        x = self.linear_head(x)
        
        return x












