import torch

from math import sqrt
from torch.autograd import Variable


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out = torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

######################
# Tracotron2 MODEL
######################
class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, 
                 padding = None, dilation = 1, bias = True, w_init_gain = 'linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) // 2)
        
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size,
                                    stride = stride, padding = padding, dilation = dilation, bias = bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain = torch.nn.init.calculate_gain(w_init_gain))
        
    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
    
    
class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias = True, w_init_gain = 'linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias = bias)
        torch.nn.init.xavier_normal_(self.linear_layer.weight, gain = torch.nn.init.calculate_gain(w_init_gain))
    
    def forward(self, x):
        return self.linear_layer(x)
        
            
class Encoder(torch.nn.Module):
    def __init__(self, para):
        super(Encoder, self).__init__()
        
        self.Conv_layer1 = torch.nn.Sequential(
            ConvNorm(para.symbols_embedding_dim, para.encoder_embedding_dim,
                     kernel_size = 5, stride = 1,
                     padding = 2, dilation = 1,
                     w_init_gain = 'relu'),
            torch.nn.BatchNorm1d(para.encoder_embedding_dim),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5)
        )
        
        self.Conv_layer2 = torch.nn.Sequential(
            ConvNorm(para.encoder_embedding_dim, para.encoder_embedding_dim,
                     kernel_size = 5, stride = 1,
                     padding = 2, dilation = 1,
                     w_init_gain = 'relu'),
            torch.nn.BatchNorm1d(para.encoder_embedding_dim),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5)
        )
        
        self.Conv_layer3 = torch.nn.Sequential(
            ConvNorm(para.encoder_embedding_dim, para.encoder_embedding_dim,
                     kernel_size = 5, stride = 1,
                     padding = 2, dilation = 1,
                     w_init_gain = 'relu'),
            torch.nn.BatchNorm1d(para.encoder_embedding_dim),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5)
        )
        
        self.lstm = torch.nn.LSTM(para.encoder_embedding_dim, int(para.encoder_embedding_dim / 2),
                                  1, batch_first = True, bidirectional = True)
        
    def forward(self, x, input_lengths):
        # x : [B, C, T]
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.Conv_layer3(x)
        
        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        
        # 对batch内的数据按照input_lengths进行压缩
        # 在进行lstm计算是，每条数据只计算input_length步就可以
        input_lengths = input_lengths.cpu().numpy()
        x = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first = True)
        self.lstm.flatten_parameters()  # 参数连续化，增加匀速速度
        outputs, _ = self.lstm(x)
        # pack_padded的反操作，将计算结构重新补0对齐
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True)
        
        return outputs
    
    def inference(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.Conv_layer3(x)
        
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        
        return outputs
    
    
class Prenet(torch.nn.Module):
    def __init__(self, in_dim, prenet_dim):
        super(Prenet, self).__init__()
        
        self.prenet_layer1 = torch.nn.Sequential(
            LinearNorm(in_dim, prenet_dim, bias = False),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5)
        )
        
        self.prenet_layer2 = torch.nn.Sequential(
            LinearNorm(prenet_dim, prenet_dim, bias = False),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5)
        )
        
    def forward(self, x):
        out = self.prenet_layer1(x)
        out = self.prenet_layer2(out)
        return out
    
    
class Postnet(torch.nn.Module):
    '''
    Postnet
    5 1-d convolution with 512 channels and kernel size 5
    '''
    def __init__(self, para):
        super(Postnet, self).__init__()
        
        self.postnet_layer_1 = torch.nn.Sequential(
            ConvNorm(para.n_mel_channels, para.postnet_embedding_dim, kernel_size = 5,
                     stride = 1, padding = 2, dilation = 1, w_init_gain = 'tanh'),
            torch.nn.BatchNorm1d(para.postnet_embedding_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5)
        )
        
        self.postnet_layer_2 = torch.nn.Sequential(
            ConvNorm(para.postnet_embedding_dim, para.postnet_embedding_dim, kernel_size = 5,
                     stride = 1, padding = 2, dilation = 1, w_init_gain = 'tanh'),
            torch.nn.BatchNorm1d(para.postnet_embedding_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5)
        )
        
        self.postnet_layer_3 = torch.nn.Sequential(
            ConvNorm(para.postnet_embedding_dim, para.postnet_embedding_dim, kernel_size = 5,
                     stride = 1, padding = 2, dilation = 1, w_init_gain = 'tanh'),
            torch.nn.BatchNorm1d(para.postnet_embedding_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5)
        )
        
        self.postnet_layer_4 = torch.nn.Sequential(
            ConvNorm(para.postnet_embedding_dim, para.postnet_embedding_dim, kernel_size = 5,
                     stride = 1, padding = 2, dilation = 1, w_init_gain = 'tanh'),
            torch.nn.BatchNorm1d(para.postnet_embedding_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5)
        )
        
        self.postnet_layer_5 = torch.nn.Sequential(
            ConvNorm(para.postnet_embedding_dim, para.n_mel_channels, kernel_size = 5,
                     stride = 1, padding = 2, dilation = 1, w_init_gain = 'linear'),
            torch.nn.BatchNorm1d(para.n_mel_channels),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5)
        )
        
    def forward(self, x):
        out = self.postnet_layer_1(x)
        out = self.postnet_layer_2(out)
        out = self.postnet_layer_3(out)
        out = self.postnet_layer_4(out)
        out = self.postnet_layer_5(out)
        return out
    
    
class LocationLayer(torch.nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters, kernel_size = attention_kernel_size,
                                      padding = padding, bias = False, stride = 1, dilation = 1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim, bias = False, w_init_gain = 'tanh')
        
    def forward(self, attention_weights_cat):
        # attention_weights_cat = [B, 2, T]
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention
    
    
class Attention(torch.nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, 
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        # 将query 即decoder的输出变换维度
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias = False, w_init_gain = 'tanh')
        
        # 将memory 即encoder的输出变换维度
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias = False, w_init_gain = 'tanh')
        
        self.v = LinearNorm(attention_dim, 1, bias = False)
        
        self.location_layer = LocationLayer(attention_location_n_filters, attention_location_kernel_size, attention_dim)
        
        self.score_mask_value = -float('inf')
        
    def get_aligment_energies(self, query, processed_memory, attention_weights_cat):
        processed_query = self.query_layer(query.unsqueeze(1))  # [B, 1, 128]
        processed_attention_weights = self.location_layer(attention_weights_cat)    #[B, T, 128]
        # processed_memory [B, T, 128]
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))
        # energies [B, T, 1]
        energies = energies.squeeze(-1)
        return energies
    
    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat, mask):
        '''
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memoty: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        '''
        aligment = self.get_aligment_energies(attention_hidden_state, processed_memory, attention_weights_cat)
        
        if mask is not None:
            aligment.data.masked_fill_(mask, self.score_mask_value)
            
        attention_weights = torch.softmax(aligment, dim = 1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        
        return attention_context, attention_weights
    
    
# 解码
class Decoder(torch.nn.Module):
    def __init__(self, para):
        super(Decoder, self).__init__()
        # 目标特征维度
        self.n_mel_channels = para.n_mel_channels
        
        # 每步解码 n_frames_per_step 帧特征
        self.n_frames_per_step = para.n_frames_per_step
        
        # 编码输出特征的维度，也就是attention-context的维度
        self.encoder_embedding_dim = para.encoder_embedding_dim
        
        # 注意力计算RNN的维度
        self.attention_rnn_dim = para.attention_rnn_dim
        
        # 解码RNN的维度
        self.decoder_rnn_dim = para.decoder_rnn_dim
        
        # pre-net的维度
        self.prenet_dim = para.prenet_dim
        
        # 测试过程中最多解码多少步
        self.max_decoder_steps = para.max_decoder_steps
        
        # 测试过程中，gate端输入多少认为解码结束
        self.gate_threshold = para.gate_threshold
        
        # 定义Prenet
        self.prenet = Prenet(
            para.n_mel_channels * para.n_frames_per_step,
            para.prenet_dim
        )
        
        # attention rnn 底层RNN
        self.attention_rnn = torch.nn.LSTMCell(
            para.prenet_dim + para.encoder_embedding_dim,
            para.attention_rnn_dim
        )
        self.dropout_attention_rnn = torch.nn.Dropout(0.1)
        
        # attention 层
        self.attention_layer = Attention(
            para.attention_rnn_dim, para.encoder_embedding_dim,
            para.attention_dim, para.attention_location_n_filters,
            para.attention_location_kernel_size
        )
        
        # decoder RNN 上层 RNN
        self.decoder_rnn = torch.nn.LSTMCell(
            para.attention_rnn_dim + para.encoder_embedding_dim,
            para.decoder_rnn_dim, 1)
        self.drop_decoder_rnn = torch.nn.Dropout(0.1)
        
        # 线性映射层
        self.linear_projection = LinearNorm(
            para.decoder_rnn_dim + para.encoder_embedding_dim,
            para.n_mel_channels * para.n_frames_per_step
        )
        
        self.gate_layer = LinearNorm(
            para.decoder_rnn_dim + para.encoder_embedding_dim, 1, 
            bias = True, w_init_gain = 'sigmoid'
        )
        
    def get_go_frame(self, memory):
        '''
        构造一个全0的矢量作为decoder第一帧的输出
        '''
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step
        ).zero_())
        return decoder_input
    
    def initialize_decoder_states(self, memory, mask):
        '''
        初始化，由于在最开始迭代的时候没有数据，所以这里将这些数据用0填充
        '''
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        
        # torch.nn.LSTM与torch.nn.LSTMCell的区别在于：torch.nn.LSTM表示整个LSTM层，包含多个时间步骤的信息，
        # 需要输入整个序列，返回整个序列的输出,所以在forward的方法中，可以不输入h_t-1和C_t-1，直接使用默认初
        # 始化的值即可。torch.nn.LSTMCell是一个更底层的模块，表示LSTM网络的一个单元，LSTMCell只处理一个时间
        # 步的输入，并且需要手动迭代每个时间步骤，以构建整个序列，通常，需要自己实现循环。所以在LSTMCell的forward
        # 方法中，需要指定h_t-1和C_t-1。这也是下面这些hidden和cell需要定义的由来。
        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim
        ).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim
        ).zero_())
        
        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim
        ).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim
        ).zero_())
        
        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME
        ).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME
        ).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim
        ).zero_())
        
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
        
    def parse_decoder_inputs(self, decoder_inputs):
        '''
        Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs : inputs used for teacher-forced training, i.e. mel-specs
        
        RETURNS
        ------
        inputs : processed decoder inputs
        '''
        
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        # (B, T_out, n_mel_channels) -> (B, T_out / 3, n_mel_channels * 3)
        decoder_inputs = decoder_inputs.reshape(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs
    
    
    def parse_decoder_outputs(self, mel_outputs, gate_outputs, aligments):
        '''
        Prepare decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs:
        aligments:
        
        RETURNS
        -------
        mel_outputs:
        gate_outputs:
        aligments:
        '''
        
        # (T_out, B) -> (B, T_out)
        aligments = torch.stack(aligments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frame per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        
        return mel_outputs, gate_outputs, aligments
         
    def decode(self, decoder_input):
        
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        self.attention_hidden = self.dropout_attention_rnn(self.attention_hidden)
        
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim = 1
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask
        )
        self.attention_weights_cum += self.attention_weights
        
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1 
        )
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = self.drop_decoder_rnn(self.decoder_hidden)
        
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim = 1
        )
        decoder_output = self.linear_projection(decoder_hidden_attention_context)
        
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        
        return decoder_output, gate_prediction, self.attention_weights 
        
        
    def forward(self, memory, decoder_inputs, memory_lengths):
        
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim = 0)
        decoder_inputs = self.prenet(decoder_inputs)
        
        self.initialize_decoder_states(memory, mask = ~get_mask_from_lengths(memory_lengths))
        
        mel_outputs, gate_outputs, aligments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weight = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            aligments += [attention_weight]
            
        mel_outputs, gate_outputs, aligments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, aligments
        )
        
        return mel_outputs, gate_outputs, aligments
    

    def inference(self, memory):
        
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask = None)
        
        mel_outputs, gate_outputs, aligments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, aligment = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            aligments += aligment
            
            if torch.sigmoid(gate_output) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print('warning! reached max decoder steps!')
                break
            
            decoder_input = mel_output
            
        mel_outputs, gate_outputs, aligments = self.parse_decoder_outputs(mel_outputs, gate_outputs, aligments)
        
        return mel_outputs, gate_outputs, aligments
    
    
class Tacotron2(torch.nn.Module):
    def __init__(self, para):
        super(Tacotron2, self).__init__()
        
        self.n_frames_per_step = para.n_frames_per_step
        self.n_mel_channels = para.n_mel_channels
        
        self.embedding = torch.nn.Embedding(para.n_symbols, para.symbols_embedding_dim)
        
        std = sqrt(2.0 / (para.n_symbols + para.symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(para)
        self.decoder = Decoder(para)
        self.postnet = Postnet(para)
        
    def forward(self, text_inputs, text_lengths, mels, output_lengths):
        
        # 进行text编码
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        # 得到encoder输出
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        
        # 得到decoder输出
        mel_outputs, gate_outputs, aligments = self.decoder(encoder_outputs, mels, memory_lengths = text_lengths)
        
        gate_outputs = gate_outputs.unsqueeze(2).repeat(1, 1, self.n_frames_per_step)
        gate_outputs = gate_outputs.view(gate_outputs.size(0), -1)
        
        # 经过postnet得到预测的mel输出
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, aligments], output_lengths)
    
    def parse_output(self, outputs, output_lengths = None):
        
        max_len = outputs[0].size(-1)
        ids = torch.arange(0, max_len, out = torch.cuda.LongTensor(max_len))
        mask = (ids < output_lengths.unsqueeze(1)).bool()
        mask = ~mask
        mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        
        outputs[0].data.masked_fill_(mask, 0.0)
        outputs[1].data.masked_fill_(mask, 0.0)
        outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)
        
        return outputs
    
    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, aligments = self.decoder.inference(encoder_outputs)
        
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        outputs = [mel_outputs, mel_outputs_postnet, gate_outputs, aligments]
        
        return outputs