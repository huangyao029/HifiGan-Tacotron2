import os

def index_unknow():
    return 0

class hparams():
    def __init__(self):
        # 数据存储相关参数
        # self.path_wav = '/mnt/storage/hy_workspace/tmp/tmp4/tmp1/tmp9/tts/Tacotron2/data/Wave'
        # self.file_trans = '/mnt/storage/hy_workspace/tmp/tmp4/tmp1/tmp9/tts/Tacotron2/dataProsodyLabeling/000001-010000.txt'
        # self.path_fea = '/mnt/storage/hy_workspace/tmp/tmp4/tmp1/tmp9/tts/Tacotron2/data/features'
        # self.path_scp = '/mnt/storage/hy_workspace/tmp/tmp4/tmp1/tmp9/tts/Tacotron2/data/scp'
        # self.train_scp = '/mnt/storage/hy_workspace/tmp/tmp4/tmp1/tmp9/tts/Tacotron2/data/scp/train.scp'
        # self.path_save = '/mnt/storage/hy_workspace/tmp/tmp4/tmp1/tmp9/tts/Tacotron2/data/save'
        # self.test_scp = '/mnt/storage/hy_workspace/tmp/tmp4/tmp1/tmp9/tts/Tacotron2/data/scp/test.scp'
        
        self.base_dir = '/ZFS4T/tts/data/Hifi_GAN/Tacotron2/'
        self.input_wav_dir = '/ZFS4T/tts/data/Hifi_GAN/Tacotron2/wave_tr/'
        self.path_fea = '/ZFS4T/tts/data/Hifi_GAN/Tacotron2/feature_tr/'
        self.static_file = '/ZFS4T/tts/data/Hifi_GAN/static.npy'
        self.path_wav = '/ZFS4T/tts/data/Hifi_GAN/Tacotron2/wave_tr/'
        self.label_file = '/ZFS4T/tts/data/ProsodyLabeling/000001-010000.txt'
        self.summary_file = '/ZFS4T/tts/data/Tacotron2_16k/train.summary.txt'
        self.symbol_to_id = '/ZFS4T/tts/data/Tacotron2_16k/symbol_to_id.json'
        self.id_to_symbol = '/ZFS4T/tts/data/Tacotron2_16k/id_to_symbol.json'
        self.train_scp = '/ZFS4T/tts/data/Tacotron2_16k/train.scp'
        self.test_scp = '/ZFS4T/tts/data/Tacotron2_16k/test.scp'
        self.path_save_test = '/ZFS4T/tts/data/Hifi_GAN/Tacotron2/test_save'
        self.path_save = '/ZFS4T/tts/data/Hifi_GAN/Tacotron2/model_save'
        
        # # 加载字典
        # vocab_file = os.path.join(self.path_scp, 'vocab')
        
        # # 设置默认值为unknow = 0
        # self.dic_phoneme = defaultdict(index_unknow)
        # with open(vocab_file, 'r', encoding = 'utf-8') as vocab:
        #     for line in vocab:
        #         word, index = line.split()
        #         index = int(index)
        #         self.dic_phoneme[word] = index
                
        # 提取特征相关
        self.fs = 16000
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 256
        self.n_mels = 80
        self.fmin = 0
        self.fmax = self.fs / 2
        
        ########################
        # 模型相关
        ########################
        # 文本相关
        self.n_frames_per_step = 3  # 解码时，每步重建3帧音频特征
        self.n_symbols = 191        # 字典内符号的数目
        self.symbols_embedding_dim = 512      # 将文本符号映射为512维的特征
        self.encoder_embedding_dim = 512      # 3个 conv 层的 out—channel，以及lstm层的 out-channel
        
        # Decoder解码相关
        self.n_mel_channels = 80    # 目标特征的维度
        self.attention_rnn_dim = 1024
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        
        # attention相关
        self.attention_dim = 256    # 在attention计算时，将encoder输出，和decoder的输出都先变换到attention_dim
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31
        
        # PosNet相关
        self.postnet_embedding_dim = 512
        
        ########################
        # 训练相关
        ########################
        self.batch_size = 64
        self.n_epoch = 1000
        self.gamma = 0.99
        self.start_lr_decay = 3000
        self.weight_decay = 1e-6
        
        self.lr = 1e-3
        self.lr_final = 1e-5
        self.grad_clip_thresh = 1.0
        
        self.step_save = 2000
        
        
        self.allow_clipping_in_normalization = True
        self.symmetric_mels = True
        self.max_abs_value = 4.
        self.min_level_db = -100
        
        
        # GTA
        self.gta_feature_dir = '/ZFS4T/tts/data/Hifi_GAN/Tacotron2/feature_gta_cv/'
        