
class Hparams():
    
    def __init__(self):
        
        # some files or directories
        ##############################################################################
        self.in_wave_dir = '/ZFS4T/tts/data/Wave/'
        self.out_feature_dir = '/ZFS4T/tts/data/Hifi_GAN/feature/'
        self.out_feature_cv_dir = '/ZFS4T/tts/data/Hifi_GAN/feature_cv/'
        self.out_wave_dir = '/ZFS4T/tts/data/Hifi_GAN/wave/'
        self.out_wave_cv_dir = '/ZFS4T/tts/data/Hifi_GAN/wave_cv/'
        self.static_mean_std = '/ZFS4T/tts/data/Hifi_GAN/static.npy'
        self.checkpoint = '/ZFS4T/tts/data/Hifi_GAN/checkpoint/'
        self.finetune_feature_tr_dir = '/ZFS4T/tts/data/Hifi_GAN/finetune/feature_tr/'
        self.finetune_feature_cv_dir = '/ZFS4T/tts/data/Hifi_GAN/finetune/feature_cv/'
        self.finetune_wave_dir = '/ZFS4T/tts/data/Hifi_GAN/finetune/wave_tr/'
        ##############################################################################
        
        # signal process parameters
        ##############################################################################
        self.sample_rate = 16000
        self.fmin = 0
        self.fmax = 8000
        self.n_fft = 1024
        self.win_size = 1024
        self.hop_size = 256
        self.n_freq = 1025
        self.n_mels = 80
        self.segment_size = 8192
        ##############################################################################
        
        # MODEL
        # Genetator
        ##############################################################################
        self.resblock = '1'
        self.upsample_rates = [8, 8, 2, 2]
        self.upsample_kernel_sizes = [16, 16, 4, 4]
        self.upsample_initial_channel = 512
        self.resblock_kernel_sizes = [3, 7, 11]
        self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.lrelu_slope = 0.1
        ##############################################################################
        
        # DiscriminatorP
        ##############################################################################
        self.DP_kernel_size = 5
        self.DP_stride = 3
        ##############################################################################
        
        # train
        ##############################################################################
        self.seed = 1234
        self.num_gpus = 2
        self.batch_size = 10
        self.learning_rate = 0.0002
        self.lr_decay = 0.999
        self.adam_b1 = 0.8
        self.adam_b2 = 0.99
        self.num_workers = 4
        self.training_epochs = 3000
        self.stdout_interval = 5
        self.checkpoint_interval = 5000
        self.validation_interval = 1000
        ##############################################################################
        
        # distribution
        ##############################################################################
        self.dist_backend = 'nccl'
        self.dist_url = 'tcp://localhost:54321'
        self.world_size = 1
        ##############################################################################
        
        # inference
        self.in_test_wave_dir = '/ZFS4T/tts/data/Hifi_GAN/wave_cv/'
        self.out_test_wave_dir = '/ZFS4T/tts/data/Hifi_GAN/out_test_wave/'
        self.out_test_wave_feature_dir = '/ZFS4T/tts/data/Hifi_GAN/out_test_wave_from_feature_ft/'
        self.infer_checkpoint = '/ZFS4T/tts/data/Hifi_GAN/checkpoint/g_00685000'