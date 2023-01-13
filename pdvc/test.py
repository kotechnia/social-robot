from video_backbone.TSP.extract_features import extract_features
import os


video_folder = '/home/jaewon/work/dense-video-captioning/1cycle_220629/videos/'
metadata_path = '/home/jaewon/work/dense-video-captioning/1cycle_220629/metadata2.csv'
feature_folder = '/home/jaewon/work/dense-video-captioning/1cycle_220629/features/'

print(__file__)

def parse_args():
    class local_args():
        def __init__(self):
            #-------------
            self.data_path = video_folder
            self.metadata_csv_filename = metadata_path
            self.output_dir = feature_folder
            
            #-------------
            self.backbone = 'r2plus1d_34'
            self.device = 'cuda'
            self.released_checkpoint = 'r2plus1d_34-tsp_on_activitynet'
            self.local_checkpoint = False
            #self.local_checkpoint = '/home/jaewon/work/dense-video-captioning/PDVC/video_backbone/TSP/models/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone.pth'
            self.clip_len = 16
            self.frame_rate = 15
            self.stride = 16
            self.batch_size = 32
            self.workers = 8
            self.shard_id = 0
            self.num_shards = 1
    args = local_args()
    return args

opt=parse_args()

extract_features(opt)


