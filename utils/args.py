import argparse
import pyaudio

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Batch Processing with TalkNet and Face Recognition")
    
    # parser.add_argument('--talknet_model', type=str, default="weights/pretrain_TalkSet.model", 
    #                     help='Path to the pretrained TalkNet model file')
    parser.add_argument('--talknet_model', type=str, default="weights/pretrain_AVA.model", 
                        help='Path to the pretrained TalkNet model file')

    parser.add_argument('--face_masking', action='store_true', help='Enable face masking')
    parser.add_argument('--no_face_masking', action='store_false', dest='face_masking', help='Disable face masking')
    parser.set_defaults(face_masking=True)

    parser.add_argument('--registered_faces', type=str, default="registeredFaces", 
                        help='Path to the pretrained TalkNet model file')

    parser.add_argument('--tmp_dir', type=str, default='output/temp/', 
                        help='Temporary directory for processing')
    
    parser.add_argument('--max_frames', type=int, default=6000, 
                        help='Total number of frames to process (termination condition)')
    
    parser.add_argument('--input_device_index', type=int, default=3, 
                        help='Audio input device index')
    
    parser.add_argument('--s3fd_conf_threshold', type=float, default=0.5, 
                        help='Confidence threshold for S3FD face detection')
    
    parser.add_argument('--s3fd_scales', type=str, default='0.5', 
                        help='Comma-separated list of scales for S3FD detection (e.g. "0.25,0.5")')
    
    parser.add_argument('--sort_max_age', type=int, default=30, 
                        help='Maximum age for SORT tracker')
    parser.add_argument('--sort_min_hits', type=int, default=1, 
                        help='Minimum hits for SORT tracker')
    parser.add_argument('--sort_iou_threshold', type=float, default=0.5, 
                        help='IOU threshold for SORT tracker')
    
    parser.add_argument('--face_model_name', type=str, default='buffalo_l', 
                        help='Model name for InsightFace')
    parser.add_argument('--face_app_ctx_id', type=int, default=0, 
                        help='Context ID for InsightFace (GPU index or -1 for CPU)')
    parser.add_argument('--face_pad_scale', type=float, default=0.25, 
                        help='Padding scale factor for cropping face region during recognition')
    
    parser.add_argument('--kernel_size', type=int, default=9, 
                        help='Kernel size for median filtering of bounding boxes (odd integer)')

    parser.add_argument('--mask_blur_kernel', type=int, default=25, 
                        help='Kernel size for Gaussian blur for unknown faces (use an odd integer)')
    
    parser.add_argument('--audio_format', type=int, default=pyaudio.paInt16, 
                        help='Audio format (e.g. pyaudio.paInt16)')
    
    parser.add_argument('--output_framerate', type=int, default=30, 
                        help='Frame rate for output video')
    parser.add_argument('--video_codec', type=str, default='libx264', 
                        help='Video codec for output video')

    parser.add_argument('--speaking_gap_threshold', type=int, default=59, 
                        help='Maximum length (in frames at 30fps) between two speaking segments to call as non-speaking segment')

    parser.add_argument('--speaking_min_frame_length', type=int, default=25, 
                        help='Minimum length (in frames at 30fps) to call a segment as speaking')

    return parser.parse_args()
