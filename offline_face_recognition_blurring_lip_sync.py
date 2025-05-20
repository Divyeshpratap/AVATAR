"""
Offline video → face recognition → masking unknown faces → lip sync → speaking segments → save video and tracks
Mandoatory arguments
  • --videoFolder    : folder name under which output is to be placed
  • --videoName      : video name should be mp4
  • --face_masking   : if face masking is required
                    or
  • --no_face_masking: if face masking is not required
Output
  • video_out.avi in subfolder pyavi
"""


import sys, time, os, glob, subprocess, warnings, cv2, pickle, numpy, math, logging, tqdm
import argparse
from scipy import signal
from scipy.signal import medfilt
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from insightface.app import FaceAnalysis
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from model.faceDetector.s3fd import S3FD
from model.talkNet import talkNet
import python_speech_features
import torch
import shutil
from collections import defaultdict, Counter
import json
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceRec")

#############################################
#  Initialisation helpers                   #
#############################################

def initialize_face_detector():
    """Initialise InsightFace detector (AntelopeV2)."""
    ctx = 0
    det_thresh = 0.51
    pack = "antelopev2"
    app = FaceAnalysis(name=pack, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=ctx, det_thresh=det_thresh)
    return app


def load_face_database(face_db_path, face_app, logger):
    """Build an in‑memory face database { person: [embeddings] }."""
    face_database = {}
    if not os.path.isdir(face_db_path):
        logger.error(f"Face database directory '{face_db_path}' does not exist.")
        return face_database
    for person in os.listdir(face_db_path):
        person_dir = os.path.join(face_db_path, person)
        if os.path.isdir(person_dir):
            embeddings = []
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"Failed to load {image_path}")
                    continue
                faces = face_app.get(img)
                if len(faces) == 0:
                    logger.warning(f"No face detected in {image_path}")
                    continue
                embeddings.extend([face.normed_embedding for face in faces])
            if embeddings:
                face_database[person] = embeddings
    logger.info(f"Face database loaded. Persons found: {list(face_database.keys())}")
    return face_database


def radial_blur(img, bb, sigma_blur=25):
    """
    Apply strong blur in the centre of bb that fades toward the border.
    bb = (x1,y1,x2,y2)  (already clipped to image)
    """
    x1,y1,x2,y2 = map(int, bb)
    roi        = img[y1:y2, x1:x2]
    h, w       = roi.shape[:2]

    # 1. Gaussian blurred
    roi_blur = cv2.GaussianBlur(roi, (0, 0), sigma_blur)

    # 2. 2-D Gaussian weight to mask the size of the ROI
    g_x = cv2.getGaussianKernel(w, w/3)    
    g_y = cv2.getGaussianKernel(h, h/3)
    mask = g_y @ g_x.T                        
    mask = (mask - mask.min()) / (mask.max()-mask.min())  
    mask = mask[..., None]                       
    mask = mask.astype(numpy.float32)

    # 3. blend: blurred * mask  +  original * (1-mask)
    roi_out = (roi_blur * mask + roi * (1-mask)).astype(numpy.uint8)
    img[y1:y2, x1:x2] = roi_out
    return img



def recognize_face(embedding, face_database, threshold=0.55):
    """Return best matching identity & similarity score (or 'Unknown')."""
    identity = "Unknown"
    best_score = -1.0
    for person, embeddings in face_database.items():
        for db_emb in embeddings:
            score = numpy.dot(embedding, db_emb)
            if score > best_score:
                best_score = score
                if score > threshold:
                    identity = person
    return identity, best_score

#############################################
#  Video‑level helpers                      #
#############################################

def scene_detect(args):
    videoManager   = VideoManager([args.videoFilePath])
    statsManager   = StatsManager()
    sceneManager   = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode   = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList      = sceneManager.get_scene_list(baseTimecode)
    savePath       = os.path.join(args.pyworkPath, 'scene.pckl')
    if not sceneList:
        sceneList = [(videoManager.get_base_timecode(), videoManager.get_current_timecode())]
    with open(savePath, 'wb') as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write('%s - scenes detected %d\n' % (args.videoFilePath, len(sceneList)))
    return sceneList


def inference_video(args, face_database, thr):
    """Run face detection on every frame **and label immediately**."""
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()

    dets = []  # Structure is list( list( detections_in_frame ) )
    for fidx, fname in enumerate(flist):
        image  = cv2.imread(fname)
        faces  = face_detector.get(image)
        dets.append([])
        for face in faces:
            bbox  = face.bbox.astype(int)
            conf  = face.score
            identity, _ = recognize_face(face.normed_embedding, face_database, thr)
            dets[-1].append({
                'frame': fidx,
                'bbox' : bbox.tolist(),  # x1,y1,x2,y2
                'conf' : conf,
                'identity': identity
            })
        sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))

    savePath = os.path.join(args.pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea  = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea  = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(args, sceneFaces):
    """Simple IOU based tracker that also propagates identities."""
    iouThres = 0.5
    tracks   = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if not track:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes   = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI   = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI  = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                   numpy.mean(bboxesI[:, 3] - bboxesI[:, 1])) > args.minFaceSize:
                tracks.append({
                    'frame'      : frameI,
                    'bbox'       : bboxesI,
                    'identities' : [f['identity'] for f in track]  # keep identity votes
                })
    return tracks

#############################################
#  Cropping / TalkNet / Utility Functions   #
#############################################

def crop_video(args, track, cropFile):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = args.cropScale
        bs = dets['s'][fidx]
        bsi = int(bs * (1 + 2 * cs))
        image = cv2.imread(flist[frame])
        frame_padded = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi
        mx = dets['x'][fidx] + bsi
        face = frame_padded[int(my - bs):int(my + bs * (1 + 2 * cs)),
                            int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + '.wav'
    audioStart = (track['frame'][0]) / 25
    audioEnd = (track['frame'][-1] + 1) / 25
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
               (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
    subprocess.call(command, shell=True, stdout=None)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" %
               (cropFile, audioTmp, args.nDataLoaderThread, cropFile))
    subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track': track, 'proc_track': dets}

# Talknet Evaluation code, directly taken from talknet
def evaluate_network(files, args):
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n" % args.pretrainModel)
    s.eval()
    allScores = []
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(os.path.basename(file))[0]
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112 - (112 / 2)):int(112 + (112 / 2)), int(112 - (112 / 2)):int(112 + (112 / 2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * 25)), :, :]
        allScore = []
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(videoFeature[i * duration * 25:(i + 1) * duration * 25, :, :]).unsqueeze(0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score, label = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
                    # scores.extend(label)
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores


def extract_speaking_segments(frame_numbers, scores, gap_threshold=59, min_segment_length=31):
    smoothed = medfilt(scores, kernel_size=gap_threshold)
    segments = run_length_encode(smoothed)
    merged_segments = []
    i = 0
    while i < len(segments):
        start_idx, end_idx, val = segments[i]
        if val == 1:
            j = i + 1
            while j + 1 < len(segments):
                gap_start, gap_end, gap_val = segments[j]
                next_start, next_end, next_val = segments[j+1]
                if gap_val == 0 and (gap_end - gap_start + 1) <= gap_threshold and next_val == 1:
                    end_idx = segments[j+1][1]
                    j += 2
                else:
                    break
            merged_segments.append((start_idx, end_idx, 1))
            i = j
        else:
            i += 1
    speaking_segments = []
    for seg in merged_segments:
        seg_start_idx, seg_end_idx, val = seg
        if (seg_end_idx - seg_start_idx + 1) >= min_segment_length:
            speaking_segments.append((frame_numbers[seg_start_idx]- 15, frame_numbers[seg_end_idx] + 15))
    return speaking_segments

def run_length_encode(seq):
    if len(seq) == 0:
        return []
    segments = []
    start = 0
    prev_val = seq[0]
    for i, val in enumerate(seq[1:], start=1):
        if val != prev_val:
            segments.append((start, i - 1, prev_val))
            start = i
            prev_val = val
    segments.append((start, len(seq) - 1, prev_val))
    return segments

def visualization(tracks, scores, args):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    flag_count = 0
    faces = [[] for _ in range(len(flist))]
    # final_scores_dict = defaultdict(defaultdict(list))
    final_scores_dict = defaultdict(lambda: defaultdict(list))
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        label = track['track'].get('track_label', 'Unknown')
        if label == 'sam' and flag_count >=2:
            label = 'jack'
            print(f'changed label to jack')
        if label == 'sam' and flag_count < 2:
            flag_count += 1
            print(f'flag count is {flag_count}')
        # final_scores_dict[label]['frames'] +=  track['track']['frame'].tolist()
        # final_scores_dict[label]['scores'] +=  score
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s_val = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
            s_mean = numpy.mean(s_val)
            final_scores_dict[label]['frames'].append(frame)
            final_scores_dict[label]['scores'].append(s_mean)
            faces[frame].append({'track': tidx, 'score': float(s_mean), 's': track['proc_track']['s'][fidx],
                                  'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx],
                                  'label': label})
    plain = { k: dict(v) for k, v in final_scores_dict.items() }
    savePath = os.path.join(args.pyworkPath, 'label_scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(plain, fil)

    textSavePath = os.path.join(args.pyworkPath, 'label_scores.txt')
    with open(textSavePath, 'w') as f:
        for label, data in plain.items():
            f.write(f"{label}\n")
            f.write(f"  frames: {data['frames']}\n")
            f.write(f"  scores: {data['scores']}\n\n")

    speaking_segments_dict = defaultdict(list)

    # speaking segments code
    for label, data in final_scores_dict.items():
        frames = data['frames']
        scores = data['scores']
        binary_scores = [1 if score >= 0.0 else 0 for score in scores]
        speaking_segments = extract_speaking_segments(frames, binary_scores)
        speaking_segments_dict[label] = speaking_segments
        print(f"Label: {label} Speaking Segments: {speaking_segments}")

    speaking_segments_path = os.path.join(args.pyworkPath, 'speaking_segments.pckl')
    with open(speaking_segments_path, 'wb') as fil:
        pickle.dump(speaking_segments_dict, fil)
    speaking_segments_json_path = os.path.join(args.pyworkPath, 'speaking_segments.json')
    try:
        with open(speaking_segments_json_path, 'w') as json_file:
            json.dump(speaking_segments_dict, json_file, indent=4)
        print(f"Speaking segments saved to {speaking_segments_json_path}")
    except Exception as e:
        print(f"Error saving speaking segments to JSON: {e}")

    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'),
                           cv2.VideoWriter_fourcc(*'XVID'), 25, (fw, fh))
    colorDict = {0: 0, 1: 255}
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            if face['label'] == 'Unknown':
                continue
            clr = colorDict[int((face['score'] >= 0))] # If using scores
            x, y, s = int(face['x']), int(face['y']), int(face['s'])
            # text = f"{person_identity} {'is speaking' if float((face['score'] >= 0)) >= 0.0 else ''}"
            # clr = colorDict[int((face['score'] >= 0.3))] # If using labels

            cv2.rectangle(image, (int(face['x'] - face['s']), int(face['y'] - face['s'])),
                          (int(face['x'] + face['s']), int(face['y'] + face['s'])), (0, clr, 255 - clr), 10)
            
            # label_text = f"{face['label'] + ': ' if face['label'] else ''}{round(face['score'], 1)}"
            label_text = f"{face['label'] if face['label'] else ''} {'is speaking' if float((face['score'] >= 0)) else ''}"
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 5)
            text_x = x - s
            text_y = y - s - 10

            if text_y < 0:
                text_y = y + s + text_height + 10
            if text_x + text_width > fw:
                text_x = fw - text_width - 10
            if text_y + text_height > fh:
                text_y = fh - text_height - 10

            # cv2.putText(image, label_text, (int(face['x'] - face['s']), int(face['y'] - face['s'])),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, clr, 255 - clr), 5)
            cv2.putText(image, label_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, clr, 255 - clr), 5)            
        vOut.write(image)
    vOut.release()
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
               (os.path.join(args.pyaviPath, 'video_only.avi'),
                os.path.join(args.pyaviPath, 'audio.wav'),
                args.nDataLoaderThread, os.path.join(args.pyaviPath, 'video_out.avi')))
    subprocess.call(command, shell=True, stdout=None)



#############################################
#  Main                                     #
#############################################

parser = argparse.ArgumentParser(description="TalkNet Demo or Columbia ASD Evaluation")
parser.add_argument('--videoName', type=str, default="001", help='Demo video name')
parser.add_argument('--videoFolder', type=str, default="demo", help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel', type=str, default="weights/pretrain_TalkSet.model", help='Path for the pretrained TalkNet model')
parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of workers')
parser.add_argument('--facedetScale', type=float, default=0, help='Scale factor for face detection; frames are scaled by this factor')
parser.add_argument('--minTrack', type=int, default=10, help='Minimum number of frames for each shot')
parser.add_argument('--numFailedDet', type=int, default=20, help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize', type=int, default=1, help='Minimum face size in pixels')
parser.add_argument('--cropScale', type=float, default=0.40, help='Crop bounding box scale')
parser.add_argument('--start', type=int, default=0, help='The start time of the video')
parser.add_argument('--duration', type=int, default=0, help='Duration of the video; if 0, use the whole video')
parser.add_argument('--evalCol', dest='evalCol', action='store_true', help='Evaluate on Columbia dataset')
parser.add_argument('--colSavePath', type=str, default="/data08/col", help='Path for inputs, tmps and outputs (Columbia)')
# Face recognition
parser.add_argument('--faceDB', type=str, default="registeredFaces", help='Path to the face database for recognition')
parser.add_argument('--facePadScale', type=float, default=0.2, help='Padding scale when cropping face for masking (unused after refactor)')
parser.add_argument('--face_masking', action='store_true', help='Enable face masking of unknown identities')
parser.add_argument('--no_face_masking', action='store_false', dest='face_masking', help='Disable face masking')
parser.add_argument('--simThreshold', type=float, default=0.25, help='Cosine‑similarity threshold (0‑1); lower = stricter')
parser.set_defaults(face_masking=True)
args = parser.parse_args()
args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
args.savePath  = os.path.join(args.videoFolder, args.videoName)

# ----------------------------------------------------------------------------
#  Folder prep
# ----------------------------------------------------------------------------
args.pyaviPath    = os.path.join(args.savePath, 'pyavi')
args.pyframesPath = os.path.join(args.savePath, 'pyframes')
args.pyworkPath   = os.path.join(args.savePath, 'pywork')
args.pycropPath   = os.path.join(args.savePath, 'pycrop')

if os.path.exists(args.savePath):
    rmtree(args.savePath)
os.makedirs(args.pyaviPath,    exist_ok=True)
os.makedirs(args.pyframesPath, exist_ok=True)
os.makedirs(args.pyworkPath,   exist_ok=True)
os.makedirs(args.pycropPath,   exist_ok=True)

face_detector = initialize_face_detector()

# ----------------------------------------------------------------------------
#  Main pipeline
# ----------------------------------------------------------------------------

def main():
    # 1. Extract video / audio / frames
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    if args.duration == 0:
        cmd = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" %
               (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
    else:
        cmd = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" %
               (args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
    subprocess.call(cmd, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " - Extracted video to %s\n" % args.videoFilePath)

    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    cmd = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" %
           (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(cmd, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " - Extracted audio to %s\n" % args.audioFilePath)

    cmd = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" %
           (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg')))
    subprocess.call(cmd, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " - Extracted frames to %s\n" % args.pyframesPath)

    # 2. Load DB **before** detection so we can label on‑the‑fly
    face_database = load_face_database(args.faceDB, face_detector, logger) if args.faceDB else {}

    # 3. Scene detection & per‑frame inference (with identities)
    scene = scene_detect(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " - Scene detection complete\n")

    faces = inference_video(args, face_database, args.simThreshold)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " - Face detection+recognition complete\n")

    # Optional masking setup
    args.pyframes2Path = os.path.join(args.savePath, 'pyframes2')
    if args.face_masking:
        if os.path.exists(args.pyframes2Path):
            rmtree(args.pyframes2Path)
        os.makedirs(args.pyframes2Path)
        for src in glob.glob(os.path.join(args.pyframesPath, '*.jpg')):
            shutil.copyfile(src, os.path.join(args.pyframes2Path, os.path.basename(src)))

    # Pre‑load frames
    frame_files = sorted(glob.glob(os.path.join(args.pyframesPath, '*.jpg')))
    frames      = [cv2.imread(f) for f in frame_files]
    fh, fw      = frames[0].shape[:2]

    # 4. Tracking + label assignment (majority vote)
    allTracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num < args.minTrack:
            continue
        shot_faces = faces[shot[0].frame_num:shot[1].frame_num]
        tracks     = track_shot(args, shot_faces)
        pad        = 0
        for track in tracks:
            # majority vote among detection identities
            if track['identities']:
                label = Counter(track['identities']).most_common(1)[0][0]
            else:
                label = 'Unknown'
            track['track_label'] = label

            # optional masking
            if args.face_masking and label == 'Unknown':
                for fno, bb in zip(track['frame'], track['bbox']):
                    fn  = os.path.basename(frame_files[fno])
                    fp2 = os.path.join(args.pyframes2Path, fn)
                    img2 = cv2.imread(fp2)
                    if img2 is None:
                        continue
                    x1,y1,x2,y2 = map(int, bb)
                    x1,x2 = max(0, x1 - pad), min(img2.shape[1], x2 + pad)
                    y1,y2 = max(0, y1 - pad), min(img2.shape[0], y2 + pad)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    img2[y1:y2, x1:x2] = cv2.GaussianBlur(img2[y1:y2, x1:x2], (0,0), 10)
                    # img2 = radial_blur(img2, (x1, y1, x2, y2), sigma_blur=35)
                    cv2.imwrite(fp2, img2)
                continue  # skip adding blurred track for ASD
            allTracks.append(track)

    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" - Tracking complete. {len(allTracks)} tracks kept\n")

    # If masking was applied, switch to new frame path
    if args.face_masking:
        args.pyframesPath = args.pyframes2Path

    # 5. Crop clips for TalkNet
    vidTracks = []
    for ii, track in enumerate(allTracks):
        cropFile = os.path.join(args.pycropPath, f"{ii:05d}")
        vidTracks.append(crop_video(args, track, cropFile))
    with open(os.path.join(args.pyworkPath, 'tracks.pckl'), 'wb') as fil:
        pickle.dump(vidTracks, fil)

    # 6. Active speaker detection (unchanged)
    files  = sorted(glob.glob(f"{args.pycropPath}/*.avi"))
    scores = evaluate_network(files, args)
    with open(os.path.join(args.pyworkPath, 'scores.pckl'), 'wb') as fil:
        pickle.dump(scores, fil)

    # 7. Visualisation
    visualization(vidTracks, scores, args)

if __name__ == '__main__':
    main()
