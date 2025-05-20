#!/usr/bin/env python3
"""
Real-time capture → face-mask / blur unknown faces → encode & mux to MP4.
  • Camera          : index 0  (change with --cam)
  • Microphone      : PyAudio device 7 (USB C920 mic)  (change with --mic)
  • Resolution/FPS  : read from camera
  • Output          : synced_output.mp4
"""

import argparse, time, os, queue, threading, fractions, signal, sys
import cv2, av, numpy as np, pyaudio
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

# ------------------------------------------------------------------ #
# 1.  CLI args
# ------------------------------------------------------------------ #
ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0, help="cv2.VideoCapture index")
ap.add_argument("--mic", type=int, default=3, help="PyAudio device index")
ap.add_argument("--faces_dir", default="registeredFaces", help="folder with sub-dirs of images")
ap.add_argument("--outfile", default="output/tmp/min_blur.mp4")
ap.add_argument("--duration", type=int, default=30,
                help="stop after N seconds (0 = until Ctrl-C)")
ap.add_argument('--face_masking', action='store_true', help='Enable face masking')
ap.add_argument('--no_face_masking', action='store_false', dest='face_masking', help='Disable face masking')
ap.set_defaults(face_masking=True)
args = ap.parse_args()

# ------------------------------------------------------------------ #
# 2.  Prepare face DB
# ------------------------------------------------------------------ #

print("[INFO] Building embeddings database …")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

database = {}          # person → list[512-D vectors]
for person in os.listdir(args.faces_dir):
    pdir = os.path.join(args.faces_dir, person)
    if not os.path.isdir(pdir): continue
    embs = []
    for f in os.listdir(pdir):
        img = cv2.imread(os.path.join(pdir, f))
        if img is None: continue
        faces = face_app.get(img)
        embs += [fc.normed_embedding for fc in faces] if faces else []
    if embs: database[person] = embs

print("[INFO] People in DB:", list(database.keys()))
THRESH   = 0.35        # cosine distance
PAD      = 0
BLUR_SIG = 10.0

def mask_unknown(frame_bgr):
    """returns processed frame"""
    faces = face_app.get(frame_bgr)
    frame   = frame_bgr.copy()
    if args.face_masking == True:
        for face in faces:
            x1,y1,x2,y2 = face.bbox.astype(int)
            x1p,y1p = max(0,x1-PAD), max(0,y1-PAD)
            x2p,y2p = min(frame.shape[1],x2+PAD), min(frame.shape[0],y2+PAD)

            emb = face.normed_embedding
            best_score, identity = -1.0, None
            for name, embs in database.items():
                for ref in embs:
                    # d = cosine(emb, ref)
                    score = np.dot(emb, ref)
                    if score > best_score:
                        best_score, best_identity = score, name

            if best_score > THRESH:
                identity = best_identity

            if identity is None:
                roi = frame[y1p:y2p, x1p:x2p]
                frame[y1p:y2p, x1p:x2p] = cv2.GaussianBlur(roi, (0,0), BLUR_SIG)
            # else:
            #     cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            #     cv2.putText(frame, f"{identity} ({best_score:.2f})",
            #                 (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, .6,
            #                 (0,255,0), 2)
    return frame

# ------------------------------------------------------------------ #
# 3.  Queues & capture threads
# ------------------------------------------------------------------ #

FrameQ = queue.Queue(maxsize=18000)   # 10 minutes
AudioQ = queue.Queue(maxsize=9375)   # 10 minutes
# stop   = threading.Event()
capture_stop = threading.Event()  

def video_capture():
    cap = cv2.VideoCapture(args.cam, cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERR] Cannot open camera", args.cam); return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened {width}×{height} @ {fps:.1f} fps")
    frame_num = 0
    while not capture_stop.is_set():
        ok, frame = cap.read()
        frame_num += 1
        
        print(f'Capturing frame number {frame_num}')
        if not ok: break 
        ts = time.monotonic()
        try: FrameQ.put_nowait((frame, ts))
        except queue.Full: pass  # drop
        if frame_num >= args.duration * 30:
            print(f'capture_stop is set now')
            capture_stop.set()
    cap.release()
    print(f'Adding none flag to Video Queue')
    FrameQ.put((None, None)) 

def audio_capture():
    pa   = pyaudio.PyAudio()
    RATE = 16000
    CHUNK= 1024      # 16 ms
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                     input=True, input_device_index=args.mic,
                     frames_per_buffer=CHUNK)

    while not capture_stop.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        # buf  = np.frombuffer(data, dtype=np.int16)
        # buf  = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)
        buf = np.frombuffer(data, dtype=np.int16)[None, :]
        ts   = time.monotonic()
        try: AudioQ.put_nowait((buf, ts))
        except queue.Full: pass
    stream.stop_stream(); stream.close(); pa.terminate()
    AudioQ.put((None, None)) 

# ------------------------------------------------------------------ #
# 4.  Encoder / muxer (runs in main thread)
# ------------------------------------------------------------------ #
def main():
    # --- start capture threads
    threading.Thread(target=video_capture, daemon=True).start()
    threading.Thread(target=audio_capture, daemon=True).start()

    # Wait for first video frame to know resolution
    # frame0, t0 = FrameQ.get()
    height, width = 480, 640
    print("[INFO] Encoder initialised")

    # --- container / streams
    container = av.open(args.outfile, "w")
    # vstream = container.add_stream("h264_nvenc", rate=30)
    try:
        vstream = container.add_stream("h264_nvenc", rate=30)
    except av.AVError:
        print("[WARN] h264_nvenc not found, falling back to libx264 (CPU).")
        vstream = container.add_stream("libx264", rate=30)
        vstream.options = {"preset": "veryfast"}   # choose speed/quality trade-off
    vstream.width, vstream.height = width, height
    vstream.pix_fmt = "yuv420p"
    vstream.time_base = fractions.Fraction(1, 30)
    # -------- AUDIO SETUP --------
    astream = container.add_stream("aac", rate=16_000)
    cc                 = astream.codec_context
    # cc.sample_rate     = 16_000
    # cc.channels        = 1
    # cc.format          = "s16p"          # s16 = signed 16-bit
    cc.layout          = "mono"         # channel layout
    astream.time_base  = fractions.Fraction(1, 16_000)
    # -----------------------------------

    # put the first frame back
    # FrameQ.put((frame0, t0))

    video_pts = 0               # counts in ticks of 1/fps  (vstream.time_base)
    audio_pts = 0               # counts in samples         (astream.time_base)
    video_eof = audio_eof = False
    while True:
        # ---------- VIDEO ----------
        try:
            frame, _ = FrameQ.get(timeout=1)
            if frame is None:         # sentinel
                print(f'received none image frame')
                video_eof = True
                if video_pts > args.duration * 30:
                    break
            else:
                vf = av.VideoFrame.from_ndarray(mask_unknown(frame),
                                                format="bgr24")
                vf.pts = video_pts
                video_pts += 1
                print(f'current frame number which is being processed is {video_pts}')
                for pkt in vstream.encode(vf):
                    container.mux(pkt)
                if video_pts >= args.duration * 30:
                    break
        except queue.Empty:
            pass

        # ---------- AUDIO ----------
        try:
            abuf, _ = AudioQ.get_nowait()
            if abuf is None:      # sentinel
                print(f'received none audio frame')
                audio_eof = True
            else:
                af = av.AudioFrame.from_ndarray(abuf, format="s16p",
                                                layout="mono")
                af.sample_rate = 16_000
                af.pts = audio_pts
                audio_pts += af.samples              # 1024 each chunk
                for pkt in astream.encode(af):
                    container.mux(pkt)
        except queue.Empty:
            pass

        # ---------- exit when fully done ----------
        # if video_eof or audio_eof or FrameQ.empty() or AudioQ.empty():
        #     print(f'exiting while true loop')
        #     break

    # --- flush
    for pkt in vstream.encode(): container.mux(pkt)
    for pkt in astream.encode(): container.mux(pkt)
    container.close()
    print("[INFO] Saved to", args.outfile)

def handler(sig,frm): capture_stop.set()
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

if __name__ == "__main__":
    main()
