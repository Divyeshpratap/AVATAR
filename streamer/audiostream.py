import wave

def capture_audio(stream, frames_per_clip, audio_rate, video_fps, logger):
    seconds = frames_per_clip / video_fps
    frames = []
    chunk = 128
    total_frames = int(audio_rate / chunk * seconds)
    for _ in range(total_frames):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
    audio_data = b''.join(frames)
    return audio_data

def capture_audio_wrapper(stream, frames_per_clip, audio_rate, video_fps, output_queue, logger):
    audio_data = capture_audio(stream, frames_per_clip, audio_rate, video_fps, logger)
    output_queue.put(audio_data)
