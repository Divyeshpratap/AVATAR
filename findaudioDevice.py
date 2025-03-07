import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    print("Available Audio Input Devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"Device {i}: {info['name']} - {info['maxInputChannels']} Input Channels")
    p.terminate()

if __name__ == "__main__":
    list_audio_devices()
