import librosa
import librosa.display
import numpy as np
import pandas as pd


def extract_volume_levels(audio_path, output_csv_path):
    # Load audio file
    y, sr = librosa.load(audio_path)
    
    # Get the amplitude envelope
    amplitude_envelope = np.abs(librosa.stft(y))
    amplitude_envelope_db = librosa.amplitude_to_db(amplitude_envelope, ref=np.max)
    
    # Calculate time for each frame
    times = librosa.frames_to_time(np.arange(amplitude_envelope.shape[1]), sr=sr)
    
    # Calculate mean dB level for each time frame
    mean_db_per_frame = np.mean(amplitude_envelope_db, axis=0)
    
    # Create a DataFrame
    df = pd.DataFrame({'time': times, 'decibel': mean_db_per_frame})
    df.to_csv(output_csv_path, index=False)
    
    print(f"Volume data of your original audio saved to {output_csv_path}")
    
    return output_csv_path




