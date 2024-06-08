import pandas as pd
from pydub import AudioSegment

def export_audio(filepath, original_audio_filepath):
    # Load the CSV data
    df = pd.read_csv(filepath)

    # Remove unwanted columns and sort by 't0'
    df = df.drop(columns=['time', 'diff']).sort_values(by='t0')

    # Save the cleaned data to a new CSV if needed
    df.to_csv('./assets/cleaned_output_stream.csv', index=False)

    audio = AudioSegment.from_file(original_audio_filepath)

    new_audio = AudioSegment.silent(duration=len(audio))

    for index, row in df.iterrows():
        start_time = int(row['t0'] * 1000)
        end_time = start_time + 1000
        if end_time > len(audio):
            end_time = len(audio)
        segment = audio[start_time:end_time]

        # Calculate the required dB change
        if row['new_decibel'] != segment.dBFS:
            # Apply a gain to normalize this segment to the desired dBFS
            gain = row['new_decibel'] - segment.dBFS
            segment = segment.apply_gain(gain)

        # Overlay the adjusted segment back to the new audio track
        new_audio = new_audio.overlay(segment, position=start_time)

    new_audio_filepath = './assets/adjusted_audio_file.wav'
    # Export the adjusted audio
    new_audio.export(new_audio_filepath, format='wav')

    return new_audio_filepath