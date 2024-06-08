# 🎧 First Steps with Pathway: Basic Audio Processing

Recently, while watching a movie, I found myself squinting at the screen, trying to understand the actors as they were whispering. Being a problem solver, I cranked the volume up to max, eagerly awaiting the next line of dialogue. Instead, I was greeted with a loud jump scare that seemed to have a personal vendetta against my eardrums.


> 💡 **Disclaimer**: This article explores my first steps with Pathway, where I developed a feature to auto-adjust audio levels in an audio file. As this is a **beginner-friendly guide for Python developers curious to explore Pathway and digital signal processing (DSP)** , this article only scratches the surface of - and is not an exhaustive guide to - Pathway's capabilities.

[Skip to the code.](#the-code)

## Contents
1. [Introduction: Let's Start with Why](#lets-start-with-why)
2. [What's the Plan?](#whats-the-plan)
3. [The Code](#the-code)
4. [Final Thoughts](#final-thoughts)
5. [Next Steps](#next-steps)

## 🗣️ Let's Start with Why

I wanted to find out why this frustrating audio imbalance occurs in the first place. So I did some digging and discovered that movies are created for big theatres with dedicated speakers and separate audio channels to create an immersive experience for the audience. This is what makes a moment _cinematic_.

<img 
src="https://i.pinimg.com/originals/70/e4/b5/70e4b567bbcced1e998cccbe8b27fd28.jpg"
style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 80%;">

The sound is then adapted to smaller screens. During this mixing process, many adjustments occur, but the resulting sound isn't optimized for every device due to the sheer variety of streaming devices. 

This is a modern problem. And it required a modern solution. So, I set out to solve this using  Pathway, for reasons including but not limited to the fact that I want to work with their team.

## 🤔 What’s the Plan?
### TL;DR: Sample an audio file. If the amplitude of an instance exceeds a threshold value, reduce it. Otherwise, apply logic to increase the value. Test this on the downloaded audio file.

[Skip to the code.](#the-code)

###  Challenges/ Considerations
> 🗣️  **Dynamic range** is the ratio between the largest and the smallest amount of a signal, and in this case, the loudest and softest sounds. It determines how various sounds are represented. For example, if normal human speech is represented as 50dB, thunder, which is roughly 10 times louder, would be 500dB.

To preserve the "movie-like" audio experience, some dynamic range must be retained when the amplitude values are adjusted, to ensure, for instance, that whispers and louder effects remain distinct.

### The Granular Breakdown:
For this experiment, I've used [a scene from the movie, _Harry Potter and the Order of the Phoenix_](https://www.youtube.com/watch?v=nx8TX0c5tgI) . In this clip, the actors whisper and fight, offering a wide range of dB levels for the application to adjust. 

Here are the steps to build your own dynamic range processor with Pathway:

1. Convert the audio file to a CSV, representing each dB level by its mean with time.
2. Pass the CSV through Pathway and perform signal manipulation.
3. Extract the output file into a table, sort by time, and overlay it on the original audio.
4. Test the program on a downloaded file.

## 👩‍💻 The Code

Now, we'll dive into the code required to process audio using Pathway. By the end of this process, you'll have a new audio file with normalized volume levels.

### Step1: Extract the input audio file using the `librosa` library.

Import the necessary libraries

```python
import librosa
import librosa.display
import numpy as np
import pandas as pd
```

Write a function to extract volume levels from the file using `librosa` Python library. The function uses the [amplitude envelope](https://en.wikipedia.org/wiki/Envelope_(waves)#/media/File:Signal_envelopes.png) to sample time frames, and then calculates the average value of the volume in each frame.

```python
def extract_volume_levels(audio_path, output_csv_path):
    # Load audio file, into time series variable, y and sampling rate
    y, sampling_rate = librosa.load(audio_path)
    
    # Get the amplitude envelope and convert it to deciBel units
    amplitude_envelope = np.abs(librosa.stft(y))
    amplitude_envelope_db = librosa.amplitude_to_db(amplitude_envelope, ref=np.max)
    
    # Convert the envelope to time indices (this is the sampling process)
    times = librosa.frames_to_time(np.arange(amplitude_envelope.shape[1]), sr=sampling_rate )
    
    # Calculate mean dB level for each time frame
    mean_db_per_frame = np.mean(amplitude_envelope_db, axis=0)
    
    # Create a DataFrame
    df = pd.DataFrame({'time': times, 'decibel': mean_db_per_frame})
    df.to_csv(output_csv_path, index=False)
    
    # print a confirmation on the terminal
    print(f"Data saved to {output_csv_path}")
    return df

extract_volume_levels('YOUR_AUDIO_FILEPATH', 'OUTPUT_CSV_FILE')
```

### Step 2: Implementing the logic with Pathway

> #### 🗣️ A Brief Explanation of the Parameters for DSP
>
> * **Threshold**: The volume or dB level at which compression starts. 
> * **Ratio**: The amount by which the dB level is reduced once it surpasses the threshold. 
> * **Attack**: The duration the compressor takes to apply the full compression effect after the incoming signal exceeds the threshold (helps in smoothing the onset of compression).
> 
> * **Release**: The time it takes for the compression to stop after the signal falls below the threshold.
> 
> * **Knee**: Makes the compression start gradually around the threshold, leading to a more natural sound.

Define parameters of DSP.

```python
import pathway as pw

# Define variables
DB_THRESHOLD = -60
RATIO = 2.0
ATTACK = 0.1
RELEASE = 0.5
KNEE = 10
```

To pass the converted CSV from Step 1 as input to Pathway's framework for further processing, you'll need to first define a Schema, a class that informs Pathway about the type of data being processed. The Schema inherits from Pathway's `pw.Schema` base class. The simplest way to think of this is as a table, and its columns are defined by the attributes of the class.

Declare two columns to represent the input CSV file - time and decibel. Both of these are declared as float values, to implement real world audio data. 

> 📑 You can also find more ways to adapt to various use case in this [descriptive article about Schema implementation](https://pathway.com/developers/user-guide/connect/schema/).

```python
# define input schema
class InputSchema(pw.Schema):
    time: float
    decibel: float
```

Next, write an input Connector - Pathway's interface for extracting input data and exporting output data. Start by writing a simple input connector to read the CSV file generated in Step 1.

>  📑 You might also want to check out this [list of supported input sources](https://pathway.com/developers/user-guide/connect/supported-data-sources/) and this [list of supported connectors](https://pathway.com/developers/user-guide/connect/pathway-connectors/) to adapt to your use case.

```python
# define table
audio_stream = pw.io.csv.read(
  'YOUR_OUTPUT_FILE_FROM_STEP_1',
  schema=InputSchema, # defined earlier in Step 2
  mode="static" # this can be changed to streaming for live input
)
```
Next, we'll define the main processing logic using transformations in Pathway. Each transformation will return a new table, leaving the input table unchanged. You can define your processing pipeline, sequentially specifying the transformations your data will go through. 

> 📑 Although there are a lot of [readily available basic operations in Pathway's library](https://pathway.com/developers/user-guide/data-transformation/table-operations/), I decided to write my custom Python function for this pipeline using [parameters explained earlier](#️-a-brief-explanation-of-the-parameters-for-dsp). 

```python
# perform operations on the rows
# Define a user-defined function to adjust decibels

@pw.udf
def adjust_decibels(decibel: float) -> float:
    if decibel > DB_THRESHOLD + KNEE / 2:
            gain_reduction = (decibel - DB_THRESHOLD) / RATIO
    elif decibel < DB_THRESHOLD - KNEE / 2:
            gain_reduction = 0

    else:
        # Apply a soft knee compression
        knee_start = DB_THRESHOLD - KNEE / 2
        knee_end = DB_THRESHOLD + KNEE / 2
        gain_reduction = ((decibel - knee_start) / 
                          (knee_end - knee_start)) * (
                                (decibel - DB_THRESHOLD) / RATIO)

    # Apply the gain reduction smoothly
    adjusted_decibel = decibel - gain_reduction
    return adjusted_decibel
```

Now, set up an output connector, similar to the input connector, to save the function results in a CSV file. The output data will be produced as streams with five columns: 

* `t0`, `decibel`, `new_decibel`: represent the initial timestamp, original decibel level, and adjusted decibel level, respectively.
* `time`: represents the timestamp when the values were updated.
* `diff`: indicates whether a value has been added `(diff=1)` or removed `(diff=-1)`.

This setup ensures that you can track changes in the data over time and understand the adjustments made to the audio levels.

Finally, run the entire program using `pw.run()`.    

```python
# Apply the UDF to the decibel column
adjusted_audio_stream = audio_stream.select(
    t0=pw.this.time, # initial timestamp
    decibel = pw.this.decibel, # initial decibel value
    new_decibel=adjust_decibels(pw.this.decibel) # adjusted decibel value
)

# Output to CSV
pw.io.csv.write(adjusted_audio_stream, "PATH_TO_OUTPUT_CSV")
pw.run()
```

### Step 3: Refining the output
To create an audio file from the table obtained in Step 2, you need to overlay the original audio with the adjusted CSV file using `pydub` library. 


```python
import pandas as pd
from pydub import AudioSegment

# Load the CSV data
df = pd.read_csv('OUTPUT_CSV_FROM_STEP_2')

# Remove unwanted columns and sort by 't0'
df = df.drop(columns=['time', 'diff']).sort_values(by='t0')

# Save the cleaned data to a new CSV if needed
df.to_csv('PATH_TO_NEW_CSV', index=False)

# create a new audio segment equal in duration to the original audio
audio = AudioSegment.from_file('PATH_TO_YOUR_AUDIO')
n = len(audio)
new_audio = AudioSegment.silent(duration=n)

# iterate over every row in the dataframe
for index, row in df.iterrows():
    # convert time to ms
    start_time = int(row['t0'] * 1000)
    end_time = start_time + 1000
    # prevent errors while extracting a segment that ends beyond the audio's length
    if end_time > n:
        end_time = n
   # isolate a specific part of the audio
    segment = audio[start_time:end_time]

    # Check if dB value needs to be updated
    if row['new_decibel'] != segment.dBFS:
        # Apply a gain to normalize this segment to the desired dBFS
        gain = row['new_decibel'] - segment.dBFS
        segment = segment.apply_gain(gain)

    # Overlay the adjusted segment back to the new audio track
    new_audio = new_audio.overlay(segment, position=start_time)
```

Finally, export the combined audio to a new music file.

```python
# Export the adjusted audio
new_audio.export('PATH_TO_NEW_AUDIO', format='wav')
```
Test this out on different audio files. You can experiment with tuning [the parameters](#️-a-brief-explanation-of-the-parameters-for-dsp) to get different results. 

> Full code at [GitHub repo](https://github.com/alt-shreya/pathway.git).

And voila! Now you have created not only a much more balanced auditory experience but also fond memories of using Pathway and DSP libraries!

<img src="assets/blog/treasure.png"
style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 80%;">


## ✍️ Final Thoughts

This project provided a hands-on introduction to Pathway for audio signal processing. By extracting volume levels from an audio file, adjusting them based on a threshold, and overlaying the results onto the original audio, we created a more balanced listening experience!

To reiterate, this project only scratched the surface of Pathway's capabilities in handling real-time data transformations and demonstrates its potential for solving modern signal processing challenges.

This project has been a valuable learning experience, marking my first steps onto the `Pathway` of real-time event processing, through the domain of audio signal processing. I'm enthusiastic to explore more possibilities with Pathway, and push the boundaries of both digital and analog signal processing!


## ⏭️ Next Steps

As I continue to explore Pathway, I will refine these techniques and potentially integrate them into more advanced applications. I want to improve this project by:

* Implementing the same function for live audio through a mic input, or through a live-stream and other sources.
* Integrating ML libraries with Pathway to detect variations in the audio file, leading to an improved decibel detection.
* Building a mechanism to be able to play this on streaming websites, such as a browser plugin.

I'm excited to continue this journey and see where it leads! 