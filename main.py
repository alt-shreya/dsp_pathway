import pathway as pw
from scripts.extract import extract_volume_levels
from scripts.audio_cleaner import export_audio
from scripts.utils import process_cli_arguments


# Define variables
DB_THRESHOLD = -60
RATIO = 2.0
ATTACK = 0.1
RELEASE = 0.5
KNEE = 10

# define final adjusted output file
OUTPUT_CSV_FILE = './assets/output_stream.csv'


# define input schema
class InputSchema(pw.Schema):
    time: float
    decibel: float


def main():
    audio_path, input_path = process_cli_arguments()
    audio_csv = extract_volume_levels(audio_path, input_path)

    # define table
    audio_stream = pw.io.csv.read(
    audio_csv,
    schema=InputSchema,
    mode="static"
    )

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
    

    # Apply the UDF to the decibel column
    adjusted_audio_stream = audio_stream.select(
    t0=pw.this.time,
    decibel = pw.this.decibel,
    new_decibel=adjust_decibels(pw.this.decibel)
)

    # Output to CSV
    pw.io.csv.write(adjusted_audio_stream, OUTPUT_CSV_FILE)
    pw.run()
    return audio_path, OUTPUT_CSV_FILE

    
    

# Call the function if this script is executed
if __name__ == '__main__':
    audio_path, OUTPUT_CSV_FILE = main()
    new_audio = export_audio(OUTPUT_CSV_FILE, audio_path)
    print(f"The audio file is available at {new_audio}")

    
