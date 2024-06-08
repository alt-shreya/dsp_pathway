import sys, os

def process_cli_arguments():
    # Check if the right number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <your_audio_filepath>")
        sys.exit(1)  # Exit the script if not the correct number of arguments
    
    # Retrieve arguments
    audio_path = sys.argv[1]
    if not os.path.isfile(audio_path):
        print("The specified file does not exist.")
        sys.exit(1)

    csv_path = os.path.splitext(audio_path)[0] + '.csv'
    return audio_path, csv_path
