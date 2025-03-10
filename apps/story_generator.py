import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.difficulty import Difficulty
from utils.text_to_story import text_to_story
from utils.text_to_audio import text_to_audio
from utils.keyword import sanitize_keyword
import soundfile as sf

audio_sample_rate_normal = 20000
audio_sample_rate_faster = 24000

import os
from datetime import datetime

current_directory = os.path.dirname(os.path.abspath(__file__))
output_file_path_parent = os.path.join(current_directory, "..", "published")


def open_file_with_dirs(file_path, mode):
    """
    Opens a file and creates any necessary parent directories if they do not exist.

    Parameters:
    file_path (str): The path to the file to be opened.
    mode (str): The mode in which the file is to be opened.

    Returns:
    file object: The opened file object.
    """
    # Create parent directories if they do not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open the file
    return open(file_path, mode)


def generate(input: str, difficulty: Difficulty, output_file_full_path: str):
    print(
        f"\n----------Generating story for [{difficulty.value}] difficulty with input [{input}]..."
    )

    story = text_to_story(input, difficulty)

    with open(f"{output_file_full_path}.story.txt", "w") as f:
        f.write(story)

    print(
        f"\n----------Generating audio for [{difficulty.value}] difficulty with input [{input}]..."
    )

    audio = text_to_audio(story)

    sf.write(f"{output_file_full_path}.normal.wav", audio, audio_sample_rate_normal)
    sf.write(f"{output_file_full_path}.faster.wav", audio, audio_sample_rate_faster)


print("\n----------Story Generator----------")

while True:
    inputText = input("input some text (q to exit): ")

    if inputText == "q":
        break

    keyword = sanitize_keyword(inputText)

    output_file_path = os.path.join(output_file_path_parent, keyword)
    file_name = datetime.now().strftime("%Y%m%d%H%M%S")

    output_file_full_path = os.path.join(output_file_path, file_name)

    with open_file_with_dirs(f"{output_file_full_path}.input.txt", "w") as f:
        f.write(inputText)

    # file_name = file_name + "_" + Difficulty.PRIMARY.value
    # generate(inputText, Difficulty.PRIMARY, output_file_full_path)
    for difficulty in Difficulty:
        output_file_full_path2 = output_file_full_path + "_" + difficulty.value
        generate(inputText, difficulty, output_file_full_path2)
