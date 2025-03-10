from kokoro import KPipeline

# from IPython.display import display, Audio
import numpy as np

def text_to_audio(text: str):
    """
    Converts text to audio using the KPipeline text-to-speech engine and concatenates the audio segments.

    Parameters:
    text (str): The text to be converted to speech.
    sample_rate (int): The sample rate for the audio output. Default is 20000.
    speech_file_path (str): The file path to save the generated speech audio. Default is 'speech'.

    Returns:
    np.ndarray: The concatenated audio array.
    """
    # Install espeak, used for English OOD fallback and some non-English languages
    # apt-get -qq -y install espeak-ng > /dev/null 2>&1

    ae_pipeline = KPipeline(lang_code="a")

    generator = ae_pipeline(
        text,
        voice="af_heart",  # <= change voice here
        speed=1,
        split_pattern="",  # don't split text
        # split_pattern=r'\n+'
    )

    # control speed, bigger number means faster

    audios = []

    for i, (gs, ps, audio) in enumerate(generator):
        print("i:", i)  # i => index
        print("gs: ", gs)  # gs => graphemes/text
        print("ps: ", ps)  # ps => phonemes
        # display(Audio(data=audio, rate=sample_rate, autoplay=i==0))

        audios.append(audio)

        # Save audio to file
        # sf.write(f'{speech_file_path}-{i}.wav', audio, sample_rate)

    # Concatenate audios
    return np.concatenate(audios, axis=None)
