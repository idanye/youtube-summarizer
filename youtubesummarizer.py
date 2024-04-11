from pytube import Search, YouTube
import cv2
import easyocr
import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import sys
import imageio.v2 as imageio
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress DeprecationWarnings

# Suppressing other warnings (e.g., CUDA/MPS availability from easyocr)
original_stderr = sys.stderr  # Backup original stderr
sys.stderr = open(os.devnull, 'w')  # Redirect stderr to devnull


def download_video(query):
    search = Search(query)

    for video in search.results:
        if video.length < 600:  # Finds videos less than 10 minutes
            yt = YouTube(video.watch_url)
            print(f"Downloading: {yt.title}")
            ys = yt.streams.get_highest_resolution()
            video_path = ys.download()
            print("Download completed!")
            return yt.title, video_path  # Return both title and file path

    print("No suitable video found.")
    return None, None


def create_animated_gif(frames_with_paths, output_gif_path='animated_summary.gif', max_duration=10):
    """
    Creates an animated GIF from the provided frame paths.

    :param frames_with_paths: List of tuples containing frame images and their paths.
    :param output_gif_path: Path where the GIF should be saved.
    :param max_duration: Maximum duration of the GIF in seconds.
    """
    images = []
    for _, path in frames_with_paths:
        images.append(imageio.imread(path))

    # Calculate frame duration to evenly distribute frames within the max_duration
    frame_duration = max_duration / len(frames_with_paths)

    imageio.mimsave(output_gif_path, images, duration=frame_duration)
    print(f"Animated GIF saved as {output_gif_path}")


def open_gif(gif_path):
    """
    Opens the GIF using the default program.
    """
    import os
    import platform
    import subprocess

    if platform.system() == "Windows":
        os.startfile(gif_path)
    else:
        opener = "open" if platform.system() == "Darwin" else "xdg-open"
        subprocess.call([opener, gif_path])


def find_scenes(video_path, threshold=30, min_scene_length_seconds=0.5):
    """
    Finds scenes in a video using PySceneDetect.

    :param video_path: Path to the video file.
    :param threshold: Sensitivity threshold for scene detection.
    :param min_scene_length_seconds: Minimum length of a scene in seconds.
    :return: A list of frame numbers where scenes change.
    """
    # First, use OpenCV to get the frame rate of the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()  # Don't forget to release the video capture

    # Now proceed with PySceneDetects VideoManager and SceneManager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    # Calculate the minimum scene length in frames using the obtained frame rate
    min_scene_length_frames = int(fps * min_scene_length_seconds)

    # Add a content detector with the specified threshold and minimum scene length
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_length_frames))

    # Start processing the video
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Return the list of detected scenes
    return scene_manager.get_scene_list(video_manager.get_base_timecode())


def extract_frames(video_path, scene_list, output_dir='./images'):
    """
    Extracts key frames based on scene list.

    :param output_dir:
    :param video_path: Path to the video file.
    :param scene_list: List of scenes to extract frames from.
    :return: A list of frame images.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for index, (start_time, _) in enumerate(scene_list):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_time.get_seconds() * 1000))
        ret, frame = cap.read()

        if ret:
            frame_path = os.path.join(output_dir, f"frame_{index + 1}.jpg")
            cv2.imwrite(frame_path, frame)  # Save frame as JPEG file
            frames.append((frame, frame_path))

    cap.release()
    return frames


def add_watermark(frames_with_paths, watermark_text="Idan Yehiel"):
    """
    Adds a watermark to the bottom right corner of each frame and saves the modified frame.

    :param frames_with_paths: List of tuples containing frame images and their paths.
    :param watermark_text: Text to use as the watermark.
    """
    for frame, path in frames_with_paths:
        cv2.putText(frame, watermark_text, (frame.shape[1] - 200, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.imwrite(path, frame)  # Save the watermarked frame back to the file


def perform_ocr_on_frames(frames_with_paths):
    """
    Performs OCR on the given list of frames using EasyOCR.

    :param frames_with_paths: A list of frame images on which OCR needs to be performed.
    :return: A list of strings extracted from each frame.
    """
    reader = easyocr.Reader(['en'])
    extracted_texts = []

    for _, path in frames_with_paths:
        frame = cv2.imread(path)  # Read the frame from file
        result = reader.readtext(frame)
        texts = ' '.join([text[1] for text in result])
        extracted_texts.append(texts)

    return extracted_texts


def main():
    subject = input("Please enter a subject for the video: ")
    video_title, video_path = download_video(subject)

    if video_title:
        print(f"Video '{video_title}' downloaded. Detecting scenes...")
        scene_list = find_scenes(video_path)
        print("Extracting frames from detected scenes...")
        frames_with_paths = extract_frames(video_path, scene_list)
        print("Performing OCR on extracted frames...")
        texts = perform_ocr_on_frames(frames_with_paths)

        for i, text in enumerate(texts, start=1):
            print(f"Text from Frame {i}: {text}")

        print("Adding watermarks to frames...")
        add_watermark(frames_with_paths)

        print("Creating animated GIF from frames...")
        create_animated_gif(frames_with_paths)

        concatenated_text = ' '.join(texts)  # Original line, results in redundant spaces
        # Normalize the spacing by splitting and re-joining the text
        normalized_text = ' '.join(concatenated_text.split())
        print("Concatenated Text from All Frames:", normalized_text)

        print("Opening animated GIF...")
        open_gif('animated_summary.gif')

        print("Summary creation complete.")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr  # Restore original stderr after execution
