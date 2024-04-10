from pytube import Search, YouTube
import cv2
import easyocr
import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


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


def find_scenes(video_path, threshold=30):
    """
    Finds scenes in a video using PySceneDetect.

    :param video_path: Path to the video file.
    :param threshold: Sensitivity threshold for scene detection.
    :return: A list of frame numbers where scenes change.
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)

    return scene_manager.get_scene_list(base_timecode=video_manager.get_base_timecode())


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

    :param frames: A list of frame images on which OCR needs to be performed.
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
        print("Adding watermarks to frames...")
        add_watermark(frames_with_paths)
        print("Performing OCR on extracted frames...")
        texts = perform_ocr_on_frames(frames_with_paths)

        for i, text in enumerate(texts, start=1):
            print(f"Text from Frame {i}: {text}")

        print("Summary creation complete.")


if __name__ == "__main__":
    main()
