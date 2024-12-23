import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

def draw_text(image, text, position, font_size=10, color=(0, 255, 0), outline_color=(0, 0, 0), outline_thickness=2):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("arial.ttf", font_size)  
    x, y = position
    for dx in range(-outline_thickness, outline_thickness + 1):
        for dy in range(-outline_thickness, outline_thickness + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    
    draw.text(position, text, font=font, fill=color)
    
    return np.array(pil_image)

def process_image(image_path):
    reader = easyocr.Reader(['en', 'ru'])
    image = cv2.imread(image_path)
    results = reader.readtext(image)
    
    for (bbox, text, prob) in results:
        print(f"Recognized text: {text} (probability: {prob:.2f})")
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        image = draw_text(image, text, (top_left[0], top_left[1] - 10), font_size=30)

    output_image_path = os.path.splitext(image_path)[0] + '_processed.jpg'
    cv2.imwrite(output_image_path, image)
    print(f"Processed image saved as: {output_image_path}")

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    reader = easyocr.Reader(['en', 'ru'])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_path = os.path.splitext(video_path)[0] + '_processed.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = reader.readtext(frame)
        for (bbox, text, prob) in results:
            try:
                print(f"Recognized text: {text} (probability: {prob:.2f})")
            except UnicodeEncodeError:
                print(f"Recognized text: {text.encode('utf-8', 'replace').decode('utf-8')} (probability: {prob:.2f})")
            
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            frame = draw_text(frame, text, (top_left[0], top_left[1] - 10), font_size=15)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved as: {output_path}")

if __name__ == "__main__":
    while True:
        input_type = input("Enter 'image' to process an image or 'video' to process a video (or 'exit' to quit): ")
        if input_type == 'exit':
            break
        input_path = input("Enter the path to the input file: ")

        if input_type == 'image':
            process_image(input_path)
        elif input_type == 'video':
            process_video(input_path)
        else:
            print("Invalid input type.")