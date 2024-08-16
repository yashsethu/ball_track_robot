import cv2
import os


def capture_images(directory):
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow(
            f'Press Enter to save the image, Delete to discard, "q" to quit ({directory})',
            frame,
        )

        # Wait for user feedback
        key = cv2.waitKey(1)

        # If Enter is pressed, save the frame to the directory
        if key == 13:  # 13 is the ASCII value for Enter
            cv2.imwrite(os.path.join(directory, f"image_{count}.jpg"), frame)
            count += 1

        # If Delete is pressed, discard the current frame
        elif key == 127:  # 127 is the ASCII value for Delete
            continue

        # If 'q' is pressed, break from the loop
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Capture images for each category
capture_images("datasets/face_images")
capture_images("datasets/no_face_images")
capture_images("datasets/my_face_images")
capture_images("datasets/other_face_images")
