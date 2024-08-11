import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_frame(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply blur
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Apply erosion and dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    processed_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_OPEN, kernel)

    # Rotate 90 degrees
    rotated_frame = cv2.rotate(processed_frame, cv2.ROTATE_90_CLOCKWISE)

    return rotated_frame


def main():
    # Define a video capture object
    vid = cv2.VideoCapture(0)

    if not vid.isOpened():
        print("Failed to open video capture.")
        exit()

    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Resize the frame
        frame = cv2.resize(frame, (640, 480))

        # Process the frame
        processed_frame = process_frame(frame)

        # Display the resulting frame
        cv2.imshow("frame", processed_frame)

        # Calculate color statistics
        color_stats = np.mean(processed_frame, axis=(0, 1))

        # Plot color statistics
        plt.bar(
            range(len(color_stats)),
            color_stats,
            color=["red", "green", "blue", "alpha"],
        )
        plt.xticks(range(len(color_stats)), ["Red", "Green", "Blue", "Alpha"])
        plt.xlabel("Color Channel")
        plt.ylabel("Mean Value")
        plt.title("Color Statistics")
        plt.show()

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Save the processed frame as an image
        cv2.imwrite("processed_frame.jpg", processed_frame)

    # Release the video capture object
    vid.release()

    # Close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
