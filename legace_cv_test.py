import cv2
import matplotlib.pyplot as plt

# define a video capture object
vid = cv2.VideoCapture(0)

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Resize the frame
    frame = cv2.resize(frame, (640, 480))

    # Change resolution
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply blur
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Apply erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame = cv2.erode(frame, kernel, iterations=1)

    # Apply dilation
    frame = cv2.dilate(frame, kernel, iterations=1)

    # Rotate 90 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # Convert frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Calculate color statistics
    color_stats = cv2.mean(frame_rgb)

    # Plot color statistics
    plt.bar(
        range(len(color_stats)), color_stats, color=["red", "green", "blue", "alpha"]
    )
    plt.xticks(range(len(color_stats)), ["Red", "Green", "Blue", "Alpha"])
    plt.xlabel("Color Channel")
    plt.ylabel("Mean Value")
    plt.title("Color Statistics")
    plt.show()

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
