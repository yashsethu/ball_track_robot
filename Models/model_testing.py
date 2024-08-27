import cv2
from picamera2 import Picamera2
import time

# Initialize the camera
camera = Picamera2()
camera.configure(
    camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
)
camera.start()
time.sleep(2)

# Allow the camera to warm up
time.sleep(0.1)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Capture frames from the camera
while True:
    # Retrieve the image array from the frame
    image = camera.capture_array()

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw rectangles around the detected faces
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Create a named window with fullscreen flag
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Face Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    # Display the output
    cv2.imshow("Face Detection", image)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera resources
camera.close()
cv2.destroyAllWindows()


# Define function to collect data from images
def collect_data_images(num_samples):
    # Create directory if it doesn't exist
    if not os.path.exists("datasets/images"):
        os.makedirs("datasets/images")

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Loop through the specified number of samples
    for i in range(num_samples):
        # Read the frame from the camera
        ret, frame = cap.read()

        # Save the frame as an image
        filename = f"datasets/images/sample_{i}.jpg"
        cv2.imwrite(filename, frame)

        # Display the frame
        cv2.imshow("frame", frame)
        cv2.waitKey(0)

    # Release the camera
    cap.release()
    # Close all windows
    cv2.destroyAllWindows()


# Define function to collect data from videos
def collect_data_videos(num_samples, duration):
    # Create directory if it doesn't exist
    if not os.path.exists("datasets/videos"):
        os.makedirs("datasets/videos")

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Loop through the specified number of samples
    for i in range(num_samples):
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        filename = f"datasets/videos/sample_{i}.avi"
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

        # Start recording time
        start_time = time.time()
        # Loop until specified duration is reached
        while int(time.time() - start_time) < duration:
            # Read the frame from the camera
            ret, frame = cap.read()
            # Write the frame to the video file
            out.write(frame)

            # Display the frame
            cv2.imshow("frame", frame)
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release the video file
        out.release()
        # Close all windows
        cv2.destroyAllWindows()

    # Release the camera
    cap.release()


# Define function to collect data from audio
def collect_data_audio(num_samples, duration):
    # Create directory if it doesn't exist
    if not os.path.exists("datasets/audio"):
        os.makedirs("datasets/audio")

    # Loop through the specified number of samples
    for i in range(num_samples):
        # Define filename
        filename = f"datasets/audio/sample_{i}.wav"
        # Convert duration to milliseconds
        duration_seconds = duration * 1000

        # Record audio using the default microphone
        audio = sd.rec(int(duration_seconds), samplerate=44100, channels=2)
        # Wait until recording is finished
        sd.wait()

        # Save the recorded audio as a WAV file
        sf.write(filename, audio, 44100)

        print(f"Audio sample {i} recorded.")

    print("Audio recording completed.")


# Define function to collect data from sensor
def collect_data_sensor(num_samples):
    # Create directory if it doesn't exist
    if not os.path.exists("datasets/sensor"):
        os.makedirs("datasets/sensor")

    # Loop through the specified number of samples
    for i in range(num_samples):
        # Read sensor data
        sensor_data = read_sensor()

        # Define filename
        filename = f"datasets/sensor/sample_{i}.txt"
        # Write sensor data to file
        with open(filename, "w") as file:
            file.write(sensor_data)

        print(f"Sensor data sample {i} collected.")

    print("Sensor data collection completed.")


# Define function to collect data from GPS
def collect_data_gps(num_samples):
    # Create directory if it doesn't exist
    if not os.path.exists("datasets/gps"):
        os.makedirs("datasets/gps")

    # Loop through the specified number of samples
    for i in range(num_samples):
        # Read GPS data
        gps_data = read_gps()

        # Define filename
        filename = f"datasets/gps/sample_{i}.txt"
        # Write GPS data to file
        with open(filename, "w") as file:
            file.write(gps_data)

        print(f"GPS data sample {i} collected.")

    print("GPS data collection completed.")


# Define function to collect custom data
def collect_data_custom(num_samples):
    # Create directory if it doesn't exist
    if not os.path.exists("datasets/custom"):
        os.makedirs("datasets/custom")

    # Loop through the specified number of samples
    for i in range(num_samples):
        # Collect custom data
        custom_data = collect_custom_data()

        # Define filename
        filename = f"datasets/custom/sample_{i}.txt"
        # Write custom data to file
        with open(filename, "w") as file:
            file.write(custom_data)

        print(f"Custom data sample {i} collected.")

    print("Custom data collection completed.")


# Define function to train regression models
def train_models(X_train, y_train):
    # Model 1: Linear Regression
    model1 = LinearRegression()
    model1.fit(X_train, y_train)

    # Model 2: Decision Tree Regressor
    model2 = DecisionTreeRegressor()
    model2.fit(X_train, y_train)

    # Model 3: Random Forest Regressor
    model3 = RandomForestRegressor()
    model3.fit(X_train, y_train)

    # Model 4: Support Vector Regressor
    model4 = SVR()
    model4.fit(X_train, y_train)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color="blue", label="Actual")
    plt.plot(X_train, model1.predict(X_train), color="red", label="Linear Regression")
    plt.plot(X_train, model2.predict(X_train), color="green", label="Decision Tree")
    plt.plot(X_train, model3.predict(X_train), color="orange", label="Random Forest")
    plt.plot(X_train, model4.predict(X_train), color="purple", label="Support Vector")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Regression Models")
    plt.legend()
    plt.show()


# Define function for live inference
def live_inference():
    # Open the default camera
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frame from the camera
        ret, frame = cap.read()
        # Preprocess the frame if needed
        # Perform inference using the trained models
        # Display the results on the frame
        cv2.imshow("Live Inference", frame)
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # Release the camera
    cap.release()
    # Close all windows
    cv2.destroyAllWindows()


# Define function to generate data from webcam
def generate_data_from_webcam(num_samples):
    # Create directory if it doesn't exist
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Loop through the specified number of samples
    for i in range(num_samples):
        # Read the frame from the camera
        ret, frame = cap.read()

        # Resize and normalize the frame
        frame = cv2.resize(frame, (200, 200))
        frame = frame / 255.0

        # Save the frame as an image
        filename = f"datasets/sample_{i}.jpg"
        cv2.imwrite(filename, frame)

        # Display the frame
        cv2.imshow("frame", frame)
        cv2.waitKey(0)

        # Pause for 0.1 seconds
        time.sleep(0.1)

    # Release the camera
    cap.release()
    # Close all windows
    cv2.destroyAllWindows()


# Define function to preprocess image
def preprocess_image(image):
    # Resize the image
    image = cv2.resize(image, (100, 100))
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
