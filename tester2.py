import cv2
from picamera2 import Picamera2
import time

camera = Picamera2()
camera.configure(
    camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
)
camera.start()
time.sleep(2)

time.sleep(0.1)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    image = camera.capture_array()

    faces = face_cascade.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Face Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    cv2.imshow("Face Detection", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.close()
cv2.destroyAllWindows()


def collect_data_images(num_samples):
    if not os.path.exists("datasets/images"):
        os.makedirs("datasets/images")

    cap = cv2.VideoCapture(0)

    for i in range(num_samples):
        ret, frame = cap.read()

        filename = f"datasets/images/sample_{i}.jpg"
        cv2.imwrite(filename, frame)

        cv2.imshow("frame", frame)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


def collect_data_videos(num_samples, duration):
    if not os.path.exists("datasets/videos"):
        os.makedirs("datasets/videos")

    cap = cv2.VideoCapture(0)

    for i in range(num_samples):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        filename = f"datasets/videos/sample_{i}.avi"
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

        start_time = time.time()
        while int(time.time() - start_time) < duration:
            ret, frame = cap.read()
            out.write(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.release()
        cv2.destroyAllWindows()

    cap.release()


def collect_data_audio(num_samples, duration):
    if not os.path.exists("datasets/audio"):
        os.makedirs("datasets/audio")

    for i in range(num_samples):
        filename = f"datasets/audio/sample_{i}.wav"
        duration_seconds = duration * 1000

        audio = sd.rec(int(duration_seconds), samplerate=44100, channels=2)
        sd.wait()

        sf.write(filename, audio, 44100)

        print(f"Audio sample {i} recorded.")

    print("Audio recording completed.")


def collect_data_sensor(num_samples):
    if not os.path.exists("datasets/sensor"):
        os.makedirs("datasets/sensor")

    for i in range(num_samples):
        sensor_data = read_sensor()

        filename = f"datasets/sensor/sample_{i}.txt"
        with open(filename, "w") as file:
            file.write(sensor_data)

        print(f"Sensor data sample {i} collected.")

    print("Sensor data collection completed.")


def collect_data_gps(num_samples):
    if not os.path.exists("datasets/gps"):
        os.makedirs("datasets/gps")

    for i in range(num_samples):
        gps_data = read_gps()

        filename = f"datasets/gps/sample_{i}.txt"
        with open(filename, "w") as file:
            file.write(gps_data)

        print(f"GPS data sample {i} collected.")

    print("GPS data collection completed.")


def collect_data_custom(num_samples):
    if not os.path.exists("datasets/custom"):
        os.makedirs("datasets/custom")

    for i in range(num_samples):
        custom_data = collect_custom_data()

        filename = f"datasets/custom/sample_{i}.txt"
        with open(filename, "w") as file:
            file.write(custom_data)

        print(f"Custom data sample {i} collected.")

    print("Custom data collection completed.")


def train_models(X_train, y_train):
    model1 = LinearRegression()
    model1.fit(X_train, y_train)

    model2 = DecisionTreeRegressor()
    model2.fit(X_train, y_train)

    model3 = RandomForestRegressor()
    model3.fit(X_train, y_train)

    model4 = SVR()
    model4.fit(X_train, y_train)

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


def live_inference():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Live Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def generate_data_from_webcam(num_samples):
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    cap = cv2.VideoCapture(0)

    for i in range(num_samples):
        ret, frame = cap.read()

        frame = cv2.resize(frame, (200, 200))
        frame = frame / 255.0

        filename = f"datasets/sample_{i}.jpg"
        cv2.imwrite(filename, frame)

        cv2.imshow("frame", frame)
        cv2.waitKey(0)

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()


def preprocess_image(image):
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
