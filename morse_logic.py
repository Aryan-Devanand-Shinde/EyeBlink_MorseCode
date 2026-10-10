import cv2
import mediapipe as mp
import time
import math


# ============================================================
# 1. MediaPipe Face Landmarker
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)


landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# 2. Eye Landmark Points
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]

RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ============================================================
# 3. EAR Calculation
# ============================================================

def calculate_ear(landmarks, eye_points, width, height):

    points = []

    for index in eye_points:

        x = landmarks[index].x * width
        y = landmarks[index].y * height

        points.append((x, y))


    vertical_1 = math.dist(
        points[1],
        points[5]
    )


    vertical_2 = math.dist(
        points[2],
        points[4]
    )


    horizontal = math.dist(
        points[0],
        points[3]
    )


    ear = (
        vertical_1 + vertical_2
    ) / (2 * horizontal)


    return ear


# ============================================================
# 4. Webcam
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("❌ Cannot access webcam")

    raise SystemExit


print("=================================")
print("        BLINK RECORDER")
print("=================================")
print("Webcam started")
print("Press Q to stop")
print()


# ============================================================
# 5. Blink Settings
# ============================================================

EAR_THRESHOLD = 0.21

# Morse timing
DOT_MAX_TIME = 0.25
SPACE_TIME = 0.5


# ============================================================
# 6. Blink Event Variables
# ============================================================

blink_active = False

blink_start_time = None


# Remember which eyes participated
left_was_closed = False
right_was_closed = False


# Store detected blinks
blinks = []


# ============================================================
# Morse Variables
# ============================================================

morse_string = ""

# Time when both eyes became open
eyes_open_start_time = None

# Prevent multiple spaces during one continuous
# eye-open period
space_added = False


# MediaPipe timestamp
timestamp_ms = 0


# ============================================================
# 7. Main Loop
# ============================================================

while True:

    # --------------------------------------------------------
    # Read webcam
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print("❌ Cannot read webcam frame")

        break


    # Mirror image
    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Create MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_ms += 33


    # --------------------------------------------------------
    # Detect face landmarks
    # --------------------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # 8. Process Face
    # ========================================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # ----------------------------------------------------
        # Calculate LEFT EAR
        # ----------------------------------------------------

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE,
            width,
            height
        )


        # ----------------------------------------------------
        # Calculate RIGHT EAR
        # ----------------------------------------------------

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE,
            width,
            height
        )


        # ----------------------------------------------------
        # Determine eye states
        # ----------------------------------------------------

        left_is_closed = (
            left_ear < EAR_THRESHOLD
        )


        right_is_closed = (
            right_ear < EAR_THRESHOLD
        )


        # ====================================================
        # 9. START BLINK EVENT
        # ====================================================

        if (
            left_is_closed or
            right_is_closed
        ) and not blink_active:

            blink_active = True

            blink_start_time = time.time()


            # Remember which eye started closing

            left_was_closed = left_is_closed

            right_was_closed = right_is_closed


            # Eyes are no longer continuously open
            eyes_open_start_time = None
            space_added = False


        # ====================================================
        # 10. BLINK EVENT IS ACTIVE
        # ====================================================

        elif blink_active:


            # -----------------------------------------------
            # Remember LEFT eye
            # -----------------------------------------------

            if left_is_closed:

                left_was_closed = True


            # -----------------------------------------------
            # Remember RIGHT eye
            # -----------------------------------------------

            if right_is_closed:

                right_was_closed = True


            # =================================================
            # Both eyes are open again
            # Blink is finished
            # =================================================

            if (
                not left_is_closed and
                not right_is_closed
            ):

                blink_end_time = time.time()


                duration = (
                    blink_end_time -
                    blink_start_time
                )


                # -------------------------------------------
                # Determine blink type
                # -------------------------------------------

                if (
                    left_was_closed and
                    right_was_closed
                ):

                    eye = "BOTH"


                elif left_was_closed:

                    eye = "LEFT"


                elif right_was_closed:

                    eye = "RIGHT"


                else:

                    eye = "UNKNOWN"


                # -------------------------------------------
                # Save blink
                # -------------------------------------------

                blinks.append(
                    (
                        eye,
                        duration
                    )
                )


                # -------------------------------------------
                # Convert blink to Morse
                # -------------------------------------------

                if duration <= DOT_MAX_TIME:

                    morse_string += "."

                    print(
                        f"👁️ {eye} blink | "
                        f"{duration:.2f} seconds | "
                        f"MORSE: ."
                    )

                else:

                    morse_string += "-"

                    print(
                        f"👁️ {eye} blink | "
                        f"{duration:.2f} seconds | "
                        f"MORSE: -"
                    )


                # -------------------------------------------
                # Reset event
                # -------------------------------------------

                blink_active = False

                blink_start_time = None

                left_was_closed = False

                right_was_closed = False


                # Eyes have just opened
                eyes_open_start_time = time.time()

                space_added = False


        # ====================================================
        # 11. SPACE DETECTION
        # ====================================================

        if (
            not left_is_closed and
            not right_is_closed
        ):

            # Start counting open-eye time
            if eyes_open_start_time is None:

                eyes_open_start_time = time.time()


            open_duration = (
                time.time() -
                eyes_open_start_time
            )


            # Add space after 0.5 seconds
            # of continuous open eyes

            if (
                open_duration >= SPACE_TIME
                and not space_added
            ):

                # Only add space if there is
                # already Morse content

                if (
                    morse_string
                    and not morse_string.endswith(" ")
                ):

                    morse_string += " "

                    print("MORSE: SPACE")


                space_added = True


        else:

            # Eyes are closed
            # Reset open-eye timer

            eyes_open_start_time = None


            # Allow another space after
            # the next blink/open period

            space_added = False


        # ====================================================
        # 12. Display Eye Status
        # ====================================================

        if (
            left_is_closed and
            right_is_closed
        ):

            eye_status = "BOTH EYES CLOSED"


        elif left_is_closed:

            eye_status = "LEFT EYE CLOSED"


        elif right_is_closed:

            eye_status = "RIGHT EYE CLOSED"


        else:

            eye_status = "EYES OPEN"


        # ====================================================
        # 13. Display EAR values
        # ====================================================

        cv2.putText(
            frame,
            f"Left EAR: {left_ear:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Right EAR: {right_ear:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ====================================================
        # 14. Display Eye Status
        # ====================================================

        cv2.putText(
            frame,
            eye_status,
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # ====================================================
        # 15. Display Blink Count
        # ====================================================

        cv2.putText(
            frame,
            f"Blinks: {len(blinks)}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ====================================================
        # 16. Display Morse String
        # ====================================================

        cv2.putText(
            frame,
            f"Morse: {morse_string}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )


    # ========================================================
    # 17. Show Webcam
    # ========================================================

    cv2.imshow(
        "Blink Recorder",
        frame
    )


    # ========================================================
    # 18. Press Q to Stop
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord('q'):

        break


    # ========================================================
    # 19. Detect Window Close
    # ========================================================

    if cv2.getWindowProperty(
        "Blink Recorder",
        cv2.WND_PROP_VISIBLE
    ) < 1:

        break


# ============================================================
# 20. Cleanup
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


# ============================================================
# 21. Display Recorded Blinks
# ============================================================

print()

print("=================================")
print("        RECORDED BLINKS")
print("=================================")


if len(blinks) == 0:

    print("No blinks detected.")


else:

    for i, (eye, duration) in enumerate(
        blinks,
        start=1
    ):

        print(
            f"Blink {i}: "
            f"{eye} | "
            f"{duration:.2f} seconds"
        )


# ============================================================
# 22. Final Morse String
# ============================================================

print()

print("=================================")
print("        MORSE STRING")
print("=================================")

print(morse_string)

print()

print("Program stopped.")

from morse_to_text import morse_to_text
m_t_t = morse_to_text(morse_string)
print(f"This the raw outpu: {m_t_t}")

from genai import generate_response

prompt = f"""
Think urself as an Expert in Decoding morse code 
so u have to get the correct output from the raw morsecode
their maybe some mistakes like missinng or an extra char, but u have to get through it and give accurate ans.

morse_code_string:
{morse_string}

Morse_to_text_coverted:
{m_t_t}

Answer:
"""
result = generate_response(prompt)
print(result)