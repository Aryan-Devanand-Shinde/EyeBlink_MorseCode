import cv2
import mediapipe as mp
import time
import math
import threading
from pathlib import Path

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from streamlit_autorefresh import st_autorefresh

from morse_to_text import morse_to_text
from genai import generate_response


# ============================================================
# Configuration — kept from your original blink code
# ============================================================
MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.21
DOT_MAX_TIME = 0.3
SPACE_TIME = 0.5


def calculate_ear(landmarks, eye_points, width, height):
    points = []
    for index in eye_points:
        x = landmarks[index].x * width
        y = landmarks[index].y * height
        points.append((x, y))

    vertical_1 = math.dist(points[1], points[5])
    vertical_2 = math.dist(points[2], points[4])
    horizontal = math.dist(points[0], points[3])

    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2 * horizontal)


class BlinkRecorder(VideoProcessorBase):
    """Streamlit browser-webcam version of the user's original blink logic."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset_state()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing {MODEL_PATH.name}. Put it beside app.py."
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def reset_state(self):
        self.blink_active = False
        self.blink_start_time = None
        self.left_was_closed = False
        self.right_was_closed = False
        self.blinks = []
        self.morse_string = ""
        self.eyes_open_start_time = None
        self.space_added = False
        self.timestamp_ms = 0
        self.left_ear = None
        self.right_ear = None
        self.eye_status = "Waiting for face"
        self.face_detected = False

    def reset(self):
        with self.lock:
            self.reset_state()

    def get_snapshot(self):
        with self.lock:
            return {
                "morse_string": self.morse_string,
                "blinks": list(self.blinks),
                "left_ear": self.left_ear,
                "right_ear": self.right_ear,
                "eye_status": self.eye_status,
                "face_detected": self.face_detected,
            }

    def recv(self, video_frame):
        frame = video_frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        self.timestamp_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        with self.lock:
            if result.face_landmarks:
                self.face_detected = True
                landmarks = result.face_landmarks[0]

                left_ear = calculate_ear(landmarks, LEFT_EYE, width, height)
                right_ear = calculate_ear(landmarks, RIGHT_EYE, width, height)
                self.left_ear = left_ear
                self.right_ear = right_ear

                left_is_closed = left_ear < EAR_THRESHOLD
                right_is_closed = right_ear < EAR_THRESHOLD

                # START BLINK EVENT — same logic as original
                if (
                    left_is_closed or right_is_closed
                ) and not self.blink_active:
                    self.blink_active = True
                    self.blink_start_time = time.time()
                    self.left_was_closed = left_is_closed
                    self.right_was_closed = right_is_closed
                    self.eyes_open_start_time = None
                    self.space_added = False

                # BLINK EVENT ACTIVE — same logic as original
                elif self.blink_active:
                    if left_is_closed:
                        self.left_was_closed = True
                    if right_is_closed:
                        self.right_was_closed = True

                    if not left_is_closed and not right_is_closed:
                        duration = time.time() - self.blink_start_time

                        if self.left_was_closed and self.right_was_closed:
                            eye = "BOTH"
                        elif self.left_was_closed:
                            eye = "LEFT"
                        elif self.right_was_closed:
                            eye = "RIGHT"
                        else:
                            eye = "UNKNOWN"

                        self.blinks.append((eye, duration))
                        if duration <= DOT_MAX_TIME:
                            self.morse_string += "."
                        else:
                            self.morse_string += "-"

                        self.blink_active = False
                        self.blink_start_time = None
                        self.left_was_closed = False
                        self.right_was_closed = False
                        self.eyes_open_start_time = time.time()
                        self.space_added = False

                # SPACE DETECTION — same 0.5-second behavior as original
                if not left_is_closed and not right_is_closed:
                    if self.eyes_open_start_time is None:
                        self.eyes_open_start_time = time.time()

                    open_duration = time.time() - self.eyes_open_start_time
                    if open_duration >= SPACE_TIME and not self.space_added:
                        if (
                            self.morse_string
                            and not self.morse_string.endswith(" ")
                        ):
                            self.morse_string += " "
                        self.space_added = True
                else:
                    self.eyes_open_start_time = None
                    self.space_added = False

                if left_is_closed and right_is_closed:
                    self.eye_status = "BOTH EYES CLOSED"
                elif left_is_closed:
                    self.eye_status = "LEFT EYE CLOSED"
                elif right_is_closed:
                    self.eye_status = "RIGHT EYE CLOSED"
                else:
                    self.eye_status = "EYES OPEN"

                cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, self.eye_status, (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Blinks: {len(self.blinks)}", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Morse: {self.morse_string[-40:]}", (20, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                self.face_detected = False
                self.eye_status = "NO FACE DETECTED"
                cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


# ============================================================
# Streamlit UI
# Important fix: no blocking `while ctx.state.playing` loop.
# Auto-refresh saves snapshots; results stay visible after Stop.
# ============================================================
st.set_page_config(page_title="Blink Morse Translator", page_icon="👁️", layout="wide")
st.title("👁️ Blink Morse Translator")
st.write(
    "Blink to generate Morse code. After you press Stop, the final Morse "
    "string and decoded text remain on the page."
)
st.info(
    "Short blink = dot (.). Long blink = dash (-). Keep both eyes open for "
    "0.5 seconds to add a separator. Stop the camera when finished."
)

if "last_snapshot" not in st.session_state:
    st.session_state.last_snapshot = {
        "morse_string": "",
        "blinks": [],
        "left_ear": None,
        "right_ear": None,
        "eye_status": "Waiting for face",
        "face_detected": False,
    }

ctx = webrtc_streamer(
    key="blink-recorder",
    video_processor_factory=BlinkRecorder,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Refresh the Streamlit script periodically while the WebRTC stream is active.
if ctx.state.playing:
    st_autorefresh(interval=500, key="blink_morse_refresh")

# Save the latest available processor snapshot on every rerun.
if ctx.video_processor is not None:
    st.session_state.last_snapshot = ctx.video_processor.get_snapshot()

snapshot = st.session_state.last_snapshot

if ctx.state.playing:
    st.caption("Camera is running. Press Stop above when you have finished.")
elif snapshot["morse_string"]:
    st.success("Camera stopped. Your final output is shown below.")
else:
    st.caption("Start the camera above. Your final output will appear here after Stop.")

st.subheader("1. Live / Final Morse String")
st.code(snapshot["morse_string"] or "(No Morse symbols recorded yet)")

left_col, right_col = st.columns(2)
left_ear = snapshot["left_ear"]
right_ear = snapshot["right_ear"]
left_col.metric("Left EAR", f"{left_ear:.3f}" if left_ear is not None else "—")
right_col.metric("Right EAR", f"{right_ear:.3f}" if right_ear is not None else "—")
st.write(f"Eye status: **{snapshot['eye_status']}**")
st.write(f"Detected blinks: **{len(snapshot['blinks'])}**")

# Decode on every rerun, including the rerun triggered by pressing Stop.
morse_string = snapshot["morse_string"]
m_t_t = morse_to_text(morse_string) if morse_string else ""

st.subheader("2. Decoded Text")
st.write("**Raw output from morse_to_text():**")
st.success(m_t_t if m_t_t else "(No text decoded yet)")

with st.expander("Recorded blink events"):
    if snapshot["blinks"]:
        for i, (eye, duration) in enumerate(snapshot["blinks"], start=1):
            st.write(f"Blink {i}: {eye} | {duration:.2f} seconds")
    else:
        st.write("No blinks recorded.")

if st.button("✨ Ask Gemini to correct/interpret the output", disabled=not morse_string):
    prompt = f"""
Think of yourself as an expert in decoding Morse code.
Find the most likely intended output from the raw Morse code. There may be
mistakes such as a missing or extra character. Do not claim certainty if the
sequence is ambiguous.

morse_code_string:
{morse_string}

Morse_to_text_converted:
{m_t_t}

Answer:
"""
    try:
        with st.spinner("Asking Gemini..."):
            result = generate_response(prompt)
        st.subheader("3. Gemini Output")
        st.write(result)
    except Exception as exc:
        st.error(
            "Gemini request failed. Check GOOGLE_API_KEY, model access, network, "
            f"and quota. Details: {exc}"
        )

if st.button("Reset session"):
    if ctx.video_processor is not None:
        ctx.video_processor.reset()
    st.session_state.last_snapshot = {
        "morse_string": "",
        "blinks": [],
        "left_ear": None,
        "right_ear": None,
        "eye_status": "Waiting for face",
        "face_detected": False,
    }
    st.rerun()
