# import os
# import time
# import math
# import threading
# from pathlib import Path

# import cv2
# import mediapipe as mp
# import streamlit as st
# from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
# import av

# from morse_to_text import morse_to_text


# # ============================================================
# # Page setup
# # ============================================================

# st.set_page_config(
#     page_title="Blink Morse Translator",
#     page_icon="👁️",
#     layout="wide",
# )

# st.title("👁️ Blink Morse Translator")
# st.write(
#     "Convert eye blinks into Morse code, translate the sequence into text, "
#     "and optionally ask Gemini to suggest a correction."
# )

# st.info(
#     "Allow camera access when your browser asks. Use short blinks for dots, "
#     "long blinks for dashes, and keep both eyes open for the configured "
#     "separator interval."
# )


# # ============================================================
# # Configuration — matching your current local script
# # ============================================================

# EAR_THRESHOLD = 0.21
# DOT_MAX_TIME = 0.25
# SPACE_TIME = 0.5

# LEFT_EYE = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"


# def calculate_ear(landmarks, eye_points, width, height):
#     """Calculate Eye Aspect Ratio (EAR) for one eye."""
#     points = []

#     for index in eye_points:
#         x = landmarks[index].x * width
#         y = landmarks[index].y * height
#         points.append((x, y))

#     vertical_1 = math.dist(points[1], points[5])
#     vertical_2 = math.dist(points[2], points[4])
#     horizontal = math.dist(points[0], points[3])

#     if horizontal == 0:
#         return 0.0

#     return (vertical_1 + vertical_2) / (2 * horizontal)


# class BlinkMorseProcessor(VideoProcessorBase):
#     """Process browser webcam frames and convert blinks into Morse code."""

#     def __init__(self):
#         self.lock = threading.Lock()
#         self.reset_state()

#         if not MODEL_PATH.exists():
#             raise FileNotFoundError(
#                 f"MediaPipe model not found: {MODEL_PATH}. "
#                 "Place face_landmarker.task beside app.py."
#             )

#         base_options = mp.tasks.BaseOptions(
#             model_asset_path=str(MODEL_PATH)
#         )
#         options = mp.tasks.vision.FaceLandmarkerOptions(
#             base_options=base_options,
#             running_mode=mp.tasks.vision.RunningMode.VIDEO,
#             num_faces=1,
#         )
#         self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
#             options
#         )

#     def reset_state(self):
#         # Called during initialization and when the user resets the app.
#         self.blink_active = False
#         self.blink_start_time = None
#         self.left_was_closed = False
#         self.right_was_closed = False
#         self.blinks = []
#         self.morse_string = ""
#         self.eyes_open_start_time = None
#         self.space_added = False
#         self.timestamp_ms = 0
#         self.left_ear = None
#         self.right_ear = None
#         self.eye_status = "Waiting for face"
#         self.face_detected = False

#     def reset(self):
#         with self.lock:
#             self.reset_state()

#     def get_snapshot(self):
#         with self.lock:
#             return {
#                 "morse_string": self.morse_string,
#                 "blinks": list(self.blinks),
#                 "left_ear": self.left_ear,
#                 "right_ear": self.right_ear,
#                 "eye_status": self.eye_status,
#                 "face_detected": self.face_detected,
#             }

#     def recv(self, video_frame):
#         frame = video_frame.to_ndarray(format="bgr24")
#         frame = cv2.flip(frame, 1)
#         height, width, _ = frame.shape

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=rgb_frame,
#         )

#         # MediaPipe VIDEO mode requires monotonically increasing timestamps.
#         self.timestamp_ms += 33
#         result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

#         with self.lock:
#             if result.face_landmarks:
#                 self.face_detected = True
#                 landmarks = result.face_landmarks[0]

#                 left_ear = calculate_ear(
#                     landmarks, LEFT_EYE, width, height
#                 )
#                 right_ear = calculate_ear(
#                     landmarks, RIGHT_EYE, width, height
#                 )

#                 self.left_ear = left_ear
#                 self.right_ear = right_ear

#                 left_is_closed = left_ear < EAR_THRESHOLD
#                 right_is_closed = right_ear < EAR_THRESHOLD
#                 now = time.monotonic()

#                 # Start a blink when either eye closes.
#                 if (
#                     left_is_closed or right_is_closed
#                 ) and not self.blink_active:
#                     self.blink_active = True
#                     self.blink_start_time = now
#                     self.left_was_closed = left_is_closed
#                     self.right_was_closed = right_is_closed
#                     self.eyes_open_start_time = None
#                     self.space_added = False

#                 # Track which eye(s) close during the blink event.
#                 elif self.blink_active:
#                     if left_is_closed:
#                         self.left_was_closed = True
#                     if right_is_closed:
#                         self.right_was_closed = True

#                     # End the event when both eyes are open again.
#                     if not left_is_closed and not right_is_closed:
#                         duration = now - self.blink_start_time

#                         if self.left_was_closed and self.right_was_closed:
#                             eye = "BOTH"
#                         elif self.left_was_closed:
#                             eye = "LEFT"
#                         elif self.right_was_closed:
#                             eye = "RIGHT"
#                         else:
#                             eye = "UNKNOWN"

#                         self.blinks.append((eye, duration))

#                         if duration <= DOT_MAX_TIME:
#                             self.morse_string += "."
#                         else:
#                             self.morse_string += "-"

#                         self.blink_active = False
#                         self.blink_start_time = None
#                         self.left_was_closed = False
#                         self.right_was_closed = False
#                         self.eyes_open_start_time = now
#                         self.space_added = False

#                 # Add one separator after a continuous open-eye interval.
#                 if not left_is_closed and not right_is_closed:
#                     if self.eyes_open_start_time is None:
#                         self.eyes_open_start_time = now

#                     open_duration = now - self.eyes_open_start_time
#                     if open_duration >= SPACE_TIME and not self.space_added:
#                         if (
#                             self.morse_string
#                             and not self.morse_string.endswith(" ")
#                         ):
#                             self.morse_string += " "
#                         self.space_added = True
#                 else:
#                     self.eyes_open_start_time = None
#                     self.space_added = False

#                 if left_is_closed and right_is_closed:
#                     self.eye_status = "BOTH EYES CLOSED"
#                 elif left_is_closed:
#                     self.eye_status = "LEFT EYE CLOSED"
#                 elif right_is_closed:
#                     self.eye_status = "RIGHT EYE CLOSED"
#                 else:
#                     self.eye_status = "EYES OPEN"

#                 # Draw status on the outgoing browser video.
#                 cv2.putText(
#                     frame,
#                     f"Left EAR: {left_ear:.2f}",
#                     (20, 35),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 255, 0),
#                     2,
#                 )
#                 cv2.putText(
#                     frame,
#                     f"Right EAR: {right_ear:.2f}",
#                     (20, 65),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 255, 0),
#                     2,
#                 )
#                 cv2.putText(
#                     frame,
#                     self.eye_status,
#                     (20, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 255, 255),
#                     2,
#                 )
#                 cv2.putText(
#                     frame,
#                     f"Blinks: {len(self.blinks)}",
#                     (20, 135),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 255, 0),
#                     2,
#                 )
#                 cv2.putText(
#                     frame,
#                     f"Morse: {self.morse_string[-40:]}",
#                     (20, 170),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (255, 255, 0),
#                     2,
#                 )
#             else:
#                 self.face_detected = False
#                 self.eye_status = "NO FACE DETECTED"

#                 cv2.putText(
#                     frame,
#                     "NO FACE DETECTED",
#                     (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 0, 255),
#                     2,
#                 )

#         return av.VideoFrame.from_ndarray(frame, format="bgr24")


# # ============================================================
# # Browser webcam stream
# # ============================================================

# st.subheader("1. Blink detection")

# ctx = webrtc_streamer(
#     key="blink-morse-stream",
#     video_processor_factory=BlinkMorseProcessor,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True,
# )

# if ctx.video_processor:
#     processor = ctx.video_processor

#     # Refresh the dashboard while streaming so the Morse output updates.
#     if ctx.state.playing:
#         status_area = st.empty()
#         morse_area = st.empty()
#         blink_area = st.empty()

#         while ctx.state.playing:
#             snapshot = processor.get_snapshot()

#             with status_area.container():
#                 if snapshot["face_detected"]:
#                     st.success(snapshot["eye_status"])
#                 else:
#                     st.warning("Face not detected. Keep your face visible.")

#                 col1, col2 = st.columns(2)
#                 left = snapshot["left_ear"]
#                 right = snapshot["right_ear"]
#                 col1.metric(
#                     "Left EAR",
#                     f"{left:.3f}" if left is not None else "—",
#                 )
#                 col2.metric(
#                     "Right EAR",
#                     f"{right:.3f}" if right is not None else "—",
#                 )

#             with morse_area.container():
#                 st.markdown("**Live Morse string**")
#                 st.code(snapshot["morse_string"] or "(waiting for blinks)")

#             with blink_area.container():
#                 st.caption(f"Detected blink events: {len(snapshot['blinks'])}")

#             time.sleep(0.2)

#     # Show final result after the user stops the stream.
#     snapshot = processor.get_snapshot()
#     morse_string = snapshot["morse_string"]

#     st.subheader("2. Results")
#     st.markdown("**Morse Code**")
#     st.code(morse_string or "(No Morse symbols recorded yet)")

#     raw_text = morse_to_text(morse_string) if morse_string else ""
#     st.markdown("**Rule-based decoded text**")
#     st.success(raw_text or "No text decoded yet.")

#     with st.expander("Recorded blink events"):
#         if snapshot["blinks"]:
#             for index, (eye, duration) in enumerate(snapshot["blinks"], start=1):
#                 st.write(f"{index}. {eye} blink — {duration:.2f} seconds")
#         else:
#             st.write("No blinks detected.")

#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("✨ Ask Gemini to suggest a correction", disabled=not morse_string):
#             try:
#                 from genai import generate_response

#                 prompt = f"""
# You are assisting with interpretation of a blink-generated Morse message.
# The blink timing may have introduced missing or extra symbols.
# Suggest a likely interpretation, but do not claim certainty. If ambiguous,
# say so and show the most likely candidate.

# Raw Morse sequence:
# {morse_string}

# Rule-based decoded text:
# {raw_text}

# Return:
# 1. Most likely interpretation
# 2. Brief reason
# 3. Any uncertainty
# """
#                 with st.spinner("Asking Gemini..."):
#                     suggestion = generate_response(prompt)
#                 st.session_state["gemini_suggestion"] = suggestion
#                 st.session_state["gemini_source_morse"] = morse_string
#             except Exception as exc:
#                 st.error(
#                     "Gemini request failed. Check the API key, model access, "
#                     f"network, and quota. Details: {exc}"
#                 )

#     with col2:
#         if st.button("🔄 Reset session"):
#             processor.reset()
#             st.session_state.pop("gemini_suggestion", None)
#             st.session_state.pop("gemini_source_morse", None)
#             st.rerun()

#     if (
#         st.session_state.get("gemini_suggestion")
#         and st.session_state.get("gemini_source_morse") == morse_string
#     ):
#         st.subheader("3. Gemini suggested interpretation")
#         st.warning(
#             "This is an AI suggestion and may be incorrect. Compare it with "
#             "the raw Morse sequence and rule-based decoding."
#         )
#         st.write(st.session_state["gemini_suggestion"])

# else:
#     st.caption(
#         "Start the camera using the control above. When finished, stop the "
#         "stream to review the final Morse sequence and decoded text."
#     )





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
