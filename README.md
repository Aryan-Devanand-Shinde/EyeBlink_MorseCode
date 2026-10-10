# Blink Morse Translator

## Technical Project Documentation

**Project type:** Computer Vision + Morse Code Translation + Generative
AI\
**Primary language:** Python\
**Current interface:** OpenCV webcam window; a Streamlit interface is
also present in the supplied files\
**AI model:** Google Gemini (`gemini-3.1-flash-lite`)

------------------------------------------------------------------------

## 1. Project Overview

Blink Morse Translator is a Python-based application that interprets eye
blinks as Morse code and converts the resulting Morse sequence into
readable text. It uses a webcam to capture video, MediaPipe Face
Landmarker to detect facial landmarks, and the Eye Aspect Ratio (EAR) to
estimate whether each eye is open or closed.

The duration of a detected blink determines whether it is interpreted as
a dot (`.`) or a dash (`-`). A sustained period with both eyes open is
used to insert a space into the Morse sequence. The sequence is then
translated using a predefined Morse-code dictionary.

The project also includes a Generative AI step using Google Gemini. It
sends the raw Morse sequence and its rule-based decoded text to the
model and asks it to infer a likely corrected message when the detected
sequence contains missing or extra symbols.

> **Important:** Gemini's result is a suggested interpretation, not
> guaranteed ground truth. It may guess incorrectly, so the raw Morse
> sequence and deterministic decoding should remain visible for
> comparison.

## 2. Problem Statement

People may find conventional typing or input methods difficult in some
situations. This project explores an alternative input method in which a
person communicates by intentionally blinking their eyes. The system
detects blink events, maps them to Morse symbols, and converts those
symbols into text.

Blink timing can be inconsistent, and a single missing or extra symbol
can change the decoded message. The project therefore explores using a
language model as a final assistance step to suggest a more plausible
interpretation.

## 3. Objectives

-   Detect facial and eye landmarks from a webcam stream.
-   Estimate eye openness using the Eye Aspect Ratio.
-   Detect blink events and measure their duration.
-   Convert short blinks to dots and longer blinks to dashes.
-   Use a sustained open-eye interval to separate Morse groups.
-   Decode Morse sequences into letters and digits.
-   Use Gemini to suggest a corrected interpretation of the raw
    decoding.
-   Provide a path toward a simple Streamlit interface.

## 4. System Architecture

``` text
Webcam
  |
  v
OpenCV: read and mirror video frames
  |
  v
Convert BGR frame to RGB
  |
  v
MediaPipe Face Landmarker
  |
  v
Extract left- and right-eye landmarks
  |
  v
Calculate EAR for each eye
  |
  v
Compare EAR with threshold
  |
  v
Detect blink start and end; measure duration
  |
  +------------------------------+
  |                              |
  v                              v
Short blink <= 0.25           Long blink > 0.25
  |                              |
  v                              v
Append "."                     Append "-"
  |                              |
  +---------------+--------------+
                  |
                  v
      Sustained open-eye interval >= 0.5
                  |
                  v
        Insert Morse separator
                  |
                  v
         Complete Morse string
                  |
                  v
      Rule-based Morse-to-text map
                  |
                  v
       Raw decoded text (baseline)
                  |
                  v
  Send Morse + raw text to Google Gemini
                  |
                  v
       Suggested corrected interpretation
```

The supplied implementation uses a local webcam through
`cv2.VideoCapture(0)` and displays the live frame using `cv2.imshow`. A
Streamlit file is also supplied, but it imports `start_blink_detection`
from `blink_detector.py`, which was not among the supplied files. The
Streamlit entry point and the standalone detection script should
therefore be checked together before packaging or deployment.

## 5. Pipeline and Data Flow

### Stage 1: Video capture

OpenCV opens the default webcam with `cv2.VideoCapture(0)`. Each frame
is read in a loop and mirrored horizontally with `cv2.flip(frame, 1)` to
provide a more natural mirror-like view.

If the camera cannot be opened or a frame cannot be read, the script
reports the problem and exits or breaks out of the loop.

### Stage 2: Frame preparation

OpenCV supplies frames in BGR channel order. The frame is converted to
RGB before being passed to MediaPipe. The frame dimensions are retained
so normalized landmark coordinates can be converted into pixel
coordinates.

A timestamp is incremented for MediaPipe's video-mode detector. In the
supplied script, it increases by 33 milliseconds per processed frame,
approximating a 30 FPS stream.

### Stage 3: Face and eye landmark detection

MediaPipe Face Landmarker is configured in video mode to detect one
face. Its model asset is loaded from `face_landmarker.task`.

The supplied eye landmark index lists are:

-   Left eye: `[362, 385, 387, 263, 373, 380]`
-   Right eye: `[33, 160, 158, 133, 153, 144]`

These landmark points are used to estimate eye openness.

### Stage 4: Eye Aspect Ratio (EAR)

The `calculate_ear()` function converts the selected normalized landmark
coordinates into pixel coordinates and calculates distances between
points.

The EAR formula used is:

\[ EAR = `\frac{d(p_2,p_6) + d(p_3,p_5)}{2d(p_1,p_4)}`{=tex} \]

Here, the numerator represents two vertical eye distances and the
denominator represents the horizontal eye width. In the code, Euclidean
distance is calculated using `math.dist()`.

A smaller EAR generally indicates a more closed eye. The script compares
each eye's EAR with `EAR_THRESHOLD = 0.21`:

-   EAR below the threshold: eye considered closed.
-   EAR at or above the threshold: eye considered open.

The threshold is a configurable heuristic and may need adjustment for
different people, lighting, camera positions, and facial
characteristics.

### Stage 5: Blink event detection

The script maintains state variables such as `blink_active`,
`blink_start_time`, `left_was_closed`, and `right_was_closed`.

1.  A blink event begins when either eye is detected as closed and no
    blink event is already active.
2.  The start time is recorded.
3.  The code remembers whether the left eye, right eye, or both
    participated in the event.
4.  The event ends when both eyes are open again.
5.  The duration is calculated from the start and end timestamps.
6.  The eye label and duration are appended to the `blinks` list.

Each recorded blink is stored as a tuple containing the eye label and
duration, for example `("BOTH", 0.18)`.

### Stage 6: Convert blink duration into Morse symbols

The current timing configuration is:

  ------------------------------------------------------------------------
  Setting                                      Value Purpose
  --------------------- ---------------------------- ---------------------
  `EAR_THRESHOLD`                             `0.21` Threshold used to
                                                     classify an eye as
                                                     closed

  `DOT_MAX_TIME`                      `0.25` seconds Maximum duration
                                                     classified as a dot

  `SPACE_TIME`                         `0.5` seconds Continuous open-eye
                                                     duration that
                                                     triggers a separator
  ------------------------------------------------------------------------

The decision rule is:

-   Blink duration less than or equal to 0.25 seconds → append `.`.
-   Blink duration greater than 0.25 seconds → append `-`.

The generated symbols are appended to `morse_string`. The code also
prints each event's eye label, duration, and Morse symbol.

These timing values are initial heuristics, not universal Morse timing
standards. They should be calibrated and tested with the intended user.

### Stage 7: Morse space detection

After both eyes are open, the script starts or continues an open-eye
timer. When both eyes remain open for at least 0.5 seconds, it appends a
space to the Morse string, provided the string is non-empty and does not
already end in a space.

The `space_added` flag prevents repeated spaces during one continuous
open-eye interval. The flag and timer are reset around subsequent blink
events.

This is the project's current custom separation rule. It should not be
confused with standard Morse timing, which distinguishes gaps between
symbols, letters, and words. The current implementation uses one
configurable open-eye interval as a separator and may require refinement
for reliable letter-versus-word boundaries.

### Stage 8: Live visual feedback

The OpenCV window displays:

-   Left-eye EAR.
-   Right-eye EAR.
-   Eye state (left closed, right closed, both closed, or open).
-   Total number of recorded blink events.
-   Current Morse string.

This information helps the developer observe how the system classifies
eye states and timing events during testing.

### Stage 9: Stop and cleanup

The capture loop stops when the user presses `Q`, the window is closed,
or the script cannot read a frame. The code releases the webcam, closes
OpenCV windows, and closes the MediaPipe landmarker.

After stopping, the script prints the recorded blink events and final
Morse string.

### Stage 10: Rule-based Morse decoding

The `morse_to_text.py` module contains a dictionary mapping standard
Morse sequences for A--Z and digits 0--9 to their corresponding
characters.

The `morse_to_text(morse)` function splits the sequence on spaces, looks
up each group, and concatenates the decoded characters. An unrecognized
group is represented as `?`.

This provides a deterministic baseline: the same Morse input produces
the same decoded result. The decoder currently includes letters and
digits; punctuation and broader Morse conventions are not included in
the supplied mapping.

### Stage 11: Gemini-assisted interpretation

The detection script imports `generate_response` from `genai.py`. The
function:

1.  Loads environment variables using `python-dotenv`.
2.  Reads `GOOGLE_API_KEY` from the environment.
3.  Configures the Google Generative AI SDK.
4.  Creates the `gemini-3.1-flash-lite` model.
5.  Sends the prompt to Gemini and returns `response.text`.

The prompt includes both the raw Morse sequence and the rule-based
decoded text. It asks the model to act as a Morse-decoding expert and
infer a likely correction when symbols may be missing or extra.

This is a language-model-assisted correction step, not a formal
error-correcting algorithm. Without additional context, multiple
messages may correspond to similar or ambiguous Morse sequences. Gemini
may produce a plausible but incorrect result. The application should
label this output as a suggestion and retain the raw sequence and
baseline text.

## 6. Module-Wise File Description

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `morse_logic.py`                    The supplied file contains the
                                      webcam processing, MediaPipe setup,
                                      EAR calculation, blink timing,
                                      space insertion, live overlays,
                                      cleanup, Morse decoding call, and
                                      Gemini prompt. Despite its
                                      filename, it currently contains
                                      much more than just Morse logic.

  `morse_to_text.py`                  Stores the Morse-to-character
                                      mapping and provides
                                      `morse_to_text()` for deterministic
                                      decoding.

  `genai.py`                          Configures the Google Generative AI
                                      SDK using `GOOGLE_API_KEY` and
                                      provides
                                      `generate_response(prompt)`.

  `app.py` (uploaded as               Provides a basic Streamlit page
  `app(20261009-051700).py`)          with a Start button and sections
                                      for Morse Code and Translated Text.
                                      It imports `start_blink_detection`
                                      from `blink_detector.py`.

  `face_landmarker.task`              MediaPipe model asset required by
                                      the face-landmark detector. The
                                      code references the relative path
                                      `face_landmarker.task`.

  `requirements.txt` (uploaded as     Lists Python packages required by
  `requirements(1).txt`)              the project.

  `readmefile.txt`                    Contains a reminder to download the
                                      face landmark task asset.
  -----------------------------------------------------------------------

**File consistency note:** The supplied Streamlit file imports
`start_blink_detection` from `blink_detector.py`, but that file was not
supplied. The supplied `morse_logic.py` script does not define
`start_blink_detection()`; it runs the webcam loop at module level. As a
result, the provided files alone do not demonstrate a fully connected
Streamlit execution path. The working local version may have an
additional or newer file. Confirm the actual project folder before
deployment.

## 7. Technology Stack

  -----------------------------------------------------------------------
  Technology                          Role in the project
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  OpenCV (`cv2`)                      Webcam capture, image conversion,
                                      overlays, and display

  MediaPipe                           Face landmark detection

  `face_landmarker.task`              Face landmark model asset

  `math`                              Euclidean distance calculations for
                                      EAR

  `time`                              Blink and open-eye duration
                                      measurement

  Streamlit                           Basic web interface in the supplied
                                      app file

  Google Generative AI SDK            Calls Gemini for suggested
                                      correction

  `python-dotenv`                     Loads environment variables from a
                                      local `.env` file
  -----------------------------------------------------------------------

## 8. Installation and Configuration

### Prerequisites

-   A supported Python installation.
-   A working webcam for local blink detection.
-   The project source files.
-   The `face_landmarker.task` model asset.
-   A Google AI API key if Gemini correction is enabled.

### Install dependencies

Create and activate a virtual environment, then install the packages
listed in `requirements.txt`:

``` bash
python -m venv .venv
```

Windows Command Prompt:

``` bat
.venv\Scripts\activate
```

Install the requirements:

``` bash
pip install -r requirements.txt
```

The supplied requirements file lists `opencv-python`, `mediapipe`,
`ipykernel`, `streamlit`, `google-generativeai`, and `python-dotenv`.

### Configure the API key

Create a local `.env` file in the project root:

``` text
GOOGLE_API_KEY=your_api_key_here
```

Do not commit the real `.env` file or API key to GitHub. Add `.env` to
`.gitignore`. The key should remain secret and should be configured
securely in the hosting platform when deployed.

### Model asset path

The code uses the relative path `face_landmarker.task`. Keep the model
file in the expected working directory or update the code to resolve the
asset path relative to the Python file.

### Running the current scripts

The standalone blink script can be run using the filename that contains
the webcam loop, for example:

``` bash
python morse_logic.py
```

The exact command depends on the actual local filenames. The supplied
Streamlit entry point can be started with:

``` bash
streamlit run app.py
```

However, the Streamlit app currently imports `start_blink_detection`
from `blink_detector.py`. That module and function must exist and
correctly return the completed Morse string for the supplied app to run.
The standalone script, as supplied, does not expose that function.

## 9. Input and Output

**Input** - Live webcam frames showing the user's face and eyes. -
Optional Google AI API key for the Gemini correction stage.

**Intermediate data** - Face landmarks. - Left and right EAR values. -
Eye-open/closed state. - Blink start and end times. - Blink duration and
eye label. - List of recorded blink events. - Raw Morse sequence. -
Rule-based decoded text.

**Output** - Live eye-state and EAR overlays. - Recorded blink event
details. - Final Morse string. - Deterministically decoded text. -
Gemini-generated suggested interpretation.

## 10. Limitations

1.  **Sensitivity to conditions:** Lighting, head pose, camera
    placement, glasses, and landmark quality can affect eye-state
    detection.
2.  **Timing variability:** A single fixed duration threshold may not
    work equally well for every user.
3.  **Space ambiguity:** One open-eye threshold is used for separation;
    it does not fully model standard Morse character and word gaps.
4.  **Morse ambiguity:** A missing or extra dot/dash can result in a
    different valid sequence.
5.  **GenAI uncertainty:** Gemini may guess a correction that sounds
    reasonable but does not match the intended message.
6.  **Model/API dependency:** Gemini correction requires API access,
    network connectivity, a valid key, and an available model.
7.  **Punctuation coverage:** The supplied Morse dictionary maps letters
    and digits only.
8.  **UI integration:** The supplied Streamlit entry point and the
    standalone webcam script do not currently share a confirmed function
    interface.
9.  **Deployment constraint:** `cv2.VideoCapture(0)` accesses a camera
    available to the Python process. On a hosted server, that is
    generally not the visitor's browser webcam. A browser webcam
    streaming approach is required for a public web demo.

## 11. Future Enhancements

-   Separate detection, Morse logic, decoding, and AI correction into
    clean functions and modules.
-   Calibrate blink timing per user or provide configurable thresholds.
-   Improve separation between letters and words using an explicit Morse
    timing model.
-   Add punctuation and additional supported symbols.
-   Add confidence or ambiguity reporting instead of returning a single
    unqualified correction.
-   Ask the user to confirm the suggested corrected message before
    treating it as final.
-   Add tests for EAR calculation, blink-duration classification,
    separator insertion, and Morse dictionary coverage.
-   Add a browser-compatible webcam stream for a deployed application.
-   Improve the Streamlit interface with Start, Stop, Reset, raw Morse,
    baseline decoded text, and suggested correction panels.
-   Evaluate accuracy with multiple users and predefined test phrases.

## 12. Suggested GitHub Repository Structure

``` text
Blink-Morse-Translator/
├── app.py
├── blink_detector.py
├── morse_logic.py
├── morse_to_text.py
├── genai.py
├── face_landmarker.task
├── requirements.txt
├── .env                 # local only; do not commit
├── .gitignore
└── README.md
```

This is a recommended organization, not a claim that every file
currently exists in the supplied version. Refactor the detection script
into a callable function before relying on this structure for Streamlit
deployment.

## 13. Conclusion

Blink Morse Translator combines real-time computer vision with a
rule-based Morse decoder and a Generative AI assistance step. MediaPipe
landmarks and EAR provide the eye-state signal; timing rules convert
blink events into Morse symbols; a dictionary decodes those symbols; and
Gemini attempts to suggest a more plausible message when the raw
decoding appears incorrect.

The key engineering challenge is to make blink timing and message
boundaries reliable while keeping the AI correction transparent. The raw
Morse sequence, baseline decoding, and AI suggestion should remain
distinguishable so users can judge the result rather than relying
blindly on the model.
