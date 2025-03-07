# app.py
import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template_string, request, send_file
from utils.args import parse_args
from pipeline.pipeline import TrackGenerator

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Set a secret key for session management

# -------------------------
# Home page: Configuration Form with Two Options
# -------------------------
@app.route("/", methods=["GET"])
def index():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Speaker Detection Setup</title>
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
      <style>
          body { background-color: #f8f9fa; }
          .container { margin-top: 50px; max-width: 700px; }
          .card { padding: 20px; margin-bottom: 20px; }
          h3 { margin-bottom: 20px; }
      </style>
    </head>
    <body>
      <div class="container">
        <!-- New Detection Section -->
        <div class="card shadow-sm">
          <h3>Run New Speaker Detection</h3>
          <form method="post" action="/start">
            <div class="form-group">
              <label for="tmp_dir_new">Temporary Directory (tmp_dir):</label>
              <input type="text" class="form-control" id="tmp_dir_new" name="tmp_dir" value="output/temp/" required>
            </div>
            <div class="form-group">
              <label for="max_frames">Max Frames:</label>
              <input type="number" class="form-control" id="max_frames" name="max_frames" value="6000" required>
            </div>
            <div class="form-group">
              <label for="input_device_index">Input Device Index:</label>
              <input type="number" class="form-control" id="input_device_index" name="input_device_index" value="3" required>
            </div>
            <div class="form-check">
              <input type="checkbox" class="form-check-input" id="no_face_masking" name="no_face_masking">
              <label class="form-check-label" for="no_face_masking">Disable Face Masking</label>
            </div>
            <br>
            <button type="submit" class="btn btn-primary btn-block">Start Detecting Speakers</button>
          </form>
        </div>
        <!-- Load Existing Setup Section -->
        <div class="card shadow-sm">
          <h3>Load Existing Inference</h3>
          <form method="post" action="/load">
            <div class="form-group">
              <label for="tmp_dir_existing">Temporary Directory (tmp_dir):</label>
              <input type="text" class="form-control" id="tmp_dir_existing" name="tmp_dir" value="output/temp/" required>
              <small class="form-text text-muted">
                This directory should contain both "annotated_video.mp4" and "speaking_segments.json".
              </small>
            </div>
            <button type="submit" class="btn btn-secondary btn-block">Load Existing Setup</button>
          </form>
        </div>
      </div>
    </body>
    </html>
    """
    return render_template_string(html)

# -------------------------
# Route for Running New Detection
# -------------------------
@app.route("/start", methods=["POST"])
def start_detection():
    # Retrieve parameters for new detection.
    tmp_dir = request.form.get("tmp_dir")
    max_frames = request.form.get("max_frames")
    input_device_index = request.form.get("input_device_index")
    no_face_masking = request.form.get("no_face_masking")
    
    try:
        max_frames = int(max_frames)
        input_device_index = int(input_device_index)
    except ValueError:
        return "Invalid numeric input provided.", 400

    # Create args object and override defaults with form values.
    args = parse_args()
    args.tmp_dir = tmp_dir
    args.max_frames = max_frames
    args.input_device_index = input_device_index
    # Face masking is enabled by default; disable if checkbox is checked.
    args.face_masking = False if no_face_masking is not None else True

    os.makedirs(tmp_dir, exist_ok=True)

    logging.getLogger('TrackGeneratorLogger').info("Starting speaker detection from Flask app (New Detection).")

    # Run the detection process (this is synchronous and may take time).
    try:
        start_time = datetime.now()
        track_generator = TrackGenerator(args)
        track_generator.run()
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logging.getLogger('TrackGeneratorLogger').info(f"Total processing time: {elapsed} seconds.")
    except Exception as e:
        logging.getLogger('TrackGeneratorLogger').exception("Error during detection process.")
        return f"Error during detection: {str(e)}", 500

    # After processing, load the annotated video and speaking segments.
    return render_results_page(tmp_dir, max_frames, elapsed)

# -------------------------
# Route for Loading Existing Inference Setup
# -------------------------
@app.route("/load", methods=["POST"])
def load_existing():
    tmp_dir = request.form.get("tmp_dir")
    if not tmp_dir:
        return "Temporary directory is required.", 400

    # For loading, we require that both the annotated video and speaking segments file exist.
    video_path = os.path.join(tmp_dir, "annotated_video.mp4")
    segments_path = os.path.join(tmp_dir, "speaking_segments.json")
    if not os.path.exists(video_path):
        return "Annotated video not found in the provided tmp_dir.", 404
    if not os.path.exists(segments_path):
        return "Speaking segments file not found in the provided tmp_dir.", 404

    # We might not have a new max_frames value here, so we use a default.
    max_frames = 6000
    elapsed = "N/A (Loaded Existing Setup)"
    return render_results_page(tmp_dir, max_frames, elapsed)

# -------------------------
# Helper Function to Render the Results Page
# -------------------------
def render_results_page(tmp_dir, max_frames, elapsed):
    video_path = os.path.join(tmp_dir, "annotated_video.mp4")
    segments_path = os.path.join(tmp_dir, "speaking_segments.json")
    try:
        with open(segments_path, "r") as f:
            segments_data = json.load(f)
    except Exception as e:
        return f"Error loading speaking segments: {str(e)}", 500

    annotations_json = json.dumps(segments_data)

    result_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Detection Complete</title>
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
      <style>
          body {{ background-color: #f8f9fa; }}
          .container {{ margin-top: 20px; max-width: 900px; }}
          .card {{ padding: 20px; }}
          video {{ width: 100%; height: auto; }}
          .timeline-container {{
              position: relative;
              width: 100%;
              height: 30px;
              background-color: #ddd;
              margin-bottom: 0px;
          }}
          /* Tick container for x-axis markers */
          #tickContainer {{
              position: relative;
              height: 25px;
              margin-bottom: 10px;
          }}
          .speaker-row {{
              margin-bottom: 15px;
          }}
          .speaker-label {{
              font-weight: bold;
              margin-bottom: 5px;
          }}
          .segment {{
              position: absolute;
              top: 0;
              height: 100%;
              opacity: 0.7;
          }}
          #currentMarker {{
              position: absolute;
              width: 2px;
              height: 100%;
              background-color: red;
          }}
          .tick-marker {{
              position: absolute;
              width: 1px;
              height: 10px;
              background-color: #333;
              top: 0;
          }}
          .tick-label {{
              position: absolute;
              top: 12px;
              transform: translateX(-50%);
              font-size: 10px;
              color: #333;
          }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="card shadow-sm">
          <h3 class="card-title">Speaker Detection Completed</h3>
          <p>
             Total frames processed: {max_frames}.<br>
             Processing time: {elapsed}.
          </p>
          <video id="annotatedVideo" controls>
            <source src="/video?tmp_dir={tmp_dir}" type="video/mp4">
            Your browser does not support the video tag.
          </video>
          <hr>
          <h5>Active Speaker Timeline</h5>
          <!-- Timeline Slider (current video frame) -->
          <div class="timeline-container" id="timelineSlider">
              <div id="currentMarker"></div>
          </div>
          <!-- Tick Markers Container -->
          <div id="tickContainer"></div>
          <!-- Annotation bars for each speaker -->
          <div id="annotationsContainer"></div>
          <br>
          <a href="/" class="btn btn-secondary">Go Back</a>
        </div>
      </div>
      <script>
        // Configuration parameters
        const fps = 30;                  // Assumed frames per second
        const timelineWindow = 1500;     // Number of frames per timeline window (~50 seconds)
        const maxFrames = {max_frames};    // Total frames processed
        // Annotation data from speaking_segments.json
        const annotations = {annotations_json};

        // Colors for speakers – extend or modify as needed.
        const speakerColors = {{"divyesh": "#28a745", "lu": "#007bff"}};

        // Create annotation rows for each speaker.
        const annotationsContainer = document.getElementById("annotationsContainer");
        for (let speaker in annotations) {{
            let row = document.createElement("div");
            row.className = "speaker-row";
            let label = document.createElement("div");
            label.className = "speaker-label";
            label.textContent = speaker;
            row.appendChild(label);
            let timelineDiv = document.createElement("div");
            timelineDiv.className = "timeline-container";
            timelineDiv.style.height = "20px";
            timelineDiv.id = "timeline_" + speaker;
            timelineDiv.style.backgroundColor = "#eee";
            row.appendChild(timelineDiv);
            annotationsContainer.appendChild(row);
        }}

        const video = document.getElementById("annotatedVideo");
        const timelineSlider = document.getElementById("timelineSlider");
        const currentMarker = document.getElementById("currentMarker");
        const tickContainer = document.getElementById("tickContainer");

        // This variable tracks the current timeline window's start frame.
        let currentWindowStart = 0;

        // Function to update tick markers along the x-axis.
        function updateTickMarks(windowStart) {{
            tickContainer.innerHTML = "";
            // Tick every 5 seconds (i.e. every 5 * fps frames)
            const tickIntervalFrames = 5 * fps;
            const numTicks = Math.floor(timelineWindow / tickIntervalFrames) + 1;
            for (let i = 0; i < numTicks; i++) {{
                let leftPerc = (i * tickIntervalFrames) / timelineWindow * 100;
                // Create tick line
                let tickDiv = document.createElement("div");
                tickDiv.className = "tick-marker";
                tickDiv.style.left = leftPerc + "%";
                tickContainer.appendChild(tickDiv);
                // Create tick label (absolute seconds)
                let label = document.createElement("div");
                label.className = "tick-label";
                label.style.left = leftPerc + "%";
                let absSeconds = Math.floor(windowStart / fps + i * 5);
                label.textContent = absSeconds + "s";
                tickContainer.appendChild(label);
            }}
        }}

        // Update timeline display based on current window.
        function updateTimelines(windowStart) {{
            currentWindowStart = windowStart;
            // Update tick marks for current window.
            updateTickMarks(windowStart);
            // Clear and update each speaker's timeline
            for (let speaker in annotations) {{
                const timelineDiv = document.getElementById("timeline_" + speaker);
                timelineDiv.innerHTML = "";
                const segments = annotations[speaker];
                segments.forEach(seg => {{
                    let segStart = seg[0], segEnd = seg[1];
                    // Only draw if the segment overlaps with the current window.
                    if (segEnd < windowStart || segStart > windowStart + timelineWindow) {{
                        return;
                    }}
                    // Clip segment to current window.
                    let displayStart = Math.max(segStart, windowStart);
                    let displayEnd = Math.min(segEnd, windowStart + timelineWindow);
                    // Calculate percentage positions relative to timelineWindow.
                    let leftPerc = ((displayStart - windowStart) / timelineWindow) * 100;
                    let widthPerc = ((displayEnd - displayStart) / timelineWindow) * 100;
                    let segmentDiv = document.createElement("div");
                    segmentDiv.className = "segment";
                    segmentDiv.style.left = leftPerc + "%";
                    segmentDiv.style.width = widthPerc + "%";
                    segmentDiv.style.backgroundColor = speakerColors[speaker] || "#888";
                    timelineDiv.appendChild(segmentDiv);
                }});
            }}
        }}

        // Update the slider (current marker) based on video current frame.
        function updateSlider(currentFrame) {{
            // If current frame is outside the current window, update the timeline.
            if (currentFrame < currentWindowStart || currentFrame > currentWindowStart + timelineWindow) {{
                let newWindowStart = Math.floor(currentFrame / timelineWindow) * timelineWindow;
                updateTimelines(newWindowStart);
            }}
            // Update marker position.
            let relativeFrame = currentFrame - currentWindowStart;
            let markerPos = (relativeFrame / timelineWindow) * timelineSlider.clientWidth;
            currentMarker.style.left = markerPos + "px";
        }}

        // Allow user to click on the timeline slider to jump to that position.
        timelineSlider.addEventListener("click", function(e) {{
            const rect = timelineSlider.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickPerc = clickX / timelineSlider.clientWidth;
            const targetFrame = currentWindowStart + clickPerc * timelineWindow;
            video.currentTime = targetFrame / fps;
        }});

        // Update slider as the video plays.
        video.addEventListener("timeupdate", function() {{
            const currentFrame = Math.floor(video.currentTime * fps);
            updateSlider(currentFrame);
        }});

        // Initialize timeline on page load.
        updateTimelines(0);
      </script>
    </body>
    </html>
    """
    return render_template_string(result_html)

# -------------------------
# Route to Serve the Annotated Video
# -------------------------
@app.route("/video")
def serve_video():
    tmp_dir = request.args.get("tmp_dir")
    if not tmp_dir:
        return "Temporary directory not provided.", 400
    video_path = os.path.join(tmp_dir, "annotated_video.mp4")
    if not os.path.exists(video_path):
        return "Annotated video not found.", 404
    return send_file(video_path, mimetype="video/mp4")

if __name__ == "__main__":
    app.run(debug=True)
