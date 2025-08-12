import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");
const anglesEl = document.getElementById("angles");
const enableBtn = document.getElementById("enableWebcam");

let faceLandmarker;
let webcamRunning = false;
const videoWidth = 480;

// Load model
(async () => {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    outputFaceBlendshapes: false,
    numFaces: 1
  });
})();

let currentStream = null;

enableBtn.addEventListener("click", async () => {
  if (!webcamRunning) {
    if (!faceLandmarker) return alert("Face Landmarker not loaded yet.");

    currentStream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = currentStream;
    video.addEventListener("loadeddata", predictLoop, { once: true });
    webcamRunning = true;
    enableBtn.textContent = "Disable Webcam";
  } else {
    // Disable webcam
    webcamRunning = false;
    enableBtn.textContent = "Enable Webcam";
    if (video.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      video.srcObject = null;
    }
    currentStream = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    anglesEl.innerText = "";
  }
});

async function predictLoop() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const now = performance.now();
  const result = faceLandmarker.detectForVideo(video, now);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (result.faceLandmarks && result.faceLandmarks.length > 0) {
    const landmarks = result.faceLandmarks[0];

    // Head pose approximation
    const noseTip = landmarks[1]; // Tip of nose
    const leftEye = landmarks[33]; // Outer corner of left eye
    const rightEye = landmarks[263]; // Outer corner of right eye
    const chin = landmarks[152]; // Chin
    const forehead = landmarks[10]; // Approx top of forehead

    // Vectors
    const dx = rightEye.x - leftEye.x;
    const dy = rightEye.y - leftEye.y;
    const dz = rightEye.z - leftEye.z;

    const eyeVectorLength = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const yaw = Math.atan2(dx, dz) * (180 / Math.PI);
    const roll = Math.atan2(dy, dx) * (180 / Math.PI);

    const dyPitch = chin.y - forehead.y;
    const dzPitch = chin.z - forehead.z;
    const pitch = Math.atan2(dyPitch, dzPitch) * (180 / Math.PI);

    anglesEl.innerText = `Yaw: ${yaw.toFixed(2)}°\nPitch: ${pitch.toFixed(2)}°\nRoll: ${roll.toFixed(2)}°`;

    // Draw landmarks
    const drawer = new DrawingUtils(ctx);
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#00FF00" });
  }

  if (webcamRunning) {
    requestAnimationFrame(predictLoop);
  }
}
