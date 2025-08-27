import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");
const anglesEl = document.getElementById("angles");
const enableBtn = document.getElementById("enableWebcam");
const loadingOverlay = document.getElementById("loadingOverlay");

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

    loadingOverlay.style.display = "flex";
    try {
      currentStream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = currentStream;
      video.addEventListener("loadeddata", () => {
        loadingOverlay.style.display = "none";
        predictLoop();
      }, { once: true });
      webcamRunning = true;
      enableBtn.textContent = "Disable Webcam";
    } catch (e) {
      loadingOverlay.style.display = "none";
      alert("Could not access webcam.");
    }
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
    loadingOverlay.style.display = "none";
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

    // Mouth openness calculation
    const upperLip = landmarks[13]; // Upper lip
    const lowerLip = landmarks[14]; // Lower lip
    const mouthOpenDist = Math.sqrt(
      Math.pow(upperLip.x - lowerLip.x, 2) +
      Math.pow(upperLip.y - lowerLip.y, 2) +
      Math.pow(upperLip.z - lowerLip.z, 2)
    ) * 100;

    // Vectors
    const dx = rightEye.x - leftEye.x;
    const dy = rightEye.y - leftEye.y;
    const dz = rightEye.z - leftEye.z;

    const eyeVectorLength = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const yaw = Math.atan2(dx, dz) * (180 / Math.PI) - 90;
    const roll = Math.atan2(dy, dx) * (180 / Math.PI);

    const dyPitch = chin.y - forehead.y;
    const dzPitch = chin.z - forehead.z;
    const pitch = Math.atan2(dyPitch, dzPitch) * (180 / Math.PI) - 90;

  anglesEl.innerText = `Yaw: ${yaw.toFixed(2)}°\nPitch: ${pitch.toFixed(2)}°\nRoll: ${roll.toFixed(2)}°\nMouth Open: ${mouthOpenDist.toFixed(4)}`;

    // Draw landmarks
    const drawer = new DrawingUtils(ctx);
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#00FF00" });
  }

  if (webcamRunning) {
    requestAnimationFrame(predictLoop);
  }
}
