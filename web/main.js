import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingEl = document.getElementById("loading");

// UI Elements
const fpsCounter = document.getElementById("fps-counter");
const handsCounter = document.getElementById("hands-counter");
const btnFlip = document.getElementById("btn-flip");
const shapeButtons = document.querySelectorAll("#shape-selector button");
const fillButtons = document.querySelectorAll("#fill-selector button");

// State
let handLandmarker = undefined;
let webcamRunning = false;
let lastVideoTime = -1;
let fps = 0;
let frameCount = 0;
let lastFpsTime = performance.now();
let facingMode = "user";

let currentShape = "web";
let currentFill = "none";
let autoCycleTime = performance.now();

// Gesture State
let bowDrawn = false;
let pullHand = null;
let bowHand = null;
let arrows = [];

// Constants
const FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"];
const TIP_IDS = [4, 8, 12, 16, 20];

// ─────────────────────────── UI LISTENERS ───────────────────────────

shapeButtons.forEach(btn => {
  btn.addEventListener("click", (e) => {
    shapeButtons.forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    currentShape = e.target.dataset.val;
  });
});

fillButtons.forEach(btn => {
  btn.addEventListener("click", (e) => {
    fillButtons.forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    currentFill = e.target.dataset.val;
    autoCycleTime = performance.now(); // Reset timer if manually clicked
  });
});

btnFlip.addEventListener("click", () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  video.style.transform = facingMode === "user" ? "scaleX(-1)" : "scaleX(1)";
  canvasElement.style.transform = facingMode === "user" ? "scaleX(-1)" : "scaleX(1)";
  startCamera();
});

// ─────────────────────────── INIT AI ───────────────────────────

async function createHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6
  });
  
  loadingEl.style.display = "none";
  startCamera();
}
createHandLandmarker();

// ─────────────────────────── CAMERA ───────────────────────────

async function startCamera() {
  if (webcamRunning && video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
  }

  const constraints = {
    video: { facingMode: facingMode, width: { ideal: 1280 }, height: { ideal: 720 } }
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
    webcamRunning = true;
  } catch (err) {
    console.error("Camera access denied or unavailable", err);
    alert("Please allow camera access to use this app.");
  }
}

// ─────────────────────────── HELPERS ───────────────────────────

function distance(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

function midpoint(p1, p2) {
  return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
}

// ─────────────────────────── MAIN LOOP ───────────────────────────

async function predictWebcam() {
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const results = handLandmarker.detectForVideo(video, startTimeMs);
    
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    processResults(results);
    updateArrows();

    canvasCtx.restore();
    
    // FPS
    frameCount++;
    if (startTimeMs - lastFpsTime >= 1000) {
      fps = frameCount;
      frameCount = 0;
      lastFpsTime = startTimeMs;
      fpsCounter.innerText = fps;
    }
    
    // Auto cycle fill every 4s
    if (startTimeMs - autoCycleTime > 4000) {
      autoCycleTime = startTimeMs;
      const currentIndex = Array.from(fillButtons).findIndex(b => b.classList.contains("active"));
      const nextIndex = (currentIndex + 1) % fillButtons.length;
      fillButtons[nextIndex].click();
    }
  }

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// ─────────────────────────── LOGIC & DRAWING ───────────────────────────

function processResults(results) {
  const hands = results.landmarks;
  handsCounter.innerText = hands ? hands.length : 0;
  
  if (!hands || hands.length === 0) return;

  const w = canvasElement.width;
  const h = canvasElement.height;

  // Convert normalized landmarks to pixel coords
  const parsedHands = hands.map(lm => {
    return {
      thumb: { x: lm[4].x * w, y: lm[4].y * h },
      index: { x: lm[8].x * w, y: lm[8].y * h },
      middle: { x: lm[12].x * w, y: lm[12].y * h },
      ring: { x: lm[16].x * w, y: lm[16].y * h },
      pinky: { x: lm[20].x * w, y: lm[20].y * h },
    };
  });

  // Sort left to right
  parsedHands.sort((a, b) => a.index.x - b.index.x);

  // GESTURE: Bow and Arrow
  if (parsedHands.length >= 2) {
    const h1 = parsedHands[0];
    const h2 = parsedHands[1];
    
    const pinch1 = distance(h1.thumb, h1.index) < 40;
    const pinch2 = distance(h2.thumb, h2.index) < 40;
    const handsDist = distance(h1.middle, h2.middle);

    if (handsDist > 150) {
      if (pinch1 && !pinch2) {
        bowDrawn = true; pullHand = h1; bowHand = h2;
      } else if (pinch2 && !pinch1) {
        bowDrawn = true; pullHand = h2; bowHand = h1;
      } else {
        if (bowDrawn) {
          // Release!
          const dx = bowHand.middle.x - pullHand.index.x;
          const dy = bowHand.middle.y - pullHand.index.y;
          shootArrow(bowHand.middle, { x: dx, y: dy });
        }
        bowDrawn = false;
      }
    } else {
      bowDrawn = false;
    }
  } else {
    bowDrawn = false;
  }

  // Draw Bow String
  if (bowDrawn) {
    canvasCtx.beginPath();
    canvasCtx.strokeStyle = "rgba(255,255,255,0.8)";
    canvasCtx.lineWidth = 3;
    canvasCtx.moveTo(bowHand.thumb.x, bowHand.thumb.y);
    canvasCtx.lineTo(pullHand.index.x, pullHand.index.y);
    canvasCtx.lineTo(bowHand.index.x, bowHand.index.y);
    canvasCtx.stroke();
  }

  // Draw Core Shapes
  if (parsedHands.length >= 2) {
    drawShapes(parsedHands[0], parsedHands[1]);
  } else if (parsedHands.length === 1) {
    drawSingleHand(parsedHands[0]);
  }
}

function applyFill(pts) {
  if (currentFill === "none" || pts.length < 3) return;

  canvasCtx.beginPath();
  canvasCtx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) canvasCtx.lineTo(pts[i].x, pts[i].y);
  canvasCtx.closePath();

  if (currentFill === "bw") {
    // Halftone-ish pattern fill
    canvasCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
    canvasCtx.fill();
    canvasCtx.globalCompositeOperation = "difference";
    canvasCtx.fillStyle = "rgba(100, 100, 100, 1)";
    canvasCtx.fill();
    canvasCtx.globalCompositeOperation = "source-over";
  } else if (currentFill === "rainbow") {
    // Heatmap/Neon fill
    const minX = Math.min(...pts.map(p => p.x));
    const maxX = Math.max(...pts.map(p => p.x));
    const gradient = canvasCtx.createLinearGradient(minX, 0, maxX, 0);
    const time = performance.now() * 0.002;
    gradient.addColorStop(0, `hsl(${time * 50}, 100%, 50%)`);
    gradient.addColorStop(0.5, `hsl(${time * 50 + 120}, 100%, 50%)`);
    gradient.addColorStop(1, `hsl(${time * 50 + 240}, 100%, 50%)`);
    
    canvasCtx.globalAlpha = 0.8;
    canvasCtx.fillStyle = gradient;
    canvasCtx.fill();
    canvasCtx.globalAlpha = 1.0;
  }
}

function drawPolygon(pts) {
  if (pts.length < 2) return;
  
  applyFill(pts);
  
  canvasCtx.beginPath();
  canvasCtx.strokeStyle = "#ffffff";
  canvasCtx.lineWidth = 4;
  canvasCtx.lineJoin = "round";
  canvasCtx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) canvasCtx.lineTo(pts[i].x, pts[i].y);
  canvasCtx.closePath();
  
  // Outer glow
  canvasCtx.shadowColor = "#00ffff";
  canvasCtx.shadowBlur = 15;
  canvasCtx.stroke();
  canvasCtx.shadowBlur = 0;
}

function drawShapes(left, right) {
  if (currentShape === "line") {
    drawPolygon([left.index, right.index]);
  } else if (currentShape === "triangle") {
    const tmid = midpoint(left.thumb, right.thumb);
    drawPolygon([left.index, right.index, tmid]);
  } else if (currentShape === "rectangle") {
    drawPolygon([left.index, right.index, right.thumb, left.thumb]);
  } else if (currentShape === "diamond") {
    const cx = (left.index.x + right.index.x) / 2;
    const cy = (left.index.y + right.index.y) / 2;
    const dx = Math.abs(right.index.x - left.index.x) / 2;
    const dy = Math.abs(right.index.y - left.index.y) / 2 + 60;
    drawPolygon([
      {x: cx, y: cy - dy}, {x: cx + dx, y: cy},
      {x: cx, y: cy + dy}, {x: cx - dx, y: cy}
    ]);
  } else if (currentShape === "web") {
    // Continuous chain for web
    const leftChain = [left.pinky, left.ring, left.middle, left.index, left.thumb];
    const rightChain = [right.thumb, right.index, right.middle, right.ring, right.pinky];
    const pts = leftChain.concat(rightChain);
    
    applyFill(pts);
    
    canvasCtx.beginPath();
    canvasCtx.strokeStyle = "#ffffff";
    canvasCtx.lineWidth = 3;
    canvasCtx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) canvasCtx.lineTo(pts[i].x, pts[i].y);
    
    canvasCtx.shadowColor = "#ff00ff";
    canvasCtx.shadowBlur = 10;
    canvasCtx.stroke();
    canvasCtx.shadowBlur = 0;
    
    // Cross connections
    canvasCtx.beginPath();
    canvasCtx.lineWidth = 1;
    FINGER_NAMES.forEach(finger => {
      canvasCtx.moveTo(left[finger].x, left[finger].y);
      canvasCtx.lineTo(right[finger].x, right[finger].y);
    });
    canvasCtx.stroke();
  }
}

function drawSingleHand(hand) {
  const pts = [hand.pinky, hand.ring, hand.middle, hand.index, hand.thumb];
  canvasCtx.beginPath();
  canvasCtx.strokeStyle = "#ffffff";
  canvasCtx.lineWidth = 3;
  canvasCtx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) canvasCtx.lineTo(pts[i].x, pts[i].y);
  
  canvasCtx.shadowColor = "#00ffff";
  canvasCtx.shadowBlur = 10;
  canvasCtx.stroke();
  canvasCtx.shadowBlur = 0;
}

// ─────────────────────────── ARROW LOGIC ───────────────────────────

function shootArrow(startPos, dirVec) {
  const speed = 30;
  const mag = Math.hypot(dirVec.x, dirVec.y);
  if (mag === 0) return;
  arrows.push({
    x: startPos.x,
    y: startPos.y,
    vx: (dirVec.x / mag) * speed,
    vy: (dirVec.y / mag) * speed,
    age: 0
  });
}

function updateArrows() {
  for (let i = arrows.length - 1; i >= 0; i--) {
    let arr = arrows[i];
    arr.x += arr.vx;
    arr.y += arr.vy;
    arr.age++;
    
    if (arr.age > 40) {
      arrows.splice(i, 1);
      continue;
    }
    
    const tx = arr.x - arr.vx * 1.5;
    const ty = arr.y - arr.vy * 1.5;
    
    // Draw arrow line
    canvasCtx.beginPath();
    canvasCtx.moveTo(tx, ty);
    canvasCtx.lineTo(arr.x, arr.y);
    canvasCtx.strokeStyle = "#fff";
    canvasCtx.lineWidth = 4;
    canvasCtx.lineCap = "round";
    canvasCtx.shadowColor = "#ff0000";
    canvasCtx.shadowBlur = 15;
    canvasCtx.stroke();
    canvasCtx.shadowBlur = 0;
  }
}
