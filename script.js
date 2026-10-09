import {
    FilesetResolver,
    GestureRecognizer,
    FaceDetector,
    ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";

const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";

const MODELS = {
    hands: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    face: "https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task",
    objects: "https://storage.googleapis.com/mediapipe-models/object_detector/object_detector/efficientdet_lite0/1/object_detector.tflite"
};

function findElement(...ids) {
    for (const id of ids) {
        const element = document.getElementById(id);
        if (element) return element;
    }
    return null;
}

const video = findElement("video", "camera", "videoElement");
const canvas = findElement("canvas", "output", "overlay", "canvasElement");

if (!video || !canvas) {
    throw new Error("SPECTRA: Camera video or canvas element not found.");
}

const ctx = canvas.getContext("2d");

const ui = {
    camera: findElement("cameraBtn", "startCamera", "startBtn"),
    hands: findElement("handBtn", "handsBtn", "handTrackingBtn", "trackingBtn"),
    target: findElement("targetBtn", "targetBoxBtn"),
    objects: findElement("objectBtn", "objectsBtn", "objectDetectionBtn"),
    sign: findElement("signBtn", "signLanguageBtn"),
    geometry: findElement("shapeBtn", "shapesBtn", "geometryBtn"),
    filter: findElement("filterBtn"),
    privacy: findElement("privacyBtn", "facePrivacyBtn"),
    gestureText: findElement("gesture", "gestureDisplay", "gestureText"),
    signText: findElement("sign", "signDisplay", "signText"),
    fps: findElement("fps", "fpsDisplay"),
    shapeStatus: findElement("shapeStatus", "geometry", "geometryDisplay"),
    filterStatus: findElement("filterStatus", "filter", "filterDisplay"),
    handStatus: findElement("handStatus"),
    faceStatus: findElement("faceStatus"),
    handsInfo: findElement("handsInfo"),
    facesInfo: findElement("facesInfo"),
    gestureInfo: findElement("gestureInfo"),
    signInfo: findElement("signInfo"),
    shapesInfo: findElement("shapesInfo"),
    objectStatus: findElement("objectStatus")
};

let cameraRunning = false;
let handTracking = true;
let targetVisible = true;
let objectDetectionEnabled = false;
let signLanguageEnabled = false;
let geometryEnabled = true;
let privacyEnabled = false;

let vision = null;
let gestureRecognizer = null;
let faceDetector = null;
let objectDetector = null;
let modelsLoading = false;

let handResults = null;
let faceResults = null;
let objectResults = null;

const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

const detectionIntervals = {
    hands: isMobile ? 40 : 25,
    face: isMobile ? 180 : 110,
    objects: isMobile ? 400 : 250
};

let lastHandDetection = 0;
let lastFaceDetection = 0;
let lastObjectDetection = 0;

const GEOMETRY_MODES = [
    "QUAD WARP",
    "DIAMOND",
    "SHARD",
    "FRAME",
    "HAND MESH",
    "HAND MESH 3D",
    "POLYGON HAND TRACKER",
    "HAND SKELETON AR",
    "HOLOGRAM HAND TRACKING",
    "CYBERPUNK HAND MESH",
    "NEON HAND GEOMETRY"
];

const FILTERS = [
    "NORMAL",
    "PIXEL BLUR",
    "GLITCH",
    "BLUEPRINT",
    "HEAVY CONTRAST",
    "CHANGING COLOUR",
    "GREEN SCREEN",
    "RED FILTER",
    "PIXEL DISPLAY",
    "BLUE HALFTONE SCAN",
    "BRIGHTNESS",
    "GRAYSCALE",
    "INVERT",
    "SATURATE",
    "SEPIA"
];

let geometryIndex = 0;
let filterIndex = 0;

let geometryMode = GEOMETRY_MODES[geometryIndex];
let currentFilter = FILTERS[filterIndex];

let previousPinching = false;
let previousIndexY = null;
let previousHandAngle = null;
let lastPinchChange = 0;
let lastVerticalChange = 0;
let lastTwistChange = 0;

const GESTURE_COOLDOWN = 420;
const geometryPointState = new Map();

let frameCounter = 0;
let fpsTimer = performance.now();
let currentFPS = 0;

const geometryCanvas = document.createElement("canvas");
const geometryCtx = geometryCanvas.getContext("2d");

function setText(element, text) {
    if (element) element.textContent = text;
}

function updateButton(button, enabled, onText, offText) {
    if (!button) return;
    button.textContent = enabled ? onText : offText;
    button.setAttribute("aria-pressed", String(enabled));
}

function updateStatuses() {
    setText(ui.shapeStatus, `GEOMETRY // ${geometryMode}`);
    setText(ui.filterStatus, `FILTER // ${currentFilter}`);
    setText(ui.handStatus, `HANDS // ${handTracking ? "ON" : "OFF"}`);
    setText(ui.faceStatus, `FACE PRIVACY // ${privacyEnabled ? "ON" : "OFF"}`);
    setText(ui.objectStatus, `OBJECTS // ${objectDetectionEnabled ? "ON" : "OFF"}`);
}

function resizeCanvas() {
    const width = video.videoWidth || window.innerWidth;
    const height = video.videoHeight || window.innerHeight;

    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        geometryCanvas.width = width;
        geometryCanvas.height = height;
    }
}

window.addEventListener("resize", resizeCanvas);

async function loadVision() {
    if (vision) return vision;

    if (modelsLoading) {
        while (modelsLoading) {
            await new Promise(resolve => setTimeout(resolve, 50));
        }
        return vision;
    }

    modelsLoading = true;

    try {
        vision = await FilesetResolver.forVisionTasks(WASM_URL);

        await Promise.allSettled([
            loadGestureRecognizer(),
            loadFaceDetector(),
            loadObjectDetector()
        ]);

        return vision;
    } finally {
        modelsLoading = false;
    }
}

async function loadGestureRecognizer() {
    if (gestureRecognizer || !vision) return;

    try {
        gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: MODELS.hands,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2
        });
    } catch (error) {
        console.warn("SPECTRA: Hand model could not use GPU; trying CPU.", error);

        gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: MODELS.hands,
                delegate: "CPU"
            },
            runningMode: "VIDEO",
            numHands: 2
        });
    }
}

async function loadFaceDetector() {
    if (faceDetector || !vision) return;

    try {
        faceDetector = await FaceDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: MODELS.face,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            minDetectionConfidence: 0.35
        });
    } catch (error) {
        console.warn("SPECTRA: Face detector unavailable.", error);
    }
}

async function loadObjectDetector() {
    if (objectDetector || !vision) return;

    try {
        objectDetector = await ObjectDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: MODELS.objects,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            scoreThreshold: 0.35,
            maxResults: 8
        });
    } catch (error) {
        console.warn("SPECTRA: Object detector unavailable.", error);
    }
}

async function startCamera() {
    if (cameraRunning) return;

    if (!navigator.mediaDevices?.getUserMedia) {
        alert("Camera access requires HTTPS or localhost.");
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        cameraRunning = true;
        updateButton(ui.camera, true, "STOP CAMERA", "START CAMERA");
        resizeCanvas();

        loadVision().catch(error => {
            console.error("SPECTRA model loading failed:", error);
        });

        requestAnimationFrame(renderLoop);
    } catch (error) {
        console.error("SPECTRA could not start the camera:", error);
        alert("Camera could not start. Check your browser permissions.");
    }
}

function stopCamera() {
    cameraRunning = false;

    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    updateButton(ui.camera, false, "STOP CAMERA", "START CAMERA");
}

if (ui.camera) {
    ui.camera.addEventListener("click", () => {
        if (cameraRunning) stopCamera();
        else startCamera();
    });
}

function drawVideo() {
    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );
}

function applyImageFilter() {
    if (currentFilter === "NORMAL") return;

    const w = canvas.width;
    const h = canvas.height;

    if (currentFilter === "PIXEL BLUR") {
        const smallW = Math.max(1, Math.floor(w / 24));
        const smallH = Math.max(1, Math.floor(h / 24));
        const small = document.createElement("canvas");
        small.width = smallW;
        small.height = smallH;
        const smallCtx = small.getContext("2d");

        smallCtx.drawImage(canvas, 0, 0, smallW, smallH);
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(small, 0, 0, w, h);
        ctx.restore();
        return;
    }

    if (currentFilter === "GLITCH") {
        const sliceHeight = Math.max(3, Math.floor(h / 55));

        for (let y = 0; y < h; y += sliceHeight) {
            if (Math.random() > 0.82) {
                const shift = (Math.random() - 0.5) * 35;
                try {
                    const strip = ctx.getImageData(0, y, w, sliceHeight);
                    ctx.putImageData(strip, shift, y);
                } catch (_) {}
            }
        }

        ctx.fillStyle = "rgba(255,0,70,0.06)";
        ctx.fillRect(0, 0, w, h);
        return;
    }

    if (currentFilter === "BLUEPRINT") {
        ctx.fillStyle = "rgba(0,70,255,0.58)";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(180,225,255,0.24)";
        ctx.lineWidth = 1;

        for (let x = 0; x < w; x += 28) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }

        for (let y = 0; y < h; y += 28) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
        return;
    }

    if (currentFilter === "HEAVY CONTRAST") {
        ctx.fillStyle = "rgba(0,0,0,0.18)";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(255,255,255,0.14)";
        ctx.lineWidth = 2;
        ctx.strokeRect(0, 0, w, h);
        return;
    }

    if (currentFilter === "CHANGING COLOUR") {
        const hue = (performance.now() / 15) % 360;
        ctx.fillStyle = `hsla(${hue},100%,50%,0.18)`;
        ctx.fillRect(0, 0, w, h);
        return;
    }

    if (currentFilter === "GREEN SCREEN") {
        ctx.fillStyle = "rgba(0,255,60,0.22)";
        ctx.fillRect(0, 0, w, h);
        return;
    }

    if (currentFilter === "RED FILTER") {
        ctx.fillStyle = "rgba(255,0,0,0.25)";
        ctx.fillRect(0, 0, w, h);
        return;
    }

    if (currentFilter === "PIXEL DISPLAY") {
        ctx.save();
        ctx.globalAlpha = 0.35;
        ctx.fillStyle = "#000";

        for (let y = 0; y < h; y += 5) {
            ctx.fillRect(0, y, w, 1);
        }

        for (let x = 0; x < w; x += 5) {
            for (let y = 0; y < h; y += 5) {
                ctx.fillRect(x, y, 1, 1);
            }
        }
        ctx.restore();
        return;
    }

    if (currentFilter === "BLUE HALFTONE SCAN") {
        ctx.fillStyle = "rgba(0,100,255,0.16)";
        ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = "rgba(120,200,255,0.3)";

        for (let y = 0; y < h; y += 8) {
            for (let x = 0; x < w; x += 8) {
                ctx.beginPath();
                ctx.arc(x, y, 1.4, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        return;
    }

    if (currentFilter === "BRIGHTNESS") {
        ctx.fillStyle = "rgba(255,255,255,0.18)";
        ctx.fillRect(0, 0, w, h);
        return;
    }

    if (currentFilter === "GRAYSCALE") {
        ctx.fillStyle = "rgba(100,100,100,0.3)";
        ctx.fillRect(0, 0, w, h);
        return;
    }

    if (currentFilter === "INVERT") {
        ctx.save();
        ctx.globalCompositeOperation = "difference";
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, w, h);
        ctx.restore();
        return;
    }

    if (currentFilter === "SATURATE") {
        ctx.fillStyle = "rgba(255,0,100,0.10)";
        ctx.fillRect(0, 0, w, h);
        return;
    }

    if (currentFilter === "SEPIA") {
        ctx.fillStyle = "rgba(140,80,25,0.25)";
        ctx.fillRect(0, 0, w, h);
    }
}

function cycleGeometryMode() {
    geometryIndex = (geometryIndex + 1) % GEOMETRY_MODES.length;
    geometryMode = GEOMETRY_MODES[geometryIndex];
    updateStatuses();
}

function cycleGeometryFilter() {
    filterIndex = (filterIndex + 1) % FILTERS.length;
    currentFilter = FILTERS[filterIndex];
    updateStatuses();
}

if (ui.filter) {
    ui.filter.addEventListener("click", cycleGeometryFilter);
}

if (ui.geometry) {
    ui.geometry.addEventListener("click", () => {
        geometryEnabled = !geometryEnabled;
        updateButton(ui.geometry, geometryEnabled, "GEOMETRY ON", "GEOMETRY OFF");
        updateStatuses();
    });
}

if (ui.hands) {
    ui.hands.addEventListener("click", () => {
        handTracking = !handTracking;
        updateButton(ui.hands, handTracking, "HANDS ON", "HANDS OFF");
        updateStatuses();
    });
}

if (ui.target) {
    ui.target.addEventListener("click", () => {
        targetVisible = !targetVisible;
        updateButton(ui.target, targetVisible, "TARGET ON", "TARGET OFF");
    });
}

if (ui.objects) {
    ui.objects.addEventListener("click", async () => {
        objectDetectionEnabled = !objectDetectionEnabled;
        updateButton(ui.objects, objectDetectionEnabled, "OBJECTS ON", "OBJECTS OFF");

        if (objectDetectionEnabled) {
            try {
                await loadVision();
                await loadObjectDetector();
            } catch (error) {
                console.error(error);
            }
        }

        updateStatuses();
    });
}

if (ui.sign) {
    ui.sign.addEventListener("click", async () => {
        signLanguageEnabled = !signLanguageEnabled;
        updateButton(ui.sign, signLanguageEnabled, "SIGN ON", "SIGN OFF");

        if (signLanguageEnabled) {
            try {
                await loadVision();
                await loadGestureRecognizer();
            } catch (error) {
                console.error(error);
            }
        }
    });
}

if (ui.privacy) {
    ui.privacy.addEventListener("click", () => {
        privacyEnabled = !privacyEnabled;
        updateButton(ui.privacy, privacyEnabled, "PRIVACY ON", "PRIVACY OFF");
        updateStatuses();
    });
}

function distance(a, b) {
    if (!a || !b) return Infinity;
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function landmarkToCanvas(point) {
    if (!point) return null;

    return {
        x: (1 - point.x) * canvas.width,
        y: point.y * canvas.height
    };
}

function smoothPoint(key, point) {
    if (!point) return null;

    let previous = geometryPointState.get(key);

    if (!previous) {
        previous = { x: point.x, y: point.y };
        geometryPointState.set(key, previous);
        return { ...previous };
    }

    previous.x += (point.x - previous.x) * 0.55;
    previous.y += (point.y - previous.y) * 0.55;

    return { x: previous.x, y: previous.y };
}

function isPinching(hand) {
    if (!hand || !hand[4] || !hand[8]) return false;
    return distance(hand[4], hand[8]) < 0.085;
}

function checkHandGestures(hands) {
    const now = performance.now();
    const anyPinching = (hands || []).some(isPinching);

    // A single pinch edge advances the filter once, whether one or both hands pinch.
    if (
        anyPinching &&
        !previousPinching &&
        now - lastPinchChange > GESTURE_COOLDOWN
    ) {
        cycleGeometryFilter();
        lastPinchChange = now;
    }

    previousPinching = anyPinching;

    if (!hands || !hands.length) {
        previousIndexY = null;
        previousHandAngle = null;
        return;
    }

    const hand = hands[0];
    const indexTip = hand[8];

    if (indexTip && !anyPinching) {
        if (previousIndexY !== null) {
            const dy = indexTip.y - previousIndexY;

            if (
                Math.abs(dy) > 0.085 &&
                now - lastVerticalChange > 520
            ) {
                cycleGeometryFilter();
                lastVerticalChange = now;
            }
        }

        previousIndexY = indexTip.y;
    } else if (anyPinching) {
        previousIndexY = indexTip?.y ?? null;
    }

    // A wrist rotation cycles geometry modes.
    const wrist = hand[0];
    const middleMcp = hand[9];

    if (wrist && middleMcp && !anyPinching) {
        const angle = Math.atan2(
            middleMcp.y - wrist.y,
            middleMcp.x - wrist.x
        );

        if (previousHandAngle !== null) {
            let delta = angle - previousHandAngle;

            while (delta > Math.PI) delta -= Math.PI * 2;
            while (delta < -Math.PI) delta += Math.PI * 2;

            if (
                Math.abs(delta) > 0.42 &&
                now - lastTwistChange > 650
            ) {
                cycleGeometryMode();
                lastTwistChange = now;
            }
        }

        previousHandAngle = angle;
    } else if (anyPinching) {
        previousHandAngle = null;
    }
}

function drawHandSkeleton(hands) {
    if (!hands) return;

    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20],
        [0, 17]
    ];

    for (const hand of hands) {
        ctx.save();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ffffff";
        ctx.fillStyle = "#ffffff";
        ctx.shadowColor = "#ffffff";
        ctx.shadowBlur = 8;

        for (const [a, b] of connections) {
            const p1 = landmarkToCanvas(hand[a]);
            const p2 = landmarkToCanvas(hand[b]);
            if (!p1 || !p2) continue;

            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }

        for (const point of hand) {
            const p = landmarkToCanvas(point);
            if (!p) continue;

            ctx.beginPath();
            ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.restore();
    }
}

function drawTargetBox(hands) {
    if (!targetVisible || !hands?.length) return;

    for (const hand of hands) {
        const points = hand
            .map(landmarkToCanvas)
            .filter(Boolean);

        if (!points.length) continue;

        const xs = points.map(point => point.x);
        const ys = points.map(point => point.y);

        const minX = Math.min(...xs) - 14;
        const maxX = Math.max(...xs) + 14;
        const minY = Math.min(...ys) - 14;
        const maxY = Math.max(...ys) + 14;

        ctx.save();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 5]);
        ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
        ctx.setLineDash([]);
        ctx.restore();
    }
}

function getGeometryPoints(hands) {
    if (!hands?.length) return [];

    const result = [];

    hands.forEach((hand, handIndex) => {
        const indexes = [0, 4, 8, 12, 16, 20];

        for (const index of indexes) {
            const point = landmarkToCanvas(hand[index]);
            if (!point) continue;

            result.push(
                smoothPoint(`hand-${handIndex}-point-${index}`, point)
            );
        }
    });

    return result;
}

function drawGeometry(hands) {
    if (!geometryEnabled || !hands?.length) return;

    const allPoints = getGeometryPoints(hands);
    if (!allPoints.length) return;

    const palette = geometryMode.includes("CYBERPUNK")
        ? ["#ff2d95", "#00f0ff"]
        : geometryMode.includes("HOLOGRAM")
            ? ["#7df9ff", "#ffffff"]
            : geometryMode.includes("NEON")
                ? ["#ffffff", "#b4fffb"]
                : geometryMode.includes("BLUEPRINT")
                    ? ["#d8efff", "#78bfff"]
                    : ["#ffffff", "#bdbdbd"];

    ctx.save();
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = palette[0];
    ctx.fillStyle = palette[1];
    ctx.shadowColor = palette[0];
    ctx.shadowBlur = geometryMode.includes("NEON") ||
        geometryMode.includes("HOLOGRAM") ||
        geometryMode.includes("CYBERPUNK") ? 12 : 4;

    if (geometryMode === "QUAD WARP" ||
        geometryMode === "DIAMOND" ||
        geometryMode === "SHARD" ||
        geometryMode === "FRAME") {
        drawBasicGeometry(allPoints);
    } else {
        for (const hand of hands) {
            drawDetailedGeometry(hand);
        }
    }

    ctx.restore();
}

function drawBasicGeometry(points) {
    if (points.length < 4) return;

    const first = points[0];
    const last = points[points.length - 1];

    const xs = points.map(point => point.x);
    const ys = points.map(point => point.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    ctx.beginPath();

    if (geometryMode === "DIAMOND") {
        ctx.moveTo((minX + maxX) / 2, minY);
        ctx.lineTo(maxX, (minY + maxY) / 2);
        ctx.lineTo((minX + maxX) / 2, maxY);
        ctx.lineTo(minX, (minY + maxY) / 2);
        ctx.closePath();
        ctx.stroke();
    } else if (geometryMode === "SHARD") {
        ctx.moveTo(first.x, first.y);
        ctx.lineTo(maxX, minY);
        ctx.lineTo(last.x, last.y);
        ctx.lineTo(minX, maxY);
        ctx.closePath();
        ctx.stroke();
    } else if (geometryMode === "FRAME") {
        const size = 18;
        drawCornerFrame(minX - size, minY - size, maxX - minX + size * 2, maxY - minY + size * 2, size);
    } else {
        ctx.moveTo(first.x, first.y);
        ctx.lineTo(maxX, minY);
        ctx.lineTo(last.x, last.y);
        ctx.lineTo(minX, maxY);
        ctx.closePath();
        ctx.stroke();
    }

    for (const point of points) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
        ctx.fill();
    }
}

function drawCornerFrame(x, y, width, height, size) {
    ctx.beginPath();

    ctx.moveTo(x, y + size);
    ctx.lineTo(x, y);
    ctx.lineTo(x + size, y);

    ctx.moveTo(x + width - size, y);
    ctx.lineTo(x + width, y);
    ctx.lineTo(x + width, y + size);

    ctx.moveTo(x, y + height - size);
    ctx.lineTo(x, y + height);
    ctx.lineTo(x + size, y + height);

    ctx.moveTo(x + width - size, y + height);
    ctx.lineTo(x + width, y + height);
    ctx.lineTo(x + width, y + height - size);

    ctx.stroke();
}

function drawDetailedGeometry(hand) {
    const p = hand.map(landmarkToCanvas);
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20],
        [5, 9], [9, 13], [13, 17], [5, 17]
    ];

    ctx.save();

    if (geometryMode === "HAND MESH 3D") {
        ctx.strokeStyle = "#ffffff";
        ctx.globalAlpha = 0.7;
    } else if (geometryMode === "POLYGON HAND TRACKER") {
        ctx.strokeStyle = "#bdbdbd";
        ctx.lineWidth = 2;
    } else if (geometryMode === "HAND SKELETON AR") {
        ctx.strokeStyle = "#ffffff";
        ctx.setLineDash([5, 4]);
    } else if (geometryMode === "HOLOGRAM HAND TRACKING") {
        ctx.strokeStyle = "#7df9ff";
        ctx.shadowColor = "#7df9ff";
        ctx.shadowBlur = 12;
    } else if (geometryMode === "CYBERPUNK HAND MESH") {
        ctx.strokeStyle = "#ff2d95";
        ctx.shadowColor = "#00f0ff";
        ctx.shadowBlur = 10;
    } else if (geometryMode === "NEON HAND GEOMETRY") {
        ctx.strokeStyle = "#ffffff";
        ctx.shadowColor = "#a0fff4";
        ctx.shadowBlur = 15;
        ctx.lineWidth = 2;
    } else {
        ctx.strokeStyle = "#ffffff";
    }

    for (const [a, b] of connections) {
        if (!p[a] || !p[b]) continue;

        ctx.beginPath();
        ctx.moveTo(p[a].x, p[a].y);
        ctx.lineTo(p[b].x, p[b].y);
        ctx.stroke();
    }

    if (
        geometryMode === "HAND MESH" ||
        geometryMode === "HAND MESH 3D" ||
        geometryMode === "POLYGON HAND TRACKER" ||
        geometryMode === "CYBERPUNK HAND MESH"
    ) {
        const triangles = [
            [0, 5, 9], [0, 9, 13], [0, 13, 17],
            [5, 6, 9], [6, 7, 9], [7, 8, 9],
            [9, 10, 13], [10, 11, 13], [11, 12, 13],
            [13, 14, 17], [14, 15, 17], [15, 16, 17],
            [1, 2, 5], [2, 3, 5], [3, 4, 5],
            [5, 9, 17]
        ];

        ctx.globalAlpha = 0.28;

        for (const [a, b, c] of triangles) {
            if (!p[a] || !p[b] || !p[c]) continue;

            ctx.beginPath();
            ctx.moveTo(p[a].x, p[a].y);
            ctx.lineTo(p[b].x, p[b].y);
            ctx.lineTo(p[c].x, p[c].y);
            ctx.closePath();
            ctx.stroke();
        }
    }

    if (geometryMode === "HAND MESH 3D") {
        // Offset wireframe gives the mesh a simple depth effect.
        ctx.globalAlpha = 0.3;
        ctx.translate(5, -5);

        for (const [a, b] of connections) {
            if (!p[a] || !p[b]) continue;

            ctx.beginPath();
            ctx.moveTo(p[a].x, p[a].y);
            ctx.lineTo(p[b].x, p[b].y);
            ctx.stroke();
        }
    }

    for (const point of p) {
        if (!point) continue;

        ctx.beginPath();
        ctx.arc(point.x, point.y, 2.4, 0, Math.PI * 2);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fill();
    }

    ctx.restore();
}

async function detectHands(timestamp) {
    if (!gestureRecognizer) return;

    if (timestamp - lastHandDetection < detectionIntervals.hands) return;
    lastHandDetection = timestamp;

    try {
        handResults = gestureRecognizer.recognizeForVideo(video, timestamp);
    } catch (error) {
        console.warn("SPECTRA hand detection error:", error);
    }
}

async function detectFace(timestamp) {
    if (!faceDetector) return;

    if (timestamp - lastFaceDetection < detectionIntervals.face) return;
    lastFaceDetection = timestamp;

    try {
        faceResults = faceDetector.detectForVideo(video, timestamp);
    } catch (error) {
        console.warn("SPECTRA face detection error:", error);
    }
}

async function detectObjects(timestamp) {
    if (!objectDetectionEnabled || !objectDetector) {
        objectResults = null;
        return;
    }

    if (timestamp - lastObjectDetection < detectionIntervals.objects) return;
    lastObjectDetection = timestamp;

    try {
        objectResults = objectDetector.detectForVideo(video, timestamp);
    } catch (error) {
        console.warn("SPECTRA object detection error:", error);
    }
}

function drawFacePrivacy(results) {
    if (!privacyEnabled || !results?.detections) return;

    for (const detection of results.detections) {
        const box = detection.boundingBox;
        if (!box) continue;

        // The preview is mirrored, so mirror the detected face box too.
        const x = canvas.width - box.originX - box.width;
        const y = box.originY;
        const width = box.width;
        const height = box.height;

        const pixelSize = 12;
        const smallCanvas = document.createElement("canvas");
        smallCanvas.width = Math.max(1, Math.floor(width / pixelSize));
        smallCanvas.height = Math.max(1, Math.floor(height / pixelSize));

        const smallCtx = smallCanvas.getContext("2d");

        try {
            const faceImage = ctx.getImageData(x, y, width, height);
            const temp = document.createElement("canvas");
            temp.width = width;
            temp.height = height;
            temp.getContext("2d").putImageData(faceImage, 0, 0);

            smallCtx.drawImage(
                temp,
                0,
                0,
                smallCanvas.width,
                smallCanvas.height
            );

            ctx.save();
            ctx.imageSmoothingEnabled = false;
            ctx.clearRect(x, y, width, height);
            ctx.drawImage(
                smallCanvas,
                x,
                y,
                width,
                height
            );
            ctx.restore();
        } catch (error) {
            console.warn("SPECTRA face privacy could not pixelate a face:", error);
        }
    }
}

function drawFaceFilter(results) {
    if (!results?.detections) return;

    for (const detection of results.detections) {
        const box = detection.boundingBox;
        if (!box) continue;

        const x = canvas.width - box.originX - box.width;
        const y = box.originY;
        const width = box.width;
        const height = box.height;

        ctx.save();

        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.strokeRect(x, y, width, height);
        ctx.setLineDash([]);

        ctx.fillStyle = "#ffffff";
        ctx.font = "12px monospace";
        ctx.fillText("FACE DETECTED", x, Math.max(14, y - 7));

        ctx.restore();
    }
}

function drawObjects(results) {
    if (!results?.detections) return;

    for (const detection of results.detections) {
        const box = detection.boundingBox;
        if (!box) continue;

        const x = canvas.width - box.originX - box.width;
        const y = box.originY;
        const width = box.width;
        const height = box.height;

        ctx.save();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x, y, width, height);

        const category = detection.categories?.[0];
        const label = category?.categoryName || "OBJECT";
        const score = category?.score;

        ctx.font = "12px monospace";
        ctx.fillStyle = "#ffffff";
        ctx.fillText(
            score == null ? label : `${label} ${(score * 100).toFixed(0)}%`,
            x,
            Math.max(14, y - 6)
        );

        ctx.restore();
    }
}

function updateGestureDisplay(results) {
    const category = results?.gestures?.[0]?.[0];
    const name = category?.categoryName || "NONE";
    const score = category?.score;

    setText(
        ui.gestureText,
        score == null
            ? `GESTURE // ${name.toUpperCase()}`
            : `GESTURE // ${name.toUpperCase()} ${(score * 100).toFixed(0)}%`
    );

    setText(ui.gestureInfo, `Gesture: ${name}`);
}

function updateSignDisplay(results) {
    if (!signLanguageEnabled) {
        setText(ui.signText, "SIGN // OFF");
        setText(ui.signInfo, "Sign recognition: OFF");
        return;
    }

    const category = results?.gestures?.[0]?.[0];
    const name = category?.categoryName || "NO SIGN";

    setText(ui.signText, `SIGN // ${name.toUpperCase()}`);
    setText(ui.signInfo, `Recognized gesture: ${name}`);
}

function updateFPSDisplay() {
    frameCounter++;

    const now = performance.now();

    if (now - fpsTimer >= 1000) {
        currentFPS = Math.round(frameCounter * 1000 / (now - fpsTimer));
        frameCounter = 0;
        fpsTimer = now;

        setText(ui.fps, `FPS // ${currentFPS}`);
    }
}

async function renderLoop() {
    if (!cameraRunning) return;

    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        requestAnimationFrame(renderLoop);
        return;
    }

    resizeCanvas();

    const timestamp = performance.now();

    drawVideo();

    if (vision) {
        if (handTracking) {
            await detectHands(timestamp);
        } else {
            handResults = null;
        }

        await detectFace(timestamp);
        await detectObjects(timestamp);
    }

    const hands = handResults?.landmarks || [];

    if (hands.length) {
        if (handTracking) {
            drawHandSkeleton(hands);
            drawTargetBox(hands);
            checkHandGestures(hands);
            drawGeometry(hands);
        }
    } else {
        previousPinching = false;
        previousIndexY = null;
        previousHandAngle = null;
    }

    updateGestureDisplay(handResults);
    updateSignDisplay(handResults);

    if (faceResults) {
        drawFacePrivacy(faceResults);
        drawFaceFilter(faceResults);
    }

    drawObjects(objectResults);

    setText(ui.handsInfo, `Hands detected: ${hands.length}`);
    setText(ui.facesInfo, `Faces detected: ${faceResults?.detections?.length || 0}`);
    setText(ui.shapesInfo, `Geometry: ${geometryMode}`);

    updateFPSDisplay();
    updateStatuses();

    requestAnimationFrame(renderLoop);
}

setText(ui.gestureText, "GESTURE // NONE");
setText(ui.signText, "SIGN // OFF");
setText(ui.fps, "FPS // --");
updateStatuses();

updateButton(ui.camera, false, "STOP CAMERA", "START CAMERA");
updateButton(ui.hands, true, "HANDS ON", "HANDS OFF");
updateButton(ui.target, true, "TARGET ON", "TARGET OFF");
updateButton(ui.objects, false, "OBJECTS ON", "OBJECTS OFF");
updateButton(ui.sign, false, "SIGN ON", "SIGN OFF");
updateButton(ui.geometry, true, "GEOMETRY ON", "GEOMETRY OFF");
updateButton(ui.privacy, false, "PRIVACY ON", "PRIVACY OFF");

console.log("%cSPECTRA READY", "font-weight:bold;font-size:18px");
console.log("Geometry modes:", GEOMETRY_MODES);
console.log("Visual filters:", FILTERS);
