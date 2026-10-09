// ============================================================
// SPECTRA - COMPLETE script.js
// Hand Geometry + Filters + Gesture + Sign + Face Privacy
// ============================================================

import {
    FilesetResolver,
    GestureRecognizer,
    FaceDetector,
    ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


// ============================================================
// MEDIAPIPE
// ============================================================

const WASM =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";

const GESTURE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

const FACE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task";

const OBJECT_MODEL =
    "https://storage.googleapis.com/mediapipe-models/object_detector/object_detector/efficientdet_lite0/1/object_detector.tflite";


// ============================================================
// ELEMENT FINDER
// ============================================================

function findElement(...ids) {
    for (const id of ids) {
        const element = document.getElementById(id);
        if (element) return element;
    }

    return null;
}


// ============================================================
// ELEMENTS
// ============================================================

const video =
    findElement("video", "camera", "videoElement");

const canvas =
    findElement("canvas", "output", "overlay", "canvasElement");

if (!video || !canvas) {
    throw new Error("SPECTRA: Camera video or canvas element not found.");
}

const ctx = canvas.getContext("2d");


// Buttons
const cameraBtn =
    findElement("cameraBtn", "startCamera", "startBtn");

const handBtn =
    findElement("handBtn", "handsBtn", "handTrackingBtn", "trackingBtn");

const targetBtn =
    findElement("targetBtn", "targetBoxBtn");

const objectBtn =
    findElement("objectBtn", "objectsBtn", "objectDetectionBtn");

const signBtn =
    findElement("signBtn", "signLanguageBtn");

const shapeBtn =
    findElement("shapeBtn", "shapesBtn", "geometryBtn");

const filterBtn = findElement("filterBtn");

const privacyBtn =
    findElement("privacyBtn", "facePrivacyBtn");

const faceFilterBtn =
    findElement("faceFilterBtn");

const fpsBoostBtn =
    findElement("fpsBoostBtn", "performanceBtn");

const resetBtn =
    findElement("resetBtn");


// Status displays
const gestureDisplay =
    findElement("gesture", "gestureDisplay", "gestureText");

const signDisplay =
    findElement("sign", "signDisplay", "signText");

const fpsDisplay =
    findElement("fps", "fpsDisplay");

const targetDisplay =
    findElement("target", "targetDisplay");

const geometryDisplay =
    findElement("geometry", "geometryDisplay", "shapeStatus");

const filterDisplay =
    findElement("filter", "filterDisplay", "filterStatus");


// ============================================================
// STATE
// ============================================================

let cameraRunning = false;

let handTracking = true;
let targetVisible = true;
let objectDetection = false;
let signLanguage = false;
let shapesEnabled = true;

let privacyMode = "OFF";
let faceFilter = "OFF";


// ============================================================
// PERFORMANCE
// ============================================================

const isMobile =
    /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

const PERFORMANCE_NORMAL = isMobile
    ? {
        hands: 32,
        face: 180,
        objects: 400
    }
    : {
        hands: 18,
        face: 110,
        objects: 250
    };

const PERFORMANCE_BOOST = isMobile
    ? {
        hands: 35,
        face: 999999,
        objects: 999999
    }
    : {
        hands: 23,
        face: 999999,
        objects: 999999
    };

let performanceConfig = PERFORMANCE_NORMAL;


// ============================================================
// MEDIAPIPE OBJECTS
// ============================================================

let vision = null;

let gestureRecognizer = null;
let faceDetector = null;
let objectDetector = null;

let modelsLoading = false;


// ============================================================
// DETECTION RESULTS
// ============================================================

let handResults = null;
let faceResults = null;
let objectResults = null;


// ============================================================
// GEOMETRY AND FILTERS
// ============================================================

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

const GEOMETRY_FILTERS = [
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

let geometryModeIndex = 0;
let geometryFilterIndex = 0;

let geometryMode =
    GEOMETRY_MODES[geometryModeIndex];

let geometryFilter =
    GEOMETRY_FILTERS[geometryFilterIndex];


// ============================================================
// GEOMETRY TRACKING
// ============================================================

const geometryPoints = new Map();

let geometryVisible = false;
let geometryLastSeen = 0;

const GEOMETRY_PERSISTENCE = 180;

const geometryPointState = new Map();


// ============================================================
// GESTURE STATE
// ============================================================

const DOUBLE_PINCH_WINDOW = 700;

let firstPinchTime = 0;
let previousBothPinching = false;
let previousAnyPinching = false;
let lastGestureChange = 0;
let previousIndexY = null;
let previousHandAngle = null;
let lastVerticalChange = 0;
let lastTwistChange = 0;

const GESTURE_COOLDOWN = 420;

const gestureEffectCanvas = document.createElement("canvas");
const gestureEffectCtx = gestureEffectCanvas.getContext("2d");


// ============================================================
// FPS AND TIMERS
// ============================================================

let fpsTimer = performance.now();
let frameCounter = 0;
let currentFPS = 0;

let lastHandDetection = 0;
let lastFaceDetection = 0;
let lastObjectDetection = 0;


// ============================================================
// OFFSCREEN CANVAS
// ============================================================

const geometryCanvas =
    document.createElement("canvas");

const geometryCtx =
    geometryCanvas.getContext("2d");


// ============================================================
// SVG URL FILTER
// ============================================================

function createURLFilter() {
    if (document.getElementById("spectraURLFilter")) {
        return;
    }

    const svgNS = "http://www.w3.org/2000/svg";

    const svg =
        document.createElementNS(svgNS, "svg");

    svg.id = "spectraURLFilter";

    svg.setAttribute("width", "0");
    svg.setAttribute("height", "0");
    svg.style.position = "absolute";
    svg.style.pointerEvents = "none";

    const filter =
        document.createElementNS(svgNS, "filter");

    filter.setAttribute("id", "spectraGlow");

    const blur =
        document.createElementNS(svgNS, "feGaussianBlur");

    blur.setAttribute("stdDeviation", "3");

    const color =
        document.createElementNS(svgNS, "feColorMatrix");

    color.setAttribute("type", "matrix");

    color.setAttribute(
        "values",
        `
        1 0 0 0 0
        0 1 0 0 0
        0 0 1 0 0
        0 0 0 1 0
        `
    );

    const merge =
        document.createElementNS(svgNS, "feMerge");

    const glow =
        document.createElementNS(svgNS, "feMergeNode");

    glow.setAttribute("in", "SourceGraphic");

    const blurNode =
        document.createElementNS(svgNS, "feMergeNode");

    blurNode.setAttribute("in", "blur");

    merge.appendChild(blurNode);
    merge.appendChild(glow);

    filter.appendChild(blur);
    filter.appendChild(color);
    filter.appendChild(merge);

    svg.appendChild(filter);

    document.body.appendChild(svg);
}

createURLFilter();


// ============================================================
// LOAD MEDIAPIPE
// ============================================================

async function loadVision() {
    if (vision) return vision;

    if (modelsLoading) {
        while (modelsLoading) {
            await new Promise(resolve =>
                setTimeout(resolve, 50)
            );
        }

        return vision;
    }

    modelsLoading = true;

    try {
        vision =
            await FilesetResolver.forVisionTasks(WASM);

        await loadGestureRecognizer();
        await loadFaceDetector();
        await loadObjectDetector();

        return vision;
    } catch (error) {
        console.error(
            "SPECTRA MediaPipe loading error:",
            error
        );

        throw error;
    } finally {
        modelsLoading = false;
    }
}


// ============================================================
// GESTURE MODEL
// ============================================================

async function loadGestureRecognizer() {
    if (gestureRecognizer) return;

    try {
        gestureRecognizer =
            await GestureRecognizer.createFromOptions(
                vision,
                {
                    baseOptions: {
                        modelAssetPath: GESTURE_MODEL,
                        delegate: "GPU"
                    },

                    runningMode: "VIDEO",

                    numHands: 2
                }
            );
    } catch (error) {
        console.warn(
            "Gesture recognizer failed:",
            error
        );

        gestureRecognizer = null;
    }
}


// ============================================================
// FACE MODEL
// ============================================================

async function loadFaceDetector() {
    if (faceDetector) return;

    try {
        faceDetector =
            await FaceDetector.createFromOptions(
                vision,
                {
                    baseOptions: {
                        modelAssetPath: FACE_MODEL,
                        delegate: "GPU"
                    },

                    runningMode: "VIDEO",

                    minDetectionConfidence: 0.35
                }
            );
    } catch (error) {
        console.warn(
            "Face detector failed:",
            error
        );

        faceDetector = null;
    }
}


// ============================================================
// OBJECT MODEL
// ============================================================

async function loadObjectDetector() {
    if (objectDetector) return;

    try {
        objectDetector =
            await ObjectDetector.createFromOptions(
                vision,
                {
                    baseOptions: {
                        modelAssetPath: OBJECT_MODEL,
                        delegate: "GPU"
                    },

                    runningMode: "VIDEO",

                    scoreThreshold: 0.35,

                    maxResults: 8
                }
            );
    } catch (error) {
        console.warn(
            "Object detector failed:",
            error
        );

        objectDetector = null;
    }
}


// ============================================================
// CAMERA
// ============================================================

async function startCamera() {
    if (cameraRunning) return;

    try {
        const stream =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "user",

                    width: {
                        ideal: 1280
                    },

                    height: {
                        ideal: 720
                    }
                },

                audio: false
            });

        video.srcObject = stream;

        await video.play();

        cameraRunning = true;

        if (cameraBtn) {
            cameraBtn.textContent = "STOP CAMERA";
        }

        resizeCanvas();

        loadVision().catch(error => {
            console.error(
                "SPECTRA model loading failed:",
                error
            );
        });

        requestAnimationFrame(renderLoop);
    } catch (error) {
        console.error(
            "Camera could not start:",
            error
        );

        cameraRunning = false;

        if (cameraBtn) {
            cameraBtn.textContent = "START CAMERA";
        }
    }
}


function stopCamera() {
    cameraRunning = false;

    if (video.srcObject) {
        const tracks =
            video.srcObject.getTracks();

        tracks.forEach(track =>
            track.stop()
        );

        video.srcObject = null;
    }

    if (cameraBtn) {
        cameraBtn.textContent = "START CAMERA";
    }
}


// ============================================================
// CANVAS SIZE
// ============================================================

function resizeCanvas() {
    const width =
        video.videoWidth ||
        window.innerWidth;

    const height =
        video.videoHeight ||
        window.innerHeight;

    if (
        canvas.width !== width ||
        canvas.height !== height
    ) {
        canvas.width = width;
        canvas.height = height;

        geometryCanvas.width = width;
        geometryCanvas.height = height;
    }
}

window.addEventListener(
    "resize",
    resizeCanvas
);


// ============================================================
// VIDEO DRAW
// ============================================================

function drawVideo() {
    ctx.save();

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.translate(
        canvas.width,
        0
    );

    ctx.scale(
        -1,
        1
    );

    ctx.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.restore();
}


// ============================================================
// LANDMARK TO CANVAS — MIRRORED CAMERA
// ============================================================

function landmarkToCanvas(landmark) {
    if (!landmark) return null;

    return {
        x: (1 - landmark.x) * canvas.width,
        y: landmark.y * canvas.height
    };
}


// ============================================================
// SMOOTH TRACKED POINT
// ============================================================

function trackedPoint(key, point) {
    if (!point) return null;

    let state =
        geometryPointState.get(key);

    if (!state) {
        state = {
            x: point.x,
            y: point.y,
            vx: 0,
            vy: 0
        };

        geometryPointState.set(
            key,
            state
        );

        return {
            x: point.x,
            y: point.y
        };
    }

    const dx =
        point.x - state.x;

    const dy =
        point.y - state.y;

    const speed =
        Math.sqrt(
            dx * dx +
            dy * dy
        );

    const normalizedSpeed =
        Math.min(
            speed / 100,
            1
        );

    const alpha =
        0.42 +
        normalizedSpeed * 0.34;

    state.vx =
        state.vx * 0.55 +
        dx * 0.45;

    state.vy =
        state.vy * 0.55 +
        dy * 0.45;

    const prediction =
        0.018 +
        normalizedSpeed * 0.025;

    const targetX =
        point.x +
        state.vx * prediction;

    const targetY =
        point.y +
        state.vy * prediction;

    state.x +=
        (targetX - state.x) *
        alpha;

    state.y +=
        (targetY - state.y) *
        alpha;

    state.x =
        Math.max(
            0,
            Math.min(
                canvas.width,
                state.x
            )
        );

    state.y =
        Math.max(
            0,
            Math.min(
                canvas.height,
                state.y
            )
        );

    return {
        x: state.x,
        y: state.y
    };
}


// ============================================================
// DISTANCE AND MIDPOINT
// ============================================================

function distance(a, b) {
    if (!a || !b) return 0;

    const dx =
        a.x - b.x;

    const dy =
        a.y - b.y;

    return Math.sqrt(
        dx * dx +
        dy * dy
    );
}

function midpoint(a, b) {
    return {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2
    };
}


// ============================================================
// PINCH DETECTION AND GESTURE SWITCHING
// ============================================================

function isPinching(hand) {
    if (!hand || hand.length < 9) {
        return false;
    }

    const thumb = hand[4];
    const index = hand[8];

    return distance(
        thumb,
        index
    ) < 0.085;
}

function checkDoublePinch(hands) {
    const now = performance.now();

    const pinchStates =
        (hands || []).map(isPinching);

    const anyPinching =
        pinchStates.some(Boolean);

    const bothPinching =
        pinchStates.length >= 2 &&
        pinchStates[0] &&
        pinchStates[1];

    // One pinch with either hand advances one filter.
    if (
        anyPinching &&
        !previousAnyPinching &&
        now - lastGestureChange > GESTURE_COOLDOWN
    ) {
        cycleGeometryFilter();
        lastGestureChange = now;
    }

    previousAnyPinching = anyPinching;
    previousBothPinching = bothPinching;

    if (!hands || !hands.length) {
        previousIndexY = null;
        previousHandAngle = null;
        return;
    }

    // Move the index fingertip up or down to change filters.
    const hand = hands[0];
    const indexTip = hand?.[8];

    if (indexTip && !anyPinching) {
        if (previousIndexY !== null) {
            const dy =
                indexTip.y - previousIndexY;

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

    // Twist the hand to cycle geometry styles.
    const wrist = hand?.[0];
    const middleMcp = hand?.[9];

    if (wrist && middleMcp && !anyPinching) {
        const angle = Math.atan2(
            middleMcp.y - wrist.y,
            middleMcp.x - wrist.x
        );

        if (previousHandAngle !== null) {
            let delta =
                angle - previousHandAngle;

            while (delta > Math.PI) {
                delta -= Math.PI * 2;
            }

            while (delta < -Math.PI) {
                delta += Math.PI * 2;
            }

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


// ============================================================
// CYCLE FILTER AND GEOMETRY
// ============================================================

function cycleGeometryFilter() {
    geometryFilterIndex++;

    if (
        geometryFilterIndex >=
        GEOMETRY_FILTERS.length
    ) {
        geometryFilterIndex = 0;
    }

    geometryFilter =
        GEOMETRY_FILTERS[
            geometryFilterIndex
        ];

    if (filterDisplay) {
        filterDisplay.textContent =
            `FILTER // ${geometryFilter}`;
    }

    console.log(
        "SPECTRA Geometry Filter:",
        geometryFilter
    );
}

function cycleGeometryMode() {
    geometryModeIndex =
        (geometryModeIndex + 1) %
        GEOMETRY_MODES.length;

    geometryMode =
        GEOMETRY_MODES[geometryModeIndex];

    if (geometryDisplay) {
        geometryDisplay.textContent =
            `GEOMETRY // ${geometryMode}`;
    }
}


// ============================================================
// GET HAND GEOMETRY POINTS
// ============================================================

function getHandGeometry(hands) {
    if (!hands || hands.length === 0) {
        return null;
    }

    // Two hands
    if (hands.length >= 2) {
        const handA = hands[0];
        const handB = hands[1];

        const aThumb =
            landmarkToCanvas(handA[4]);

        const aIndex =
            landmarkToCanvas(handA[8]);

        const bThumb =
            landmarkToCanvas(handB[4]);

        const bIndex =
            landmarkToCanvas(handB[8]);

        if (
            !aThumb ||
            !aIndex ||
            !bThumb ||
            !bIndex
        ) {
            return null;
        }

        return [
            trackedPoint("A_INDEX", aIndex),
            trackedPoint("B_INDEX", bIndex),
            trackedPoint("B_THUMB", bThumb),
            trackedPoint("A_THUMB", aThumb)
        ];
    }

    // One hand
    const hand = hands[0];

    const index =
        landmarkToCanvas(hand[8]);

    const middle =
        landmarkToCanvas(hand[12]);

    const ring =
        landmarkToCanvas(hand[16]);

    const thumb =
        landmarkToCanvas(hand[4]);

    if (
        !index ||
        !middle ||
        !ring ||
        !thumb
    ) {
        return null;
    }

    return [
        trackedPoint("ONE_INDEX", index),
        trackedPoint("ONE_MIDDLE", middle),
        trackedPoint("ONE_RING", ring),
        trackedPoint("ONE_THUMB", thumb)
    ];
}


// ============================================================
// GEOMETRY UTILITIES
// ============================================================

function averagePoint(points) {
    let x = 0;
    let y = 0;

    for (const point of points) {
        x += point.x;
        y += point.y;
    }

    return {
        x: x / points.length,
        y: y / points.length
    };
}

function createPolygonPath(targetCtx, points) {
    targetCtx.beginPath();

    targetCtx.moveTo(
        points[0].x,
        points[0].y
    );

    for (
        let i = 1;
        i < points.length;
        i++
    ) {
        targetCtx.lineTo(
            points[i].x,
            points[i].y
        );
    }

    targetCtx.closePath();
}


// ============================================================
// FILTER VALUE
// ============================================================

function getGeometryFilterValue() {
    switch (geometryFilter) {
        case "PIXEL BLUR":
            return "blur(3px) contrast(1.35)";

        case "GLITCH":
            return "contrast(1.8) saturate(2.8) hue-rotate(145deg)";

        case "BLUEPRINT":
            return "grayscale(1) contrast(2.1) brightness(1.2) sepia(1) hue-rotate(165deg) saturate(4)";

        case "HEAVY CONTRAST":
            return "contrast(2.8) brightness(1.08) saturate(1.3)";

        case "CHANGING COLOUR":
            return `hue-rotate(${(performance.now() / 7) % 360}deg) saturate(2.4)`;

        case "GREEN SCREEN":
            return "grayscale(1) sepia(1) hue-rotate(65deg) saturate(5) contrast(1.4)";

        case "RED FILTER":
            return "grayscale(.35) sepia(1) hue-rotate(315deg) saturate(4) contrast(1.25)";

        case "PIXEL DISPLAY":
            return "contrast(1.6) saturate(1.6)";

        case "BLUE HALFTONE SCAN":
            return "grayscale(1) sepia(1) hue-rotate(155deg) saturate(5) contrast(1.8)";

        case "BLUR":
            return "blur(7px)";

        case "BRIGHTNESS":
            return "brightness(1.8)";

        case "CONTRAST":
            return "contrast(2)";

        case "GRAYSCALE":
            return "grayscale(1)";

        case "INVERT":
            return "invert(1)";

        case "SATURATE":
            return "saturate(3)";

        case "SEPIA":
            return "sepia(1)";

        default:
            return "none";
    }
}


// ============================================================
// DRAW FILTERED GEOMETRY
// ============================================================

function drawFilteredGeometry(points) {
    if (!points || points.length < 3) {
        return;
    }

    geometryCtx.clearRect(
        0,
        0,
        geometryCanvas.width,
        geometryCanvas.height
    );

    geometryCtx.save();

    createPolygonPath(
        geometryCtx,
        points
    );

    geometryCtx.clip();

    geometryCtx.filter =
        getGeometryFilterValue();

    if (geometryFilter === "PIXEL DISPLAY") {
        const smallW =
            Math.max(
                24,
                Math.floor(geometryCanvas.width / 24)
            );

        const smallH =
            Math.max(
                18,
                Math.floor(geometryCanvas.height / 24)
            );

        gestureEffectCanvas.width = smallW;
        gestureEffectCanvas.height = smallH;

        gestureEffectCtx.filter = "none";

        gestureEffectCtx.clearRect(
            0,
            0,
            smallW,
            smallH
        );

        gestureEffectCtx.save();

        gestureEffectCtx.translate(
            smallW,
            0
        );

        gestureEffectCtx.scale(-1, 1);

        gestureEffectCtx.drawImage(
            video,
            0,
            0,
            smallW,
            smallH
        );

        gestureEffectCtx.restore();

        geometryCtx.imageSmoothingEnabled = false;

        geometryCtx.drawImage(
            gestureEffectCanvas,
            0,
            0,
            geometryCanvas.width,
            geometryCanvas.height
        );

        geometryCtx.imageSmoothingEnabled = true;
    } else {
        geometryCtx.translate(
            geometryCanvas.width,
            0
        );

        geometryCtx.scale(
            -1,
            1
        );

        geometryCtx.drawImage(
            video,
            0,
            0,
            geometryCanvas.width,
            geometryCanvas.height
        );
    }

    geometryCtx.restore();

    ctx.save();

    ctx.globalCompositeOperation =
        "source-over";

    ctx.drawImage(
        geometryCanvas,
        0,
        0
    );

    if ([
        "BLUEPRINT",
        "GREEN SCREEN",
        "RED FILTER",
        "BLUE HALFTONE SCAN",
        "GLITCH"
    ].includes(geometryFilter)) {
        ctx.save();

        createPolygonPath(ctx, points);
        ctx.clip();

        if (geometryFilter === "BLUEPRINT") {
            ctx.fillStyle = "rgba(0,105,255,0.34)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.strokeStyle = "rgba(120,210,255,0.7)";
            ctx.lineWidth = 1;

            for (let x = 0; x < canvas.width; x += 18) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }

            for (let y = 0; y < canvas.height; y += 18) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
        } else if (geometryFilter === "GREEN SCREEN") {
            ctx.fillStyle = "rgba(0,255,70,0.24)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        } else if (geometryFilter === "RED FILTER") {
            ctx.fillStyle = "rgba(255,0,18,0.28)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        } else if (geometryFilter === "BLUE HALFTONE SCAN") {
            ctx.fillStyle = "rgba(0,145,255,0.2)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = "rgba(130,220,255,0.55)";

            for (let y = 0; y < canvas.height; y += 6) {
                for (let x = 0; x < canvas.width; x += 6) {
                    ctx.beginPath();
                    ctx.arc(x, y, 1.1, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            ctx.fillStyle = "rgba(80,200,255,0.22)";

            const scanY =
                (performance.now() * 0.18) % canvas.height;

            ctx.fillRect(0, scanY, canvas.width, 3);
        } else if (geometryFilter === "GLITCH") {
            ctx.fillStyle = "rgba(255,0,80,0.16)";

            for (let y = 0; y < canvas.height; y += 23) {
                if (
                    (y + Math.floor(performance.now() / 90)) % 4 < 2
                ) {
                    ctx.fillRect(0, y, canvas.width, 3);
                }
            }
        }

        ctx.restore();
    }

    ctx.restore();
}


// ============================================================
// GEOMETRY BORDER
// ============================================================

function drawGeometryBorder(points) {
    if (!points || points.length < 3) {
        return;
    }

    ctx.save();

    createPolygonPath(ctx, points);

    ctx.strokeStyle =
        "rgba(255,255,255,0.95)";

    ctx.lineWidth = 2;

    ctx.shadowColor =
        "rgba(255,255,255,0.75)";

    ctx.shadowBlur = 10;

    ctx.stroke();

    ctx.restore();
}


// ============================================================
// CORNER POINTS
// ============================================================

function drawGeometryCorners(points) {
    ctx.save();

    for (const point of points) {
        ctx.beginPath();

        ctx.arc(
            point.x,
            point.y,
            5,
            0,
            Math.PI * 2
        );

        ctx.fillStyle =
            "rgba(255,255,255,0.95)";

        ctx.fill();

        ctx.beginPath();

        ctx.arc(
            point.x,
            point.y,
            10,
            0,
            Math.PI * 2
        );

        ctx.strokeStyle =
            "rgba(255,255,255,0.35)";

        ctx.lineWidth = 1;

        ctx.stroke();
    }

    ctx.restore();
}


// ============================================================
// MEASUREMENT LINE
// ============================================================

function drawMeasurementOutside(
    a,
    b,
    label,
    offset = 24
) {
    const dx =
        b.x - a.x;

    const dy =
        b.y - a.y;

    const length =
        Math.sqrt(
            dx * dx +
            dy * dy
        );

    if (length < 1) return;

    const nx =
        -dy / length;

    const ny =
        dx / length;

    const ax =
        a.x + nx * offset;

    const ay =
        a.y + ny * offset;

    const bx =
        b.x + nx * offset;

    const by =
        b.y + ny * offset;

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.75)";

    ctx.lineWidth = 1;

    ctx.setLineDash([5, 5]);

    ctx.beginPath();

    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);

    ctx.stroke();

    ctx.setLineDash([]);

    const tick = 5;

    ctx.beginPath();

    ctx.moveTo(
        ax - nx * tick,
        ay - ny * tick
    );

    ctx.lineTo(
        ax + nx * tick,
        ay + ny * tick
    );

    ctx.moveTo(
        bx - nx * tick,
        by - ny * tick
    );

    ctx.lineTo(
        bx + nx * tick,
        by + ny * tick
    );

    ctx.stroke();

    const mx =
        (ax + bx) / 2;

    const my =
        (ay + by) / 2;

    ctx.font =
        "bold 11px monospace";

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const padding = 7;

    const width =
        ctx.measureText(label).width +
        padding * 2;

    const height = 22;

    ctx.fillStyle =
        "rgba(0,0,0,0.72)";

    ctx.fillRect(
        mx - width / 2,
        my - height / 2,
        width,
        height
    );

    ctx.strokeStyle =
        "rgba(255,255,255,0.5)";

    ctx.strokeRect(
        mx - width / 2,
        my - height / 2,
        width,
        height
    );

    ctx.fillStyle = "white";

    ctx.fillText(
        label,
        mx,
        my
    );

    ctx.restore();
}


// ============================================================
// GEOMETRY MEASUREMENTS
// ============================================================

function drawPolygonMeasurements(points) {
    if (!points || points.length < 3) {
        return;
    }

    if (points.length === 4) {
        const top =
            distance(points[0], points[1]);

        const bottom =
            distance(points[2], points[3]);

        const left =
            distance(points[0], points[3]);

        const right =
            distance(points[1], points[2]);

        const length =
            Math.round((top + bottom) / 2);

        const breadth =
            Math.round((left + right) / 2);

        drawMeasurementOutside(
            points[0],
            points[1],
            `LENGTH ${length}px`,
            22
        );

        drawMeasurementOutside(
            points[1],
            points[2],
            `BREADTH ${breadth}px`,
            22
        );

        return;
    }

    let minX = Infinity;
    let maxX = -Infinity;

    let minY = Infinity;
    let maxY = -Infinity;

    for (const point of points) {
        minX = Math.min(minX, point.x);
        maxX = Math.max(maxX, point.x);

        minY = Math.min(minY, point.y);
        maxY = Math.max(maxY, point.y);
    }

    const width =
        Math.round(maxX - minX);

    const height =
        Math.round(maxY - minY);

    const center =
        averagePoint(points);

    ctx.save();

    ctx.font = "bold 11px monospace";
    ctx.textAlign = "center";

    ctx.fillStyle = "rgba(0,0,0,0.75)";

    ctx.fillText(
        `WIDTH ${width}px`,
        center.x,
        minY - 18
    );

    ctx.fillText(
        `HEIGHT ${height}px`,
        center.x,
        maxY + 22
    );

    ctx.restore();
}


// ============================================================
// GEOMETRY MODES
// ============================================================

function drawGeometryMode(points) {
    if (!points || points.length < 3) {
        return;
    }

    if (geometryMode === "QUAD WARP") {
        drawFilteredGeometry(points);
        drawGeometryBorder(points);
        return;
    }

    if (geometryMode === "DIAMOND") {
        const center =
            averagePoint(points);

        const diamond =
            points.map(point => ({
                x: center.x + (point.x - center.x) * 1.12,
                y: center.y + (point.y - center.y) * 1.12
            }));

        drawFilteredGeometry(diamond);
        drawGeometryBorder(diamond);
        return;
    }

    if (geometryMode === "SHARD") {
        const center =
            averagePoint(points);

        const shard =
            points.map((point, index) => {
                const amount =
                    index % 2 === 0 ? 1.25 : 0.75;

                return {
                    x: center.x + (point.x - center.x) * amount,
                    y: center.y + (point.y - center.y) * amount
                };
            });

        drawFilteredGeometry(shard);
        drawGeometryBorder(shard);
        return;
    }

    if ([
        "HAND MESH",
        "HAND MESH 3D",
        "POLYGON HAND TRACKER",
        "HAND SKELETON AR",
        "HOLOGRAM HAND TRACKING",
        "CYBERPUNK HAND MESH",
        "NEON HAND GEOMETRY"
    ].includes(geometryMode)) {
        drawFilteredGeometry(points);

        const styles = {
            "HAND MESH": {
                stroke: "rgba(255,255,255,.88)",
                fill: "rgba(255,255,255,.08)",
                width: 1.4
            },

            "HAND MESH 3D": {
                stroke: "rgba(180,220,255,.9)",
                fill: "rgba(30,120,255,.12)",
                width: 1.7
            },

            "POLYGON HAND TRACKER": {
                stroke: "rgba(255,255,255,.95)",
                fill: "rgba(255,255,255,.04)",
                width: 2.2
            },

            "HAND SKELETON AR": {
                stroke: "rgba(180,255,230,.95)",
                fill: "rgba(0,255,170,.07)",
                width: 1.6
            },

            "HOLOGRAM HAND TRACKING": {
                stroke: "rgba(0,235,255,.95)",
                fill: "rgba(0,170,255,.12)",
                width: 1.5
            },

            "CYBERPUNK HAND MESH": {
                stroke: "rgba(255,45,180,.95)",
                fill: "rgba(255,0,100,.1)",
                width: 1.8
            },

            "NEON HAND GEOMETRY": {
                stroke: "rgba(90,255,120,1)",
                fill: "rgba(0,255,90,.1)",
                width: 2.4
            }
        }[geometryMode];

        ctx.save();

        createPolygonPath(ctx, points);

        ctx.fillStyle = styles.fill;
        ctx.fill();

        ctx.strokeStyle = styles.stroke;
        ctx.lineWidth = styles.width;
        ctx.shadowColor = styles.stroke;

        ctx.shadowBlur =
            geometryMode.includes("HOLOGRAM") ||
            geometryMode.includes("NEON") ||
            geometryMode.includes("CYBERPUNK")
                ? 14
                : 4;

        ctx.stroke();

        ctx.shadowBlur = 0;

        const center = averagePoint(points);

        for (let i = 0; i < points.length; i++) {
            const a = points[i];
            const b = points[(i + 1) % points.length];

            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(center.x, center.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
        }

        if (
            geometryMode === "HAND MESH 3D" ||
            geometryMode === "CYBERPUNK HAND MESH"
        ) {
            const offset =
                geometryMode === "HAND MESH 3D" ? 8 : 5;

            const shifted =
                points.map(p => ({
                    x: p.x + offset,
                    y: p.y - offset
                }));

            createPolygonPath(ctx, shifted);

            ctx.strokeStyle = "rgba(255,255,255,.42)";
            ctx.stroke();

            for (let i = 0; i < points.length; i++) {
                ctx.beginPath();

                ctx.moveTo(
                    points[i].x,
                    points[i].y
                );

                ctx.lineTo(
                    shifted[i].x,
                    shifted[i].y
                );

                ctx.stroke();
            }
        }

        if (geometryMode === "HOLOGRAM HAND TRACKING") {
            ctx.setLineDash([4, 6]);

            for (let y = 0; y < canvas.height; y += 8) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }

            ctx.setLineDash([]);
        }

        ctx.restore();

        drawGeometryBorder(points);
        return;
    }

    if (geometryMode === "FRAME") {
        const center =
            averagePoint(points);

        const outer =
            points.map(point => ({
                x: center.x + (point.x - center.x) * 1.18,
                y: center.y + (point.y - center.y) * 1.18
            }));

        drawFilteredGeometry(outer);

        ctx.save();

        createPolygonPath(ctx, points);

        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 2;

        ctx.stroke();

        createPolygonPath(ctx, outer);

        ctx.strokeStyle = "rgba(255,255,255,0.35)";
        ctx.stroke();

        ctx.restore();
        return;
    }
}


// ============================================================
// HAND SKELETON
// ============================================================

function drawHandSkeleton(hands) {
    if (!hands) return;

    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20],
        [5, 9], [9, 13], [13, 17]
    ];

    ctx.save();

    ctx.lineWidth = 1.4;
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.fillStyle = "rgba(255,255,255,0.9)";

    for (const hand of hands) {
        for (const [startIndex, endIndex] of connections) {
            const a =
                landmarkToCanvas(hand[startIndex]);

            const b =
                landmarkToCanvas(hand[endIndex]);

            if (!a || !b) continue;

            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
        }

        for (const landmark of hand) {
            const point =
                landmarkToCanvas(landmark);

            if (!point) continue;

            ctx.beginPath();

            ctx.arc(
                point.x,
                point.y,
                2.5,
                0,
                Math.PI * 2
            );

            ctx.fill();
        }
    }

    ctx.restore();
}


// ============================================================
// TARGET BOX
// ============================================================

function drawTargetBox(hands) {
    if (!targetVisible) return;
    if (!hands || !hands.length) return;

    let minX = Infinity;
    let minY = Infinity;

    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const hand of hands) {
        for (const landmark of hand) {
            const point =
                landmarkToCanvas(landmark);

            if (!point) continue;

            minX = Math.min(minX, point.x);
            minY = Math.min(minY, point.y);

            maxX = Math.max(maxX, point.x);
            maxY = Math.max(maxY, point.y);
        }
    }

    const padding = 18;

    minX -= padding;
    minY -= padding;

    maxX += padding;
    maxY += padding;

    ctx.save();

    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1;

    ctx.setLineDash([5, 7]);

    ctx.strokeRect(
        minX,
        minY,
        maxX - minX,
        maxY - minY
    );

    ctx.restore();
}


// ============================================================
// FACE PIXEL PRIVACY
// ============================================================

function drawFacePrivacy(detections) {
    if (privacyMode === "OFF" || !detections) return;

    const list = detections.detections || [];
    if (!list.length) return;

    for (const detection of list) {
        const box = detection.boundingBox;
        if (!box) continue;

        const sx =
            canvas.width / (video.videoWidth || canvas.width);

        const sy =
            canvas.height / (video.videoHeight || canvas.height);

        const width = Math.max(1, box.width * sx);
        const height = Math.max(1, box.height * sy);

        // Adjust for the mirrored camera.
        const x = Math.max(
            0,
            Math.min(
                canvas.width - width,
                canvas.width - (box.originX * sx) - width
            )
        );

        const y = Math.max(
            0,
            Math.min(
                canvas.height - height,
                box.originY * sy
            )
        );

        const padX = width * 0.12;
        const padY = height * 0.16;

        const rx = Math.max(0, x - padX);
        const ry = Math.max(0, y - padY);

        const rw = Math.min(
            canvas.width - rx,
            width + padX * 2
        );

        const rh = Math.min(
            canvas.height - ry,
            height + padY * 2
        );

        // Pixelate the actual camera pixels within the face region.
        const block =
            Math.max(5, Math.floor(Math.min(rw, rh) / 12));

        const sw = Math.max(2, Math.ceil(rw / block));
        const sh = Math.max(2, Math.ceil(rh / block));

        gestureEffectCanvas.width = sw;
        gestureEffectCanvas.height = sh;

        gestureEffectCtx.clearRect(0, 0, sw, sh);

        gestureEffectCtx.drawImage(
            canvas,
            rx,
            ry,
            rw,
            rh,
            0,
            0,
            sw,
            sh
        );

        ctx.save();

        ctx.imageSmoothingEnabled = false;

        ctx.drawImage(
            gestureEffectCanvas,
            0,
            0,
            sw,
            sh,
            rx,
            ry,
            rw,
            rh
        );

        ctx.imageSmoothingEnabled = true;

        ctx.fillStyle = "rgba(0,8,18,0.18)";
        ctx.fillRect(rx, ry, rw, rh);

        ctx.strokeStyle = "rgba(130,235,255,0.95)";
        ctx.lineWidth = Math.max(1.5, canvas.width / 700);

        ctx.strokeRect(rx, ry, rw, rh);

        const corner = Math.max(8, rw * 0.12);

        ctx.lineWidth = Math.max(2, canvas.width / 500);

        ctx.beginPath();

        ctx.moveTo(rx, ry + corner);
        ctx.lineTo(rx, ry);
        ctx.lineTo(rx + corner, ry);

        ctx.moveTo(rx + rw - corner, ry);
        ctx.lineTo(rx + rw, ry);
        ctx.lineTo(rx + rw, ry + corner);

        ctx.moveTo(rx, ry + rh - corner);
        ctx.lineTo(rx, ry + rh);
        ctx.lineTo(rx + corner, ry + rh);

        ctx.moveTo(rx + rw - corner, ry + rh);
        ctx.lineTo(rx + rw, ry + rh);
        ctx.lineTo(rx + rw, ry + rh - corner);

        ctx.stroke();

        ctx.font = `${Math.max(10, canvas.width / 95)}px monospace`;
        ctx.fillStyle = "rgba(130,235,255,0.98)";

        ctx.fillText(
            "FACE PRIVACY // ACTIVE",
            rx,
            Math.max(12, ry - 7)
        );

        ctx.restore();
    }
}


// ============================================================
// FACE FILTER
// ============================================================

function drawFaceFilter(detections) {
    if (
        faceFilter === "OFF" ||
        !detections
    ) {
        return;
    }

    const list =
        detections.detections || [];

    for (const detection of list) {
        const box = detection.boundingBox;
        if (!box) continue;

        const centerX =
            box.originX + box.width / 2;

        const centerY =
            box.originY + box.height / 2;

        const radius =
            Math.max(box.width, box.height) * 0.45;

        ctx.save();

        ctx.beginPath();

        ctx.arc(
            centerX,
            centerY,
            radius,
            0,
            Math.PI * 2
        );

        if (faceFilter === "INVERT") {
            ctx.globalCompositeOperation = "difference";
            ctx.fillStyle = "white";
            ctx.fill();
        } else if (faceFilter === "SCAN") {
            ctx.strokeStyle = "rgba(255,255,255,0.9)";
            ctx.lineWidth = 3;
            ctx.stroke();

            for (let i = -radius; i < radius; i += 8) {
                ctx.beginPath();

                ctx.moveTo(
                    centerX - radius,
                    centerY + i
                );

                ctx.lineTo(
                    centerX + radius,
                    centerY + i
                );

                ctx.strokeStyle = "rgba(255,255,255,0.15)";
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }

        ctx.restore();
    }
}


// ============================================================
// OBJECT DETECTION
// ============================================================

function drawObjects(results) {
    if (!objectDetection || !results) {
        return;
    }

    const detections =
        results.detections || [];

    ctx.save();

    for (const detection of detections) {
        const box = detection.boundingBox;
        if (!box) continue;

        const x = box.originX;
        const y = box.originY;
        const width = box.width;
        const height = box.height;

        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.lineWidth = 1;

        ctx.strokeRect(
            x,
            y,
            width,
            height
        );

        const category =
            detection.categories &&
            detection.categories[0];

        const name =
            category?.categoryName || "OBJECT";

        const score =
            category?.score != null
                ? Math.round(category.score * 100)
                : 0;

        ctx.font = "bold 11px monospace";
        ctx.fillStyle = "rgba(0,0,0,0.75)";

        const label =
            `${name.toUpperCase()} ${score}%`;

        const labelWidth =
            ctx.measureText(label).width + 10;

        ctx.fillRect(
            x,
            y - 19,
            labelWidth,
            19
        );

        ctx.fillStyle = "white";

        ctx.fillText(
            label,
            x + 5,
            y - 6
        );
    }

    ctx.restore();
}


// ============================================================
// GESTURE DISPLAY
// ============================================================

function updateGestureDisplay(results) {
    if (!gestureDisplay) return;

    const gestures =
        results?.gestures || [];

    if (!gestures.length) {
        gestureDisplay.textContent =
            "GESTURE // NONE";

        return;
    }

    const names = [];

    for (const gestureList of gestures) {
        const top = gestureList?.[0];

        if (top) {
            names.push(
                `${top.categoryName} ${Math.round(top.score * 100)}%`
            );
        }
    }

    gestureDisplay.textContent =
        names.length
            ? `GESTURE // ${names.join(" | ")}`
            : "GESTURE // NONE";
}


// ============================================================
// SIGN LANGUAGE
// ============================================================

function getSignName(gestureName) {
    const signs = {
        Open_Palm: "OPEN HAND",
        Closed_Fist: "FIST",
        Pointing_Up: "POINT UP",
        Thumb_Up: "THUMBS UP",
        Thumb_Down: "THUMBS DOWN",
        Victory: "VICTORY",
        ILoveYou: "I LOVE YOU",
        None: "UNKNOWN"
    };

    return (
        signs[gestureName] ||
        gestureName ||
        "UNKNOWN"
    );
}

function updateSignDisplay(results) {
    if (!signDisplay) return;

    if (!signLanguage) {
        signDisplay.textContent =
            "SIGN // OFF";

        return;
    }

    const gestures =
        results?.gestures || [];

    if (!gestures.length) {
        signDisplay.textContent =
            "SIGN // WAITING";

        return;
    }

    const names = [];

    for (const gestureList of gestures) {
        const top = gestureList?.[0];

        if (top) {
            names.push(
                getSignName(top.categoryName)
            );
        }
    }

    signDisplay.textContent =
        names.length
            ? `SIGN // ${names.join(" | ")}`
            : "SIGN // UNKNOWN";
}


// ============================================================
// FPS
// ============================================================

function updateFPSDisplay() {
    const now = performance.now();

    frameCounter++;

    if (now - fpsTimer >= 1000) {
        currentFPS =
            Math.round(
                frameCounter *
                1000 /
                (now - fpsTimer)
            );

        frameCounter = 0;
        fpsTimer = now;

        if (fpsDisplay) {
            fpsDisplay.textContent =
                `FPS // ${currentFPS}`;
        }
    }
}


// ============================================================
// HAND DETECTION
// ============================================================

async function detectHands(timestamp) {
    if (!gestureRecognizer) {
        return;
    }

    if (
        timestamp - lastHandDetection <
        performanceConfig.hands
    ) {
        return;
    }

    lastHandDetection = timestamp;

    try {
        handResults =
            gestureRecognizer.recognizeForVideo(
                video,
                timestamp
            );
    } catch (error) {
        console.warn(
            "Hand detection error:",
            error
        );
    }
}


// ============================================================
// FACE DETECTION
// ============================================================

async function detectFace(timestamp) {
    if (!faceDetector) {
        return;
    }

    if (
        timestamp - lastFaceDetection <
        performanceConfig.face
    ) {
        return;
    }

    lastFaceDetection = timestamp;

    try {
        faceResults =
            faceDetector.detectForVideo(
                video,
                timestamp
            );
    } catch (error) {
        console.warn(
            "Face detection error:",
            error
        );
    }
}


// ============================================================
// OBJECT DETECTION
// ============================================================

async function detectObjects(timestamp) {
    if (
        !objectDetector ||
        !objectDetection
    ) {
        return;
    }

    if (
        timestamp - lastObjectDetection <
        performanceConfig.objects
    ) {
        return;
    }

    lastObjectDetection = timestamp;

    try {
        objectResults =
            objectDetector.detectForVideo(
                video,
                timestamp
            );
    } catch (error) {
        console.warn(
            "Object detection error:",
            error
        );
    }
}


// ============================================================
// GEOMETRY RENDER
// ============================================================

function renderGeometry() {
    if (!shapesEnabled) {
        return;
    }

    const hands =
        handResults?.landmarks;

    if (!hands || !hands.length) {
        if (
            geometryVisible &&
            performance.now() - geometryLastSeen <
            GEOMETRY_PERSISTENCE
        ) {
            const points =
                geometryPoints.get("current");

            if (points) {
                drawGeometryMode(points);
                drawGeometryCorners(points);
                drawPolygonMeasurements(points);
            }
        }

        return;
    }

    const points =
        getHandGeometry(hands);

    if (!points || points.length < 3) {
        return;
    }

    geometryPoints.set("current", points);

    geometryVisible = true;

    geometryLastSeen = performance.now();

    drawGeometryMode(points);
    drawGeometryCorners(points);
    drawPolygonMeasurements(points);

    if (geometryDisplay) {
        geometryDisplay.textContent =
            `GEOMETRY // ${geometryMode}`;
    }

    if (filterDisplay) {
        filterDisplay.textContent =
            `FILTER // ${geometryFilter}`;
    }
}


// ============================================================
// BUTTON HELPER
// ============================================================

function updateButton(
    button,
    enabled,
    onText,
    offText
) {
    if (!button) return;

    button.textContent =
        enabled ? onText : offText;
}


// ============================================================
// CAMERA BUTTON
// ============================================================

if (cameraBtn) {
    cameraBtn.addEventListener(
        "click",
        () => {
            if (cameraRunning) {
                stopCamera();
            } else {
                startCamera();
            }
        }
    );
}


// ============================================================
// HAND BUTTON
// ============================================================

if (handBtn) {
    handBtn.addEventListener(
        "click",
        () => {
            handTracking = !handTracking;

            updateButton(
                handBtn,
                handTracking,
                "HANDS ON",
                "HANDS OFF"
            );
        }
    );
}


// ============================================================
// TARGET BUTTON
// ============================================================

if (targetBtn) {
    targetBtn.addEventListener(
        "click",
        () => {
            targetVisible = !targetVisible;

            updateButton(
                targetBtn,
                targetVisible,
                "TARGET ON",
                "TARGET OFF"
            );
        }
    );
}


// ============================================================
// OBJECT BUTTON
// ============================================================

if (objectBtn) {
    objectBtn.addEventListener(
        "click",
        async () => {
            objectDetection = !objectDetection;

            if (objectDetection) {
                await loadVision();
                await loadObjectDetector();
            }

            updateButton(
                objectBtn,
                objectDetection,
                "OBJECTS ON",
                "OBJECTS OFF"
            );
        }
    );
}


// ============================================================
// SIGN LANGUAGE BUTTON
// ============================================================

if (signBtn) {
    signBtn.addEventListener(
        "click",
        async () => {
            signLanguage = !signLanguage;

            if (signLanguage) {
                await loadVision();
                await loadGestureRecognizer();
            }

            updateButton(
                signBtn,
                signLanguage,
                "SIGN ON",
                "SIGN OFF"
            );
        }
    );
}


// ============================================================
// FILTER BUTTON
// ============================================================

if (filterBtn) {
    filterBtn.addEventListener(
        "click",
        () => cycleGeometryFilter()
    );
}


// ============================================================
// GEOMETRY BUTTON
// ============================================================

if (shapeBtn) {
    shapeBtn.addEventListener(
        "click",
        () => {
            shapesEnabled = !shapesEnabled;

            updateButton(
                shapeBtn,
                shapesEnabled,
                "GEOMETRY ON",
                "GEOMETRY OFF"
            );
        }
    );
}


// ============================================================
// PRIVACY BUTTON
// ============================================================

if (privacyBtn) {
    privacyBtn.addEventListener(
        "click",
        () => {
            privacyMode =
                privacyMode === "OFF"
                    ? "PIXEL"
                    : "OFF";

            updateButton(
                privacyBtn,
                privacyMode !== "OFF",
                "PRIVACY ON",
                "PRIVACY OFF"
            );

            privacyBtn.title =
                privacyMode === "OFF"
                    ? "Face privacy disabled"
                    : "Pixelated face privacy active";
        }
    );
}


// ============================================================
// FACE FILTER BUTTON
// ============================================================

if (faceFilterBtn) {
    faceFilterBtn.addEventListener(
        "click",
        () => {
            const filters = [
                "OFF",
                "INVERT",
                "SCAN"
            ];

            const current =
                filters.indexOf(faceFilter);

            const next =
                (current + 1) % filters.length;

            faceFilter = filters[next];

            faceFilterBtn.textContent =
                `FACE // ${faceFilter}`;
        }
    );
}


// ============================================================
// FPS BOOST BUTTON
// ============================================================

if (fpsBoostBtn) {
    fpsBoostBtn.addEventListener(
        "click",
        () => {
            if (performanceConfig === PERFORMANCE_NORMAL) {
                performanceConfig = PERFORMANCE_BOOST;

                fpsBoostBtn.textContent =
                    "FPS BOOST ON";
            } else {
                performanceConfig = PERFORMANCE_NORMAL;

                fpsBoostBtn.textContent =
                    "FPS BOOST OFF";
            }
        }
    );
}


// ============================================================
// RESET BUTTON
// ============================================================

if (resetBtn) {
    resetBtn.addEventListener(
        "click",
        () => {
            geometryModeIndex = 0;
            geometryFilterIndex = 0;

            geometryMode = GEOMETRY_MODES[0];
            geometryFilter = GEOMETRY_FILTERS[0];

            geometryPointState.clear();
            geometryPoints.clear();

            firstPinchTime = 0;
            previousBothPinching = false;
            previousAnyPinching = false;
            previousIndexY = null;
            previousHandAngle = null;

            if (geometryDisplay) {
                geometryDisplay.textContent =
                    "GEOMETRY // QUAD WARP";
            }

            if (filterDisplay) {
                filterDisplay.textContent =
                    "FILTER // NORMAL";
            }
        }
    );
}


// ============================================================
// KEYBOARD CONTROLS
// ============================================================

window.addEventListener(
    "keydown",
    event => {
        if (event.key.toLowerCase() === "g") {
            cycleGeometryMode();
        }

        if (event.key.toLowerCase() === "f") {
            cycleGeometryFilter();
        }

        if (event.key.toLowerCase() === "r") {
            geometryModeIndex = 0;
            geometryFilterIndex = 0;

            geometryMode = GEOMETRY_MODES[0];
            geometryFilter = GEOMETRY_FILTERS[0];
        }
    }
);


// ============================================================
// MAIN RENDER LOOP
// ============================================================

async function renderLoop() {
    if (!cameraRunning) {
        return;
    }

    if (
        video.readyState <
        HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
        requestAnimationFrame(renderLoop);
        return;
    }

    resizeCanvas();

    const timestamp = performance.now();

    // Draw camera.
    drawVideo();

    // Detection.
    if (handTracking) {
        await detectHands(timestamp);
    } else {
        handResults = null;
    }

    await detectFace(timestamp);
    await detectObjects(timestamp);

    // Hand visuals and gestures.
    if (
        handTracking &&
        handResults?.landmarks
    ) {
        drawHandSkeleton(
            handResults.landmarks
        );

        drawTargetBox(
            handResults.landmarks
        );

        checkDoublePinch(
            handResults.landmarks
        );

        renderGeometry();
    }

    // Gesture and sign status.
    updateGestureDisplay(handResults);
    updateSignDisplay(handResults);

    // Face privacy and face effects.
    if (faceResults) {
        drawFacePrivacy(faceResults);
        drawFaceFilter(faceResults);
    }

    // Objects.
    drawObjects(objectResults);

    // FPS.
    updateFPSDisplay();

    // Next frame.
    requestAnimationFrame(renderLoop);
}


// ============================================================
// INITIAL UI
// ============================================================

if (gestureDisplay) {
    gestureDisplay.textContent =
        "GESTURE // NONE";
}

if (signDisplay) {
    signDisplay.textContent =
        "SIGN // OFF";
}

if (fpsDisplay) {
    fpsDisplay.textContent =
        "FPS // --";
}

if (geometryDisplay) {
    geometryDisplay.textContent =
        "GEOMETRY // QUAD WARP";
}

if (filterDisplay) {
    filterDisplay.textContent =
        "FILTER // NORMAL";
}

if (targetDisplay) {
    targetDisplay.textContent =
        "TARGET // READY";
}

if (cameraBtn) {
    cameraBtn.textContent =
        "START CAMERA";
}

if (handBtn) {
    handBtn.textContent =
        "HANDS ON";
}

if (targetBtn) {
    targetBtn.textContent =
        "TARGET ON";
}

if (objectBtn) {
    objectBtn.textContent =
        "OBJECTS OFF";
}

if (signBtn) {
    signBtn.textContent =
        "SIGN OFF";
}

if (shapeBtn) {
    shapeBtn.textContent =
        "GEOMETRY ON";
}

if (privacyBtn) {
    privacyBtn.textContent =
        "PRIVACY OFF";
}

if (fpsBoostBtn) {
    fpsBoostBtn.textContent =
        "FPS BOOST OFF";
}


// ============================================================
// READY
// ============================================================

console.log(
    "%cSPECTRA READY",
    "font-weight:bold;font-size:18px"
);

console.log(
    "Geometry filters:",
    GEOMETRY_FILTERS
);
