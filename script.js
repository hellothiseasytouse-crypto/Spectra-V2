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
    findElement("handBtn", "handsBtn", "handTrackingBtn");

const targetBtn =
    findElement("targetBtn", "targetBoxBtn");

const objectBtn =
    findElement("objectBtn", "objectsBtn", "objectDetectionBtn");

const signBtn =
    findElement("signBtn", "signLanguageBtn");

const shapeBtn =
    findElement("shapeBtn", "shapesBtn", "geometryBtn");

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
    findElement("geometry", "geometryDisplay");

const filterDisplay =
    findElement("filter", "filterDisplay");


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
// IMPORTANT:
// DO NOT NAME THIS VARIABLE "performance"
// because browser performance.now() would break.
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
// GEOMETRY
// ============================================================

const GEOMETRY_MODES = [
    "QUAD WARP",
    "DIAMOND",
    "SHARD",
    "FRAME"
];

const GEOMETRY_FILTERS = [
    "NORMAL",
    "BLUR",
    "BRIGHTNESS",
    "CONTRAST",
    "GRAYSCALE",
    "HUE-ROTATE",
    "INVERT",
    "OPACITY",
    "SATURATE",
    "SEPIA",
    "DROP-SHADOW",
    "URL"
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
// DOUBLE PINCH
// ============================================================

const DOUBLE_PINCH_WINDOW = 700;

let firstPinchTime = 0;
let previousBothPinching = false;


// ============================================================
// FPS
// ============================================================

let fpsTimer = performance.now();
let frameCounter = 0;
let currentFPS = 0;


// ============================================================
// TIMERS
// ============================================================

let lastHandDetection = 0;
let lastFaceDetection = 0;
let lastObjectDetection = 0;


// ============================================================
// OFFSCREEN CANVAS
// Used for filtered geometry
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

    color.setAttribute(
        "type",
        "matrix"
    );

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

    glow.setAttribute(
        "in",
        "SourceGraphic"
    );

    const blurNode =
        document.createElementNS(svgNS, "feMergeNode");

    blurNode.setAttribute(
        "in",
        "blur"
    );

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
// LANDMARK -> CANVAS
// ============================================================

function landmarkToCanvas(landmark) {

    if (!landmark) return null;

    return {
        x: landmark.x * canvas.width,
        y: landmark.y * canvas.height
    };
}


// ============================================================
// SMOOTH TRACKED POINT
// ============================================================

function trackedPoint(
    key,
    point
) {

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
// DISTANCE
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


// ============================================================
// MIDPOINT
// ============================================================

function midpoint(a, b) {

    return {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2
    };
}


// ============================================================
// HAND PINCH
// ============================================================

function isPinching(hand) {

    if (!hand || hand.length < 9) {
        return false;
    }

    const thumb =
        hand[4];

    const index =
        hand[8];

    return distance(
        thumb,
        index
    ) < 0.085;
}


// ============================================================
// DOUBLE PINCH
// ============================================================

function checkDoublePinch(hands) {

    if (!hands || hands.length < 2) {

        if (
            firstPinchTime &&
            performance.now() -
            firstPinchTime >
            DOUBLE_PINCH_WINDOW
        ) {
            firstPinchTime = 0;
        }

        previousBothPinching = false;

        return;
    }

    const bothPinching =
        isPinching(hands[0]) &&
        isPinching(hands[1]);

    if (
        bothPinching &&
        !previousBothPinching
    ) {

        const now =
            performance.now();

        if (
            firstPinchTime &&
            now - firstPinchTime <=
            DOUBLE_PINCH_WINDOW
        ) {

            cycleGeometryFilter();

            firstPinchTime = 0;

        } else {

            firstPinchTime = now;
        }
    }

    previousBothPinching =
        bothPinching;
}


// ============================================================
// CYCLE GEOMETRY FILTER
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


// ============================================================
// GET HAND GEOMETRY POINTS
//
// THIS IS THE IMPORTANT PART.
//
// The geometry is now directly attached to:
// thumb tip = 4
// index tip = 8
// middle tip = 12
// ring tip = 16
//
// No floating arbitrary points.
// ============================================================

function getHandGeometry(hands) {

    if (!hands || hands.length === 0) {
        return null;
    }


    // --------------------------------------------------------
    // TWO HANDS
    //
    // Shape connects directly to the thumb/index fingertips
    // of both hands.
    // --------------------------------------------------------

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

        const points = [
            trackedPoint(
                "A_INDEX",
                aIndex
            ),

            trackedPoint(
                "B_INDEX",
                bIndex
            ),

            trackedPoint(
                "B_THUMB",
                bThumb
            ),

            trackedPoint(
                "A_THUMB",
                aThumb
            )
        ];

        return points;
    }


    // --------------------------------------------------------
    // ONE HAND
    //
    // Shape is literally built from the hand's fingertips.
    //
    // index
    // middle
    // ring
    // thumb
    // --------------------------------------------------------

    const hand =
        hands[0];

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
        trackedPoint(
            "ONE_INDEX",
            index
        ),

        trackedPoint(
            "ONE_MIDDLE",
            middle
        ),

        trackedPoint(
            "ONE_RING",
            ring
        ),

        trackedPoint(
            "ONE_THUMB",
            thumb
        )
    ];
}


// ============================================================
// GEOMETRY CENTROID
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


// ============================================================
// DRAW POLYGON PATH
// ============================================================

function createPolygonPath(
    targetCtx,
    points
) {

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
// GEOMETRY FILTER VALUE
// ============================================================

function getGeometryFilterValue() {

    switch (geometryFilter) {

        case "NORMAL":
            return "none";

        case "BLUR":
            return "blur(7px)";

        case "BRIGHTNESS":
            return "brightness(1.8)";

        case "CONTRAST":
            return "contrast(2)";

        case "GRAYSCALE":
            return "grayscale(1)";

        case "HUE-ROTATE":
            return "hue-rotate(135deg)";

        case "INVERT":
            return "invert(1)";

        case "OPACITY":
            return "opacity(0.55)";

        case "SATURATE":
            return "saturate(3)";

        case "SEPIA":
            return "sepia(1)";

        case "DROP-SHADOW":
            return "drop-shadow(0 0 12px white)";

        case "URL":
            return "url(#spectraGlow)";

        default:
            return "none";
    }
}


// ============================================================
// DRAW FILTERED GEOMETRY
// ============================================================

function drawFilteredGeometry(
    points
) {

    if (!points || points.length < 3) {
        return;
    }

    geometryCtx.clearRect(
        0,
        0,
        geometryCanvas.width,
        geometryCanvas.height
    );


    // --------------------------------------------------------
    // CLIP TO HAND GEOMETRY
    // --------------------------------------------------------

    geometryCtx.save();

    createPolygonPath(
        geometryCtx,
        points
    );

    geometryCtx.clip();


    // --------------------------------------------------------
    // APPLY CSS-STYLE FILTER
    // --------------------------------------------------------

    geometryCtx.filter =
        getGeometryFilterValue();


    // --------------------------------------------------------
    // DRAW MIRRORED VIDEO
    // --------------------------------------------------------

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

    geometryCtx.restore();


    // --------------------------------------------------------
    // COMPOSITE FILTERED AREA
    // --------------------------------------------------------

    ctx.save();

    ctx.globalCompositeOperation =
        "source-over";

    ctx.drawImage(
        geometryCanvas,
        0,
        0
    );

    ctx.restore();
}


// ============================================================
// GEOMETRY BORDER
// ============================================================

function drawGeometryBorder(
    points
) {

    if (!points || points.length < 3) {
        return;
    }

    ctx.save();

    createPolygonPath(
        ctx,
        points
    );

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

function drawGeometryCorners(
    points
) {

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

    ctx.setLineDash([
        5,
        5
    ]);

    ctx.beginPath();

    ctx.moveTo(
        ax,
        ay
    );

    ctx.lineTo(
        bx,
        by
    );

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

    ctx.textAlign =
        "center";

    ctx.textBaseline =
        "middle";

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

    ctx.fillStyle =
        "white";

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

function drawPolygonMeasurements(
    points
) {

    if (!points || points.length < 3) {
        return;
    }

    if (points.length === 4) {

        const top =
            distance(
                points[0],
                points[1]
            );

        const bottom =
            distance(
                points[2],
                points[3]
            );

        const left =
            distance(
                points[0],
                points[3]
            );

        const right =
            distance(
                points[1],
                points[2]
            );

        const length =
            Math.round(
                (top + bottom) / 2
            );

        const breadth =
            Math.round(
                (left + right) / 2
            );

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


    // Generic measurement for other shapes

    let minX = Infinity;
    let maxX = -Infinity;

    let minY = Infinity;
    let maxY = -Infinity;

    for (const point of points) {

        minX =
            Math.min(
                minX,
                point.x
            );

        maxX =
            Math.max(
                maxX,
                point.x
            );

        minY =
            Math.min(
                minY,
                point.y
            );

        maxY =
            Math.max(
                maxY,
                point.y
            );
    }

    const width =
        Math.round(
            maxX - minX
        );

    const height =
        Math.round(
            maxY - minY
        );

    const center =
        averagePoint(points);

    ctx.save();

    ctx.font =
        "bold 11px monospace";

    ctx.textAlign =
        "center";

    ctx.fillStyle =
        "rgba(0,0,0,0.75)";

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

function drawGeometryMode(
    points
) {

    if (!points || points.length < 3) {
        return;
    }


    // --------------------------------------------------------
    // QUAD
    // --------------------------------------------------------

    if (geometryMode === "QUAD WARP") {

        drawFilteredGeometry(
            points
        );

        drawGeometryBorder(
            points
        );

        return;
    }


    // --------------------------------------------------------
    // DIAMOND
    // --------------------------------------------------------

    if (geometryMode === "DIAMOND") {

        const center =
            averagePoint(points);

        const diamond = points.map(
            point => {

                return {
                    x:
                        center.x +
                        (point.x - center.x) *
                        1.12,

                    y:
                        center.y +
                        (point.y - center.y) *
                        1.12
                };
            }
        );

        drawFilteredGeometry(
            diamond
        );

        drawGeometryBorder(
            diamond
        );

        return;
    }


    // --------------------------------------------------------
    // SHARD
    // --------------------------------------------------------

    if (geometryMode === "SHARD") {

        const center =
            averagePoint(points);

        const shard =
            points.map(
                (point, index) => {

                    const amount =
                        index % 2 === 0
                            ? 1.25
                            : 0.75;

                    return {
                        x:
                            center.x +
                            (point.x - center.x) *
                            amount,

                        y:
                            center.y +
                            (point.y - center.y) *
                            amount
                    };
                }
            );

        drawFilteredGeometry(
            shard
        );

        drawGeometryBorder(
            shard
        );

        return;
    }


    // --------------------------------------------------------
    // FRAME
    // --------------------------------------------------------

    if (geometryMode === "FRAME") {

        const center =
            averagePoint(points);

        const outer =
            points.map(
                point => {

                    return {
                        x:
                            center.x +
                            (point.x - center.x) *
                            1.18,

                        y:
                            center.y +
                            (point.y - center.y) *
                            1.18
                    };
                }
            );

        drawFilteredGeometry(
            outer
        );

        ctx.save();

        createPolygonPath(
            ctx,
            points
        );

        ctx.strokeStyle =
            "rgba(255,255,255,0.9)";

        ctx.lineWidth = 2;

        ctx.stroke();

        createPolygonPath(
            ctx,
            outer
        );

        ctx.strokeStyle =
            "rgba(255,255,255,0.35)";

        ctx.stroke();

        ctx.restore();

        return;
    }
}


// ============================================================
// HAND SKELETON
// ============================================================

function drawHandSkeleton(
    hands
) {

    if (!hands) return;

    const connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],

        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],

        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],

        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],

        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],

        [5, 9],
        [9, 13],
        [13, 17]
    ];

    ctx.save();

    ctx.lineWidth = 1.4;

    ctx.strokeStyle =
        "rgba(255,255,255,0.55)";

    ctx.fillStyle =
        "rgba(255,255,255,0.9)";

    for (const hand of hands) {

        for (const [
            startIndex,
            endIndex
        ] of connections) {

            const a =
                landmarkToCanvas(
                    hand[startIndex]
                );

            const b =
                landmarkToCanvas(
                    hand[endIndex]
                );

            if (!a || !b) continue;

            ctx.beginPath();

            ctx.moveTo(
                a.x,
                a.y
            );

            ctx.lineTo(
                b.x,
                b.y
            );

            ctx.stroke();
        }

        for (const landmark of hand) {

            const point =
                landmarkToCanvas(
                    landmark
                );

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

function drawTargetBox(
    hands
) {

    if (!targetVisible) return;

    if (!hands || !hands.length) return;

    let minX = Infinity;
    let minY = Infinity;

    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const hand of hands) {

        for (const landmark of hand) {

            const point =
                landmarkToCanvas(
                    landmark
                );

            minX =
                Math.min(
                    minX,
                    point.x
                );

            minY =
                Math.min(
                    minY,
                    point.y
                );

            maxX =
                Math.max(
                    maxX,
                    point.x
                );

            maxY =
                Math.max(
                    maxY,
                    point.y
                );
        }
    }

    const padding = 18;

    minX -= padding;
    minY -= padding;

    maxX += padding;
    maxY += padding;

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.35)";

    ctx.lineWidth = 1;

    ctx.setLineDash([
        5,
        7
    ]);

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

function drawFacePrivacy(
    detections
) {

    if (
        privacyMode === "OFF" ||
        !detections
    ) {
        return;
    }

    const detectionsArray =
        detections.detections ||
        [];

    for (
        const detection
        of detectionsArray
    ) {

        const box =
            detection.boundingBox;

        if (!box) continue;

        const x =
            box.originX;

        const y =
            box.originY;

        const width =
            box.width;

        const height =
            box.height;

        ctx.save();

        // Old pixel-style privacy look

        ctx.fillStyle =
            "rgba(0,0,0,0.94)";

        ctx.fillRect(
            x,
            y,
            width,
            height
        );

        const pixelSize =
            Math.max(
                9,
                Math.floor(
                    width / 9
                )
            );

        for (
            let py = y;
            py < y + height;
            py += pixelSize
        ) {

            for (
                let px = x;
                px < x + width;
                px += pixelSize
            ) {

                const row =
                    Math.floor(
                        (py - y) /
                        pixelSize
                    );

                const col =
                    Math.floor(
                        (px - x) /
                        pixelSize
                    );

                ctx.fillStyle =
                    (row + col) % 2 === 0
                        ? "rgba(255,255,255,0.18)"
                        : "rgba(70,70,70,0.5)";

                ctx.fillRect(
                    px,
                    py,
                    pixelSize - 1,
                    pixelSize - 1
                );
            }
        }


        // Corners

        ctx.strokeStyle =
            "rgba(255,255,255,0.9)";

        ctx.lineWidth = 2;

        const corner = 13;

        ctx.beginPath();

        ctx.moveTo(x, y + corner);
        ctx.lineTo(x, y);
        ctx.lineTo(x + corner, y);

        ctx.moveTo(
            x + width - corner,
            y
        );

        ctx.lineTo(
            x + width,
            y
        );

        ctx.lineTo(
            x + width,
            y + corner
        );

        ctx.moveTo(
            x,
            y + height - corner
        );

        ctx.lineTo(
            x,
            y + height
        );

        ctx.lineTo(
            x + corner,
            y + height
        );

        ctx.moveTo(
            x + width - corner,
            y + height
        );

        ctx.lineTo(
            x + width,
            y + height
        );

        ctx.lineTo(
            x + width,
            y + height - corner
        );

        ctx.stroke();


        ctx.font =
            "bold 10px monospace";

        ctx.fillStyle =
            "white";

        ctx.textAlign =
            "left";

        ctx.fillText(
            "SPECTRA VISION",
            x + 8,
            y + 14
        );

        ctx.fillText(
            "FACE // PIXEL",
            x + 8,
            y + height - 8
        );

        ctx.restore();
    }
}


// ============================================================
// FACE FILTER
// ============================================================

function drawFaceFilter(
    detections
) {

    if (
        faceFilter === "OFF" ||
        !detections
    ) {
        return;
    }

    const list =
        detections.detections ||
        [];

    for (
        const detection
        of list
    ) {

        const box =
            detection.boundingBox;

        if (!box) continue;

        const centerX =
            box.originX +
            box.width / 2;

        const centerY =
            box.originY +
            box.height / 2;

        const radius =
            Math.max(
                box.width,
                box.height
            ) * 0.45;

        ctx.save();

        ctx.beginPath();

        ctx.arc(
            centerX,
            centerY,
            radius,
            0,
            Math.PI * 2
        );

        if (
            faceFilter === "INVERT"
        ) {

            ctx.globalCompositeOperation =
                "difference";

            ctx.fillStyle =
                "white";

            ctx.fill();

        } else if (
            faceFilter === "SCAN"
        ) {

            ctx.strokeStyle =
                "rgba(255,255,255,0.9)";

            ctx.lineWidth = 3;

            ctx.stroke();

            for (
                let i = -radius;
                i < radius;
                i += 8
            ) {

                ctx.beginPath();

                ctx.moveTo(
                    centerX - radius,
                    centerY + i
                );

                ctx.lineTo(
                    centerX + radius,
                    centerY + i
                );

                ctx.strokeStyle =
                    "rgba(255,255,255,0.15)";

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

function drawObjects(
    results
) {

    if (
        !objectDetection ||
        !results
    ) {
        return;
    }

    const detections =
        results.detections ||
        [];

    ctx.save();

    for (
        const detection
        of detections
    ) {

        const box =
            detection.boundingBox;

        if (!box) continue;

        const x =
            box.originX;

        const y =
            box.originY;

        const width =
            box.width;

        const height =
            box.height;

        ctx.strokeStyle =
            "rgba(255,255,255,0.8)";

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
            category?.categoryName ||
            "OBJECT";

        const score =
            category?.score != null
                ? Math.round(
                    category.score * 100
                )
                : 0;

        ctx.font =
            "bold 11px monospace";

        ctx.fillStyle =
            "rgba(0,0,0,0.75)";

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

        ctx.fillStyle =
            "white";

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

function updateGestureDisplay(
    results
) {

    if (!gestureDisplay) return;

    const gestures =
        results?.gestures || [];

    if (!gestures.length) {

        gestureDisplay.textContent =
            "GESTURE // NONE";

        return;
    }

    const names = [];

    for (
        const gestureList
        of gestures
    ) {

        const top =
            gestureList?.[0];

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

function getSignName(
    gestureName
) {

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


function updateSignDisplay(
    results
) {

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

    for (
        const gestureList
        of gestures
    ) {

        const top =
            gestureList?.[0];

        if (top) {

            names.push(
                getSignName(
                    top.categoryName
                )
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

    const now =
        performance.now();

    frameCounter++;

    if (
        now - fpsTimer >=
        1000
    ) {

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
// DETECTION
// ============================================================

async function detectHands(
    timestamp
) {

    if (!gestureRecognizer) {
        return;
    }

    if (
        timestamp -
        lastHandDetection <
        performanceConfig.hands
    ) {
        return;
    }

    lastHandDetection =
        timestamp;

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

async function detectFace(
    timestamp
) {

    if (!faceDetector) {
        return;
    }

    if (
        timestamp -
        lastFaceDetection <
        performanceConfig.face
    ) {
        return;
    }

    lastFaceDetection =
        timestamp;

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

async function detectObjects(
    timestamp
) {

    if (
        !objectDetector ||
        !objectDetection
    ) {
        return;
    }

    if (
        timestamp -
        lastObjectDetection <
        performanceConfig.objects
    ) {
        return;
    }

    lastObjectDetection =
        timestamp;

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
            performance.now() -
            geometryLastSeen <
            GEOMETRY_PERSISTENCE
        ) {

            // Keep last geometry very briefly
            // so fast movements don't make it pop.
            const points =
                geometryPoints.get(
                    "current"
                );

            if (points) {

                drawGeometryMode(
                    points
                );

                drawGeometryCorners(
                    points
                );

                drawPolygonMeasurements(
                    points
                );
            }
        }

        return;
    }


    const points =
        getHandGeometry(
            hands
        );

    if (!points || points.length < 3) {
        return;
    }


    geometryPoints.set(
        "current",
        points
    );

    geometryVisible = true;

    geometryLastSeen =
        performance.now();


    // Main filtered shape

    drawGeometryMode(
        points
    );


    // Fingertip anchors

    drawGeometryCorners(
        points
    );


    // Measurements

    drawPolygonMeasurements(
        points
    );


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
// BUTTON HELPERS
// ============================================================

function updateButton(
    button,
    enabled,
    onText,
    offText
) {

    if (!button) return;

    button.textContent =
        enabled
            ? onText
            : offText;
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

            handTracking =
                !handTracking;

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

            targetVisible =
                !targetVisible;

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

            objectDetection =
                !objectDetection;

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

            signLanguage =
                !signLanguage;

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
// SHAPE BUTTON
// ============================================================

if (shapeBtn) {

    shapeBtn.addEventListener(
        "click",
        () => {

            shapesEnabled =
                !shapesEnabled;

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
                filters.indexOf(
                    faceFilter
                );

            const next =
                (current + 1) %
                filters.length;

            faceFilter =
                filters[next];

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

            if (
                performanceConfig ===
                PERFORMANCE_NORMAL
            ) {

                performanceConfig =
                    PERFORMANCE_BOOST;

                fpsBoostBtn.textContent =
                    "FPS BOOST ON";

            } else {

                performanceConfig =
                    PERFORMANCE_NORMAL;

                fpsBoostBtn.textContent =
                    "FPS BOOST OFF";
            }
        }
    );
}


// ============================================================
// RESET
// ============================================================

if (resetBtn) {

    resetBtn.addEventListener(
        "click",
        () => {

            geometryModeIndex = 0;

            geometryFilterIndex = 0;

            geometryMode =
                GEOMETRY_MODES[0];

            geometryFilter =
                GEOMETRY_FILTERS[0];

            geometryPointState.clear();

            geometryPoints.clear();

            firstPinchTime = 0;

            previousBothPinching = false;

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

        if (event.key === "g") {

            geometryModeIndex++;

            if (
                geometryModeIndex >=
                GEOMETRY_MODES.length
            ) {
                geometryModeIndex = 0;
            }

            geometryMode =
                GEOMETRY_MODES[
                    geometryModeIndex
                ];
        }


        if (event.key === "f") {

            cycleGeometryFilter();
        }


        if (event.key === "r") {

            geometryModeIndex = 0;

            geometryFilterIndex = 0;

            geometryMode =
                GEOMETRY_MODES[0];

            geometryFilter =
                GEOMETRY_FILTERS[0];
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

        requestAnimationFrame(
            renderLoop
        );

        return;
    }

    resizeCanvas();

    const timestamp =
        performance.now();


    // --------------------------------------------------------
    // DRAW CAMERA
    // --------------------------------------------------------

    drawVideo();


    // --------------------------------------------------------
    // DETECTION
    // --------------------------------------------------------

    if (handTracking) {

        await detectHands(
            timestamp
        );
    } else {

        handResults = null;
    }

    await detectFace(
        timestamp
    );

    await detectObjects(
        timestamp
    );


    // --------------------------------------------------------
    // HAND VISUALS
    // --------------------------------------------------------

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


    // --------------------------------------------------------
    // GESTURES
    // --------------------------------------------------------

    updateGestureDisplay(
        handResults
    );

    updateSignDisplay(
        handResults
    );


    // --------------------------------------------------------
    // FACE
    // --------------------------------------------------------

    if (faceResults) {

        drawFacePrivacy(
            faceResults
        );

        drawFaceFilter(
            faceResults
        );
    }


    // --------------------------------------------------------
    // OBJECTS
    // --------------------------------------------------------

    drawObjects(
        objectResults
    );


    // --------------------------------------------------------
    // FPS
    // --------------------------------------------------------

    updateFPSDisplay();


    // --------------------------------------------------------
    // NEXT FRAME
    // --------------------------------------------------------

    requestAnimationFrame(
        renderLoop
    );
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

console.log(
    "Geometry is anchored directly to hand landmarks."
);
