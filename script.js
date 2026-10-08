import {
    FilesetResolver,
    GestureRecognizer,
    FaceDetector,
    ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";

// ============================================================
// ELEMENTS
// ============================================================

function findElement(...ids) {
    for (const id of ids) {
        const el = document.getElementById(id);
        if (el) return el;
    }
    return null;
}

const video = findElement("camera", "video");
const canvas = findElement("overlay", "canvas");

if (!video || !canvas) {
    throw new Error("SPECTRA: Camera video or overlay canvas was not found.");
}

const ctx = canvas.getContext("2d");

const startBtn = findElement("startBtn", "startCamera");
const trackingBtn = findElement("trackingBtn", "handTrackingBtn");
const targetBtn = findElement("targetBtn", "targetBoxBtn");
const objectBtn = findElement("objectBtn", "objectDetectionBtn");
const privacyBtn = findElement("privacyBtn", "facePrivacyBtn");
const signBtn = findElement("signBtn", "signLanguageBtn");
const filterBtn = findElement("filterBtn", "faceFilterBtn");
const shapeBtn = findElement("shapeBtn", "shapesBtn");

// ============================================================
// STATUS UI
// ============================================================

function createStatusElement(id, text) {
    let el = document.getElementById(id);

    if (!el) {
        el = document.createElement("button");
        el.id = id;
        el.type = "button";
        el.className = "spectra-status";
        el.textContent = text;

        const controls =
            document.querySelector(".controls") ||
            document.querySelector("#controls") ||
            document.body;

        controls.appendChild(el);
    }

    return el;
}

const gestureDisplay = createStatusElement(
    "gestureDisplay",
    "GESTURE // WAITING"
);

const signDisplay = createStatusElement(
    "signDisplay",
    "SIGN // OFF"
);

const fpsBoostBtn = createStatusElement(
    "fpsBoostBtn",
    "FPS BOOST OFF"
);

const shapeDisplay = createStatusElement(
    "shapeDisplay",
    "GEOMETRY // READY"
);

// ============================================================
// MEDIAPIPE URLS
// ============================================================

const WASM =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";

const GESTURE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

const FACE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task";

const OBJECT_MODEL =
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/object_detector.tflite";

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

let gestureRecognizer = null;
let faceDetector = null;
let objectDetector = null;

let visionFileset = null;

let loadingVision = false;
let loadingFace = false;
let loadingObjects = false;

// ============================================================
// PERFORMANCE
// IMPORTANT:
// Do NOT call this variable "performance" because the browser
// already has window.performance / performance.now().
// ============================================================

const isMobile =
    /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
    window.innerWidth < 800;

let fpsBoost = false;

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
// DETECTION RESULTS
// ============================================================

let handResults = null;
let faceResults = null;
let objectResults = null;

let lastHandTime = 0;
let lastFaceTime = 0;
let lastObjectTime = 0;

let detectingHands = false;
let detectingFace = false;
let detectingObjects = false;

// ============================================================
// FPS
// ============================================================

let frames = 0;
let fps = 0;
let fpsTimer = performance.now();

// ============================================================
// GEOMETRY
// ============================================================

let geometryMode = 0;

const GEOMETRY_MODES = [
    "QUAD WARP",
    "DIAMOND",
    "SHARD",
    "FRAME"
];

let geometryFilter = 0;

const GEOMETRY_FILTERS = [
    "NORMAL",
    "INVERT",
    "CYBER",
    "BLUEPRINT",
    "CONTRAST"
];

const GEOMETRY_SMOOTHING_SLOW = 0.38;
const GEOMETRY_SMOOTHING_FAST = 0.72;

const geometryPoints = new Map();

let geometryLastSeen = 0;
let lastGeometryDrawData = null;

const GEOMETRY_HOLD_TIME = 180;

// ============================================================
// DOUBLE PINCH
// ============================================================

let bothHandsPinching = false;
let pinchSequenceCount = 0;
let lastPinchSequence = 0;

const DOUBLE_PINCH_WINDOW = 700;

// ============================================================
// CAMERA
// ============================================================

async function startCamera() {
    if (cameraRunning) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("SPECTRA: getUserMedia is not supported.");
        setButtonText(startBtn, "CAMERA UNSUPPORTED");
        return;
    }

    try {
        setButtonText(startBtn, "STARTING...");

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: {
                    ideal: 1280
                },
                height: {
                    ideal: 720
                },
                frameRate: {
                    ideal: 30,
                    max: 60
                }
            },
            audio: false
        });

        video.srcObject = stream;

        await video.play();

        cameraRunning = true;

        setButtonText(startBtn, "CAMERA ON");

        resizeCanvas();

        if (!visionFileset) {
            await loadVision();
        }

        requestAnimationFrame(renderLoop);
    } catch (error) {
        console.error("SPECTRA camera error:", error);

        cameraRunning = false;

        setButtonText(
            startBtn,
            "START CAMERA"
        );
    }
}

async function stopCamera() {
    cameraRunning = false;

    if (video.srcObject) {
        for (const track of video.srcObject.getTracks()) {
            track.stop();
        }

        video.srcObject = null;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    setButtonText(startBtn, "START CAMERA");
}

// ============================================================
// LOAD VISION
// ============================================================

async function loadVision() {
    if (visionFileset) return visionFileset;
    if (loadingVision) return null;

    loadingVision = true;

    try {
        visionFileset = await FilesetResolver.forVisionTasks(WASM);

        gestureRecognizer =
            await GestureRecognizer.createFromOptions(
                visionFileset,
                {
                    baseOptions: {
                        modelAssetPath: GESTURE_MODEL,
                        delegate: "GPU"
                    },
                    runningMode: "VIDEO",
                    numHands: 2,
                    minHandDetectionConfidence: 0.4,
                    minHandPresenceConfidence: 0.4,
                    minTrackingConfidence: 0.4
                }
            );

        return visionFileset;
    } catch (error) {
        console.error("SPECTRA vision loading error:", error);

        gestureRecognizer = null;
        visionFileset = null;

        return null;
    } finally {
        loadingVision = false;
    }
}

async function loadFaceDetector() {
    if (faceDetector) return faceDetector;
    if (loadingFace) return null;

    loadingFace = true;

    try {
        if (!visionFileset) {
            await loadVision();
        }

        if (!visionFileset) return null;

        faceDetector =
            await FaceDetector.createFromOptions(
                visionFileset,
                {
                    baseOptions: {
                        modelAssetPath: FACE_MODEL,
                        delegate: "GPU"
                    },
                    runningMode: "VIDEO",
                    minDetectionConfidence: 0.4
                }
            );

        return faceDetector;
    } catch (error) {
        console.error("SPECTRA face detector error:", error);

        faceDetector = null;

        return null;
    } finally {
        loadingFace = false;
    }
}

async function loadObjectDetector() {
    if (objectDetector) return objectDetector;
    if (loadingObjects) return null;

    loadingObjects = true;

    try {
        if (!visionFileset) {
            await loadVision();
        }

        if (!visionFileset) return null;

        objectDetector =
            await ObjectDetector.createFromOptions(
                visionFileset,
                {
                    baseOptions: {
                        modelAssetPath: OBJECT_MODEL,
                        delegate: "GPU"
                    },
                    runningMode: "VIDEO",
                    scoreThreshold: 0.35,
                    maxResults: 5
                }
            );

        return objectDetector;
    } catch (error) {
        console.error("SPECTRA object detector error:", error);

        objectDetector = null;

        return null;
    } finally {
        loadingObjects = false;
    }
}

// ============================================================
// BUTTON HELPERS
// ============================================================

function setButtonText(button, text) {
    if (button) {
        button.textContent = text;
    }
}

// ============================================================
// BUTTONS
// ============================================================

if (startBtn) {
    startBtn.addEventListener("click", async () => {
        if (cameraRunning) {
            await stopCamera();
        } else {
            await startCamera();
        }
    });
}

if (trackingBtn) {
    trackingBtn.addEventListener("click", async () => {
        handTracking = !handTracking;

        setButtonText(
            trackingBtn,
            handTracking
                ? "HAND TRACKING ON"
                : "HAND TRACKING OFF"
        );

        if (handTracking && !gestureRecognizer) {
            await loadVision();
        }

        if (!handTracking) {
            gestureDisplay.textContent = "GESTURE // OFF";
            signDisplay.textContent = signLanguage
                ? "SIGN // WAITING"
                : "SIGN // OFF";
            shapeDisplay.textContent = "GEOMETRY // OFF";
        }
    });
}

if (targetBtn) {
    targetBtn.addEventListener("click", () => {
        targetVisible = !targetVisible;

        setButtonText(
            targetBtn,
            targetVisible
                ? "TARGET BOX ON"
                : "TARGET BOX OFF"
        );
    });
}

if (objectBtn) {
    objectBtn.addEventListener("click", async () => {
        objectDetection = !objectDetection;

        setButtonText(
            objectBtn,
            objectDetection
                ? "OBJECT DETECTION ON"
                : "OBJECT DETECTION OFF"
        );

        if (objectDetection) {
            const detector = await loadObjectDetector();

            if (!detector) {
                objectDetection = false;

                setButtonText(
                    objectBtn,
                    "OBJECT DETECTION OFF"
                );
            }
        }
    });
}

if (privacyBtn) {
    privacyBtn.addEventListener("click", async () => {
        if (privacyMode === "OFF") {
            privacyMode = "PIXEL";

            setButtonText(
                privacyBtn,
                "FACE PRIVACY ON"
            );

            const detector = await loadFaceDetector();

            if (!detector) {
                privacyMode = "OFF";

                setButtonText(
                    privacyBtn,
                    "FACE PRIVACY OFF"
                );
            }
        } else {
            privacyMode = "OFF";

            setButtonText(
                privacyBtn,
                "FACE PRIVACY OFF"
            );
        }
    });
}

if (signBtn) {
    signBtn.addEventListener("click", async () => {
        signLanguage = !signLanguage;

        setButtonText(
            signBtn,
            signLanguage
                ? "SIGN LANGUAGE ON"
                : "SIGN LANGUAGE OFF"
        );

        if (signLanguage && !gestureRecognizer) {
            await loadVision();
        }

        updateSignLanguage();
    });
}

if (filterBtn) {
    filterBtn.addEventListener("click", async () => {
        if (faceFilter === "OFF") {
            faceFilter = "CYBER";
        } else if (faceFilter === "CYBER") {
            faceFilter = "SCAN";
        } else if (faceFilter === "SCAN") {
            faceFilter = "CROWN";
        } else {
            faceFilter = "OFF";
        }

        setButtonText(
            filterBtn,
            faceFilter === "OFF"
                ? "FACE FILTER OFF"
                : `FACE FILTER ${faceFilter}`
        );

        if (faceFilter !== "OFF") {
            const detector = await loadFaceDetector();

            if (!detector) {
                faceFilter = "OFF";

                setButtonText(
                    filterBtn,
                    "FACE FILTER OFF"
                );
            }
        }
    });
}

if (shapeBtn) {
    shapeBtn.addEventListener("click", () => {
        geometryMode =
            (geometryMode + 1) %
            GEOMETRY_MODES.length;

        shapesEnabled = true;

        shapeDisplay.textContent =
            `GEOMETRY // ${GEOMETRY_MODES[geometryMode]}`;
    });
}

if (fpsBoostBtn) {
    fpsBoostBtn.addEventListener("click", () => {
        fpsBoost = !fpsBoost;

        performanceConfig = fpsBoost
            ? PERFORMANCE_BOOST
            : PERFORMANCE_NORMAL;

        setButtonText(
            fpsBoostBtn,
            fpsBoost
                ? "FPS BOOST ON"
                : "FPS BOOST OFF"
        );
    });
}

// ============================================================
// MAIN LOOP
// ============================================================

async function renderLoop(timestamp) {
    if (!cameraRunning) return;

    resizeCanvas();

    drawCamera();

    if (handTracking) {
        await runHands(timestamp);
    }

    if (
        privacyMode !== "OFF" ||
        faceFilter !== "OFF"
    ) {
        await runFace(timestamp);
    }

    if (objectDetection) {
        await runObjects(timestamp);
    }

    drawFacePrivacy();
    drawFaceFilter();
    drawObjects();

    if (handTracking) {
        drawHandVFX(timestamp);
        drawHands();
    }

    updateFPS();

    requestAnimationFrame(renderLoop);
}

// ============================================================
// CAMERA DRAW
// ============================================================

function resizeCanvas() {
    if (!video.videoWidth || !video.videoHeight) return;

    if (
        canvas.width !== video.videoWidth ||
        canvas.height !== video.videoHeight
    ) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
}

function drawCamera() {
    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.save();

    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

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
// HAND DETECTION
// ============================================================

async function runHands(timestamp) {
    if (!gestureRecognizer) {
        if (!loadingVision) {
            await loadVision();
        }

        return;
    }

    if (
        timestamp - lastHandTime <
        performanceConfig.hands
    ) {
        return;
    }

    if (detectingHands) return;

    detectingHands = true;
    lastHandTime = timestamp;

    try {
        const result =
            gestureRecognizer.recognizeForVideo(
                video,
                timestamp
            );

        handResults = result;

        updateGesture();
        updateSignLanguage();

        if (result?.landmarks?.length) {
            checkDoublePinch(
                result.landmarks,
                timestamp
            );
        } else {
            checkDoublePinch([], timestamp);
        }
    } catch (error) {
        console.error(
            "SPECTRA hand detection error:",
            error
        );
    } finally {
        detectingHands = false;
    }
}

// ============================================================
// FACE DETECTION
// ============================================================

async function runFace(timestamp) {
    if (!faceDetector) {
        if (!loadingFace) {
            await loadFaceDetector();
        }

        return;
    }

    if (
        timestamp - lastFaceTime <
        performanceConfig.face
    ) {
        return;
    }

    if (detectingFace) return;

    detectingFace = true;
    lastFaceTime = timestamp;

    try {
        faceResults =
            faceDetector.detectForVideo(
                video,
                timestamp
            );
    } catch (error) {
        console.error(
            "SPECTRA face detection error:",
            error
        );
    } finally {
        detectingFace = false;
    }
}

// ============================================================
// OBJECT DETECTION
// ============================================================

async function runObjects(timestamp) {
    if (!objectDetector) {
        if (!loadingObjects) {
            await loadObjectDetector();
        }

        return;
    }

    if (
        timestamp - lastObjectTime <
        performanceConfig.objects
    ) {
        return;
    }

    if (detectingObjects) return;

    detectingObjects = true;
    lastObjectTime = timestamp;

    try {
        objectResults =
            objectDetector.detectForVideo(
                video,
                timestamp
            );
    } catch (error) {
        console.error(
            "SPECTRA object detection error:",
            error
        );
    } finally {
        detectingObjects = false;
    }
}

// ============================================================
// GESTURE DISPLAY
// ============================================================

function updateGesture() {
    if (!handTracking) {
        gestureDisplay.textContent =
            "GESTURE // OFF";
        return;
    }

    if (
        !handResults ||
        !handResults.gestures ||
        !handResults.gestures.length
    ) {
        gestureDisplay.textContent =
            "GESTURE // NONE";
        return;
    }

    const names = [];

    for (const gestureList of handResults.gestures) {
        if (!gestureList?.length) continue;

        const gesture = gestureList[0];

        if (gesture?.categoryName) {
            names.push(
                formatGestureName(
                    gesture.categoryName
                )
            );
        }
    }

    gestureDisplay.textContent =
        names.length
            ? `GESTURE // ${names.join(" + ")}`
            : "GESTURE // NONE";
}

function formatGestureName(name) {
    if (!name) return "UNKNOWN";

    return String(name)
        .replace(/_/g, " ")
        .toUpperCase();
}

// ============================================================
// SIGN LANGUAGE
// ============================================================

function updateSignLanguage() {
    if (!signLanguage) {
        signDisplay.textContent =
            "SIGN // OFF";
        return;
    }

    if (
        !handResults ||
        !handResults.landmarks ||
        !handResults.landmarks.length
    ) {
        signDisplay.textContent =
            "SIGN // WAITING";
        return;
    }

    const signs = [];

    for (const hand of handResults.landmarks) {
        signs.push(getSign(hand));
    }

    signDisplay.textContent =
        `SIGN // ${signs.join(" + ")}`;
}

function getSign(hand) {
    if (!hand || hand.length < 21) {
        return "UNKNOWN";
    }

    const thumb = fingerUp(
        hand,
        4,
        3
    );

    const index = fingerUp(
        hand,
        8,
        6
    );

    const middle = fingerUp(
        hand,
        12,
        10
    );

    const ring = fingerUp(
        hand,
        16,
        14
    );

    const pinky = fingerUp(
        hand,
        20,
        18
    );

    if (
        index &&
        middle &&
        !ring &&
        !pinky
    ) {
        return "V";
    }

    if (
        index &&
        !middle &&
        !ring &&
        !pinky
    ) {
        return "POINT";
    }

    if (
        pinky &&
        !index &&
        !middle &&
        !ring
    ) {
        return "I";
    }

    if (
        thumb &&
        index &&
        middle &&
        ring &&
        pinky
    ) {
        return "OPEN";
    }

    if (
        !thumb &&
        !index &&
        !middle &&
        !ring &&
        !pinky
    ) {
        return "FIST";
    }

    if (
        thumb &&
        !index &&
        !middle &&
        !ring &&
        !pinky
    ) {
        return "THUMBS UP";
    }

    return "UNKNOWN";
}

function fingerUp(hand, tipIndex, pipIndex) {
    return hand[tipIndex].y < hand[pipIndex].y;
}

// ============================================================
// DOUBLE PINCH
// ============================================================

function isPinchingFast(hand) {
    if (!hand || hand.length < 21) {
        return false;
    }

    const thumb = hand[4];
    const index = hand[8];

    const distanceValue = Math.hypot(
        thumb.x - index.x,
        thumb.y - index.y
    );

    return distanceValue < 0.085;
}

function checkDoublePinch(hands, timestamp) {
    if (timestamp - lastPinchSequence > DOUBLE_PINCH_WINDOW) {
        pinchSequenceCount = 0;
    }

    if (!hands || hands.length < 2) {
        bothHandsPinching = false;
        return;
    }

    const first = isPinchingFast(hands[0]);
    const second = isPinchingFast(hands[1]);

    const both = first && second;

    if (both && !bothHandsPinching) {
        if (
            timestamp - lastPinchSequence <=
            DOUBLE_PINCH_WINDOW
        ) {
            pinchSequenceCount++;
        } else {
            pinchSequenceCount = 1;
        }

        lastPinchSequence = timestamp;

        if (pinchSequenceCount >= 2) {
            cycleGeometryFilter();

            pinchSequenceCount = 0;
            lastPinchSequence = 0;
        }
    }

    bothHandsPinching = both;
}

function cycleGeometryFilter() {
    geometryFilter =
        (geometryFilter + 1) %
        GEOMETRY_FILTERS.length;

    shapeDisplay.textContent =
        `GEOMETRY // ${GEOMETRY_MODES[geometryMode]} // ${GEOMETRY_FILTERS[geometryFilter]}`;
}

// ============================================================
// GEOMETRY MAIN
// ============================================================

function drawHandVFX(timestamp) {
    if (!shapesEnabled) return;

    const hands =
        handResults?.landmarks || [];

    if (hands.length >= 2) {
        drawTwoHandGeometry(
            hands[0],
            hands[1],
            timestamp
        );

        geometryLastSeen = timestamp;

        return;
    }

    if (hands.length === 1) {
        drawLandmarkGeometry(
            hands[0],
            timestamp
        );

        geometryLastSeen = timestamp;

        return;
    }

    if (
        lastGeometryDrawData &&
        timestamp - geometryLastSeen <
        GEOMETRY_HOLD_TIME
    ) {
        const fade =
            1 -
            (
                timestamp -
                geometryLastSeen
            ) /
            GEOMETRY_HOLD_TIME;

        ctx.save();

        ctx.globalAlpha =
            Math.max(
                0,
                Math.min(1, fade)
            );

        drawStoredGeometry(
            lastGeometryDrawData,
            timestamp
        );

        ctx.restore();

        shapeDisplay.textContent =
            `GEOMETRY // HOLD`;

        return;
    }

    shapeDisplay.textContent =
        `GEOMETRY // ${GEOMETRY_MODES[geometryMode]} // ${GEOMETRY_FILTERS[geometryFilter]}`;
}

// ============================================================
// TWO HAND GEOMETRY
// ============================================================

function drawTwoHandGeometry(
    handA,
    handB,
    timestamp
) {
    const centerA = getPalmCenter(handA);
    const centerB = getPalmCenter(handB);

    let leftHand = handA;
    let rightHand = handB;

    if (centerA.x > centerB.x) {
        leftHand = handB;
        rightHand = handA;
    }

    const leftIndex = trackedPoint(
        "leftIndex",
        point(leftHand[8]),
        timestamp
    );

    const leftThumb = trackedPoint(
        "leftThumb",
        point(leftHand[4]),
        timestamp
    );

    const rightIndex = trackedPoint(
        "rightIndex",
        point(rightHand[8]),
        timestamp
    );

    const rightThumb = trackedPoint(
        "rightThumb",
        point(rightHand[4]),
        timestamp
    );

    const leftTop =
        leftIndex.y < leftThumb.y
            ? leftIndex
            : leftThumb;

    const leftBottom =
        leftIndex.y < leftThumb.y
            ? leftThumb
            : leftIndex;

    const rightTop =
        rightIndex.y < rightThumb.y
            ? rightIndex
            : rightThumb;

    const rightBottom =
        rightIndex.y < rightThumb.y
            ? rightThumb
            : rightIndex;

    const corners = {
        tl: leftTop,
        tr: rightTop,
        br: rightBottom,
        bl: leftBottom
    };

    const quad = [
        corners.tl,
        corners.tr,
        corners.br,
        corners.bl
    ];

    lastGeometryDrawData = {
        type: "quad",
        points: quad.map(clonePoint)
    };

    drawQuadSurface(
        quad,
        timestamp
    );

    drawGeometryAnchors(
        quad
    );

    drawQuadGrid(
        quad
    );

    drawPolygonMeasurements(
        quad,
        "QUAD"
    );

    shapeDisplay.textContent =
        `GEOMETRY // ${GEOMETRY_MODES[geometryMode]} // ${GEOMETRY_FILTERS[geometryFilter]}`;
}

// ============================================================
// ONE HAND GEOMETRY
// ============================================================

function drawLandmarkGeometry(
    hand,
    timestamp
) {
    const wrist = trackedPoint(
        "wrist",
        point(hand[0]),
        timestamp
    );

    const thumb = trackedPoint(
        "thumb",
        point(hand[4]),
        timestamp
    );

    const index = trackedPoint(
        "index",
        point(hand[8]),
        timestamp
    );

    const middle = trackedPoint(
        "middle",
        point(hand[12]),
        timestamp
    );

    const ring = trackedPoint(
        "ring",
        point(hand[16]),
        timestamp
    );

    const pinky = trackedPoint(
        "pinky",
        point(hand[20]),
        timestamp
    );

    const pinch =
        distance(
            thumb,
            index
        );

    if (geometryMode === 0) {
        drawQuadWarp(
            wrist,
            thumb,
            index,
            pinky,
            pinch,
            timestamp
        );
    } else if (geometryMode === 1) {
        drawLandmarkDiamond(
            thumb,
            index,
            middle,
            ring,
            timestamp
        );
    } else if (geometryMode === 2) {
        drawLandmarkShard(
            wrist,
            thumb,
            index,
            middle,
            ring,
            pinky,
            timestamp
        );
    } else {
        drawLandmarkFrame(
            wrist,
            thumb,
            index,
            middle,
            ring,
            pinky,
            timestamp
        );
    }

    shapeDisplay.textContent =
        `GEOMETRY // ${GEOMETRY_MODES[geometryMode]} // ${GEOMETRY_FILTERS[geometryFilter]}`;
}

// ============================================================
// QUAD WARP
// ============================================================

function drawQuadWarp(
    wrist,
    thumb,
    index,
    pinky,
    pinch,
    timestamp
) {
    const rawPoints = [
        wrist,
        thumb,
        index,
        pinky
    ];

    const quad = orderAroundCenter(
        rawPoints
    );

    lastGeometryDrawData = {
        type: "polygon",
        points: quad.map(clonePoint)
    };

    drawPolygonSurface(
        quad,
        timestamp
    );

    drawQuadGrid(
        quad
    );

    drawGeometryAnchors(
        quad
    );

    drawPolygonMeasurements(
        quad,
        "HAND"
    );
}

// ============================================================
// DIAMOND
// ============================================================

function drawLandmarkDiamond(
    thumb,
    index,
    middle,
    ring,
    timestamp
) {
    const raw = [
        index,
        middle,
        ring,
        thumb
    ];

    const diamond =
        orderAroundCenter(raw);

    lastGeometryDrawData = {
        type: "polygon",
        points: diamond.map(clonePoint)
    };

    drawPolygonSurface(
        diamond,
        timestamp
    );

    const center =
        getPolygonCenter(diamond);

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.45)";

    ctx.lineWidth = 1;

    for (const p of diamond) {
        ctx.beginPath();
        ctx.moveTo(center.x, center.y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
    }

    ctx.restore();

    drawGeometryAnchors(
        diamond
    );

    drawPolygonMeasurements(
        diamond,
        "DIAMOND"
    );
}

// ============================================================
// SHARD
// ============================================================

function drawLandmarkShard(
    wrist,
    thumb,
    index,
    middle,
    ring,
    pinky,
    timestamp
) {
    const raw = [
        wrist,
        thumb,
        index,
        middle,
        ring,
        pinky
    ];

    const center =
        getPolygonCenter(raw);

    const shard = raw
        .slice()
        .sort(
            (a, b) =>
                Math.atan2(
                    a.y - center.y,
                    a.x - center.x
                ) -
                Math.atan2(
                    b.y - center.y,
                    b.x - center.x
                )
        );

    lastGeometryDrawData = {
        type: "polygon",
        points: shard.map(clonePoint)
    };

    drawPolygonSurface(
        shard,
        timestamp
    );

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.4)";

    ctx.lineWidth = 1;

    for (let i = 1; i < shard.length - 1; i++) {
        ctx.beginPath();
        ctx.moveTo(
            shard[0].x,
            shard[0].y
        );
        ctx.lineTo(
            shard[i].x,
            shard[i].y
        );
        ctx.stroke();
    }

    ctx.restore();

    drawGeometryAnchors(
        shard
    );

    drawPolygonMeasurements(
        shard,
        "SHARD"
    );
}

// ============================================================
// FRAME
// ============================================================

function drawLandmarkFrame(
    wrist,
    thumb,
    index,
    middle,
    ring,
    pinky,
    timestamp
) {
    const raw = [
        wrist,
        thumb,
        index,
        pinky
    ];

    const frame =
        orderAroundCenter(raw);

    lastGeometryDrawData = {
        type: "polygon",
        points: frame.map(clonePoint)
    };

    drawPolygonSurface(
        frame,
        timestamp
    );

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.35)";

    ctx.lineWidth = 1;

    const center =
        getPolygonCenter(frame);

    ctx.beginPath();
    ctx.moveTo(
        center.x,
        center.y
    );
    ctx.lineTo(
        middle.x,
        middle.y
    );
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(
        ring.x,
        ring.y
    );
    ctx.lineTo(
        middle.x,
        middle.y
    );
    ctx.stroke();

    ctx.restore();

    drawGeometryAnchors(
        frame
    );

    drawPolygonMeasurements(
        frame,
        "FRAME"
    );
}

// ============================================================
// POLYGON SURFACE
// ============================================================

function drawPolygonSurface(
    points,
    timestamp
) {
    if (!points || points.length < 3) {
        return;
    }

    drawGeometryFilter(
        points,
        timestamp
    );

    ctx.save();

    ctx.beginPath();

    ctx.moveTo(
        points[0].x,
        points[0].y
    );

    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(
            points[i].x,
            points[i].y
        );
    }

    ctx.closePath();

    ctx.strokeStyle =
        "rgba(255,255,255,0.95)";

    ctx.lineWidth = 2;

    ctx.shadowColor =
        "rgba(255,255,255,0.25)";

    ctx.shadowBlur = 8;

    ctx.stroke();

    ctx.restore();
}

function drawQuadSurface(
    quad,
    timestamp
) {
    if (!quad || quad.length !== 4) {
        return;
    }

    drawGeometryFilter(
        quad,
        timestamp
    );

    ctx.save();

    ctx.beginPath();

    ctx.moveTo(
        quad[0].x,
        quad[0].y
    );

    ctx.lineTo(
        quad[1].x,
        quad[1].y
    );

    ctx.lineTo(
        quad[2].x,
        quad[2].y
    );

    ctx.lineTo(
        quad[3].x,
        quad[3].y
    );

    ctx.closePath();

    ctx.strokeStyle =
        "rgba(255,255,255,0.95)";

    ctx.lineWidth = 2;

    ctx.stroke();

    ctx.restore();
}

// ============================================================
// GEOMETRY FILTERS
// ============================================================

function drawGeometryFilter(
    points,
    timestamp
) {
    if (!points || points.length < 3) {
        return;
    }

    ctx.save();

    ctx.beginPath();

    ctx.moveTo(
        points[0].x,
        points[0].y
    );

    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(
            points[i].x,
            points[i].y
        );
    }

    ctx.closePath();

    ctx.clip();

    const bounds =
        getBounds(points);

    if (geometryFilter === 0) {
        ctx.fillStyle =
            "rgba(255,255,255,0.055)";

        ctx.fillRect(
            bounds.minX,
            bounds.minY,
            bounds.width,
            bounds.height
        );
    }

    if (geometryFilter === 1) {
        // REAL BLACK/WHITE INVERSION
        // Since the camera has already been drawn onto
        // the same canvas, white + difference inverts
        // everything underneath inside the geometry.

        ctx.globalCompositeOperation =
            "difference";

        ctx.fillStyle =
            "#ffffff";

        ctx.fillRect(
            bounds.minX,
            bounds.minY,
            bounds.width,
            bounds.height
        );
    }

    if (geometryFilter === 2) {
        // CYBER

        ctx.fillStyle =
            "rgba(0,255,255,0.08)";

        ctx.fillRect(
            bounds.minX,
            bounds.minY,
            bounds.width,
            bounds.height
        );

        ctx.strokeStyle =
            "rgba(255,255,255,0.5)";

        ctx.lineWidth = 1;

        const spacing = 14;

        for (
            let x = bounds.minX - bounds.height;
            x < bounds.maxX + bounds.height;
            x += spacing
        ) {
            ctx.beginPath();

            ctx.moveTo(
                x,
                bounds.minY
            );

            ctx.lineTo(
                x + bounds.height,
                bounds.maxY
            );

            ctx.stroke();
        }
    }

    if (geometryFilter === 3) {
        // BLUEPRINT

        ctx.fillStyle =
            "rgba(80,150,255,0.09)";

        ctx.fillRect(
            bounds.minX,
            bounds.minY,
            bounds.width,
            bounds.height
        );

        ctx.strokeStyle =
            "rgba(120,180,255,0.38)";

        ctx.lineWidth = 1;

        const grid = 16;

        for (
            let x = bounds.minX;
            x <= bounds.maxX;
            x += grid
        ) {
            ctx.beginPath();
            ctx.moveTo(
                x,
                bounds.minY
            );
            ctx.lineTo(
                x,
                bounds.maxY
            );
            ctx.stroke();
        }

        for (
            let y = bounds.minY;
            y <= bounds.maxY;
            y += grid
        ) {
            ctx.beginPath();
            ctx.moveTo(
                bounds.minX,
                y
            );
            ctx.lineTo(
                bounds.maxX,
                y
            );
            ctx.stroke();
        }
    }

    if (geometryFilter === 4) {
        // CONTRAST

        ctx.fillStyle =
            "rgba(255,255,255,0.16)";

        ctx.fillRect(
            bounds.minX,
            bounds.minY,
            bounds.width,
            bounds.height
        );

        ctx.strokeStyle =
            "rgba(0,0,0,0.65)";

        ctx.lineWidth = 2;

        const spacing = 18;

        for (
            let x = bounds.minX - bounds.height;
            x < bounds.maxX + bounds.height;
            x += spacing
        ) {
            ctx.beginPath();

            ctx.moveTo(
                x,
                bounds.minY
            );

            ctx.lineTo(
                x + bounds.height,
                bounds.maxY
            );

            ctx.stroke();
        }
    }

    // Moving scan line
    const scan =
        (
            timestamp * 0.12
        ) %
        Math.max(
            1,
            bounds.height + 80
        );

    ctx.fillStyle =
        "rgba(255,255,255,0.15)";

    ctx.fillRect(
        bounds.minX,
        bounds.minY + scan - 40,
        bounds.width,
        2
    );

    ctx.restore();
}

// ============================================================
// QUAD GRID
// ============================================================

function drawQuadGrid(quad) {
    if (!quad || quad.length !== 4) {
        return;
    }

    const top =
        lerpPoint(
            quad[0],
            quad[1],
            0.5
        );

    const bottom =
        lerpPoint(
            quad[3],
            quad[2],
            0.5
        );

    const left =
        lerpPoint(
            quad[0],
            quad[3],
            0.5
        );

    const right =
        lerpPoint(
            quad[1],
            quad[2],
            0.5
        );

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.18)";

    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(
        top.x,
        top.y
    );
    ctx.lineTo(
        bottom.x,
        bottom.y
    );
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(
        left.x,
        left.y
    );
    ctx.lineTo(
        right.x,
        right.y
    );
    ctx.stroke();

    ctx.restore();
}

// ============================================================
// GEOMETRY ANCHORS
// ============================================================

function drawGeometryAnchors(points) {
    ctx.save();

    for (const p of points) {
        ctx.beginPath();

        ctx.arc(
            p.x,
            p.y,
            4,
            0,
            Math.PI * 2
        );

        ctx.fillStyle =
            "#ffffff";

        ctx.fill();

        ctx.beginPath();

        ctx.arc(
            p.x,
            p.y,
            8,
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
// MEASUREMENTS
// ============================================================

function drawPolygonMeasurements(
    points,
    prefix
) {
    if (!points || points.length < 3) {
        return;
    }

    const bounds =
        getBounds(points);

    const width =
        bounds.width;

    const height =
        bounds.height;

    if (points.length === 4) {
        const length =
            (
                distance(
                    points[0],
                    points[1]
                ) +
                distance(
                    points[3],
                    points[2]
                )
            ) / 2;

        const breadth =
            (
                distance(
                    points[0],
                    points[3]
                ) +
                distance(
                    points[1],
                    points[2]
                )
            ) / 2;

        const center =
            getPolygonCenter(points);

        drawMeasurementOutside(
            points[0],
            points[1],
            center,
            `LENGTH ${Math.round(length)} PX`,
            30
        );

        drawMeasurementOutside(
            points[0],
            points[3],
            center,
            `BREADTH ${Math.round(breadth)} PX`,
            30
        );

        return;
    }

    drawHudLabel(
        bounds.minX + 8,
        bounds.minY - 28,
        `${prefix} WIDTH ${Math.round(width)} PX`
    );

    drawHudLabel(
        bounds.minX + 8,
        bounds.minY - 2,
        `${prefix} HEIGHT ${Math.round(height)} PX`
    );
}

function drawMeasurementOutside(
    a,
    b,
    polygonCenter,
    label,
    offset
) {
    const dx =
        b.x - a.x;

    const dy =
        b.y - a.y;

    const len =
        Math.hypot(dx, dy) || 1;

    let nx =
        -dy / len;

    let ny =
        dx / len;

    const midX =
        (a.x + b.x) / 2;

    const midY =
        (a.y + b.y) / 2;

    const toCenterX =
        polygonCenter.x - midX;

    const toCenterY =
        polygonCenter.y - midY;

    // Make the normal point away from the polygon.
    if (
        nx * toCenterX +
        ny * toCenterY >
        0
    ) {
        nx *= -1;
        ny *= -1;
    }

    const p1 = {
        x: a.x + nx * offset,
        y: a.y + ny * offset
    };

    const p2 = {
        x: b.x + nx * offset,
        y: b.y + ny * offset
    };

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.72)";

    ctx.lineWidth = 1;

    ctx.setLineDash([5, 4]);

    ctx.beginPath();

    ctx.moveTo(
        p1.x,
        p1.y
    );

    ctx.lineTo(
        p2.x,
        p2.y
    );

    ctx.stroke();

    ctx.setLineDash([]);

    drawMeasurementTick(
        p1,
        dx,
        dy
    );

    drawMeasurementTick(
        p2,
        dx,
        dy
    );

    ctx.restore();

    const labelX =
        (
            p1.x +
            p2.x
        ) / 2;

    const labelY =
        (
            p1.y +
            p2.y
        ) / 2;

    drawHudLabel(
        labelX,
        labelY,
        label,
        true
    );
}

function drawMeasurementTick(
    pointValue,
    dx,
    dy
) {
    const len =
        Math.hypot(dx, dy) || 1;

    const nx =
        -dy / len;

    const ny =
        dx / len;

    const size = 6;

    ctx.beginPath();

    ctx.moveTo(
        pointValue.x - nx * size,
        pointValue.y - ny * size
    );

    ctx.lineTo(
        pointValue.x + nx * size,
        pointValue.y + ny * size
    );

    ctx.stroke();
}

function drawHudLabel(
    x,
    y,
    text,
    centered = false
) {
    ctx.save();

    ctx.font =
        "bold 11px monospace";

    const paddingX = 8;
    const paddingY = 6;

    const width =
        ctx.measureText(text).width +
        paddingX * 2;

    const height = 22;

    let drawX =
        centered
            ? x - width / 2
            : x;

    let drawY =
        y - height / 2;

    drawX = clamp(
        drawX,
        4,
        canvas.width - width - 4
    );

    drawY = clamp(
        drawY,
        4,
        canvas.height - height - 4
    );

    if (ctx.roundRect) {
        ctx.beginPath();

        ctx.roundRect(
            drawX,
            drawY,
            width,
            height,
            5
        );

        ctx.fillStyle =
            "rgba(0,0,0,0.72)";

        ctx.fill();

        ctx.strokeStyle =
            "rgba(255,255,255,0.55)";

        ctx.lineWidth = 1;

        ctx.stroke();
    } else {
        ctx.fillStyle =
            "rgba(0,0,0,0.72)";

        ctx.fillRect(
            drawX,
            drawY,
            width,
            height
        );
    }

    ctx.fillStyle =
        "#ffffff";

    ctx.textBaseline =
        "middle";

    ctx.fillText(
        text,
        drawX + paddingX,
        drawY + height / 2
    );

    ctx.restore();
}

// ============================================================
// STORED GEOMETRY
// ============================================================

function drawStoredGeometry(
    data,
    timestamp
) {
    if (!data?.points) return;

    if (data.type === "quad") {
        drawQuadSurface(
            data.points,
            timestamp
        );

        drawQuadGrid(
            data.points
        );

        drawGeometryAnchors(
            data.points
        );

        drawPolygonMeasurements(
            data.points,
            "QUAD"
        );
    } else {
        drawPolygonSurface(
            data.points,
            timestamp
        );

        drawGeometryAnchors(
            data.points
        );

        drawPolygonMeasurements(
            data.points,
            "HAND"
        );
    }
}

// ============================================================
// TRACKING / SMOOTHING
// ============================================================

function trackedPoint(
    id,
    target,
    timestamp
) {
    let state =
        geometryPoints.get(id);

    if (!state) {
        state = {
            x: target.x,
            y: target.y,
            vx: 0,
            vy: 0,
            lastX: target.x,
            lastY: target.y,
            lastTime: timestamp
        };

        geometryPoints.set(
            id,
            state
        );

        return {
            x: target.x,
            y: target.y
        };
    }

    const dt =
        Math.max(
            0.001,
            Math.min(
                0.1,
                (
                    timestamp -
                    state.lastTime
                ) / 1000
            )
        );

    const rawVx =
        (
            target.x -
            state.lastX
        ) / dt;

    const rawVy =
        (
            target.y -
            state.lastY
        ) / dt;

    state.vx =
        state.vx * 0.65 +
        rawVx * 0.35;

    state.vy =
        state.vy * 0.65 +
        rawVy * 0.35;

    const speed =
        Math.hypot(
            state.vx,
            state.vy
        );

    const normalizedSpeed =
        Math.min(
            1,
            speed / 1400
        );

    const alpha =
        GEOMETRY_SMOOTHING_SLOW +
        (
            GEOMETRY_SMOOTHING_FAST -
            GEOMETRY_SMOOTHING_SLOW
        ) *
        normalizedSpeed;

    // Small prediction helps quick movements,
    // but avoids the giant overshoot from the old version.
    const prediction =
        Math.min(
            0.045,
            0.012 +
            normalizedSpeed * 0.033
        );

    let predictedX =
        target.x +
        state.vx *
        prediction;

    let predictedY =
        target.y +
        state.vy *
        prediction;

    predictedX =
        clamp(
            predictedX,
            0,
            canvas.width
        );

    predictedY =
        clamp(
            predictedY,
            0,
            canvas.height
        );

    state.x =
        state.x +
        (
            predictedX -
            state.x
        ) *
        alpha;

    state.y =
        state.y +
        (
            predictedY -
            state.y
        ) *
        alpha;

    state.lastX =
        target.x;

    state.lastY =
        target.y;

    state.lastTime =
        timestamp;

    return {
        x: state.x,
        y: state.y
    };
}

// ============================================================
// HAND SKELETON
// ============================================================

function drawHands() {
    if (
        !handResults ||
        !handResults.landmarks
    ) {
        return;
    }

    for (const hand of handResults.landmarks) {
        drawSingleHand(hand);
    }
}

function drawSingleHand(hand) {
    if (!hand || hand.length < 21) {
        return;
    }

    const connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],

        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],

        [5, 9],
        [9, 10],
        [10, 11],
        [11, 12],

        [9, 13],
        [13, 14],
        [14, 15],
        [15, 16],

        [13, 17],
        [17, 18],
        [18, 19],
        [19, 20],

        [0, 17]
    ];

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.5)";

    ctx.lineWidth = 1;

    for (const [a, b] of connections) {
        const p1 = mirrorPoint(
            hand[a]
        );

        const p2 = mirrorPoint(
            hand[b]
        );

        ctx.beginPath();

        ctx.moveTo(
            p1.x,
            p1.y
        );

        ctx.lineTo(
            p2.x,
            p2.y
        );

        ctx.stroke();
    }

    for (const landmark of hand) {
        const p =
            mirrorPoint(
                landmark
            );

        ctx.beginPath();

        ctx.arc(
            p.x,
            p.y,
            2.2,
            0,
            Math.PI * 2
        );

        ctx.fillStyle =
            "#ffffff";

        ctx.fill();
    }

    ctx.restore();
}

// ============================================================
// FACE PRIVACY
// ============================================================

function drawFacePrivacy() {
    if (
        privacyMode !== "PIXEL"
    ) {
        return;
    }

    if (
        !faceResults ||
        !faceResults.detections
    ) {
        return;
    }

    for (const detection of faceResults.detections) {
        const box =
            detection.boundingBox;

        if (!box) continue;

        const mirrored =
            mirrorBoundingBox(box);

        drawPixelPrivacy(
            mirrored
        );
    }
}

function drawPixelPrivacy(box) {
    const x =
        box.originX;

    const y =
        box.originY;

    const width =
        box.width;

    const height =
        box.height;

    ctx.save();

    // Old pixel-style privacy base
    ctx.fillStyle =
        "rgba(0,0,0,0.88)";

    ctx.fillRect(
        x,
        y,
        width,
        height
    );

    const blockSize =
        Math.max(
            10,
            Math.floor(
                Math.min(
                    width,
                    height
                ) / 8
            )
        );

    for (
        let py = y;
        py < y + height;
        py += blockSize
    ) {
        for (
            let px = x;
            px < x + width;
            px += blockSize
        ) {
            const checker =
                (
                    Math.floor(
                        (px - x) /
                        blockSize
                    ) +
                    Math.floor(
                        (py - y) /
                        blockSize
                    )
                ) % 2;

            ctx.fillStyle =
                checker === 0
                    ? "rgba(255,255,255,0.14)"
                    : "rgba(80,80,80,0.25)";

            ctx.fillRect(
                px,
                py,
                Math.min(
                    blockSize,
                    x + width - px
                ),
                Math.min(
                    blockSize,
                    y + height - py
                )
            );
        }
    }

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";

    ctx.lineWidth = 1;

    const corner =
        Math.min(
            18,
            width * 0.18,
            height * 0.18
        );

    // Top left
    ctx.beginPath();
    ctx.moveTo(x, y + corner);
    ctx.lineTo(x, y);
    ctx.lineTo(x + corner, y);
    ctx.stroke();

    // Top right
    ctx.beginPath();
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
    ctx.stroke();

    // Bottom left
    ctx.beginPath();
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
    ctx.stroke();

    // Bottom right
    ctx.beginPath();
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
        "#ffffff";

    ctx.fillText(
        "SPECTRA VISION",
        x + 8,
        Math.max(
            y + 14,
            y - 5
        )
    );

    ctx.font =
        "bold 9px monospace";

    ctx.fillText(
        "FACE // PIXEL",
        x + 8,
        y + height - 8
    );

    ctx.restore();
}

// ============================================================
// FACE FILTER
// ============================================================

function drawFaceFilter() {
    if (
        faceFilter === "OFF"
    ) {
        return;
    }

    if (
        !faceResults ||
        !faceResults.detections
    ) {
        return;
    }

    for (const detection of faceResults.detections) {
        const box =
            detection.boundingBox;

        if (!box) continue;

        const mirrored =
            mirrorBoundingBox(box);

        if (faceFilter === "CYBER") {
            drawCyberFace(
                mirrored
            );
        }

        if (faceFilter === "SCAN") {
            drawScanFace(
                mirrored
            );
        }

        if (faceFilter === "CROWN") {
            drawCrownFace(
                mirrored
            );
        }
    }
}

function drawCyberFace(box) {
    ctx.save();

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
        ) * 0.55;

    ctx.strokeStyle =
        "rgba(255,255,255,0.8)";

    ctx.lineWidth = 1.5;

    ctx.beginPath();

    ctx.arc(
        centerX,
        centerY,
        radius,
        0,
        Math.PI * 2
    );

    ctx.stroke();

    for (let i = 0; i < 3; i++) {
        ctx.beginPath();

        ctx.arc(
            centerX,
            centerY,
            radius -
                i * 10,
            0,
            Math.PI * 2
        );

        ctx.strokeStyle =
            `rgba(255,255,255,${0.35 - i * 0.07})`;

        ctx.stroke();
    }

    ctx.restore();
}

function drawScanFace(box) {
    ctx.save();

    ctx.beginPath();

    ctx.rect(
        box.originX,
        box.originY,
        box.width,
        box.height
    );

    ctx.clip();

    ctx.fillStyle =
        "rgba(255,255,255,0.05)";

    ctx.fillRect(
        box.originX,
        box.originY,
        box.width,
        box.height
    );

    ctx.strokeStyle =
        "rgba(255,255,255,0.28)";

    ctx.lineWidth = 1;

    const spacing = 10;

    for (
        let y = box.originY;
        y < box.originY + box.height;
        y += spacing
    ) {
        ctx.beginPath();

        ctx.moveTo(
            box.originX,
            y
        );

        ctx.lineTo(
            box.originX + box.width,
            y
        );

        ctx.stroke();
    }

    ctx.restore();
}

function drawCrownFace(box) {
    ctx.save();

    const x =
        box.originX;

    const y =
        box.originY;

    const w =
        box.width;

    const h =
        box.height;

    ctx.strokeStyle =
        "rgba(255,255,255,0.85)";

    ctx.lineWidth = 2;

    ctx.beginPath();

    ctx.moveTo(
        x + w * 0.15,
        y
    );

    ctx.lineTo(
        x + w * 0.25,
        y - h * 0.25
    );

    ctx.lineTo(
        x + w * 0.4,
        y
    );

    ctx.lineTo(
        x + w * 0.5,
        y - h * 0.3
    );

    ctx.lineTo(
        x + w * 0.6,
        y
    );

    ctx.lineTo(
        x + w * 0.75,
        y - h * 0.25
    );

    ctx.lineTo(
        x + w * 0.85,
        y
    );

    ctx.stroke();

    ctx.restore();
}

// ============================================================
// OBJECT DETECTION
// ============================================================

function drawObjects() {
    if (
        !objectDetection ||
        !objectResults ||
        !objectResults.detections
    ) {
        return;
    }

    for (const detection of objectResults.detections) {
        const box =
            detection.boundingBox;

        if (!box) continue;

        const mirrored =
            mirrorBoundingBox(box);

        ctx.save();

        ctx.strokeStyle =
            "#ffffff";

        ctx.lineWidth = 1.5;

        ctx.strokeRect(
            mirrored.originX,
            mirrored.originY,
            mirrored.width,
            mirrored.height
        );

        const category =
            detection.categories?.[0];

        const name =
            category?.categoryName ||
            "OBJECT";

        const score =
            category?.score != null
                ? Math.round(
                    category.score * 100
                )
                : 0;

        const label =
            `${name.toUpperCase()} ${score}%`;

        drawHudLabel(
            mirrored.originX,
            mirrored.originY - 12,
            label
        );

        ctx.restore();
    }
}

// ============================================================
// TARGET BOX
// ============================================================

function drawTargetBox() {
    if (!targetVisible) return;

    const size =
        Math.min(
            canvas.width,
            canvas.height
        ) * 0.2;

    const x =
        canvas.width / 2 -
        size / 2;

    const y =
        canvas.height / 2 -
        size / 2;

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.35)";

    ctx.lineWidth = 1;

    ctx.setLineDash([
        5,
        7
    ]);

    ctx.strokeRect(
        x,
        y,
        size,
        size
    );

    ctx.setLineDash([]);

    ctx.restore();
}

// ============================================================
// FPS
// ============================================================

function updateFPS() {
    frames++;

    const now =
        performance.now();

    if (
        now - fpsTimer >=
        1000
    ) {
        fps =
            Math.round(
                frames *
                1000 /
                (
                    now -
                    fpsTimer
                )
            );

        frames = 0;
        fpsTimer = now;

        updateFPSDisplay();
    }
}

function updateFPSDisplay() {
    if (!fpsBoostBtn) return;

    const boostText =
        fpsBoost
            ? "BOOST"
            : "NORMAL";

    // Only update the FPS portion if
    // the button isn't currently being used
    // as a normal toggle label.
    if (fpsBoost) {
        fpsBoostBtn.textContent =
            `FPS ${fps} // BOOST`;
    } else {
        fpsBoostBtn.textContent =
            `FPS ${fps} // BOOST OFF`;
    }
}

// ============================================================
// HELPERS
// ============================================================

function point(landmark) {
    return {
        x:
            landmark.x *
            canvas.width,
        y:
            landmark.y *
            canvas.height
    };
}

function mirrorPoint(landmark) {
    return {
        x:
            canvas.width -
            landmark.x *
            canvas.width,
        y:
            landmark.y *
            canvas.height
    };
}

function mirrorBoundingBox(box) {
    return {
        originX:
            canvas.width -
            box.originX -
            box.width,

        originY:
            box.originY,

        width:
            box.width,

        height:
            box.height
    };
}

function getPalmCenter(hand) {
    const ids = [
        0,
        5,
        9,
        13,
        17
    ];

    let x = 0;
    let y = 0;

    for (const id of ids) {
        x +=
            hand[id].x *
            canvas.width;

        y +=
            hand[id].y *
            canvas.height;
    }

    return {
        x:
            x / ids.length,
        y:
            y / ids.length
    };
}

function distance(a, b) {
    return Math.hypot(
        a.x - b.x,
        a.y - b.y
    );
}

function lerpPoint(
    a,
    b,
    t
) {
    return {
        x:
            a.x +
            (
                b.x -
                a.x
            ) * t,

        y:
            a.y +
            (
                b.y -
                a.y
            ) * t
    };
}

function scaleAround(
    pointValue,
    center,
    scale
) {
    return {
        x:
            center.x +
            (
                pointValue.x -
                center.x
            ) * scale,

        y:
            center.y +
            (
                pointValue.y -
                center.y
            ) * scale
    };
}

function averagePoint(...points) {
    if (!points.length) {
        return {
            x: 0,
            y: 0
        };
    }

    let x = 0;
    let y = 0;

    for (const p of points) {
        x += p.x;
        y += p.y;
    }

    return {
        x:
            x / points.length,

        y:
            y / points.length
    };
}

function getPolygonCenter(points) {
    return averagePoint(
        ...points
    );
}

function orderAroundCenter(points) {
    const center =
        getPolygonCenter(points);

    return points
        .slice()
        .sort(
            (a, b) =>
                Math.atan2(
                    a.y - center.y,
                    a.x - center.x
                ) -
                Math.atan2(
                    b.y - center.y,
                    b.x - center.x
                )
        );
}

function getBounds(points) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const p of points) {
        minX =
            Math.min(
                minX,
                p.x
            );

        minY =
            Math.min(
                minY,
                p.y
            );

        maxX =
            Math.max(
                maxX,
                p.x
            );

        maxY =
            Math.max(
                maxY,
                p.y
            );
    }

    return {
        minX,
        minY,
        maxX,
        maxY,
        width:
            maxX - minX,
        height:
            maxY - minY
    };
}

function clonePoint(p) {
    return {
        x: p.x,
        y: p.y
    };
}

function clamp(
    value,
    min,
    max
) {
    return Math.max(
        min,
        Math.min(
            max,
            value
        )
    );
}

// ============================================================
// STARTUP
// ============================================================

setButtonText(
    startBtn,
    "START CAMERA"
);

setButtonText(
    trackingBtn,
    "HAND TRACKING ON"
);

setButtonText(
    targetBtn,
    "TARGET BOX ON"
);

setButtonText(
    objectBtn,
    "OBJECT DETECTION OFF"
);

setButtonText(
    privacyBtn,
    "FACE PRIVACY OFF"
);

setButtonText(
    signBtn,
    "SIGN LANGUAGE OFF"
);

setButtonText(
    filterBtn,
    "FACE FILTER OFF"
);

shapeDisplay.textContent =
    `GEOMETRY // ${GEOMETRY_MODES[geometryMode]} // ${GEOMETRY_FILTERS[geometryFilter]}`;

console.log(
    "SPECTRA initialized successfully."
);
