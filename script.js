import {
    FilesetResolver,
    HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


/* =========================================================
   SPECTRA v0.3
   HAND TRACKING SYSTEM
========================================================= */


/* =========================
   DOM
========================= */

const video =
    document.getElementById("camera");

const canvas =
    document.getElementById("overlay");

const ctx =
    canvas.getContext("2d");

const startButton =
    document.getElementById("startButton");

const trackingButton =
    document.getElementById("trackingButton");

const statusText =
    document.getElementById("status");

const cameraMessage =
    document.getElementById("cameraMessage");

const systemStatus =
    document.getElementById("systemStatus");

const statusDot =
    document.getElementById("statusDot");

const handState =
    document.getElementById("handState");

const targetNumber =
    document.getElementById("targetNumber");

const xValue =
    document.getElementById("xValue");

const yValue =
    document.getElementById("yValue");

const fpsValue =
    document.getElementById("fpsValue");

const trackingState =
    document.getElementById("trackingState");

const gestureText =
    document.getElementById("gestureText");

const handsValue =
    document.getElementById("handsValue");

const indexValue =
    document.getElementById("indexValue");

const pinchValue =
    document.getElementById("pinchValue");

const gestureValue =
    document.getElementById("gestureValue");


/* =========================
   STATE
========================= */

let handLandmarker = null;

let cameraStream = null;

let cameraRunning = false;

let trackingEnabled = true;

let lastVideoTime = -1;

let lastFrameTime = performance.now();

let frameCounter = 0;

let displayedFPS = 0;

let previousIndex = null;


/* =========================
   MEDIAPIPE
========================= */

const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";


async function createHandLandmarker() {

    statusText.textContent =
        "Loading hand tracking model...";

    try {

        const vision =
            await FilesetResolver.forVisionTasks(
                WASM_URL
            );


        handLandmarker =
            await HandLandmarker.createFromOptions(
                vision,
                {
                    baseOptions: {
                        modelAssetPath: MODEL_URL,

                        delegate: "GPU"
                    },

                    runningMode: "VIDEO",

                    numHands: 2,

                    minHandDetectionConfidence: 0.55,

                    minHandPresenceConfidence: 0.55,

                    minTrackingConfidence: 0.55
                }
            );


        statusText.textContent =
            "Hand tracking system ready.";

        trackingButton.disabled = false;

    } catch (error) {

        console.error(error);

        statusText.textContent =
            "Hand tracking failed to load.";

        handState.textContent =
            "HAND TRACKER // ERROR";
    }
}


/* =========================
   CAMERA
========================= */

startButton.addEventListener(
    "click",
    startCamera
);


async function startCamera() {

    if (cameraRunning) {

        stopCamera();

        return;
    }


    try {

        statusText.textContent =
            "Requesting camera access...";


        cameraStream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    facingMode: {
                        ideal: "environment"
                    },

                    width: {
                        ideal: 1280
                    },

                    height: {
                        ideal: 720
                    }
                },

                audio: false
            });


        video.srcObject =
            cameraStream;


        await video.play();


        cameraRunning = true;


        cameraMessage.style.display =
            "none";


        systemStatus.textContent =
            "ONLINE";

        statusDot.classList.add(
            "active"
        );


        startButton.textContent =
            "STOP CAMERA";


        statusText.textContent =
            "Camera online.";


        setupCanvas();


        if (!handLandmarker) {

            await createHandLandmarker();
        }


        requestAnimationFrame(
            detectionLoop
        );


    } catch (error) {

        console.error(error);

        statusText.textContent =
            "Camera access failed.";

        cameraMessage.textContent =
            "CAMERA ERROR";
    }
}


function stopCamera() {

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );
    }


    cameraStream = null;

    video.srcObject = null;

    cameraRunning = false;


    systemStatus.textContent =
        "OFFLINE";

    statusDot.classList.remove(
        "active"
    );


    startButton.textContent =
        "START CAMERA";


    cameraMessage.style.display =
        "flex";

    cameraMessage.textContent =
        "CAMERA OFF";


    clearOverlay();


    resetInterface();


    statusText.textContent =
        "Camera stopped.";
}


/* =========================
   CANVAS
========================= */

function setupCanvas() {

    if (
        video.videoWidth === 0 ||
        video.videoHeight === 0
    ) {
        return;
    }


    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;
}


function clearOverlay() {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );
}


/* =========================
   TRACKING BUTTON
========================= */

trackingButton.addEventListener(
    "click",
    () => {

        trackingEnabled =
            !trackingEnabled;


        trackingButton.classList.toggle(
            "active",
            trackingEnabled
        );


        trackingButton.textContent =
            trackingEnabled
                ? "HAND TRACKING ON"
                : "HAND TRACKING OFF";


        if (!trackingEnabled) {

            clearOverlay();

            handState.textContent =
                "HAND TRACKING // OFF";
        }
    }
);


/* =========================
   DETECTION LOOP
========================= */

function detectionLoop() {

    if (!cameraRunning) {
        return;
    }


    if (
        video.readyState >=
        HTMLMediaElement.HAVE_CURRENT_DATA
    ) {

        if (
            video.currentTime !==
            lastVideoTime
        ) {

            lastVideoTime =
                video.currentTime;


            if (
                trackingEnabled &&
                handLandmarker
            ) {

                const now =
                    performance.now();


                try {

                    const results =
                        handLandmarker.detectForVideo(
                            video,
                            now
                        );


                    processResults(
                        results
                    );


                } catch (error) {

                    console.error(
                        "Detection error:",
                        error
                    );
                }
            }
        }
    }


    updateFPS();

    requestAnimationFrame(
        detectionLoop
    );
}


/* =========================
   RESULTS
========================= */

function processResults(results) {

    clearOverlay();


    if (
        !results ||
        !results.landmarks ||
        results.landmarks.length === 0
    ) {

        showNoHand();

        return;
    }


    const handCount =
        results.landmarks.length;


    handsValue.textContent =
        handCount;


    handState.textContent =
        handCount === 1
            ? "HAND // LOCKED"
            : "HANDS // " + handCount;


    trackingState.textContent =
        "ACTIVE";


    targetNumber.textContent =
        "01";


    /*
     * Draw every detected hand.
     */

    results.landmarks.forEach(
        (landmarks, index) => {

            drawHand(
                landmarks,
                index
            );
        }
    );


    /*
     * Use the first hand as the
     * primary interaction target.
     */

    const primaryHand =
        results.landmarks[0];


    const indexFinger =
        primaryHand[8];

    const thumb =
        primaryHand[4];


    const screenX =
        indexFinger.x *
        canvas.width;


    const screenY =
        indexFinger.y *
        canvas.height;


    /*
     * Smooth the fingertip.
     */

    const smoothed =
        smoothPoint(
            previousIndex,
            {
                x: screenX,
                y: screenY
            },
            0.35
        );


    previousIndex =
        smoothed;


    /*
     * Update HUD.
     */

    xValue.textContent =
        Math.round(smoothed.x);


    yValue.textContent =
        Math.round(smoothed.y);


    indexValue.textContent =
        `${Math.round(smoothed.x)}, ${Math.round(smoothed.y)}`;


    /*
     * Pinch detection.
     */

    const pinchDistance =
        distance(
            indexFinger,
            thumb
        );


    const pinch =
        pinchDistance < 0.075;


    pinchValue.textContent =
        pinch
            ? "YES"
            : "NO";


    /*
     * Gesture estimation.
     */

    const gesture =
        detectGesture(
            primaryHand,
            pinch
        );


    gestureText.textContent =
        "GESTURE // " +
        gesture;


    gestureValue.textContent =
        gesture;


    /*
     * Draw virtual cursor.
     */

    drawCursor(
        smoothed.x,
        smoothed.y,
        pinch
    );


    /*
     * Draw target box around hand.
     */

    drawTargetBox(
        primaryHand
    );
}


/* =========================
   NO HAND
========================= */

function showNoHand() {

    handsValue.textContent =
        "0";


    handState.textContent =
        "HAND // NOT DETECTED";


    trackingState.textContent =
        "SEARCH";


    targetNumber.textContent =
        "00";


    xValue.textContent =
        "---";


    yValue.textContent =
        "---";


    indexValue.textContent =
        "---";


    pinchValue.textContent =
        "---";


    gestureValue.textContent =
        "NONE";


    gestureText.textContent =
        "GESTURE // NONE";


    previousIndex = null;
}


/* =========================
   HAND DRAWING
========================= */

const CONNECTIONS = [

    /* Thumb */

    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],

    /* Index */

    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],

    /* Middle */

    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],

    /* Ring */

    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],

    /* Pinky */

    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],

    /* Palm */

    [5, 9],
    [9, 13],
    [13, 17]
];


function drawHand(
    landmarks,
    handIndex
) {

    const points =
        landmarks.map(
            point => ({
                x: point.x *
                    canvas.width,

                y: point.y *
                    canvas.height
            })
        );


    /*
     * Connections
     */

    ctx.save();

    ctx.lineWidth = 1.5;

    ctx.strokeStyle =
        "rgba(255,255,255,0.75)";

    ctx.setLineDash([]);

    CONNECTIONS.forEach(
        connection => {

            const start =
                points[
                    connection[0]
                ];

            const end =
                points[
                    connection[1]
                ];


            ctx.beginPath();

            ctx.moveTo(
                start.x,
                start.y
            );

            ctx.lineTo(
                end.x,
                end.y
            );

            ctx.stroke();
        }
    );


    /*
     * Landmark points
     */

    points.forEach(
        (point, index) => {

            ctx.beginPath();

            ctx.arc(
                point.x,
                point.y,
                index === 8
                    ? 5
                    : 3,
                0,
                Math.PI * 2
            );


            ctx.fillStyle =
                index === 8
                    ? "#ffffff"
                    : "rgba(255,255,255,0.75)";


            ctx.fill();
        }
    );


    ctx.restore();
}


/* =========================
   VIRTUAL CURSOR
========================= */

function drawCursor(
    x,
    y,
    isPinching
) {

    const size =
        isPinching
            ? 24
            : 34;


    ctx.save();


    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";

    ctx.lineWidth = 1.5;

    ctx.setLineDash([]);


    /*
     * Horizontal line
     */

    ctx.beginPath();

    ctx.moveTo(
        x - size,
        y
    );

    ctx.lineTo(
        x + size,
        y
    );

    ctx.stroke();


    /*
     * Vertical line
     */

    ctx.beginPath();

    ctx.moveTo(
        x,
        y - size
    );

    ctx.lineTo(
        x,
        y + size
    );

    ctx.stroke();


    /*
     * Center
     */

    ctx.beginPath();

    ctx.arc(
        x,
        y,
        isPinching ? 7 : 4,
        0,
        Math.PI * 2
    );

    ctx.stroke();


    ctx.restore();
}


/* =========================
   TARGET BOX
========================= */

function drawTargetBox(
    landmarks
) {

    const xs =
        landmarks.map(
            point =>
                point.x *
                canvas.width
        );


    const ys =
        landmarks.map(
            point =>
                point.y *
                canvas.height
        );


    const minX =
        Math.min(...xs);

    const maxX =
        Math.max(...xs);

    const minY =
        Math.min(...ys);

    const maxY =
        Math.max(...ys);


    const padding = 22;


    const x =
        minX - padding;

    const y =
        minY - padding;

    const width =
        maxX -
        minX +
        padding * 2;

    const height =
        maxY -
        minY +
        padding * 2;


    ctx.save();


    /*
     * Target rectangle
     */

    ctx.strokeStyle =
        "rgba(255,255,255,0.75)";

    ctx.lineWidth = 1;

    ctx.setLineDash([]);


    ctx.strokeRect(
        x,
        y,
        width,
        height
    );


    /*
     * Dashed center lines
     */

    const centerX =
        x + width / 2;

    const centerY =
        y + height / 2;


    ctx.strokeStyle =
        "rgba(255,255,255,0.25)";

    ctx.setLineDash(
        [8, 8]
    );


    ctx.beginPath();

    ctx.moveTo(
        centerX,
        0
    );

    ctx.lineTo(
        centerX,
        canvas.height
    );

    ctx.stroke();


    ctx.beginPath();

    ctx.moveTo(
        0,
        centerY
    );

    ctx.lineTo(
        canvas.width,
        centerY
    );

    ctx.stroke();


    ctx.restore();


    /*
     * Target label
     */

    ctx.save();

    ctx.fillStyle =
        "rgba(0,0,0,0.65)";

    ctx.fillRect(
        x,
        y - 22,
        115,
        18
    );


    ctx.fillStyle =
        "#fff";

    ctx.font =
        "11px Courier New";


    ctx.fillText(
        "TARGET // 01",
        x + 6,
        y - 9
    );


    ctx.restore();
}


/* =========================
   GESTURE DETECTION
========================= */

function detectGesture(
    landmarks,
    pinch
) {

    if (pinch) {

        return "PINCH";
    }


    const indexExtended =
        isFingerExtended(
            landmarks,
            8,
            6
        );


    const middleExtended =
        isFingerExtended(
            landmarks,
            12,
            10
        );


    const ringExtended =
        isFingerExtended(
            landmarks,
            16,
            14
        );


    const pinkyExtended =
        isFingerExtended(
            landmarks,
            20,
            18
        );


    if (
        indexExtended &&
        !middleExtended &&
        !ringExtended &&
        !pinkyExtended
    ) {

        return "POINT";
    }


    if (
        indexExtended &&
        middleExtended &&
        !ringExtended &&
        !pinkyExtended
    ) {

        return "TWO FINGER";
    }


    if (
        indexExtended &&
        middleExtended &&
        ringExtended &&
        pinkyExtended
    ) {

        return "OPEN PALM";
    }


    if (
        !indexExtended &&
        !middleExtended &&
        !ringExtended &&
        !pinkyExtended
    ) {

        return "FIST";
    }


    return "TRACKING";
}


/* =========================
   FINGER EXTENSION
========================= */

function isFingerExtended(
    landmarks,
    tipIndex,
    pipIndex
) {

    const wrist =
        landmarks[0];

    const tip =
        landmarks[tipIndex];

    const pip =
        landmarks[pipIndex];


    const tipDistance =
        distance3D(
            tip,
            wrist
        );


    const pipDistance =
        distance3D(
            pip,
            wrist
        );


    return tipDistance >
        pipDistance * 1.12;
}


/* =========================
   DISTANCE
========================= */

function distance(
    a,
    b
) {

    const dx =
        a.x - b.x;

    const dy =
        a.y - b.y;

    return Math.sqrt(
        dx * dx +
        dy * dy
    );
}


function distance3D(
    a,
    b
) {

    const dx =
        a.x - b.x;

    const dy =
        a.y - b.y;

    const dz =
        (a.z || 0) -
        (b.z || 0);


    return Math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    );
}


/* =========================
   SMOOTHING
========================= */

function smoothPoint(
    previous,
    current,
    amount
) {

    if (!previous) {

        return current;
    }


    return {

        x:
            previous.x +
            (current.x -
                previous.x) *
            amount,

        y:
            previous.y +
            (current.y -
                previous.y) *
            amount
    };
}


/* =========================
   FPS
========================= */

function updateFPS() {

    frameCounter++;


    const now =
        performance.now();


    const elapsed =
        now -
        lastFrameTime;


    if (elapsed >= 1000) {

        displayedFPS =
            Math.round(
                frameCounter *
                1000 /
                elapsed
            );


        fpsValue.textContent =
            displayedFPS;


        frameCounter = 0;

        lastFrameTime =
            now;
    }
}


/* =========================
   RESET
========================= */

function resetInterface() {

    handsValue.textContent =
        "0";

    indexValue.textContent =
        "---";

    pinchValue.textContent =
        "---";

    gestureValue.textContent =
        "NONE";

    gestureText.textContent =
        "GESTURE // NONE";

    handState.textContent =
        "HAND // NOT DETECTED";

    trackingState.textContent =
        "IDLE";

    targetNumber.textContent =
        "00";

    xValue.textContent =
        "---";

    yValue.textContent =
        "---";

    previousIndex = null;
}


/* =========================
   INITIALIZATION
========================= */

trackingButton.textContent =
    "HAND TRACKING ON";

trackingButton.classList.add(
    "active"
);


createHandLandmarker();
