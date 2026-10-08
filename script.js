import {
    FilesetResolver,
    GestureRecognizer,
    FaceLandmarker,
    ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


/* =========================================================
   SPECTRA V0.7
   PERFORMANCE + FACE PRIVACY UPDATE
========================================================= */

const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");

const ctx = canvas.getContext("2d", {
    alpha: false
});


/* =========================================================
   UI
========================================================= */

const startBtn = document.getElementById("startBtn");
const trackingBtn = document.getElementById("trackingBtn");
const targetBtn = document.getElementById("targetBtn");
const objectBtn = document.getElementById("objectBtn");
const privacyBtn = document.getElementById("privacyBtn");
const signBtn = document.getElementById("signBtn");
const filterBtn = document.getElementById("filterBtn");
const shapeBtn = document.getElementById("shapeBtn");

const handStatus = document.getElementById("handStatus");
const faceStatus = document.getElementById("faceStatus");
const fpsText = document.getElementById("fps");

const gestureText = document.getElementById("gesture");
const signText = document.getElementById("sign");
const shapeStatus = document.getElementById("shapeStatus");

const objectStatus = document.getElementById("objectStatus");
const filterStatus = document.getElementById("filterStatus");

const handsInfo = document.getElementById("handsInfo");
const facesInfo = document.getElementById("facesInfo");
const gestureInfo = document.getElementById("gestureInfo");
const signInfo = document.getElementById("signInfo");
const shapesInfo = document.getElementById("shapesInfo");


/* =========================================================
   MODELS
========================================================= */

const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";

const GESTURE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

const FACE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const OBJECT_MODEL =
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite";


/* =========================================================
   DEVICE PERFORMANCE
========================================================= */

const isMobile =
    /Android|iPhone|iPad|iPod/i.test(
        navigator.userAgent
    ) ||
    window.innerWidth < 800;


/*
    Detection does NOT need to run at 60 FPS.

    The canvas still renders at the browser's
    normal refresh rate.

    AI inference happens less often.
*/

const PERFORMANCE = isMobile
    ? {
        hand: 55,       // ~18 FPS
        face: 140,      // ~7 FPS
        object: 300     // ~3 FPS
    }
    : {
        hand: 40,       // ~25 FPS
        face: 100,      // ~10 FPS
        object: 220     // ~4.5 FPS
    };


/* =========================================================
   STATE
========================================================= */

let vision = null;

let gestureRecognizer = null;
let faceLandmarker = null;
let objectDetector = null;

let cameraRunning = false;
let handTracking = true;

let targetVisible = true;
let objectDetection = false;

let signLanguage = false;
let shapesEnabled = true;


/* =========================================================
   FACE PRIVACY
========================================================= */

/*
    0 = OFF
    1 = BLUR
    2 = PIXEL
    3 = REDACT
    4 = SCAN
    5 = BOX
*/

const privacyModes = [
    "OFF",
    "BLUR",
    "PIXEL",
    "REDACT",
    "SCAN",
    "BOX"
];

let privacyIndex = 0;

let facePrivacy = false;


/* =========================================================
   FACE FILTERS
========================================================= */

const filters = [
    "OFF",
    "CYBER",
    "DOG",
    "CROWN",
    "NEON"
];

let filterIndex = 0;


/* =========================================================
   DETECTION RESULTS
========================================================= */

let handResults = null;
let faceResults = null;
let objectResults = null;


/* =========================================================
   TIMERS
========================================================= */

let lastHandTime = 0;
let lastFaceTime = 0;
let lastObjectTime = 0;


/* =========================================================
   FPS
========================================================= */

let fpsFrames = 0;
let fpsLastTime = performance.now();


/* =========================================================
   SHAPES
========================================================= */

const shapes = [];

let drawing = false;
let currentPath = [];

let grabbedShape = null;

let pinchWasActive = false;

let shapeId = 0;

let lastPoint = null;


/* =========================================================
   OFFSCREEN CANVAS
========================================================= */

/*
    Used for face privacy.

    This avoids repeatedly modifying the main
    camera canvas in a destructive way.
*/

const privacyCanvas =
    document.createElement("canvas");

const privacyCtx =
    privacyCanvas.getContext("2d");


/* =========================================================
   CAMERA
========================================================= */

async function startCamera() {

    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({

                video: {

                    facingMode: {
                        ideal: "user"
                    },

                    width: {
                        ideal: 640
                    },

                    height: {
                        ideal: 360
                    },

                    frameRate: {
                        ideal: 30,
                        max: 30
                    }
                },

                audio: false
            });


        video.srcObject = stream;

        await video.play();


        canvas.width =
            video.videoWidth || 640;

        canvas.height =
            video.videoHeight || 360;


        privacyCanvas.width =
            canvas.width;

        privacyCanvas.height =
            canvas.height;


        cameraRunning = true;


        startBtn.textContent =
            "CAMERA RUNNING";


        /*
            IMPORTANT:

            Start rendering FIRST.

            We do NOT wait for MediaPipe.
        */

        requestAnimationFrame(renderLoop);


        /*
            Load hand AI in the background.
        */

        loadVision();


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );

        startBtn.textContent =
            "CAMERA ERROR";

        alert(
            "SPECTRA could not access the camera.\n\n" +
            "Allow camera permission and make sure you are using HTTPS."
        );
    }
}


/* =========================================================
   LOAD CORE VISION
========================================================= */

async function loadVision() {

    if (vision) {
        return;
    }


    handStatus.textContent =
        "LOADING";


    try {

        vision =
            await FilesetResolver.forVisionTasks(
                WASM_URL
            );


        gestureRecognizer =
            await GestureRecognizer.createFromOptions(
                vision,
                {

                    baseOptions: {

                        modelAssetPath:
                            GESTURE_MODEL
                    },

                    runningMode:
                        "VIDEO",

                    numHands:
                        2,

                    minHandDetectionConfidence:
                        0.45,

                    minHandPresenceConfidence:
                        0.45,

                    minTrackingConfidence:
                        0.45
                }
            );


        handStatus.textContent =
            "READY";

        console.log(
            "SPECTRA hand AI ready"
        );


    } catch (error) {

        console.error(
            "Hand AI failed:",
            error
        );

        handStatus.textContent =
            "ERROR";
    }
}


/* =========================================================
   FACE MODEL
========================================================= */

async function loadFaceModel() {

    if (faceLandmarker) {
        return true;
    }


    if (!vision) {

        await loadVision();
    }


    faceStatus.textContent =
        "LOADING";


    try {

        faceLandmarker =
            await FaceLandmarker.createFromOptions(
                vision,
                {

                    baseOptions: {

                        modelAssetPath:
                            FACE_MODEL
                    },

                    runningMode:
                        "VIDEO",

                    numFaces:
                        2,

                    minFaceDetectionConfidence:
                        0.45,

                    minFacePresenceConfidence:
                        0.45,

                    minTrackingConfidence:
                        0.45,

                    outputFaceBlendshapes:
                        true
                }
            );


        faceStatus.textContent =
            "READY";

        console.log(
            "SPECTRA face AI ready"
        );

        return true;


    } catch (error) {

        console.error(
            "Face AI failed:",
            error
        );

        faceStatus.textContent =
            "ERROR";

        return false;
    }
}


/* =========================================================
   OBJECT MODEL
========================================================= */

async function loadObjectModel() {

    if (objectDetector) {
        return true;
    }


    if (!vision) {
        await loadVision();
    }


    objectStatus.textContent =
        "LOADING";


    try {

        objectDetector =
            await ObjectDetector.createFromOptions(
                vision,
                {

                    baseOptions: {

                        modelAssetPath:
                            OBJECT_MODEL
                    },

                    runningMode:
                        "VIDEO",

                    maxResults:
                        isMobile ? 3 : 5,

                    scoreThreshold:
                        0.50
                }
            );


        objectStatus.textContent =
            "READY";

        return true;


    } catch (error) {

        console.error(
            "Object detector failed:",
            error
        );

        objectStatus.textContent =
            "ERROR";

        return false;
    }
}


/* =========================================================
   BUTTONS
========================================================= */

startBtn.onclick = async () => {

    if (!cameraRunning) {

        await startCamera();
    }
};


trackingBtn.onclick = () => {

    handTracking =
        !handTracking;

    trackingBtn.textContent =
        handTracking
            ? "HAND TRACKING ON"
            : "HAND TRACKING OFF";
};


targetBtn.onclick = () => {

    targetVisible =
        !targetVisible;

    targetBtn.textContent =
        targetVisible
            ? "TARGET BOX ON"
            : "TARGET BOX OFF";
};


objectBtn.onclick = async () => {

    if (!objectDetection) {

        objectDetection =
            true;

        objectBtn.textContent =
            "LOADING OBJECT AI...";

        const loaded =
            await loadObjectModel();

        if (!loaded) {

            objectDetection =
                false;

            objectBtn.textContent =
                "OBJECT DETECTION OFF";

            return;
        }

    } else {

        objectDetection =
            false;

        objectBtn.textContent =
            "OBJECT DETECTION OFF";
    }
};


/* =========================================================
   PRIVACY BUTTON
========================================================= */

privacyBtn.onclick = async () => {

    privacyIndex++;

    if (
        privacyIndex >=
        privacyModes.length
    ) {

        privacyIndex = 0;
    }


    const mode =
        privacyModes[privacyIndex];


    facePrivacy =
        mode !== "OFF";


    privacyBtn.textContent =
        `PRIVACY ${mode}`;


    if (facePrivacy) {

        await loadFaceModel();
    }
};


/* =========================================================
   SIGN LANGUAGE
========================================================= */

signBtn.onclick = () => {

    signLanguage =
        !signLanguage;

    signBtn.textContent =
        signLanguage
            ? "SIGN LANGUAGE ON"
            : "SIGN LANGUAGE OFF";
};


/* =========================================================
   FILTER BUTTON
========================================================= */

filterBtn.onclick = async () => {

    filterIndex++;

    if (
        filterIndex >=
        filters.length
    ) {

        filterIndex = 0;
    }


    const filter =
        filters[filterIndex];


    filterBtn.textContent =
        `FILTER ${filter}`;


    filterStatus.textContent =
        filter;


    if (
        filter !== "OFF"
    ) {

        await loadFaceModel();
    }
};


/* =========================================================
   SHAPE BUTTON
========================================================= */

shapeBtn.onclick = () => {

    shapesEnabled =
        !shapesEnabled;

    shapeBtn.textContent =
        shapesEnabled
            ? "SHAPES ON"
            : "SHAPES OFF";


    if (!shapesEnabled) {

        drawing = false;

        currentPath = [];

        grabbedShape = null;
    }
};


/* =========================================================
   RENDER LOOP
========================================================= */

function renderLoop(timestamp) {

    if (!cameraRunning) {
        return;
    }


    /*
        CAMERA IS ALWAYS DRAWN.

        AI loading cannot hide the camera anymore.
    */

    drawCamera();


    /*
        AI systems run independently.
    */

    runHandTracking(timestamp);

    runFaceTracking(timestamp);

    runObjectDetection(timestamp);


    /*
        Visual effects.
    */

    drawFacePrivacy();

    drawFaceTracker();

    drawFaceFilter();

    drawObjects();

    drawShapes();

    drawHands();


    updateFPS();


    requestAnimationFrame(
        renderLoop
    );
}


/* =========================================================
   CAMERA DRAW
========================================================= */

function drawCamera() {

    const width =
        canvas.width;

    const height =
        canvas.height;


    ctx.clearRect(
        0,
        0,
        width,
        height
    );


    /*
        Mirror selfie camera.
    */

    ctx.save();

    ctx.translate(
        width,
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
        width,
        height
    );


    ctx.restore();
}


/* =========================================================
   HAND DETECTION
========================================================= */

function runHandTracking(timestamp) {

    if (
        !handTracking ||
        !gestureRecognizer
    ) {

        return;
    }


    if (
        timestamp -
        lastHandTime <
        PERFORMANCE.hand
    ) {

        return;
    }


    lastHandTime =
        timestamp;


    try {

        handResults =
            gestureRecognizer.recognizeForVideo(
                video,
                timestamp
            );


        processHands();


    } catch (error) {

        console.error(
            "Hand detection error:",
            error
        );
    }
}


/* =========================================================
   FACE DETECTION
========================================================= */

function runFaceTracking(timestamp) {

    /*
        Don't run face AI unless something
        actually needs it.
    */

    if (
        !faceLandmarker
    ) {

        return;
    }


    if (
        timestamp -
        lastFaceTime <
        PERFORMANCE.face
    ) {

        return;
    }


    lastFaceTime =
        timestamp;


    try {

        faceResults =
            faceLandmarker.detectForVideo(
                video,
                timestamp
            );


        const faces =
            faceResults?.faceLandmarks ||
            [];


        facesInfo.textContent =
            faces.length;


        faceStatus.textContent =
            faces.length
                ? `${faces.length} DETECTED`
                : "SEARCHING";


    } catch (error) {

        console.error(
            "Face detection error:",
            error
        );
    }
}


/* =========================================================
   OBJECT DETECTION
========================================================= */

function runObjectDetection(timestamp) {

    if (
        !objectDetection ||
        !objectDetector
    ) {

        return;
    }


    if (
        timestamp -
        lastObjectTime <
        PERFORMANCE.object
    ) {

        return;
    }


    lastObjectTime =
        timestamp;


    try {

        objectResults =
            objectDetector.detectForVideo(
                video,
                timestamp
            );

    } catch (error) {

        console.error(
            "Object detection error:",
            error
        );
    }
}


/* =========================================================
   FACE PRIVACY
========================================================= */

function drawFacePrivacy() {

    const mode =
        privacyModes[privacyIndex];


    if (
        mode === "OFF"
    ) {

        return;
    }


    const faces =
        faceResults?.faceLandmarks ||
        [];


    if (!faces.length) {
        return;
    }


    for (
        const face
        of faces
    ) {

        const points =
            face.map(
                p => ({
                    x:
                        (1 - p.x) *
                        canvas.width,

                    y:
                        p.y *
                        canvas.height
                })
            );


        const bounds =
            getBounds(points);


        /*
            Give the privacy area some
            breathing room around the face.
        */

        const paddingX =
            Math.max(
                25,
                bounds.width * 0.12
            );

        const paddingY =
            Math.max(
                30,
                bounds.height * 0.18
            );


        const x =
            Math.max(
                0,
                bounds.minX -
                paddingX
            );

        const y =
            Math.max(
                0,
                bounds.minY -
                paddingY
            );

        const width =
            Math.min(
                canvas.width - x,
                bounds.width +
                paddingX * 2
            );

        const height =
            Math.min(
                canvas.height - y,
                bounds.height +
                paddingY * 2
            );


        if (
            mode === "BLUR"
        ) {

            drawBlurPrivacy(
                x,
                y,
                width,
                height
            );
        }


        if (
            mode === "PIXEL"
        ) {

            drawPixelPrivacy(
                x,
                y,
                width,
                height
            );
        }


        if (
            mode === "REDACT"
        ) {

            drawRedactPrivacy(
                x,
                y,
                width,
                height
            );
        }


        if (
            mode === "SCAN"
        ) {

            drawScanPrivacy(
                x,
                y,
                width,
                height
            );
        }


        if (
            mode === "BOX"
        ) {

            drawBoxPrivacy(
                x,
                y,
                width,
                height
            );
        }
    }
}


/* =========================================================
   BLUR PRIVACY
========================================================= */

function drawBlurPrivacy(
    x,
    y,
    width,
    height
) {

    ctx.save();

    ctx.beginPath();

    ctx.roundRect(
        x,
        y,
        width,
        height,
        20
    );

    ctx.clip();

    ctx.filter =
        "blur(22px)";

    /*
        Draw a slightly expanded copy
        of the already mirrored camera.
    */

    ctx.drawImage(
        canvas,
        x - 20,
        y - 20,
        width + 40,
        height + 40
    );

    ctx.restore();


    drawPrivacyFrame(
        x,
        y,
        width,
        height,
        "FACE // BLURRED"
    );
}


/* =========================================================
   PIXEL PRIVACY
========================================================= */

function drawPixelPrivacy(
    x,
    y,
    width,
    height
) {

    const smallWidth =
        Math.max(
            8,
            Math.floor(width / 18)
        );

    const smallHeight =
        Math.max(
            8,
            Math.floor(height / 18)
        );


    privacyCanvas.width =
        smallWidth;

    privacyCanvas.height =
        smallHeight;


    privacyCtx.imageSmoothingEnabled =
        false;


    privacyCtx.drawImage(
        canvas,
        x,
        y,
        width,
        height,
        0,
        0,
        smallWidth,
        smallHeight
    );


    ctx.save();

    ctx.imageSmoothingEnabled =
        false;


    ctx.drawImage(
        privacyCanvas,
        0,
        0,
        smallWidth,
        smallHeight,
        x,
        y,
        width,
        height
    );


    ctx.restore();


    drawPrivacyFrame(
        x,
        y,
        width,
        height,
        "FACE // PIXEL"
    );
}


/* =========================================================
   REDACT PRIVACY
========================================================= */

function drawRedactPrivacy(
    x,
    y,
    width,
    height
) {

    ctx.save();

    ctx.fillStyle =
        "rgba(0,0,0,0.88)";

    ctx.fillRect(
        x,
        y,
        width,
        height
    );


    ctx.strokeStyle =
        "rgba(255,255,255,0.8)";

    ctx.lineWidth =
        2;

    ctx.strokeRect(
        x,
        y,
        width,
        height
    );


    ctx.font =
        "bold 12px monospace";

    ctx.fillStyle =
        "#ffffff";

    ctx.textAlign =
        "center";

    ctx.fillText(
        "FACE HIDDEN",
        x + width / 2,
        y + height / 2 + 4
    );


    ctx.restore();
}


/* =========================================================
   SCAN PRIVACY
========================================================= */

function drawScanPrivacy(
    x,
    y,
    width,
    height
) {

    ctx.save();

    ctx.fillStyle =
        "rgba(0,0,0,0.60)";

    ctx.fillRect(
        x,
        y,
        width,
        height
    );


    ctx.beginPath();

    ctx.rect(
        x,
        y,
        width,
        height
    );

    ctx.clip();


    /*
        Horizontal scan lines.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.35)";

    ctx.lineWidth = 1;


    for (
        let yy = y;
        yy < y + height;
        yy += 8
    ) {

        ctx.beginPath();

        ctx.moveTo(
            x,
            yy
        );

        ctx.lineTo(
            x + width,
            yy
        );

        ctx.stroke();
    }


    /*
        Face tracking corners.
    */

    drawCornerBrackets(
        x,
        y,
        width,
        height
    );


    ctx.restore();


    drawPrivacyLabel(
        x,
        y,
        "FACE // SCANNED"
    );
}


/* =========================================================
   BOX PRIVACY
========================================================= */

function drawBoxPrivacy(
    x,
    y,
    width,
    height
) {

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";

    ctx.lineWidth = 2;


    drawCornerBrackets(
        x,
        y,
        width,
        height
    );


    ctx.font =
        "11px monospace";

    ctx.fillStyle =
        "#ffffff";

    ctx.fillText(
        "FACE TRACK",
        x,
        Math.max(
            14,
            y - 7
        )
    );


    ctx.restore();
}


/* =========================================================
   PRIVACY CORNERS
========================================================= */

function drawCornerBrackets(
    x,
    y,
    width,
    height
) {

    const size = 18;


    ctx.beginPath();

    /*
        Top-left
    */

    ctx.moveTo(
        x,
        y + size
    );

    ctx.lineTo(
        x,
        y
    );

    ctx.lineTo(
        x + size,
        y
    );


    /*
        Top-right
    */

    ctx.moveTo(
        x + width - size,
        y
    );

    ctx.lineTo(
        x + width,
        y
    );

    ctx.lineTo(
        x + width,
        y + size
    );


    /*
        Bottom-left
    */

    ctx.moveTo(
        x,
        y + height - size
    );

    ctx.lineTo(
        x,
        y + height
    );

    ctx.lineTo(
        x + size,
        y + height
    );


    /*
        Bottom-right
    */

    ctx.moveTo(
        x + width - size,
        y + height
    );

    ctx.lineTo(
        x + width,
        y + height
    );

    ctx.lineTo(
        x + width,
        y + height - size
    );


    ctx.stroke();
}


/* =========================================================
   PRIVACY FRAME
========================================================= */

function drawPrivacyFrame(
    x,
    y,
    width,
    height,
    label
) {

    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.8)";

    ctx.lineWidth =
        1.5;


    drawCornerBrackets(
        x,
        y,
        width,
        height
    );


    ctx.font =
        "10px monospace";

    ctx.fillStyle =
        "#ffffff";


    ctx.fillText(
        label,
        x,
        Math.max(
            12,
            y - 6
        )
    );


    ctx.restore();
}


/* =========================================================
   PRIVACY LABEL
========================================================= */

function drawPrivacyLabel(
    x,
    y,
    label
) {

    ctx.save();

    ctx.font =
        "10px monospace";

    ctx.fillStyle =
        "#ffffff";

    ctx.fillText(
        label,
        x,
        Math.max(
            12,
            y - 6
        )
    );

    ctx.restore();
}


/* =========================================================
   FACE FILTER
========================================================= */

function drawFaceFilter() {

    const filter =
        filters[filterIndex];


    if (
        filter === "OFF"
    ) {

        return;
    }


    const faces =
        faceResults?.faceLandmarks ||
        [];


    for (
        const face
        of faces
    ) {

        if (
            filter === "CYBER"
        ) {

            drawCyberFilter(
                face
            );
        }


        if (
            filter === "DOG"
        ) {

            drawDogFilter(
                face
            );
        }


        if (
            filter === "CROWN"
        ) {

            drawCrownFilter(
                face
            );
        }


        if (
            filter === "NEON"
        ) {

            drawNeonFilter(
                face
            );
        }
    }
}


/* =========================================================
   FACE TRACKER
========================================================= */

function drawFaceTracker() {

    /*
        Don't draw 478 landmarks every frame.

        Only draw a lightweight tracker when
        a filter or privacy mode is active.
    */

    if (
        filterIndex === 0 &&
        !facePrivacy
    ) {

        return;
    }


    const faces =
        faceResults?.faceLandmarks ||
        [];


    for (
        const face
        of faces
    ) {

        ctx.save();

        ctx.fillStyle =
            "rgba(255,255,255,0.55)";


        /*
            Sample the landmarks instead of
            drawing all of them.
        */

        for (
            let i = 0;
            i < face.length;
            i += 6
        ) {

            const p =
                face[i];


            const x =
                (1 - p.x) *
                canvas.width;

            const y =
                p.y *
                canvas.height;


            ctx.fillRect(
                x - 1,
                y - 1,
                2,
                2
            );
        }


        ctx.restore();
    }
}


/* =========================================================
   CYBER FILTER
========================================================= */

function drawCyberFilter(face) {

    const leftEye =
        point(face, 33);

    const rightEye =
        point(face, 263);

    const nose =
        point(face, 1);


    const eyeDistance =
        Math.abs(
            leftEye.x -
            rightEye.x
        );


    const width =
        eyeDistance * 2.2;


    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";

    ctx.lineWidth = 2;

    ctx.shadowBlur = 15;

    ctx.shadowColor =
        "rgba(255,255,255,0.8)";


    ctx.beginPath();


    ctx.roundRect(
        leftEye.x -
        width * 0.42,

        leftEye.y -
        width * 0.15,

        width * 0.35,

        width * 0.22,

        8
    );


    ctx.roundRect(
        rightEye.x -
        width * 0.0,

        rightEye.y -
        width * 0.15,

        width * 0.35,

        width * 0.22,

        8
    );


    ctx.stroke();


    /*
        Center connection.
    */

    ctx.beginPath();

    ctx.moveTo(
        leftEye.x +
        width * 0.15,

        leftEye.y
    );

    ctx.lineTo(
        rightEye.x -
        width * 0.15,

        rightEye.y
    );

    ctx.stroke();


    /*
        Nose marker.
    */

    ctx.beginPath();

    ctx.arc(
        nose.x,
        nose.y,
        5,
        0,
        Math.PI * 2
    );

    ctx.stroke();


    ctx.restore();
}


/* =========================================================
   DOG FILTER
========================================================= */

function drawDogFilter(face) {

    const top =
        point(face, 10);

    const left =
        point(face, 234);

    const right =
        point(face, 454);

    const nose =
        point(face, 1);


    const width =
        Math.abs(
            right.x -
            left.x
        );


    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";

    ctx.fillStyle =
        "rgba(255,255,255,0.12)";

    ctx.lineWidth = 2;


    /*
        Ears.
    */

    ctx.beginPath();

    ctx.moveTo(
        top.x -
        width * 0.55,

        top.y
    );

    ctx.lineTo(
        top.x -
        width * 0.8,

        top.y -
        width * 0.65
    );

    ctx.lineTo(
        top.x -
        width * 0.2,

        top.y -
        width * 0.35
    );

    ctx.closePath();

    ctx.fill();
    ctx.stroke();


    ctx.beginPath();

    ctx.moveTo(
        top.x +
        width * 0.55,

        top.y
    );

    ctx.lineTo(
        top.x +
        width * 0.8,

        top.y -
        width * 0.65
    );

    ctx.lineTo(
        top.x +
        width * 0.2,

        top.y -
        width * 0.35
    );

    ctx.closePath();

    ctx.fill();
    ctx.stroke();


    /*
        Nose.
    */

    ctx.beginPath();

    ctx.arc(
        nose.x,
        nose.y,
        width * 0.07,
        0,
        Math.PI * 2
    );

    ctx.fill();

    ctx.restore();
}


/* =========================================================
   CROWN FILTER
========================================================= */

function drawCrownFilter(face) {

    const top =
        point(face, 10);

    const left =
        point(face, 234);

    const right =
        point(face, 454);


    const width =
        Math.abs(
            right.x -
            left.x
        );


    const crownWidth =
        width * 0.8;


    const x =
        top.x -
        crownWidth / 2;


    const y =
        top.y -
        width * 0.45;


    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";

    ctx.fillStyle =
        "rgba(255,255,255,0.12)";

    ctx.lineWidth = 2;

    ctx.shadowBlur = 20;

    ctx.shadowColor =
        "rgba(255,255,255,0.8)";


    ctx.beginPath();

    ctx.moveTo(
        x,
        y + 35
    );

    ctx.lineTo(
        x + crownWidth * 0.15,
        y
    );

    ctx.lineTo(
        x + crownWidth * 0.35,
        y + 25
    );

    ctx.lineTo(
        x + crownWidth * 0.50,
        y - 15
    );

    ctx.lineTo(
        x + crownWidth * 0.65,
        y + 25
    );

    ctx.lineTo(
        x + crownWidth * 0.85,
        y
    );

    ctx.lineTo(
        x + crownWidth,
        y + 35
    );

    ctx.closePath();

    ctx.fill();
    ctx.stroke();

    ctx.restore();
}


/* =========================================================
   NEON FILTER
========================================================= */

function drawNeonFilter(face) {

    const points =
        face.map(
            p => ({
                x:
                    (1 - p.x) *
                    canvas.width,

                y:
                    p.y *
                    canvas.height
            })
        );


    const bounds =
        getBounds(points);


    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";

    ctx.lineWidth = 2;

    ctx.shadowBlur = 20;

    ctx.shadowColor =
        "rgba(255,255,255,0.9)";


    ctx.strokeRect(
        bounds.minX,
        bounds.minY,
        bounds.width,
        bounds.height
    );


    /*
        Face tracking cross.
    */

    const cx =
        bounds.cx;

    const cy =
        bounds.cy;


    ctx.beginPath();

    ctx.moveTo(
        cx - 12,
        cy
    );

    ctx.lineTo(
        cx + 12,
        cy
    );

    ctx.moveTo(
        cx,
        cy - 12
    );

    ctx.lineTo(
        cx,
        cy + 12
    );

    ctx.stroke();


    ctx.restore();
}


/* =========================================================
   HAND PROCESSING
========================================================= */

function processHands() {

    const hands =
        handResults?.handLandmarks ||
        [];


    handsInfo.textContent =
        hands.length;


    handStatus.textContent =
        hands.length
            ? `${hands.length} DETECTED`
            : "READY";


    if (!hands.length) {

        gestureText.textContent =
            "NONE";

        gestureInfo.textContent =
            "NONE";

        if (drawing) {
            finishShape();
        }

        pinchWasActive =
            false;

        return;
    }


    const hand =
        hands[0];


    const index =
        hand[8];

    const thumb =
        hand[4];


    const indexPoint = {

        x:
            (1 - index.x) *
            canvas.width,

        y:
            index.y *
            canvas.height
    };


    const pinch =
        distance(
            thumb,
            index
        ) < 0.055;


    let gesture =
        "TRACKING";


    if (
        handResults.gestures &&
        handResults.gestures[0] &&
        handResults.gestures[0][0]
    ) {

        gesture =
            handResults.gestures[0][0]
                .categoryName ||
            "TRACKING";
    }


    if (pinch) {
        gesture = "PINCH";
    }


    gestureText.textContent =
        gesture;

    gestureInfo.textContent =
        gesture;


    if (signLanguage) {

        const sign =
            detectSign(hand);

        signText.textContent =
            sign;

        signInfo.textContent =
            sign;
    }


    if (shapesEnabled) {

        handleShapeInteraction(
            indexPoint,
            pinch
        );
    }


    if (
        gesture === "Thumb_Down" ||
        gesture === "THUMBS_DOWN"
    ) {

        deleteNearestShape(
            indexPoint
        );
    }


    lastPoint =
        indexPoint;
}


/* =========================================================
   SHAPE INTERACTION
========================================================= */

function handleShapeInteraction(
    point,
    pinch
) {

    if (
        pinch &&
        !pinchWasActive
    ) {

        const existing =
            findShapeAtPoint(
                point
            );


        if (existing) {

            grabbedShape =
                existing;

            grabbedShape.offsetX =
                point.x -
                grabbedShape.x;

            grabbedShape.offsetY =
                point.y -
                grabbedShape.y;

            shapeStatus.textContent =
                "GRABBING";


        } else {

            drawing =
                true;

            currentPath = [
                {
                    x: point.x,
                    y: point.y
                }
            ];

            shapeStatus.textContent =
                "DRAWING";
        }
    }


    if (pinch) {

        if (grabbedShape) {

            grabbedShape.x =
                point.x -
                grabbedShape.offsetX;

            grabbedShape.y =
                point.y -
                grabbedShape.offsetY;


        } else if (drawing) {

            if (
                !lastPoint ||
                distance2D(
                    point,
                    lastPoint
                ) > 4
            ) {

                currentPath.push({
                    x: point.x,
                    y: point.y
                });
            }
        }
    }


    if (
        !pinch &&
        pinchWasActive
    ) {

        if (grabbedShape) {

            grabbedShape =
                null;

            shapeStatus.textContent =
                "RELEASED";


        } else if (drawing) {

            finishShape();
        }
    }


    pinchWasActive =
        pinch;
}


/* =========================================================
   FINISH SHAPE
========================================================= */

function finishShape() {

    if (
        currentPath.length < 4
    ) {

        drawing =
            false;

        currentPath =
            [];

        shapeStatus.textContent =
            "READY";

        return;
    }


    const bounds =
        getBounds(
            currentPath
        );


    if (
        bounds.width < 35 &&
        bounds.height < 35
    ) {

        shapes.push({

            id:
                ++shapeId,

            type:
                "circle",

            x:
                bounds.cx,

            y:
                bounds.cy,

            radius:
                25
        });


    } else {

        shapes.push({

            id:
                ++shapeId,

            type:
                "polygon",

            points:
                simplifyPath(
                    currentPath,
                    10
                ),

            x:
                bounds.cx,

            y:
                bounds.cy,

            centerX:
                bounds.cx,

            centerY:
                bounds.cy
        });
    }


    drawing =
        false;

    currentPath =
        [];

    shapeStatus.textContent =
        "CREATED";
}


/* =========================================================
   DRAW SHAPES
========================================================= */

function drawShapes() {

    /*
        Current drawing.
    */

    if (
        drawing &&
        currentPath.length > 1
    ) {

        ctx.save();

        ctx.beginPath();

        ctx.moveTo(
            currentPath[0].x,
            currentPath[0].y
        );


        for (
            let i = 1;
            i < currentPath.length;
            i++
        ) {

            ctx.lineTo(
                currentPath[i].x,
                currentPath[i].y
            );
        }


        ctx.strokeStyle =
            "rgba(255,255,255,0.9)";

        ctx.lineWidth = 3;

        ctx.shadowBlur = 15;

        ctx.shadowColor =
            "rgba(255,255,255,0.8)";

        ctx.stroke();

        ctx.restore();
    }


    /*
        Finished shapes.
    */

    for (
        const shape
        of shapes
    ) {

        ctx.save();

        ctx.strokeStyle =
            shape === grabbedShape
                ? "#ffffff"
                : "rgba(255,255,255,0.75)";

        ctx.fillStyle =
            "rgba(255,255,255,0.04)";

        ctx.lineWidth =
            shape === grabbedShape
                ? 3
                : 2;


        if (
            shape.type === "circle"
        ) {

            ctx.beginPath();

            ctx.arc(
                shape.x,
                shape.y,
                shape.radius,
                0,
                Math.PI * 2
            );

            ctx.fill();

            ctx.stroke();


        } else {

            ctx.beginPath();

            for (
                let i = 0;
                i < shape.points.length;
                i++
            ) {

                const p =
                    shape.points[i];


                const px =
                    shape.x +
                    (
                        p.x -
                        shape.centerX
                    );

                const py =
                    shape.y +
                    (
                        p.y -
                        shape.centerY
                    );


                if (i === 0) {

                    ctx.moveTo(
                        px,
                        py
                    );

                } else {

                    ctx.lineTo(
                        px,
                        py
                    );
                }
            }


            ctx.closePath();

            ctx.fill();

            ctx.stroke();
        }


        ctx.restore();
    }


    shapesInfo.textContent =
        shapes.length;
}


/* =========================================================
   DRAW HANDS
========================================================= */

function drawHands() {

    if (!handTracking) {
        return;
    }


    const hands =
        handResults?.handLandmarks ||
        [];


    for (
        const hand
        of hands
    ) {

        drawHandSkeleton(
            hand
        );


        const index =
            point(hand, 8);

        const thumb =
            point(hand, 4);


        ctx.save();

        ctx.beginPath();

        ctx.arc(
            index.x,
            index.y,
            7,
            0,
            Math.PI * 2
        );


        ctx.strokeStyle =
            "rgba(255,255,255,0.9)";

        ctx.lineWidth = 2;

        ctx.stroke();


        if (
            distance(
                hand[4],
                hand[8]
            ) < 0.055
        ) {

            ctx.beginPath();

            ctx.arc(
                (
                    index.x +
                    thumb.x
                ) / 2,

                (
                    index.y +
                    thumb.y
                ) / 2,

                12,

                0,
                Math.PI * 2
            );

            ctx.stroke();
        }


        ctx.restore();


        if (targetVisible) {

            const bounds =
                getBounds(
                    hand.map(
                        p => ({
                            x:
                                (1 - p.x) *
                                canvas.width,

                            y:
                                p.y *
                                canvas.height
                        })
                    )
                );


            ctx.save();

            ctx.setLineDash([
                5,
                5
            ]);

            ctx.strokeStyle =
                "rgba(255,255,255,0.35)";


            ctx.strokeRect(
                bounds.minX - 8,
                bounds.minY - 8,
                bounds.width + 16,
                bounds.height + 16
            );


            ctx.restore();
        }
    }
}


/* =========================================================
   HAND SKELETON
========================================================= */

function drawHandSkeleton(hand) {

    const connections = [

        [0,1],[1,2],[2,3],[3,4],

        [0,5],[5,6],[6,7],[7,8],

        [5,9],[9,10],[10,11],[11,12],

        [9,13],[13,14],[14,15],[15,16],

        [13,17],[17,18],[18,19],[19,20],

        [0,17]
    ];


    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.5)";

    ctx.lineWidth = 1.5;


    for (
        const [a,b]
        of connections
    ) {

        const p1 =
            point(hand,a);

        const p2 =
            point(hand,b);


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


    ctx.fillStyle =
        "rgba(255,255,255,0.9)";


    for (
        const p
        of hand
    ) {

        const x =
            (1 - p.x) *
            canvas.width;

        const y =
            p.y *
            canvas.height;


        ctx.beginPath();

        ctx.arc(
            x,
            y,
            2,
            0,
            Math.PI * 2
        );

        ctx.fill();
    }


    ctx.restore();
}


/* =========================================================
   OBJECT DRAW
========================================================= */

function drawObjects() {

    if (!objectDetection) {

        objectStatus.textContent =
            "OFF";

        return;
    }


    const detections =
        objectResults?.detections ||
        [];


    objectStatus.textContent =
        detections.length
            ? `${detections.length} FOUND`
            : "SEARCHING";


    for (
        const detection
        of detections
    ) {

        const box =
            detection.boundingBox;


        if (!box) {
            continue;
        }


        const x =
            canvas.width -
            box.originX -
            box.width;


        const y =
            box.originY;


        const category =
            detection.categories?.[0];


        const label =
            category?.categoryName ||
            "OBJECT";


        const score =
            category?.score ||
            0;


        ctx.save();


        ctx.strokeStyle =
            "rgba(255,255,255,0.75)";

        ctx.lineWidth = 2;


        ctx.strokeRect(
            x,
            y,
            box.width,
            box.height
        );


        ctx.font =
            "11px monospace";


        ctx.fillStyle =
            "rgba(0,0,0,0.7)";


        ctx.fillRect(
            x,
            Math.max(
                0,
                y - 18
            ),
            140,
            18
        );


        ctx.fillStyle =
            "#ffffff";


        ctx.fillText(
            `${label} ${Math.round(score * 100)}%`,
            x + 5,
            Math.max(
                13,
                y - 5
            )
        );


        ctx.restore();
    }
}


/* =========================================================
   SIGN LANGUAGE
========================================================= */

function detectSign(hand) {

    const extended =
        countExtendedFingers(
            hand
        );


    if (
        isFingerExtended(
            hand,
            8
        ) &&
        !isFingerExtended(
            hand,
            12
        ) &&
        !isFingerExtended(
            hand,
            16
        ) &&
        !isFingerExtended(
            hand,
            20
        )
    ) {

        return "D";
    }


    if (
        isFingerExtended(hand,8) &&
        isFingerExtended(hand,12) &&
        !isFingerExtended(hand,16) &&
        !isFingerExtended(hand,20)
    ) {

        return "V";
    }


    if (
        !isFingerExtended(hand,8) &&
        !isFingerExtended(hand,12) &&
        !isFingerExtended(hand,16) &&
        isFingerExtended(hand,20)
    ) {

        return "I";
    }


    if (
        extended === 0
    ) {

        return "A / FIST";
    }


    if (
        extended === 4 &&
        isThumbExtended(hand)
    ) {

        return "I LOVE YOU";
    }


    if (
        extended === 4
    ) {

        return "B / OPEN";
    }


    return "UNKNOWN";
}


/* =========================================================
   SIGN HELPERS
========================================================= */

function isFingerExtended(
    hand,
    tip
) {

    return (
        hand[tip].y <
        hand[tip - 2].y
    );
}


function isThumbExtended(hand) {

    return (
        Math.abs(
            hand[4].x -
            hand[3].x
        ) > 0.045
    );
}


function countExtendedFingers(
    hand
) {

    let count = 0;


    if (
        isFingerExtended(hand,8)
    ) count++;


    if (
        isFingerExtended(hand,12)
    ) count++;


    if (
        isFingerExtended(hand,16)
    ) count++;


    if (
        isFingerExtended(hand,20)
    ) count++;


    return count;
}


/* =========================================================
   SHAPE FIND
========================================================= */

function findShapeAtPoint(point) {

    for (
        let i =
            shapes.length - 1;

        i >= 0;

        i--
    ) {

        const shape =
            shapes[i];


        const x =
            shape.x ??
            shape.centerX;


        const y =
            shape.y ??
            shape.centerY;


        const distance =
            Math.hypot(
                point.x - x,
                point.y - y
            );


        if (
            shape.type === "circle" &&
            distance <
            shape.radius + 25
        ) {

            return shape;
        }


        if (
            shape.type === "polygon"
        ) {

            const bounds =
                getBounds(
                    shape.points
                );


            if (
                point.x >
                    bounds.minX - 25 &&

                point.x <
                    bounds.maxX + 25 &&

                point.y >
                    bounds.minY - 25 &&

                point.y <
                    bounds.maxY + 25
            ) {

                return shape;
            }
        }
    }


    return null;
}


/* =========================================================
   DELETE
========================================================= */

function deleteNearestShape(
    point
) {

    let nearestIndex = -1;

    let smallest =
        Infinity;


    shapes.forEach(
        (shape,index) => {

            const x =
                shape.x ??
                shape.centerX;

            const y =
                shape.y ??
                shape.centerY;


            const d =
                Math.hypot(
                    point.x - x,
                    point.y - y
                );


            if (
                d < smallest
            ) {

                smallest =
                    d;

                nearestIndex =
                    index;
            }
        }
    );


    if (
        nearestIndex !== -1 &&
        smallest < 120
    ) {

        shapes.splice(
            nearestIndex,
            1
        );


        shapeStatus.textContent =
            "DELETED";
    }
}


/* =========================================================
   MATH
========================================================= */

function distance(a,b) {

    return Math.sqrt(

        Math.pow(
            a.x - b.x,
            2
        ) +

        Math.pow(
            a.y - b.y,
            2
        ) +

        Math.pow(
            (a.z || 0) -
            (b.z || 0),
            2
        )
    );
}


function distance2D(a,b) {

    return Math.hypot(
        a.x - b.x,
        a.y - b.y
    );
}


function point(
    points,
    index
) {

    const p =
        points[index];


    return {

        x:
            (1 - p.x) *
            canvas.width,

        y:
            p.y *
            canvas.height
    };
}


/* =========================================================
   BOUNDS
========================================================= */

function getBounds(points) {

    let minX =
        Infinity;

    let minY =
        Infinity;

    let maxX =
        -Infinity;

    let maxY =
        -Infinity;


    for (
        const p
        of points
    ) {

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
            maxY - minY,

        cx:
            (minX + maxX) / 2,

        cy:
            (minY + maxY) / 2
    };
}


/* =========================================================
   PATH SIMPLIFICATION
========================================================= */

function simplifyPath(
    points,
    spacing
) {

    if (
        points.length <= 2
    ) {

        return points;
    }


    const result =
        [points[0]];


    let last =
        points[0];


    for (
        let i = 1;
        i < points.length;
        i++
    ) {

        if (
            distance2D(
                points[i],
                last
            ) >= spacing
        ) {

            result.push(
                points[i]
            );

            last =
                points[i];
        }
    }


    return result;
}


/* =========================================================
   FPS
========================================================= */

function updateFPS() {

    fpsFrames++;


    const now =
        performance.now();


    if (
        now -
        fpsLastTime >=
        1000
    ) {

        fpsText.textContent =
            fpsFrames;


        fpsFrames =
            0;


        fpsLastTime =
            now;
    }
}


/* =========================================================
   STARTUP
========================================================= */

console.log(
    "SPECTRA v0.7"
);

console.log(
    isMobile
        ? "Mobile performance mode"
        : "Desktop performance mode"
);
