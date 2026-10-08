import {
    FilesetResolver,
    GestureRecognizer,
    FaceDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


/* =========================================================
   SPECTRA v0.4
   FAST HAND TRACKING // GESTURES // OBJECTS // PRIVACY
========================================================= */


/* =========================================================
   DOM
========================================================= */

const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const startButton = document.getElementById("startButton");
const trackingButton = document.getElementById("trackingButton");

const statusText = document.getElementById("status");
const cameraMessage = document.getElementById("cameraMessage");

const systemStatus = document.getElementById("systemStatus");
const statusDot = document.getElementById("statusDot");

const handState = document.getElementById("handState");
const targetNumber = document.getElementById("targetNumber");

const xValue = document.getElementById("xValue");
const yValue = document.getElementById("yValue");

const fpsValue = document.getElementById("fpsValue");
const trackingState = document.getElementById("trackingState");

const gestureText = document.getElementById("gestureText");

const handsValue = document.getElementById("handsValue");
const indexValue = document.getElementById("indexValue");
const pinchValue = document.getElementById("pinchValue");
const gestureValue = document.getElementById("gestureValue");


/* =========================================================
   STATE
========================================================= */

let gestureRecognizer = null;
let faceDetector = null;

let cameraStream = null;
let cameraRunning = false;

let trackingEnabled = true;

let showTargetBox = true;
let facePrivacy = false;
let signLanguageMode = false;
let objectMode = true;

let lastVideoTime = -1;
let lastDetectionTime = 0;

let previousIndex = null;

let frameCounter = 0;
let displayedFPS = 0;
let lastFPSUpdate = performance.now();

let latestFaceResults = null;
let lastFaceDetection = 0;


/* =========================================================
   PERFORMANCE
========================================================= */

/*
   Hand recognition target.

   30 FPS is normally more than enough for hand interaction
   and prevents the browser from getting overloaded.
*/

const HAND_DETECTION_INTERVAL = 1000 / 30;


/*
   Face detection doesn't need to happen every frame.

   We update it around 10 times per second.
*/

const FACE_DETECTION_INTERVAL = 100;


/*
   Lower inference resolution = faster tracking.

   The displayed camera can still remain large.
*/

const INFERENCE_WIDTH = 640;
const INFERENCE_HEIGHT = 360;


/* =========================================================
   MODELS
========================================================= */

const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";


const GESTURE_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";


const FACE_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task";


/* =========================================================
   VIRTUAL OBJECTS
========================================================= */

const virtualObjects = [];

let nextObjectID = 1;

let grabbedObject = null;


/*
   These objects are intentionally simple for v0.4.

   Later we can make them 3D / glowing / animated.
*/

function createVirtualObject(
    x,
    y,
    type = "BOX"
) {

    const object = {

        id: nextObjectID++,

        x,
        y,

        width: 110,
        height: 110,

        rotation: 0,

        type,

        grabbed: false
    };


    virtualObjects.push(object);

    return object;
}


/*
   Create a starting object when the system begins.
*/

function createInitialObject() {

    if (virtualObjects.length > 0) {
        return;
    }


    createVirtualObject(
        canvas.width / 2,
        canvas.height / 2,
        "BOX"
    );
}


/* =========================================================
   CREATE EXTRA CONTROLS
========================================================= */

/*
   We don't modify your existing HTML.

   These buttons are created automatically.
*/

function createExtraControls() {

    const controls =
        document.querySelector(".controls");

    if (!controls) {
        return;
    }


    const boxButton =
        document.createElement("button");

    boxButton.id =
        "boxVisibilityButton";

    boxButton.textContent =
        "TARGET BOX ON";


    const privacyButton =
        document.createElement("button");

    privacyButton.id =
        "privacyButton";

    privacyButton.textContent =
        "FACE PRIVACY OFF";


    const signButton =
        document.createElement("button");

    signButton.id =
        "signButton";

    signButton.textContent =
        "SIGN LANGUAGE OFF";


    const objectButton =
        document.createElement("button");

    objectButton.id =
        "objectButton";

    objectButton.textContent =
        "OBJECT MODE ON";


    controls.appendChild(
        boxButton
    );

    controls.appendChild(
        privacyButton
    );

    controls.appendChild(
        signButton
    );

    controls.appendChild(
        objectButton
    );


    /*
       Target box toggle
    */

    boxButton.addEventListener(
        "click",
        () => {

            showTargetBox =
                !showTargetBox;


            boxButton.textContent =
                showTargetBox
                    ? "TARGET BOX ON"
                    : "TARGET BOX OFF";


            boxButton.classList.toggle(
                "active",
                showTargetBox
            );
        }
    );


    /*
       Face privacy
    */

    privacyButton.addEventListener(
        "click",
        () => {

            facePrivacy =
                !facePrivacy;


            privacyButton.textContent =
                facePrivacy
                    ? "FACE PRIVACY ON"
                    : "FACE PRIVACY OFF";


            privacyButton.classList.toggle(
                "active",
                facePrivacy
            );
        }
    );


    /*
       Sign language
    */

    signButton.addEventListener(
        "click",
        () => {

            signLanguageMode =
                !signLanguageMode;


            signButton.textContent =
                signLanguageMode
                    ? "SIGN LANGUAGE ON"
                    : "SIGN LANGUAGE OFF";


            signButton.classList.toggle(
                "active",
                signLanguageMode
            );
        }
    );


    /*
       Virtual objects
    */

    objectButton.addEventListener(
        "click",
        () => {

            objectMode =
                !objectMode;


            objectButton.textContent =
                objectMode
                    ? "OBJECT MODE ON"
                    : "OBJECT MODE OFF";


            objectButton.classList.toggle(
                "active",
                objectMode
            );
        }
    );
}


/* =========================================================
   MEDIAPIPE INITIALIZATION
========================================================= */

async function createModels() {

    statusText.textContent =
        "Loading SPECTRA vision systems...";


    try {

        const vision =
            await FilesetResolver.forVisionTasks(
                WASM_URL
            );


        /*
           Gesture Recognizer

           This replaces our old homemade gesture system.
        */

        gestureRecognizer =
            await GestureRecognizer.createFromOptions(
                vision,
                {

                    baseOptions: {

                        modelAssetPath:
                            GESTURE_MODEL_URL,

                        delegate:
                            "GPU"
                    },

                    runningMode:
                        "VIDEO",

                    numHands:
                        2,

                    minHandDetectionConfidence:
                        0.5,

                    minHandPresenceConfidence:
                        0.5,

                    minTrackingConfidence:
                        0.5
                }
            );


        /*
           Face detector

           Used only when Face Privacy is enabled.
        */

        faceDetector =
            await FaceDetector.createFromOptions(
                vision,
                {

                    baseOptions: {

                        modelAssetPath:
                            FACE_MODEL_URL,

                        delegate:
                            "GPU"
                    },

                    runningMode:
                        "VIDEO",

                    minDetectionConfidence:
                        0.5
                }
            );


        statusText.textContent =
            "SPECTRA vision systems ready.";

        trackingButton.disabled =
            false;


    } catch (error) {

        console.error(error);

        statusText.textContent =
            "Vision system failed to load.";

        handState.textContent =
            "VISION SYSTEM // ERROR";
    }
}


/* =========================================================
   CAMERA
========================================================= */

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

                    /*
                       640x360 is much easier to process
                       than forcing 1280x720 inference.
                    */

                    width: {
                        ideal: INFERENCE_WIDTH
                    },

                    height: {
                        ideal: INFERENCE_HEIGHT
                    },

                    frameRate: {
                        ideal: 30,
                        max: 30
                    },

                    facingMode: {
                        ideal: "user"
                    }
                },

                audio: false
            });


        video.srcObject =
            cameraStream;


        await video.play();


        cameraRunning =
            true;


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


        if (!gestureRecognizer) {

            await createModels();
        }


        createInitialObject();


        lastVideoTime = -1;


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


/* =========================================================
   STOP CAMERA
========================================================= */

function stopCamera() {

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );
    }


    cameraStream =
        null;

    video.srcObject =
        null;

    cameraRunning =
        false;


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


/* =========================================================
   CANVAS
========================================================= */

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


/* =========================================================
   TRACKING BUTTON
========================================================= */

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


/* =========================================================
   DETECTION LOOP
========================================================= */

function detectionLoop() {

    if (!cameraRunning) {
        return;
    }


    const now =
        performance.now();


    /*
       Only process a new video frame.
    */

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


            /*
               Limit expensive AI processing.
            */

            if (
                now -
                lastDetectionTime
                >=
                HAND_DETECTION_INTERVAL
            ) {

                lastDetectionTime =
                    now;


                if (
                    trackingEnabled &&
                    gestureRecognizer
                ) {

                    detectHands(
                        now
                    );
                }


                if (
                    facePrivacy &&
                    faceDetector &&
                    now -
                    lastFaceDetection
                    >=
                    FACE_DETECTION_INTERVAL
                ) {

                    detectFace(
                        now
                    );
                }
            }
        }
    }


    /*
       Drawing still runs every browser frame.
       This makes movement look smoother.
    */

    render();


    updateFPS();


    requestAnimationFrame(
        detectionLoop
    );
}


/* =========================================================
   HAND DETECTION
========================================================= */

function detectHands(
    timestamp
) {

    try {

        const results =
            gestureRecognizer.recognizeForVideo(
                video,
                timestamp
            );


        processResults(
            results
        );


    } catch (error) {

        console.error(
            "Gesture recognition error:",
            error
        );
    }
}


/* =========================================================
   FACE DETECTION
========================================================= */

function detectFace(
    timestamp
) {

    try {

        latestFaceResults =
            faceDetector.detectForVideo(
                video,
                timestamp
            );


        lastFaceDetection =
            timestamp;


    } catch (error) {

        console.error(
            "Face detection error:",
            error
        );
    }
}


/* =========================================================
   PROCESS HAND RESULTS
========================================================= */

function processResults(
    results
) {

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
            : "HANDS // " +
              handCount;


    trackingState.textContent =
        "ACTIVE";


    targetNumber.textContent =
        "01";


    /*
       Draw every detected hand.
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
       Primary hand.
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
       Faster smoothing.

       Old version used 0.35 which created noticeable delay.

       0.70 follows the finger much more closely.
    */

    const smoothed =
        smoothPoint(
            previousIndex,
            {
                x: screenX,
                y: screenY
            },
            0.70
        );


    previousIndex =
        smoothed;


    xValue.textContent =
        Math.round(
            smoothed.x
        );


    yValue.textContent =
        Math.round(
            smoothed.y
        );


    indexValue.textContent =
        `${Math.round(smoothed.x)}, ${Math.round(smoothed.y)}`;


    /*
       Pinch detection.
    */

    const pinchDistance =
        distance(
            indexFinger,
            thumb
        );


    const pinch =
        pinchDistance <
        0.075;


    pinchValue.textContent =
        pinch
            ? "YES"
            : "NO";


    /*
       Get MediaPipe gesture.
    */

    let gesture =
        getMediaPipeGesture(
            results,
            0
        );


    /*
       Override with pinch when necessary.
    */

    if (pinch) {

        gesture =
            "PINCH";
    }


    /*
       Sign language mode.
    */

    if (signLanguageMode) {

        const sign =
            detectSign(
                primaryHand
            );


        if (sign) {

            gesture =
                "SIGN " +
                sign;
        }
    }


    gestureText.textContent =
        "GESTURE // " +
        gesture;


    gestureValue.textContent =
        gesture;


    /*
       Cursor.
    */

    drawCursor(
        smoothed.x,
        smoothed.y,
        pinch
    );


    /*
       Target box.
    */

    if (showTargetBox) {

        drawTargetBox(
            primaryHand
        );
    }


    /*
       Virtual object interaction.
    */

    if (objectMode) {

        updateObjectInteraction(
            smoothed.x,
            smoothed.y,
            pinch
        );
    }
}


/* =========================================================
   MEDIAPIPE GESTURE NAME
========================================================= */

function getMediaPipeGesture(
    results,
    handIndex
) {

    if (
        !results.gestures ||
        !results.gestures[handIndex] ||
        results.gestures[handIndex].length === 0
    ) {

        return "TRACKING";
    }


    const gesture =
        results.gestures[
            handIndex
        ][0];


    const name =
        gesture.categoryName;


    switch (name) {

        case "Open_Palm":
            return "OPEN PALM";

        case "Closed_Fist":
            return "FIST";

        case "Pointing_Up":
            return "POINT";

        case "Thumb_Up":
            return "THUMBS UP";

        case "Thumb_Down":
            return "THUMBS DOWN";

        case "Victory":
            return "VICTORY";

        case "ILoveYou":
            return "I LOVE YOU";

        default:
            return name
                ? name.toUpperCase()
                : "TRACKING";
    }
}


/* =========================================================
   SIGN LANGUAGE
========================================================= */

/*
   This is a FUN starter system.

   It is NOT a complete sign-language translator.

   These are simple static-sign approximations.
*/

function detectSign(
    landmarks
) {

    const index =
        isFingerExtended(
            landmarks,
            8,
            6
        );

    const middle =
        isFingerExtended(
            landmarks,
            12,
            10
        );

    const ring =
        isFingerExtended(
            landmarks,
            16,
            14
        );

    const pinky =
        isFingerExtended(
            landmarks,
            20,
            18
        );


    /*
       B
       Four fingers extended.
    */

    if (
        index &&
        middle &&
        ring &&
        pinky
    ) {

        return "B";
    }


    /*
       D
       Index up, other fingers closed.
    */

    if (
        index &&
        !middle &&
        !ring &&
        !pinky
    ) {

        return "D";
    }


    /*
       V
       Index + middle extended.
    */

    if (
        index &&
        middle &&
        !ring &&
        !pinky
    ) {

        return "V";
    }


    /*
       FIST
    */

    if (
        !index &&
        !middle &&
        !ring &&
        !pinky
    ) {

        return "A / FIST";
    }


    /*
       I
       Pinky extended.
    */

    if (
        !index &&
        !middle &&
        !ring &&
        pinky
    ) {

        return "I";
    }


    /*
       I LOVE YOU style:
       index + pinky extended.
    */

    if (
        index &&
        !middle &&
        !ring &&
        pinky
    ) {

        return "I LOVE YOU";
    }


    return null;
}


/* =========================================================
   FINGER EXTENSION
========================================================= */

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


    return (
        tipDistance >
        pipDistance * 1.12
    );
}


/* =========================================================
   VIRTUAL OBJECT INTERACTION
========================================================= */

function updateObjectInteraction(
    x,
    y,
    pinch
) {

    /*
       If we aren't holding anything and
       pinch begins, find the closest object.
    */

    if (
        pinch &&
        !grabbedObject
    ) {

        grabbedObject =
            findObjectAt(
                x,
                y
            );


        if (grabbedObject) {

            grabbedObject.grabbed =
                true;
        }
    }


    /*
       Move grabbed object.
    */

    if (
        pinch &&
        grabbedObject
    ) {

        grabbedObject.x =
            x;

        grabbedObject.y =
            y;
    }


    /*
       Release object.
    */

    if (
        !pinch &&
        grabbedObject
    ) {

        grabbedObject.grabbed =
            false;

        grabbedObject =
            null;
    }
}


/* =========================================================
   FIND OBJECT
========================================================= */

function findObjectAt(
    x,
    y
) {

    /*
       Search backwards so the newest object
       is selected first.
    */

    for (
        let i =
            virtualObjects.length - 1;

        i >= 0;

        i--
    ) {

        const object =
            virtualObjects[i];


        const halfWidth =
            object.width / 2;

        const halfHeight =
            object.height / 2;


        if (
            x >=
                object.x -
                halfWidth &&

            x <=
                object.x +
                halfWidth &&

            y >=
                object.y -
                halfHeight &&

            y <=
                object.y +
                halfHeight
        ) {

            return object;
        }
    }


    return null;
}


/* =========================================================
   RENDER
========================================================= */

function render() {

    clearOverlay();


    /*
       Face privacy is rendered first.
    */

    if (facePrivacy) {

        drawFacePrivacy();
    }


    /*
       Virtual objects.
    */

    if (objectMode) {

        virtualObjects.forEach(
            drawVirtualObject
        );
    }
}


/* =========================================================
   VIRTUAL OBJECT DRAWING
========================================================= */

function drawVirtualObject(
    object
) {

    ctx.save();


    ctx.translate(
        object.x,
        object.y
    );


    ctx.rotate(
        object.rotation
    );


    ctx.lineWidth =
        object.grabbed
            ? 3
            : 1.5;


    ctx.strokeStyle =
        object.grabbed
            ? "rgba(255,255,255,1)"
            : "rgba(255,255,255,0.75)";


    ctx.fillStyle =
        object.grabbed
            ? "rgba(255,255,255,0.08)"
            : "rgba(255,255,255,0.03)";


    if (
        object.type ===
        "BOX"
    ) {

        ctx.fillRect(
            -object.width / 2,
            -object.height / 2,
            object.width,
            object.height
        );


        ctx.strokeRect(
            -object.width / 2,
            -object.height / 2,
            object.width,
            object.height
        );


        /*
           Corner markers.
        */

        const s = 12;


        ctx.beginPath();

        ctx.moveTo(
            -object.width / 2,
            -object.height / 2 + s
        );

        ctx.lineTo(
            -object.width / 2,
            -object.height / 2
        );

        ctx.lineTo(
            -object.width / 2 + s,
            -object.height / 2
        );

        ctx.stroke();


        ctx.beginPath();

        ctx.moveTo(
            object.width / 2 - s,
            object.height / 2
        );

        ctx.lineTo(
            object.width / 2,
            object.height / 2
        );

        ctx.lineTo(
            object.width / 2,
            object.height / 2 - s
        );

        ctx.stroke();
    }


    ctx.restore();
}


/* =========================================================
   FACE PRIVACY
========================================================= */

function drawFacePrivacy() {

    if (
        !latestFaceResults ||
        !latestFaceResults.detections
    ) {

        return;
    }


    latestFaceResults.detections.forEach(
        detection => {

            const box =
                detection.boundingBox;


            if (!box) {
                return;
            }


            const x =
                box.originX;

            const y =
                box.originY;

            const width =
                box.width;

            const height =
                box.height;


            /*
               Expand the face area slightly
               so edges aren't visible.
            */

            const padding =
                20;


            const expandedX =
                Math.max(
                    0,
                    x - padding
                );


            const expandedY =
                Math.max(
                    0,
                    y - padding
                );


            const expandedWidth =
                Math.min(
                    canvas.width -
                    expandedX,

                    width +
                    padding * 2
                );


            const expandedHeight =
                Math.min(
                    canvas.height -
                    expandedY,

                    height +
                    padding * 2
                );


            ctx.save();


            /*
               Draw a blurred copy of the
               camera image over the face.
            */

            ctx.filter =
                "blur(22px)";


            ctx.drawImage(
                video,

                expandedX,
                expandedY,
                expandedWidth,
                expandedHeight,

                expandedX,
                expandedY,
                expandedWidth,
                expandedHeight
            );


            ctx.restore();


            /*
               Extra blur layer.
            */

            ctx.save();


            ctx.fillStyle =
                "rgba(0,0,0,0.18)";


            ctx.fillRect(
                expandedX,
                expandedY,
                expandedWidth,
                expandedHeight
            );


            ctx.restore();
        }
    );
}


/* =========================================================
   HAND DRAWING
========================================================= */

const CONNECTIONS = [

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


function drawHand(
    landmarks,
    handIndex
) {

    const points =
        landmarks.map(
            point => ({

                x:
                    point.x *
                    canvas.width,

                y:
                    point.y *
                    canvas.height
            })
        );


    ctx.save();


    ctx.lineWidth =
        1.5;


    ctx.strokeStyle =
        "rgba(255,255,255,0.75)";


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


/* =========================================================
   CURSOR
========================================================= */

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


    ctx.lineWidth =
        1.5;


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


    ctx.beginPath();


    ctx.arc(
        x,
        y,
        isPinching
            ? 7
            : 4,
        0,
        Math.PI * 2
    );


    ctx.stroke();


    ctx.restore();
}


/* =========================================================
   TARGET BOX
========================================================= */

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


    const padding =
        22;


    const x =
        minX -
        padding;


    const y =
        minY -
        padding;


    const width =
        maxX -
        minX +
        padding * 2;


    const height =
        maxY -
        minY +
        padding * 2;


    ctx.save();


    ctx.strokeStyle =
        "rgba(255,255,255,0.75)";


    ctx.lineWidth =
        1;


    ctx.strokeRect(
        x,
        y,
        width,
        height
    );


    const centerX =
        x +
        width / 2;


    const centerY =
        y +
        height / 2;


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


/* =========================================================
   DISTANCE
========================================================= */

function distance(
    a,
    b
) {

    const dx =
        a.x -
        b.x;


    const dy =
        a.y -
        b.y;


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
        a.x -
        b.x;


    const dy =
        a.y -
        b.y;


    const dz =
        (a.z || 0) -
        (b.z || 0);


    return Math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    );
}


/* =========================================================
   SMOOTHING
========================================================= */

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
            (
                current.x -
                previous.x
            ) *
            amount,

        y:
            previous.y +
            (
                current.y -
                previous.y
            ) *
            amount
    };
}


/* =========================================================
   FPS
========================================================= */

function updateFPS() {

    frameCounter++;


    const now =
        performance.now();


    const elapsed =
        now -
        lastFPSUpdate;


    if (
        elapsed >=
        1000
    ) {

        displayedFPS =
            Math.round(
                frameCounter *
                1000 /
                elapsed
            );


        fpsValue.textContent =
            displayedFPS;


        frameCounter =
            0;


        lastFPSUpdate =
            now;
    }
}


/* =========================================================
   NO HAND
========================================================= */

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


    previousIndex =
        null;


    /*
       Release object if hand disappears.
    */

    if (grabbedObject) {

        grabbedObject.grabbed =
            false;

        grabbedObject =
            null;
    }
}


/* =========================================================
   RESET
========================================================= */

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


    previousIndex =
        null;


    grabbedObject =
        null;
}


/* =========================================================
   INITIALIZATION
========================================================= */

trackingButton.textContent =
    "HAND TRACKING ON";


trackingButton.classList.add(
    "active"
);


createExtraControls();


createModels();
