import {
    FilesetResolver,
    GestureRecognizer,
    FaceDetector,
    ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


/* =========================================================
   SPECTRA v0.5
   VISION INTERACTION SYSTEM
========================================================= */


/* =========================================================
   DOM
========================================================= */

const video =
    document.getElementById(
        "camera"
    );


const canvas =
    document.getElementById(
        "overlay"
    );


const ctx =
    canvas.getContext(
        "2d"
    );


const startButton =
    document.getElementById(
        "startButton"
    );


const trackingButton =
    document.getElementById(
        "trackingButton"
    );


const boxButton =
    document.getElementById(
        "boxButton"
    );


const objectButton =
    document.getElementById(
        "objectButton"
    );


const privacyButton =
    document.getElementById(
        "privacyButton"
    );


const signButton =
    document.getElementById(
        "signButton"
    );


const objectModeButton =
    document.getElementById(
        "objectModeButton"
    );


const statusText =
    document.getElementById(
        "status"
    );


const cameraMessage =
    document.getElementById(
        "cameraMessage"
    );


const systemStatus =
    document.getElementById(
        "systemStatus"
    );


const statusDot =
    document.getElementById(
        "statusDot"
    );


const handState =
    document.getElementById(
        "handState"
    );


const targetNumber =
    document.getElementById(
        "targetNumber"
    );


const xValue =
    document.getElementById(
        "xValue"
    );


const yValue =
    document.getElementById(
        "yValue"
    );


const fpsValue =
    document.getElementById(
        "fpsValue"
    );


const trackingState =
    document.getElementById(
        "trackingState"
    );


const gestureText =
    document.getElementById(
        "gestureText"
    );


const handsValue =
    document.getElementById(
        "handsValue"
    );


const indexValue =
    document.getElementById(
        "indexValue"
    );


const pinchValue =
    document.getElementById(
        "pinchValue"
    );


const gestureValue =
    document.getElementById(
        "gestureValue"
    );


/* =========================================================
   MODELS
========================================================= */

let gestureRecognizer = null;

let faceDetector = null;

let objectDetector = null;


/* =========================================================
   CAMERA STATE
========================================================= */

let cameraStream = null;

let cameraRunning = false;

let trackingEnabled = true;


/* =========================================================
   FEATURE STATE
========================================================= */

let showTargetBox = true;

let objectDetectionEnabled = false;

let facePrivacy = false;

let signLanguageMode = false;

let objectMode = true;


/* =========================================================
   RESULTS
========================================================= */

let latestHandResults = null;

let latestFaceResults = null;

let latestObjectResults = null;


/* =========================================================
   PERFORMANCE
========================================================= */

let lastVideoTime = -1;

let lastHandDetection = 0;

let lastObjectDetection = 0;

let lastFaceDetection = 0;

let previousIndex = null;

let frameCounter = 0;

let lastFPSUpdate =
    performance.now();


/*
   Hand tracking.

   30 FPS gives a good balance between
   responsiveness and CPU/GPU usage.
*/

const HAND_INTERVAL =
    1000 / 30;


/*
   Object recognition doesn't need
   to run 30 times per second.

   Around 8 FPS is enough to feel live.
*/

const OBJECT_INTERVAL =
    1000 / 8;


/*
   Face detection can be slower.
*/

const FACE_INTERVAL =
    1000 / 10;


/* =========================================================
   MODEL URLS
========================================================= */

const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";


const GESTURE_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";


const FACE_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task";


/*
   EfficientDet Lite 0.

   This is a lightweight general object detector
   suitable for browser use.
*/

const OBJECT_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0.tflite";


/* =========================================================
   VIRTUAL OBJECTS
========================================================= */

const virtualObjects = [];

let grabbedObject = null;

let nextObjectID = 1;


/* =========================================================
   INITIALIZE MODELS
========================================================= */

async function createModels() {

    statusText.textContent =
        "Loading SPECTRA vision systems...";


    try {

        const vision =
            await FilesetResolver.forVisionTasks(
                WASM_URL
            );


        /* ---------------------------------------------
           HAND / GESTURE
        --------------------------------------------- */

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


        /* ---------------------------------------------
           FACE
        --------------------------------------------- */

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


        /* ---------------------------------------------
           OBJECT DETECTOR
        --------------------------------------------- */

        objectDetector =
            await ObjectDetector.createFromOptions(
                vision,
                {

                    baseOptions: {

                        modelAssetPath:
                            OBJECT_MODEL_URL,

                        delegate:
                            "GPU"
                    },


                    runningMode:
                        "VIDEO",


                    maxResults:
                        8,


                    scoreThreshold:
                        0.45
                }
            );


        statusText.textContent =
            "SPECTRA vision systems ready.";


        trackingButton.disabled =
            false;


    } catch (error) {

        console.error(
            error
        );


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

                    width: {
                        ideal: 640
                    },


                    height: {
                        ideal: 360
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


        if (
            !gestureRecognizer ||
            !faceDetector ||
            !objectDetector
        ) {

            await createModels();
        }


        createInitialObject();


        lastVideoTime =
            -1;


        requestAnimationFrame(
            detectionLoop
        );


    } catch (error) {

        console.error(
            error
        );


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

            latestHandResults =
                null;


            grabbedObject =
                null;
        }
    }
);


/* =========================================================
   TARGET BOX
========================================================= */

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


/* =========================================================
   OBJECT DETECTION BUTTON
========================================================= */

objectButton.addEventListener(
    "click",
    () => {

        objectDetectionEnabled =
            !objectDetectionEnabled;


        objectButton.textContent =
            objectDetectionEnabled
                ? "OBJECT DETECTION ON"
                : "OBJECT DETECTION OFF";


        objectButton.classList.toggle(
            "active",
            objectDetectionEnabled
        );


        if (!objectDetectionEnabled) {

            latestObjectResults =
                null;
        }
    }
);


/* =========================================================
   FACE PRIVACY
========================================================= */

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


        if (!facePrivacy) {

            latestFaceResults =
                null;
        }
    }
);


/* =========================================================
   SIGN LANGUAGE
========================================================= */

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


/* =========================================================
   OBJECT MODE
========================================================= */

objectModeButton.addEventListener(
    "click",
    () => {

        objectMode =
            !objectMode;


        objectModeButton.textContent =
            objectMode
                ? "OBJECT MODE ON"
                : "OBJECT MODE OFF";


        objectModeButton.classList.toggle(
            "active",
            objectMode
        );
    }
);


/* =========================================================
   MAIN LOOP
========================================================= */

function detectionLoop() {

    if (!cameraRunning) {

        return;
    }


    const now =
        performance.now();


    /*
       Only process a frame when the video
       actually has a new frame.
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


            /* -----------------------------------------
               HANDS
            ----------------------------------------- */

            if (
                trackingEnabled &&
                gestureRecognizer &&
                now -
                lastHandDetection >=
                HAND_INTERVAL
            ) {

                lastHandDetection =
                    now;


                detectHands(
                    now
                );
            }


            /* -----------------------------------------
               OBJECTS
            ----------------------------------------- */

            if (
                objectDetectionEnabled &&
                objectDetector &&
                now -
                lastObjectDetection >=
                OBJECT_INTERVAL
            ) {

                lastObjectDetection =
                    now;


                detectObjects(
                    now
                );
            }


            /* -----------------------------------------
               FACE
            ----------------------------------------- */

            if (
                facePrivacy &&
                faceDetector &&
                now -
                lastFaceDetection >=
                FACE_INTERVAL
            ) {

                lastFaceDetection =
                    now;


                detectFace(
                    now
                );
            }
        }
    }


    /*
       Rendering happens every browser frame.

       AI doesn't need to run every render frame.
       This keeps movement visually smoother.
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

        latestHandResults =
            gestureRecognizer.recognizeForVideo(
                video,
                timestamp
            );


        processHands(
            latestHandResults
        );


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

function detectFace(
    timestamp
) {

    try {

        latestFaceResults =
            faceDetector.detectForVideo(
                video,
                timestamp
            );


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

function detectObjects(
    timestamp
) {

    try {

        latestObjectResults =
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
   PROCESS HANDS
========================================================= */

function processHands(
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
       0.70 = much more responsive
       than our old 0.35.
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


    /* ---------------------------------------------
       PINCH
    --------------------------------------------- */

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


    /* ---------------------------------------------
       GESTURE
    --------------------------------------------- */

    let gesture =
        getGesture(
            results,
            0
        );


    if (pinch) {

        gesture =
            "PINCH";
    }


    /* ---------------------------------------------
       SIGN LANGUAGE
    --------------------------------------------- */

    if (
        signLanguageMode
    ) {

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


    /* ---------------------------------------------
       OBJECT INTERACTION
    --------------------------------------------- */

    if (objectMode) {

        updateObjectInteraction(
            smoothed.x,
            smoothed.y,
            pinch
        );
    }
}


/* =========================================================
   GET GESTURE
========================================================= */

function getGesture(
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
       I LOVE YOU
    */

    if (
        index &&
        !middle &&
        !ring &&
        pinky
    ) {

        return "I LOVE YOU";
    }


    /*
       V
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
       B
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
       I
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
       FIST / A-style approximation
    */

    if (
        !index &&
        !middle &&
        !ring &&
        !pinky
    ) {

        return "A / FIST";
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
        landmarks[
            tipIndex
        ];


    const pip =
        landmarks[
            pipIndex
        ];


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
   OBJECT INTERACTION
========================================================= */

function updateObjectInteraction(
    x,
    y,
    pinch
) {

    /*
       Pinch starts.
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
       Move.
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
       Release.
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
   FIND VIRTUAL OBJECT
========================================================= */

function findObjectAt(
    x,
    y
) {

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
   CREATE VIRTUAL OBJECT
========================================================= */

function createVirtualObject(
    x,
    y,
    type = "BOX"
) {

    const object = {

        id:
            nextObjectID++,

        x,

        y,

        width:
            110,

        height:
            110,

        rotation:
            0,

        type,

        grabbed:
            false
    };


    virtualObjects.push(
        object
    );


    return object;
}


/* =========================================================
   INITIAL OBJECT
========================================================= */

function createInitialObject() {

    if (
        virtualObjects.length === 0
    ) {

        createVirtualObject(
            canvas.width / 2,
            canvas.height / 2
        );
    }
}


/* =========================================================
   RENDER
========================================================= */

function render() {

    clearOverlay();


    /*
       FACE PRIVACY
    */

    if (facePrivacy) {

        drawFacePrivacy();
    }


    /*
       OBJECT RECOGNITION
    */

    if (
        objectDetectionEnabled &&
        latestObjectResults
    ) {

        drawRecognizedObjects();
    }


    /*
       VIRTUAL OBJECTS
    */

    if (objectMode) {

        virtualObjects.forEach(
            drawVirtualObject
        );
    }


    /*
       HANDS
    */

    if (
        trackingEnabled &&
        latestHandResults &&
        latestHandResults.landmarks
    ) {

        latestHandResults.landmarks.forEach(
            (landmarks, index) => {

                drawHand(
                    landmarks,
                    index
                );
            }
        );


        const primaryHand =
            latestHandResults
                .landmarks[0];


        if (primaryHand) {

            const indexFinger =
                primaryHand[8];


            const thumb =
                primaryHand[4];


            const x =
                indexFinger.x *
                canvas.width;


            const y =
                indexFinger.y *
                canvas.height;


            const pinch =
                distance(
                    indexFinger,
                    thumb
                ) <
                0.075;


            drawCursor(
                x,
                y,
                pinch
            );


            if (showTargetBox) {

                drawTargetBox(
                    primaryHand
                );
            }
        }
    }
}


/* =========================================================
   DRAW RECOGNIZED OBJECTS
========================================================= */

function drawRecognizedObjects() {

    if (
        !latestObjectResults ||
        !latestObjectResults.detections
    ) {

        return;
    }


    latestObjectResults.detections.forEach(
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


            let label =
                "OBJECT";


            let confidence =
                0;


            if (
                detection.categories &&
                detection.categories.length
            ) {

                const category =
                    detection.categories[0];


                if (
                    category.categoryName
                ) {

                    label =
                        category.categoryName;
                }


                confidence =
                    category.score || 0;
            }


            const confidenceText =
                Math.round(
                    confidence * 100
                ) +
                "%";


            /* -----------------------------------------
               BOX
            ----------------------------------------- */

            ctx.save();


            ctx.strokeStyle =
                "rgba(255,255,255,0.85)";


            ctx.lineWidth =
                1.5;


            ctx.strokeRect(
                x,
                y,
                width,
                height
            );


            /* -----------------------------------------
               LABEL BACKGROUND
            ----------------------------------------- */

            const labelText =
                label.toUpperCase() +
                " // " +
                confidenceText;


            ctx.font =
                "11px Courier New";


            const textWidth =
                ctx.measureText(
                    labelText
                ).width;


            ctx.fillStyle =
                "rgba(0,0,0,0.8)";


            ctx.fillRect(
                x,
                Math.max(
                    0,
                    y - 22
                ),
                textWidth + 12,
                18
            );


            /* -----------------------------------------
               LABEL
            ----------------------------------------- */

            ctx.fillStyle =
                "#ffffff";


            ctx.fillText(
                labelText,
                x + 6,
                Math.max(
                    13,
                    y - 9
                )
            );


            ctx.restore();
        }
    );
}


/* =========================================================
   VIRTUAL OBJECT
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
            ? "#ffffff"
            : "rgba(255,255,255,0.75)";


    ctx.fillStyle =
        object.grabbed
            ? "rgba(255,255,255,0.08)"
            : "rgba(255,255,255,0.03)";


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


    const s =
        12;


    /*
       Top-left corner.
    */

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


    /*
       Bottom-right corner.
    */

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


            const padding =
                25;


            const x =
                Math.max(
                    0,
                    box.originX -
                    padding
                );


            const y =
                Math.max(
                    0,
                    box.originY -
                    padding
                );


            const width =
                Math.min(
                    canvas.width - x,
                    box.width +
                    padding * 2
                );


            const height =
                Math.min(
                    canvas.height - y,
                    box.height +
                    padding * 2
                );


            /*
               Blur the corresponding
               camera region.
            */

            ctx.save();


            ctx.filter =
                "blur(24px)";


            ctx.drawImage(
                video,

                x,
                y,
                width,
                height,

                x,
                y,
                width,
                height
            );


            ctx.restore();


            /*
               Dark translucent layer
               to strengthen privacy.
            */

            ctx.save();


            ctx.fillStyle =
                "rgba(0,0,0,0.12)";


            ctx.fillRect(
                x,
                y,
                width,
                height
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


    ctx.setLineDash([]);


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

        fpsValue.textContent =
            Math.round(
                frameCounter *
                1000 /
                elapsed
            );


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


    grabbedObject =
        null;
}


/* =========================================================
   RESET
========================================================= */

function resetInterface() {

    latestHandResults =
        null;


    latestFaceResults =
        null;


    latestObjectResults =
        null;


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


objectModeButton.classList.add(
    "active"
);


createModels();
