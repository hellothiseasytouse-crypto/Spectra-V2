import {
    FilesetResolver,
    GestureRecognizer,
    FaceDetector,
    ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


/* =========================================================
   SPECTRA
   HAND VFX / FACE / GESTURES / FPS BUILD
========================================================= */


/* =========================================================
   ELEMENTS
========================================================= */

const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");

if (!video || !canvas) {
    throw new Error("SPECTRA: camera or overlay canvas missing.");
}

const ctx = canvas.getContext("2d");


/* =========================================================
   SAFE UI FINDER
========================================================= */

function findElement(...names) {

    for (const name of names) {

        const el = document.getElementById(name);

        if (el) return el;
    }

    return null;
}


const startBtn =
    findElement("startBtn", "startCamera");

const trackingBtn =
    findElement("trackingBtn", "handTrackingBtn");

const targetBtn =
    findElement("targetBtn", "targetBoxBtn");

const objectBtn =
    findElement("objectBtn", "objectDetectionBtn");

const privacyBtn =
    findElement("privacyBtn", "facePrivacyBtn");

const signBtn =
    findElement("signBtn", "signLanguageBtn");

const filterBtn =
    findElement("filterBtn", "faceFilterBtn");

const shapeBtn =
    findElement("shapeBtn", "shapesBtn");


/* =========================================================
   CREATE / FIND GESTURE DISPLAY
========================================================= */

function createStatusElement(id, text) {

    let el = document.getElementById(id);

    if (el) return el;

    const controls =
        document.querySelector(".controls");

    if (!controls) return null;

    el = document.createElement("button");

    el.id = id;
    el.textContent = text;

    controls.appendChild(el);

    return el;
}


const gestureDisplay =
    createStatusElement(
        "gestureDisplay",
        "GESTURE // WAITING"
    );


const fpsBoostBtn =
    createStatusElement(
        "fpsBoostBtn",
        "FPS BOOST OFF"
    );


const shapeDisplay =
    createStatusElement(
        "shapeDisplay",
        "VFX // READY"
    );


/* =========================================================
   MEDIAPIPE
========================================================= */

const WASM =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";


const GESTURE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";


const FACE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task";


const OBJECT_MODEL =
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite";


let vision = null;

let gestureRecognizer = null;

let faceDetector = null;

let objectDetector = null;


/* =========================================================
   CAMERA STATE
========================================================= */

let cameraRunning = false;

let handTracking = true;

let targetVisible = true;

let objectDetection = false;

let signLanguage = false;

let shapesEnabled = true;


/* =========================================================
   FACE STATE
========================================================= */

let privacyMode = "OFF";

let faceFilter = "OFF";


/* =========================================================
   PERFORMANCE
========================================================= */

const isMobile =
    /Android|iPhone|iPad|iPod/i.test(
        navigator.userAgent
    ) ||
    window.innerWidth < 800;


let fpsBoost = false;


/*
    Hand tracking ALWAYS gets priority.

    The other systems are deliberately slower.
*/

const PERFORMANCE_NORMAL = isMobile
    ? {
        hands: 55,
        face: 180,
        objects: 400
    }
    : {
        hands: 32,
        face: 110,
        objects: 250
    };


const PERFORMANCE_BOOST = isMobile
    ? {
        hands: 65,
        face: 999999,
        objects: 999999
    }
    : {
        hands: 45,
        face: 999999,
        objects: 999999
    };


let performance =
    PERFORMANCE_NORMAL;


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

let frames = 0;

let fps = 0;

let fpsTimer =
    performanceNow();


function performanceNow() {
    return window.performance.now();
}


/* =========================================================
   HAND VFX STATE
========================================================= */

let vfxMode = "NONE";

let vfxEnergy = 0;

let vfxRotation = 0;

let vfxPulse = 0;


/* =========================================================
   START CAMERA
========================================================= */

async function startCamera() {

    if (cameraRunning) return;

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


        video.srcObject =
            stream;

        video.muted = true;
        video.playsInline = true;

        await video.play();


        canvas.width =
            video.videoWidth || 640;

        canvas.height =
            video.videoHeight || 360;


        cameraRunning = true;


        if (startBtn) {

            startBtn.textContent =
                "CAMERA RUNNING";
        }


        requestAnimationFrame(render);


        /*
            IMPORTANT:

            Only the HAND model loads
            at startup.

            Face/object models are lazy-loaded.
        */

        await loadVision();

    } catch (error) {

        console.error(
            "SPECTRA CAMERA ERROR:",
            error
        );


        if (startBtn) {

            startBtn.textContent =
                "CAMERA ERROR";
        }
    }
}


/* =========================================================
   LOAD CORE VISION
========================================================= */

async function loadVision() {

    if (vision) return;

    try {

        vision =
            await FilesetResolver
                .forVisionTasks(WASM);


        await loadGestureRecognizer();

    } catch (error) {

        console.error(
            "SPECTRA VISION ERROR:",
            error
        );
    }
}


/* =========================================================
   HAND MODEL
========================================================= */

async function loadGestureRecognizer() {

    if (gestureRecognizer) return;

    try {

        gestureRecognizer =
            await GestureRecognizer
                .createFromOptions(
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


        console.log(
            "SPECTRA // HAND AI READY"
        );

    } catch (error) {

        console.error(
            "HAND AI ERROR:",
            error
        );
    }
}


/* =========================================================
   FACE MODEL
========================================================= */

async function loadFaceDetector() {

    if (faceDetector) return;

    if (!vision) {
        await loadVision();
    }

    try {

        faceDetector =
            await FaceDetector
                .createFromOptions(
                    vision,
                    {

                        baseOptions: {

                            modelAssetPath:
                                FACE_MODEL
                        },

                        runningMode:
                            "VIDEO",

                        minDetectionConfidence:
                            0.5
                    }
                );


        console.log(
            "SPECTRA // FACE AI READY"
        );

    } catch (error) {

        console.error(
            "FACE AI ERROR:",
            error
        );
    }
}


/* =========================================================
   OBJECT MODEL
========================================================= */

async function loadObjectDetector() {

    if (objectDetector) return;

    if (!vision) {
        await loadVision();
    }

    try {

        objectDetector =
            await ObjectDetector
                .createFromOptions(
                    vision,
                    {

                        baseOptions: {

                            modelAssetPath:
                                OBJECT_MODEL
                        },

                        runningMode:
                            "VIDEO",

                        maxResults:
                            isMobile
                                ? 3
                                : 5,

                        scoreThreshold:
                            0.5
                    }
                );


        console.log(
            "SPECTRA // OBJECT AI READY"
        );

    } catch (error) {

        console.error(
            "OBJECT AI ERROR:",
            error
        );
    }
}


/* =========================================================
   FPS BOOST BUTTON
========================================================= */

if (fpsBoostBtn) {

    fpsBoostBtn.onclick = () => {

        fpsBoost =
            !fpsBoost;


        performance =
            fpsBoost
                ? PERFORMANCE_BOOST
                : PERFORMANCE_NORMAL;


        if (fpsBoost) {

            fpsBoostBtn.textContent =
                "FPS BOOST ON";


            /*
                Expensive systems OFF.
            */

            objectDetection =
                false;

            privacyMode =
                "OFF";

            faceFilter =
                "OFF";


            /*
                Hand tracking stays ON.
            */

            handTracking =
                true;


            /*
                Keep shapes.
            */

            shapesEnabled =
                true;


            targetVisible =
                false;


            if (objectBtn) {
                objectBtn.textContent =
                    "OBJECT DETECTION OFF";
            }


            if (privacyBtn) {
                privacyBtn.textContent =
                    "PRIVACY OFF";
            }


            if (filterBtn) {
                filterBtn.textContent =
                    "FILTER OFF";
            }


            if (targetBtn) {
                targetBtn.textContent =
                    "TARGET BOX OFF";
            }


        } else {

            fpsBoostBtn.textContent =
                "FPS BOOST OFF";


            targetVisible =
                true;


            if (targetBtn) {
                targetBtn.textContent =
                    "TARGET BOX ON";
            }
        }
    };
}


/* =========================================================
   NORMAL BUTTONS
========================================================= */

if (startBtn) {
    startBtn.onclick =
        startCamera;
}


if (trackingBtn) {

    trackingBtn.onclick =
        () => {

            handTracking =
                !handTracking;


            trackingBtn.textContent =
                handTracking
                    ? "HAND TRACKING ON"
                    : "HAND TRACKING OFF";
        };
}


if (targetBtn) {

    targetBtn.onclick =
        () => {

            targetVisible =
                !targetVisible;


            targetBtn.textContent =
                targetVisible
                    ? "TARGET BOX ON"
                    : "TARGET BOX OFF";
        };
}


/* =========================================================
   FACE PRIVACY
========================================================= */

if (privacyBtn) {

    privacyBtn.onclick =
        async () => {

            if (fpsBoost) return;


            const modes = [
                "OFF",
                "PIXEL"
            ];


            const index =
                modes.indexOf(
                    privacyMode
                );


            privacyMode =
                modes[
                    (index + 1) %
                    modes.length
                ];


            privacyBtn.textContent =
                `PRIVACY ${privacyMode}`;


            if (
                privacyMode !==
                "OFF"
            ) {

                await loadFaceDetector();
            }
        };
}


/* =========================================================
   FACE FILTER
========================================================= */

if (filterBtn) {

    filterBtn.onclick =
        async () => {

            if (fpsBoost) return;


            const filters = [
                "OFF",
                "CYBER",
                "SCAN",
                "CROWN"
            ];


            const index =
                filters.indexOf(
                    faceFilter
                );


            faceFilter =
                filters[
                    (index + 1) %
                    filters.length
                ];


            filterBtn.textContent =
                `FILTER ${faceFilter}`;


            if (
                faceFilter !==
                "OFF"
            ) {

                await loadFaceDetector();
            }
        };
}


/* =========================================================
   OBJECT DETECTION
========================================================= */

if (objectBtn) {

    objectBtn.onclick =
        async () => {

            if (fpsBoost) return;


            objectDetection =
                !objectDetection;


            if (objectDetection) {

                objectBtn.textContent =
                    "OBJECT AI LOADING...";


                await loadObjectDetector();


                if (objectDetector) {

                    objectBtn.textContent =
                        "OBJECT DETECTION ON";

                } else {

                    objectDetection =
                        false;

                    objectBtn.textContent =
                        "OBJECT DETECTION OFF";
                }

            } else {

                objectBtn.textContent =
                    "OBJECT DETECTION OFF";
            }
        };
}


/* =========================================================
   SIGN LANGUAGE
========================================================= */

if (signBtn) {

    signBtn.onclick =
        () => {

            signLanguage =
                !signLanguage;


            signBtn.textContent =
                signLanguage
                    ? "SIGN LANGUAGE ON"
                    : "SIGN LANGUAGE OFF";
        };
}


/* =========================================================
   SHAPES
========================================================= */

if (shapeBtn) {

    shapeBtn.onclick =
        () => {

            shapesEnabled =
                !shapesEnabled;


            shapeBtn.textContent =
                shapesEnabled
                    ? "SHAPES ON"
                    : "SHAPES OFF";
        };
}


/* =========================================================
   MAIN LOOP
========================================================= */

function render(timestamp) {

    if (!cameraRunning) return;


    /*
        CAMERA
    */

    drawCamera();


    /*
        HAND AI = TOP PRIORITY
    */

    if (handTracking) {

        runHands(timestamp);
    }


    /*
        FACE AI
    */

    if (
        !fpsBoost &&
        (
            privacyMode !== "OFF" ||
            faceFilter !== "OFF"
        )
    ) {

        runFace(timestamp);
    }


    /*
        OBJECT AI
    */

    if (
        !fpsBoost &&
        objectDetection
    ) {

        runObjects(timestamp);
    }


    /*
        FACE EFFECTS
    */

    if (!fpsBoost) {

        drawFacePrivacy();

        drawFaceFilter();

        drawObjects();
    }


    /*
        MAIN NEW FEATURE:
        HAND VFX
    */

    if (
        shapesEnabled &&
        handTracking
    ) {

        drawHandVFX(timestamp);
    }


    /*
        HAND SKELETON
    */

    drawHands();


    /*
        FPS
    */

    updateFPS();


    requestAnimationFrame(
        render
    );
}


/* =========================================================
   CAMERA DRAW
========================================================= */

function drawCamera() {

    ctx.save();


    /*
        Mirror the front camera.
    */

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


/* =========================================================
   HAND DETECTION
========================================================= */

function runHands(timestamp) {

    if (!gestureRecognizer) return;


    if (
        timestamp -
        lastHandTime <
        performance.hands
    ) {
        return;
    }


    lastHandTime =
        timestamp;


    try {

        handResults =
            gestureRecognizer
                .recognizeForVideo(
                    video,
                    timestamp
                );


        updateGesture();


    } catch (error) {

        console.error(
            "HAND DETECTION:",
            error
        );
    }
}


/* =========================================================
   FACE DETECTION
========================================================= */

function runFace(timestamp) {

    if (!faceDetector) return;


    if (
        timestamp -
        lastFaceTime <
        performance.face
    ) {
        return;
    }


    lastFaceTime =
        timestamp;


    try {

        faceResults =
            faceDetector
                .detectForVideo(
                    video,
                    timestamp
                );

    } catch (error) {

        console.error(
            "FACE DETECTION:",
            error
        );
    }
}


/* =========================================================
   OBJECT DETECTION
========================================================= */

function runObjects(timestamp) {

    if (!objectDetector) return;


    if (
        timestamp -
        lastObjectTime <
        performance.objects
    ) {
        return;
    }


    lastObjectTime =
        timestamp;


    try {

        objectResults =
            objectDetector
                .detectForVideo(
                    video,
                    timestamp
                );

    } catch (error) {

        console.error(
            "OBJECT DETECTION:",
            error
        );
    }
}


/* =========================================================
   GESTURE DISPLAY
========================================================= */

function updateGesture() {

    if (
        !handResults ||
        !handResults.gestures
    ) {

        setGesture(
            "NONE"
        );

        return;
    }


    const hands =
        handResults.gestures;


    if (!hands.length) {

        setGesture(
            "NONE"
        );

        return;
    }


    /*
        MediaPipe's built-in gesture.
    */

    const first =
        hands[0]?.[0];


    let gesture =
        first?.categoryName ||
        "TRACKING";


    /*
        Improve naming.
    */

    const names = {

        Open_Palm:
            "OPEN PALM",

        Closed_Fist:
            "FIST",

        Pointing_Up:
            "POINT",

        Thumb_Up:
            "THUMBS UP",

        Thumb_Down:
            "THUMBS DOWN",

        Victory:
            "VICTORY",

        ILoveYou:
            "I LOVE YOU"
    };


    gesture =
        names[gesture] ||
        gesture;


    setGesture(
        gesture
    );
}


/* =========================================================
   SET GESTURE UI
========================================================= */

function setGesture(value) {

    if (!gestureDisplay) return;

    gestureDisplay.textContent =
        `GESTURE // ${value}`;
}


/* =========================================================
   THE NEW SHAPE SYSTEM
========================================================= */

function drawHandVFX(timestamp) {

    const hands =
        handResults?.landmarks ||
        [];


    if (!hands.length) {

        setShapeStatus(
            "VFX // WAITING FOR HANDS"
        );

        return;
    }


    /*
        ONE HAND
        = small floating energy shape
    */

    if (hands.length === 1) {

        const hand =
            hands[0];


        const index =
            point(hand[8]);


        const thumb =
            point(hand[4]);


        const palm =
            point(hand[9]);


        const pinch =
            distance(
                index,
                thumb
            ) <
            45;


        if (pinch) {

            drawSingleHandPortal(
                palm,
                index,
                timestamp
            );


            setShapeStatus(
                "VFX // PINCH PORTAL"
            );

        } else {

            drawSingleHandOrb(
                index,
                timestamp
            );


            setShapeStatus(
                "VFX // HAND FIELD"
            );
        }


        return;
    }


    /*
        TWO HANDS
        = MAIN REFERENCE-STYLE EFFECT
    */

    const left =
        getPalmCenter(
            hands[0]
        );


    const right =
        getPalmCenter(
            hands[1]
        );


    const center = {

        x:
            (left.x +
            right.x) / 2,

        y:
            (left.y +
            right.y) / 2
    };


    const dx =
        right.x -
        left.x;


    const dy =
        right.y -
        left.y;


    const distanceBetween =
        Math.hypot(
            dx,
            dy
        );


    const angle =
        Math.atan2(
            dy,
            dx
        );


    vfxRotation =
        angle;


    /*
        Smooth energy based on
        distance between hands.
    */

    const targetEnergy =
        Math.min(
            1,
            distanceBetween /
            450
        );


    vfxEnergy +=
        (
            targetEnergy -
            vfxEnergy
        ) * 0.12;


    vfxPulse =
        Math.sin(
            timestamp * 0.006
        );


    /*
        Detect gestures.
    */

    const gesture1 =
        getGesture(0);


    const gesture2 =
        getGesture(1);


    /*
        DIFFERENT HAND COMBINATIONS
    */

    if (
        gesture1 === "Victory" ||
        gesture2 === "Victory"
    ) {

        vfxMode =
            "TRIANGLE";

    } else if (
        gesture1 === "Closed_Fist" &&
        gesture2 === "Closed_Fist"
    ) {

        vfxMode =
            "CORE";

    } else if (
        isPinching(hands[0]) &&
        isPinching(hands[1])
    ) {

        vfxMode =
            "PORTAL";

    } else if (
        gesture1 === "Open_Palm" ||
        gesture2 === "Open_Palm"
    ) {

        vfxMode =
            "RIBBON";

    } else {

        vfxMode =
            "DIAMOND";
    }


    /*
        DRAW THE ACTUAL VIRTUAL OBJECT
    */

    ctx.save();

    ctx.translate(
        center.x,
        center.y
    );


    ctx.rotate(
        angle
    );


    const width =
        Math.max(
            80,
            distanceBetween
        );


    const height =
        Math.max(
            45,
            distanceBetween *
            0.32
        );


    /*
        Outer geometry
    */

    drawVFXFrame(
        width,
        height,
        vfxMode,
        timestamp
    );


    /*
        Inner geometry
    */

    drawVFXCore(
        width,
        height,
        vfxMode,
        timestamp
    );


    /*
        connecting lines to hands
    */

    ctx.restore();


    drawHandConnections(
        left,
        right,
        center,
        timestamp
    );


    setShapeStatus(
        `VFX // ${vfxMode}`
    );
}


/* =========================================================
   VFX OUTER FRAME
========================================================= */

function drawVFXFrame(
    width,
    height,
    mode,
    timestamp
) {

    ctx.save();


    const pulse =
        1 +
        Math.sin(
            timestamp * 0.005
        ) * 0.04;


    ctx.scale(
        pulse,
        pulse
    );


    ctx.strokeStyle =
        "rgba(255,255,255,0.85)";


    ctx.lineWidth =
        fpsBoost
            ? 2
            : 3;


    ctx.setLineDash([
        10,
        8
    ]);


    if (
        mode ===
        "TRIANGLE"
    ) {

        ctx.beginPath();

        ctx.moveTo(
            0,
            -height
        );

        ctx.lineTo(
            width,
            height
        );

        ctx.lineTo(
            -width,
            height
        );

        ctx.closePath();

        ctx.stroke();


    } else if (
        mode ===
        "CORE"
    ) {

        ctx.beginPath();

        ctx.rect(
            -width * 0.35,
            -height,
            width * 0.7,
            height * 2
        );

        ctx.stroke();


    } else if (
        mode ===
        "PORTAL"
    ) {

        ctx.beginPath();

        ctx.ellipse(
            0,
            0,
            width * 0.5,
            height,
            0,
            0,
            Math.PI * 2
        );

        ctx.stroke();


        ctx.beginPath();

        ctx.ellipse(
            0,
            0,
            width * 0.34,
            height * 0.65,
            0,
            0,
            Math.PI * 2
        );

        ctx.stroke();


    } else if (
        mode ===
        "RIBBON"
    ) {

        ctx.beginPath();

        ctx.moveTo(
            -width,
            0
        );


        ctx.quadraticCurveTo(
            -width * 0.3,
            -height * 2,
            0,
            0
        );


        ctx.quadraticCurveTo(
            width * 0.3,
            height * 2,
            width,
            0
        );


        ctx.stroke();


    } else {

        /*
            DIAMOND
        */

        ctx.beginPath();

        ctx.moveTo(
            -width,
            0
        );

        ctx.lineTo(
            0,
            -height
        );

        ctx.lineTo(
            width,
            0
        );

        ctx.lineTo(
            0,
            height
        );

        ctx.closePath();

        ctx.stroke();
    }


    ctx.setLineDash([]);


    ctx.restore();
}


/* =========================================================
   VFX CORE
========================================================= */

function drawVFXCore(
    width,
    height,
    mode,
    timestamp
) {

    ctx.save();


    const pulse =
        1 +
        Math.sin(
            timestamp * 0.009
        ) * 0.12;


    ctx.scale(
        pulse,
        pulse
    );


    ctx.strokeStyle =
        "rgba(255,255,255,0.95)";


    ctx.lineWidth =
        2;


    /*
        Central energy lines.
    */

    for (
        let i = -2;
        i <= 2;
        i++
    ) {

        const offset =
            i *
            height *
            0.22;


        ctx.beginPath();

        ctx.moveTo(
            -width * 0.5,
            offset
        );

        ctx.lineTo(
            width * 0.5,
            -offset
        );

        ctx.stroke();
    }


    /*
        Center core.
    */

    const radius =
        Math.max(
            8,
            height *
            0.16
        );


    ctx.beginPath();

    ctx.arc(
        0,
        0,
        radius,
        0,
        Math.PI * 2
    );

    ctx.stroke();


    /*
        Rotating mini square.
    */

    ctx.rotate(
        timestamp * 0.001
    );


    ctx.strokeRect(
        -radius * 0.65,
        -radius * 0.65,
        radius * 1.3,
        radius * 1.3
    );


    ctx.restore();
}


/* =========================================================
   HAND CONNECTIONS
========================================================= */

function drawHandConnections(
    left,
    right,
    center,
    timestamp
) {

    ctx.save();


    ctx.strokeStyle =
        "rgba(255,255,255,0.35)";


    ctx.lineWidth =
        1.5;


    ctx.setLineDash([
        6,
        10
    ]);


    ctx.beginPath();

    ctx.moveTo(
        left.x,
        left.y
    );

    ctx.lineTo(
        center.x,
        center.y
    );

    ctx.lineTo(
        right.x,
        right.y
    );

    ctx.stroke();


    ctx.setLineDash([]);


    /*
        Small tracking points.
    */

    for (
        const p
        of [left, right]
    ) {

        ctx.beginPath();

        ctx.arc(
            p.x,
            p.y,
            5 +
            Math.sin(
                timestamp * 0.006
            ) * 2,
            0,
            Math.PI * 2
        );

        ctx.stroke();
    }


    ctx.restore();
}


/* =========================================================
   SINGLE HAND ORB
========================================================= */

function drawSingleHandOrb(
    position,
    timestamp
) {

    const radius =
        22 +
        Math.sin(
            timestamp * 0.006
        ) * 5;


    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.8)";

    ctx.lineWidth = 2;


    ctx.beginPath();

    ctx.arc(
        position.x,
        position.y,
        radius,
        0,
        Math.PI * 2
    );

    ctx.stroke();


    ctx.beginPath();

    ctx.arc(
        position.x,
        position.y,
        radius * 0.45,
        0,
        Math.PI * 2
    );

    ctx.stroke();


    ctx.restore();
}


/* =========================================================
   SINGLE HAND PORTAL
========================================================= */

function drawSingleHandPortal(
    palm,
    finger,
    timestamp
) {

    const radius =
        35 +
        Math.sin(
            timestamp * 0.01
        ) * 6;


    ctx.save();


    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";


    ctx.lineWidth = 2;


    ctx.beginPath();

    ctx.ellipse(
        finger.x,
        finger.y,
        radius,
        radius * 0.55,
        timestamp * 0.002,
        0,
        Math.PI * 2
    );

    ctx.stroke();


    ctx.beginPath();

    ctx.moveTo(
        palm.x,
        palm.y
    );

    ctx.lineTo(
        finger.x,
        finger.y
    );

    ctx.stroke();


    ctx.restore();
}


/* =========================================================
   HAND SKELETON
========================================================= */

function drawHands() {

    if (!handTracking) return;


    const hands =
        handResults?.landmarks ||
        [];


    for (
        const hand
        of hands
    ) {

        const connections = [

            [0,1],
            [1,2],
            [2,3],
            [3,4],

            [0,5],
            [5,6],
            [6,7],
            [7,8],

            [5,9],
            [9,10],
            [10,11],
            [11,12],

            [9,13],
            [13,14],
            [14,15],
            [15,16],

            [13,17],
            [17,18],
            [18,19],
            [19,20],

            [0,17]
        ];


        ctx.save();


        ctx.strokeStyle =
            "rgba(255,255,255,0.65)";


        ctx.lineWidth =
            fpsBoost
                ? 1
                : 1.5;


        for (
            const [a,b]
            of connections
        ) {

            const p1 =
                point(hand[a]);

            const p2 =
                point(hand[b]);


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


        /*
            Don't overload mobile GPU
            with every joint in boost.
        */

        if (!fpsBoost) {

            ctx.fillStyle =
                "rgba(255,255,255,0.9)";


            for (
                const p
                of hand
            ) {

                const pos =
                    point(p);


                ctx.beginPath();

                ctx.arc(
                    pos.x,
                    pos.y,
                    2,
                    0,
                    Math.PI * 2
                );

                ctx.fill();
            }
        }


        /*
            Index cursor.
        */

        const index =
            point(hand[8]);


        ctx.strokeStyle =
            "rgba(255,255,255,0.9)";


        ctx.beginPath();

        ctx.arc(
            index.x,
            index.y,
            7,
            0,
            Math.PI * 2
        );

        ctx.stroke();


        /*
            Target box.
        */

        if (
            targetVisible &&
            !fpsBoost
        ) {

            const points =
                hand.map(point);


            const bounds =
                getBounds(points);


            ctx.setLineDash([
                5,
                5
            ]);


            ctx.strokeStyle =
                "rgba(255,255,255,0.3)";


            ctx.strokeRect(
                bounds.minX - 8,
                bounds.minY - 8,
                bounds.width + 16,
                bounds.height + 16
            );


            ctx.setLineDash([]);
        }


        ctx.restore();
    }
}


/* =========================================================
   FACE PIXEL PRIVACY
   KEEPING THE STYLE YOU LIKED
========================================================= */

function drawFacePrivacy() {

    if (
        privacyMode !==
        "PIXEL"
    ) {
        return;
    }


    if (
        !faceResults ||
        !faceResults.detections
    ) {
        return;
    }


    for (
        const detection
        of faceResults.detections
    ) {

        const box =
            detection.boundingBox;


        if (!box) continue;


        /*
            FaceDetector uses the original
            camera coordinates.

            Convert to mirrored canvas.
        */

        const x =
            canvas.width -
            box.originX -
            box.width;


        const y =
            box.originY;


        const w =
            box.width;


        const h =
            box.height;


        const padX =
            w * 0.12;


        const padY =
            h * 0.15;


        const px =
            x - padX;


        const py =
            y - padY;


        const pw =
            w + padX * 2;


        const ph =
            h + padY * 2;


        /*
            Pixel blocks.
        */

        const block =
            Math.max(
                12,
                Math.floor(
                    w / 8
                )
            );


        ctx.save();


        /*
            Dark translucent base.
        */

        ctx.fillStyle =
            "rgba(0,0,0,0.22)";


        ctx.fillRect(
            px,
            py,
            pw,
            ph
        );


        /*
            Pixel grid.
        */

        for (
            let yy = py;
            yy < py + ph;
            yy += block
        ) {

            for (
                let xx = px;
                xx < px + pw;
                xx += block
            ) {

                const even =
                    (
                        Math.floor(
                            (xx - px) /
                            block
                        ) +
                        Math.floor(
                            (yy - py) /
                            block
                        )
                    ) % 2;


                ctx.fillStyle =
                    even
                        ? "rgba(30,30,30,0.78)"
                        : "rgba(100,100,100,0.65)";


                ctx.fillRect(
                    xx,
                    yy,
                    block,
                    block
                );
            }
        }


        /*
            SPECTRA label.
        */

        ctx.fillStyle =
            "rgba(255,255,255,0.45)";


        ctx.font =
            "10px monospace";


        ctx.fillText(
            "SPECTRA VISION",
            px + 10,
            py + ph / 2
        );


        /*
            Corner brackets.
        */

        ctx.strokeStyle =
            "rgba(255,255,255,0.9)";


        ctx.lineWidth = 2;


        drawCorners(
            px,
            py,
            pw,
            ph
        );


        ctx.fillStyle =
            "#ffffff";


        ctx.font =
            "11px monospace";


        ctx.fillText(
            "FACE // PIXEL",
            px,
            py - 8
        );


        ctx.restore();
    }
}


/* =========================================================
   FACE FILTERS
========================================================= */

function drawFaceFilter() {

    if (
        faceFilter ===
        "OFF"
    ) {
        return;
    }


    if (
        !faceResults ||
        !faceResults.detections
    ) {
        return;
    }


    for (
        const detection
        of faceResults.detections
    ) {

        const box =
            detection.boundingBox;


        if (!box) continue;


        const x =
            canvas.width -
            box.originX -
            box.width;


        const y =
            box.originY;


        const w =
            box.width;


        const h =
            box.height;


        ctx.save();


        if (
            faceFilter ===
            "CYBER"
        ) {

            ctx.strokeStyle =
                "rgba(255,255,255,0.9)";

            ctx.lineWidth = 2;


            ctx.strokeRect(
                x - 8,
                y - 8,
                w + 16,
                h + 16
            );


            ctx.beginPath();

            ctx.moveTo(
                x,
                y + h * 0.45
            );

            ctx.lineTo(
                x + w,
                y + h * 0.45
            );

            ctx.stroke();


        } else if (
            faceFilter ===
            "SCAN"
        ) {

            ctx.strokeStyle =
                "rgba(255,255,255,0.85)";

            ctx.lineWidth = 2;


            for (
                let i = 0;
                i < 5;
                i++
            ) {

                const yy =
                    y +
                    (
                        h *
                        i /
                        5
                    );


                ctx.beginPath();

                ctx.moveTo(
                    x,
                    yy
                );

                ctx.lineTo(
                    x + w,
                    yy
                );

                ctx.stroke();
            }


        } else if (
            faceFilter ===
            "CROWN"
        ) {

            ctx.strokeStyle =
                "rgba(255,255,255,0.9)";

            ctx.lineWidth = 2;


            ctx.beginPath();

            ctx.moveTo(
                x + w * 0.1,
                y
            );

            ctx.lineTo(
                x + w * 0.25,
                y - h * 0.3
            );

            ctx.lineTo(
                x + w * 0.5,
                y
            );

            ctx.lineTo(
                x + w * 0.75,
                y - h * 0.3
            );

            ctx.lineTo(
                x + w * 0.9,
                y
            );

            ctx.stroke();
        }


        ctx.restore();
    }
}


/* =========================================================
   OBJECTS
========================================================= */

function drawObjects() {

    if (
        !objectDetection ||
        !objectResults
    ) {
        return;
    }


    for (
        const detection
        of objectResults.detections || []
    ) {

        const box =
            detection.boundingBox;


        const category =
            detection.categories?.[0];


        if (!box) continue;


        const x =
            canvas.width -
            box.originX -
            box.width;


        const y =
            box.originY;


        ctx.save();


        ctx.strokeStyle =
            "rgba(255,255,255,0.8)";


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
            "#ffffff";


        ctx.fillText(
            `${category?.categoryName || "OBJECT"} ${Math.round((category?.score || 0) * 100)}%`,
            x,
            Math.max(
                12,
                y - 5
            )
        );


        ctx.restore();
    }
}


/* =========================================================
   SIGN LANGUAGE
========================================================= */

function getSign(hand) {

    const index =
        fingerUp(hand, 8);


    const middle =
        fingerUp(hand, 12);


    const ring =
        fingerUp(hand, 16);


    const pinky =
        fingerUp(hand, 20);


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
        !index &&
        !middle &&
        !ring &&
        pinky
    ) {

        return "I";
    }


    if (
        index &&
        middle &&
        ring &&
        pinky
    ) {

        return "OPEN";
    }


    if (
        !index &&
        !middle &&
        !ring &&
        !pinky
    ) {

        return "FIST";
    }


    return "UNKNOWN";
}


function fingerUp(
    hand,
    tip
) {

    return (
        hand[tip].y <
        hand[tip - 2].y
    );
}


/* =========================================================
   GET GESTURE
========================================================= */

function getGesture(index) {

    return (
        handResults
            ?.gestures
            ?.[
                index
            ]
            ?.[
                0
            ]
            ?.categoryName ||
        "None"
    );
}


/* =========================================================
   PINCH
========================================================= */

function isPinching(hand) {

    const thumb =
        point(hand[4]);


    const index =
        point(hand[8]);


    return (
        distance(
            thumb,
            index
        ) < 45
    );
}


/* =========================================================
   PALM CENTER
========================================================= */

function getPalmCenter(hand) {

    const a =
        point(hand[0]);


    const b =
        point(hand[5]);


    const c =
        point(hand[9]);


    const d =
        point(hand[17]);


    return {

        x:
            (
                a.x +
                b.x +
                c.x +
                d.x
            ) / 4,

        y:
            (
                a.y +
                b.y +
                c.y +
                d.y
            ) / 4
    };
}


/* =========================================================
   LANDMARK → CANVAS
========================================================= */

function point(p) {

    return {

        /*
            Mirror X because the camera
            is mirrored.
        */

        x:
            (
                1 -
                p.x
            ) *
            canvas.width,

        y:
            p.y *
            canvas.height
    };
}


/* =========================================================
   DISTANCE
========================================================= */

function distance(a, b) {

    return Math.hypot(
        a.x - b.x,
        a.y - b.y
    );
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
            maxX -
            minX,

        height:
            maxY -
            minY
    };
}


/* =========================================================
   CORNER BRACKETS
========================================================= */

function drawCorners(
    x,
    y,
    width,
    height
) {

    const s = 18;


    ctx.beginPath();


    ctx.moveTo(
        x,
        y + s
    );

    ctx.lineTo(
        x,
        y
    );

    ctx.lineTo(
        x + s,
        y
    );


    ctx.moveTo(
        x + width - s,
        y
    );

    ctx.lineTo(
        x + width,
        y
    );

    ctx.lineTo(
        x + width,
        y + s
    );


    ctx.moveTo(
        x,
        y + height - s
    );

    ctx.lineTo(
        x,
        y + height
    );

    ctx.lineTo(
        x + s,
        y + height
    );


    ctx.moveTo(
        x + width - s,
        y + height
    );

    ctx.lineTo(
        x + width,
        y + height
    );

    ctx.lineTo(
        x + width,
        y + height - s
    );


    ctx.stroke();
}


/* =========================================================
   SHAPE STATUS
========================================================= */

function setShapeStatus(text) {

    if (!shapeDisplay) return;

    shapeDisplay.textContent =
        text;
}


/* =========================================================
   FPS
========================================================= */

function updateFPS() {

    frames++;


    const now =
        performanceNow();


    if (
        now -
        fpsTimer >=
        1000
    ) {

        fps =
            frames;


        frames =
            0;


        fpsTimer =
            now;


        const fpsElement =
            findElement(
                "fps",
                "fpsText"
            );


        if (fpsElement) {

            fpsElement.textContent =
                fps;
        }
    }
}


/* =========================================================
   STARTUP
========================================================= */

console.log(
    "================================"
);

console.log(
    "SPECTRA // READY"
);

console.log(
    isMobile
        ? "DEVICE // MOBILE"
        : "DEVICE // DESKTOP"
);

console.log(
    "HAND VFX // ENABLED"
);

console.log(
    "FACE PIXEL // AVAILABLE"
);

console.log(
    "FPS BOOST // AVAILABLE"
);

console.log(
    "================================"
);
