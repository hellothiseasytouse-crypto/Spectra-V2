import {
    FilesetResolver,
    GestureRecognizer,
    FaceDetector,
    ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


/* =========================================================
   SPECTRA
   HAND TRACKING / LANDMARK GEOMETRY / FACE / FPS
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
   CREATE / FIND STATUS ELEMENTS
========================================================= */

function createStatusElement(id, text) {

    let el =
        document.getElementById(id);

    if (el) return el;


    const controls =
        document.querySelector(".controls");

    if (!controls) return null;


    el =
        document.createElement("button");

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
        "GEOMETRY // READY"
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
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/object_detector.tflite";


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
   LANDMARK GEOMETRY STATE
========================================================= */

/*
    The old:

        RIBBON
        PORTAL
        CORE
        TRIANGLE
        ORB

    system is completely removed.

    Geometry is now controlled directly by
    MediaPipe hand landmarks.
*/


let geometryMode = 0;


const GEOMETRY_MODES = [
    "QUAD WARP",
    "DIAMOND",
    "SHARD",
    "FRAME"
];


const geometrySmooth =
    new Map();


const GEOMETRY_SMOOTHING = 0.22;


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
            video.videoWidth ||
            640;


        canvas.height =
            video.videoHeight ||
            360;


        cameraRunning = true;


        if (startBtn) {

            startBtn.textContent =
                "CAMERA RUNNING";
        }


        requestAnimationFrame(
            render
        );


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
                .forVisionTasks(
                    WASM
                );


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
   FPS BOOST
========================================================= */

if (fpsBoostBtn) {

    fpsBoostBtn.onclick =
        () => {

            fpsBoost =
                !fpsBoost;


            performance =
                fpsBoost
                    ? PERFORMANCE_BOOST
                    : PERFORMANCE_NORMAL;


            if (fpsBoost) {

                fpsBoostBtn.textContent =
                    "FPS BOOST ON";


                objectDetection =
                    false;


                privacyMode =
                    "OFF";


                faceFilter =
                    "OFF";


                handTracking =
                    true;


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
   START BUTTON
========================================================= */

if (startBtn) {

    startBtn.onclick =
        startCamera;
}


/* =========================================================
   HAND TRACKING BUTTON
========================================================= */

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


/* =========================================================
   TARGET BOX
========================================================= */

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
                    (
                        index + 1
                    ) %
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
                    (
                        index + 1
                    ) %
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
   SHAPE / GEOMETRY BUTTON
========================================================= */

if (shapeBtn) {

    shapeBtn.onclick =
        () => {

            /*
                First click after disabled:
                turn geometry back on.
            */

            if (!shapesEnabled) {

                shapesEnabled =
                    true;


                shapeBtn.textContent =
                    `SHAPE // ${GEOMETRY_MODES[geometryMode]}`;


                return;
            }


            /*
                Cycle geometry type.
            */

            geometryMode =
                (
                    geometryMode + 1
                ) %
                GEOMETRY_MODES.length;


            shapeBtn.textContent =
                `SHAPE // ${GEOMETRY_MODES[geometryMode]}`;
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
        HAND AI
    */

    if (handTracking) {

        runHands(
            timestamp
        );
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

        runFace(
            timestamp
        );
    }


    /*
        OBJECT AI
    */

    if (
        !fpsBoost &&
        objectDetection
    ) {

        runObjects(
            timestamp
        );
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
        LANDMARK GEOMETRY
    */

    if (
        shapesEnabled &&
        handTracking
    ) {

        drawHandVFX(
            timestamp
        );
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
        Mirror front camera.
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

function runHands(
    timestamp
) {

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

function runFace(
    timestamp
) {

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

function runObjects(
    timestamp
) {

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


    const first =
        hands[0]?.[0];


    let gesture =
        first?.categoryName ||
        "TRACKING";


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
   SET GESTURE
========================================================= */

function setGesture(
    value
) {

    if (!gestureDisplay) return;


    gestureDisplay.textContent =
        `GESTURE // ${value}`;
}


/* =========================================================
   NEW LANDMARK GEOMETRY
========================================================= */

function drawHandVFX(
    timestamp
) {

    const hands =
        handResults?.landmarks ||
        [];


    if (!hands.length) {

        setShapeStatus(
            "GEOMETRY // WAITING"
        );


        geometrySmooth.clear();


        return;
    }


    /*
        ONE HAND
    */

    if (hands.length === 1) {

        drawLandmarkGeometry(
            hands[0],
            timestamp
        );


        setShapeStatus(
            `GEOMETRY // ${GEOMETRY_MODES[geometryMode]}`
        );


        return;
    }


    /*
        TWO HANDS
    */

    drawTwoHandGeometry(
        hands[0],
        hands[1],
        timestamp
    );


    setShapeStatus(
        `GEOMETRY // ${GEOMETRY_MODES[geometryMode]}`
    );
}


/* =========================================================
   ONE HAND GEOMETRY
========================================================= */

function drawLandmarkGeometry(
    hand,
    timestamp
) {

    const wrist =
        smoothPoint(
            "wrist",
            point(hand[0])
        );


    const thumb =
        smoothPoint(
            "thumb",
            point(hand[4])
        );


    const index =
        smoothPoint(
            "index",
            point(hand[8])
        );


    const middle =
        smoothPoint(
            "middle",
            point(hand[12])
        );


    const ring =
        smoothPoint(
            "ring",
            point(hand[16])
        );


    const pinky =
        smoothPoint(
            "pinky",
            point(hand[20])
        );


    /*
        Normalized pinch distance.

        This is deliberately based on MediaPipe's
        normalized coordinates instead of pixels.
    */

    const pinchDistance =
        Math.hypot(
            hand[8].x -
            hand[4].x,

            hand[8].y -
            hand[4].y
        );


    const pinchAmount =
        Math.min(
            1,
            pinchDistance /
            0.22
        );


    ctx.save();


    if (
        geometryMode === 0
    ) {

        drawQuadWarp(
            wrist,
            thumb,
            index,
            pinky,
            pinchAmount,
            timestamp
        );


    } else if (
        geometryMode === 1
    ) {

        drawLandmarkDiamond(
            thumb,
            index,
            middle,
            ring,
            pinchAmount,
            timestamp
        );


    } else if (
        geometryMode === 2
    ) {

        drawLandmarkShard(
            thumb,
            index,
            middle,
            ring,
            pinky,
            wrist,
            pinchAmount,
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
            pinchAmount,
            timestamp
        );
    }


    ctx.restore();
}


/* =========================================================
   QUAD WARP
========================================================= */

function drawQuadWarp(
    wrist,
    thumb,
    index,
    pinky,
    pinchAmount,
    timestamp
) {

    /*
        Four actual MediaPipe-controlled corners.
    */

    const p1 =
        thumb;


    const p2 =
        index;


    const p3 =
        pinky;


    const p4 =
        wrist;


    const pulse =
        1 +
        Math.sin(
            timestamp * 0.004
        ) *
        0.025;


    const center =
        averagePoint(
            p1,
            p2,
            p3,
            p4
        );


    /*
        Pinching compresses the geometry.
    */

    const compression =
        0.82 +
        pinchAmount *
        0.18;


    const a =
        scaleAround(
            p1,
            center,
            compression *
            pulse
        );


    const b =
        scaleAround(
            p2,
            center,
            compression *
            pulse
        );


    const c =
        scaleAround(
            p3,
            center,
            compression *
            pulse
        );


    const d =
        scaleAround(
            p4,
            center,
            compression *
            pulse
        );


    /*
        OUTER QUAD
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";


    ctx.lineWidth =
        fpsBoost
            ? 2
            : 2.5;


    ctx.beginPath();


    ctx.moveTo(
        a.x,
        a.y
    );


    ctx.lineTo(
        b.x,
        b.y
    );


    ctx.lineTo(
        c.x,
        c.y
    );


    ctx.lineTo(
        d.x,
        d.y
    );


    ctx.closePath();


    ctx.stroke();


    /*
        INNER QUAD
    */

    const inner = [

        lerpPoint(
            a,
            center,
            0.22
        ),

        lerpPoint(
            b,
            center,
            0.22
        ),

        lerpPoint(
            c,
            center,
            0.22
        ),

        lerpPoint(
            d,
            center,
            0.22
        )
    ];


    ctx.strokeStyle =
        "rgba(255,255,255,0.45)";


    ctx.lineWidth = 1;


    ctx.beginPath();


    ctx.moveTo(
        inner[0].x,
        inner[0].y
    );


    ctx.lineTo(
        inner[1].x,
        inner[1].y
    );


    ctx.lineTo(
        inner[2].x,
        inner[2].y
    );


    ctx.lineTo(
        inner[3].x,
        inner[3].y
    );


    ctx.closePath();


    ctx.stroke();


    /*
        Deforming grid.
    */

    drawQuadGrid(
        a,
        b,
        c,
        d
    );


    drawGeometryPoint(a);

    drawGeometryPoint(b);

    drawGeometryPoint(c);

    drawGeometryPoint(d);


    /*
        Pinch center.
    */

    if (
        pinchAmount <
        0.35
    ) {

        ctx.strokeStyle =
            "rgba(255,255,255,0.95)";


        ctx.lineWidth = 2;


        ctx.beginPath();


        ctx.arc(
            center.x,
            center.y,
            8,
            0,
            Math.PI * 2
        );


        ctx.stroke();
    }
}


/* =========================================================
   QUAD GRID
========================================================= */

function drawQuadGrid(
    a,
    b,
    c,
    d
) {

    ctx.save();


    ctx.strokeStyle =
        "rgba(255,255,255,0.18)";


    ctx.lineWidth = 1;


    /*
        Horizontal lines.
    */

    for (
        let i = 1;
        i < 4;
        i++
    ) {

        const t =
            i / 4;


        const left =
            lerpPoint(
                a,
                d,
                t
            );


        const right =
            lerpPoint(
                b,
                c,
                t
            );


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
    }


    /*
        Vertical lines.
    */

    for (
        let i = 1;
        i < 4;
        i++
    ) {

        const t =
            i / 4;


        const top =
            lerpPoint(
                a,
                b,
                t
            );


        const bottom =
            lerpPoint(
                d,
                c,
                t
            );


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
    }


    ctx.restore();
}


/* =========================================================
   DIAMOND
========================================================= */

function drawLandmarkDiamond(
    thumb,
    index,
    middle,
    ring,
    pinchAmount,
    timestamp
) {

    const top =
        index;


    const right =
        middle;


    const bottom =
        ring;


    const left =
        thumb;


    const center =
        averagePoint(
            top,
            right,
            bottom,
            left
        );


    const compression =
        0.78 +
        pinchAmount *
        0.22;


    const points = [

        scaleAround(
            top,
            center,
            compression
        ),

        scaleAround(
            right,
            center,
            compression
        ),

        scaleAround(
            bottom,
            center,
            compression
        ),

        scaleAround(
            left,
            center,
            compression
        )
    ];


    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";


    ctx.lineWidth = 2.5;


    ctx.beginPath();


    ctx.moveTo(
        points[0].x,
        points[0].y
    );


    for (
        let i = 1;
        i < points.length;
        i++
    ) {

        ctx.lineTo(
            points[i].x,
            points[i].y
        );
    }


    ctx.closePath();


    ctx.stroke();


    /*
        Internal deformation lines.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.28)";


    ctx.lineWidth = 1;


    ctx.beginPath();


    ctx.moveTo(
        points[0].x,
        points[0].y
    );


    ctx.lineTo(
        points[2].x,
        points[2].y
    );


    ctx.moveTo(
        points[1].x,
        points[1].y
    );


    ctx.lineTo(
        points[3].x,
        points[3].y
    );


    ctx.stroke();


    /*
        Center.
    */

    ctx.beginPath();


    ctx.arc(
        center.x,
        center.y,
        5,
        0,
        Math.PI * 2
    );


    ctx.stroke();


    for (
        const p
        of points
    ) {

        drawGeometryPoint(
            p
        );
    }
}


/* =========================================================
   SHARD
========================================================= */

function drawLandmarkShard(
    thumb,
    index,
    middle,
    ring,
    pinky,
    wrist,
    pinchAmount,
    timestamp
) {

    /*
        Six actual hand landmarks.
    */

    const points = [

        thumb,
        index,
        middle,
        ring,
        pinky,
        wrist
    ];


    const center =
        averagePoint(
            ...points
        );


    const compression =
        0.82 +
        pinchAmount *
        0.18;


    const warped =
        points.map(
            p =>
                scaleAround(
                    p,
                    center,
                    compression
                )
        );


    /*
        Main irregular geometry.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";


    ctx.lineWidth = 2.5;


    ctx.beginPath();


    ctx.moveTo(
        warped[0].x,
        warped[0].y
    );


    for (
        let i = 1;
        i < warped.length;
        i++
    ) {

        ctx.lineTo(
            warped[i].x,
            warped[i].y
        );
    }


    ctx.closePath();


    ctx.stroke();


    /*
        Internal triangulation.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.3)";


    ctx.lineWidth = 1;


    for (
        let i = 0;
        i < warped.length;
        i++
    ) {

        const next =
            warped[
                (
                    i + 2
                ) %
                warped.length
            ];


        ctx.beginPath();


        ctx.moveTo(
            warped[i].x,
            warped[i].y
        );


        ctx.lineTo(
            next.x,
            next.y
        );


        ctx.stroke();
    }


    /*
        Center.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.8)";


    ctx.beginPath();


    ctx.arc(
        center.x,
        center.y,
        5,
        0,
        Math.PI * 2
    );


    ctx.stroke();


    for (
        const p
        of warped
    ) {

        drawGeometryPoint(
            p
        );
    }
}


/* =========================================================
   FRAME
========================================================= */

function drawLandmarkFrame(
    wrist,
    thumb,
    index,
    middle,
    ring,
    pinky,
    pinchAmount,
    timestamp
) {

    /*
        Irregular four-point surface.
    */

    const topLeft =
        thumb;


    const topRight =
        index;


    const bottomRight =
        middle;


    const bottomLeft =
        wrist;


    const center =
        averagePoint(
            topLeft,
            topRight,
            bottomRight,
            bottomLeft
        );


    const compression =
        0.8 +
        pinchAmount *
        0.2;


    const a =
        scaleAround(
            topLeft,
            center,
            compression
        );


    const b =
        scaleAround(
            topRight,
            center,
            compression
        );


    const c =
        scaleAround(
            bottomRight,
            center,
            compression
        );


    const d =
        scaleAround(
            bottomLeft,
            center,
            compression
        );


    /*
        Outer frame.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";


    ctx.lineWidth = 2.5;


    ctx.beginPath();


    ctx.moveTo(
        a.x,
        a.y
    );


    ctx.lineTo(
        b.x,
        b.y
    );


    ctx.lineTo(
        c.x,
        c.y
    );


    ctx.lineTo(
        d.x,
        d.y
    );


    ctx.closePath();


    ctx.stroke();


    /*
        Cross deformation.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.35)";


    ctx.lineWidth = 1;


    ctx.beginPath();


    ctx.moveTo(
        a.x,
        a.y
    );


    ctx.lineTo(
        c.x,
        c.y
    );


    ctx.moveTo(
        b.x,
        b.y
    );


    ctx.lineTo(
        d.x,
        d.y
    );


    ctx.stroke();


    /*
        Extra tracked fingers.
    */

    ctx.beginPath();


    ctx.moveTo(
        center.x,
        center.y
    );


    ctx.lineTo(
        ring.x,
        ring.y
    );


    ctx.moveTo(
        center.x,
        center.y
    );


    ctx.lineTo(
        pinky.x,
        pinky.y
    );


    ctx.stroke();


    for (
        const p
        of [
            a,
            b,
            c,
            d,
            ring,
            pinky
        ]
    ) {

        drawGeometryPoint(
            p
        );
    }
}


/* =========================================================
   TWO-HAND GEOMETRY
========================================================= */

function drawTwoHandGeometry(
    handA,
    handB,
    timestamp
) {

    /*
        Two hands create one deformable surface.

        The geometry is controlled by the actual
        thumb/index landmarks of both hands.
    */

    const aIndex =
        smoothPoint(
            "A_INDEX",
            point(handA[8])
        );


    const aThumb =
        smoothPoint(
            "A_THUMB",
            point(handA[4])
        );


    const aPinky =
        smoothPoint(
            "A_PINKY",
            point(handA[20])
        );


    const bIndex =
        smoothPoint(
            "B_INDEX",
            point(handB[8])
        );


    const bThumb =
        smoothPoint(
            "B_THUMB",
            point(handB[4])
        );


    const bPinky =
        smoothPoint(
            "B_PINKY",
            point(handB[20])
        );


    const topLeft =
        aIndex;


    const topRight =
        bIndex;


    const bottomRight =
        bThumb;


    const bottomLeft =
        aThumb;


    ctx.save();


    if (
        geometryMode === 0
    ) {

        drawQuadWarp(
            topLeft,
            topRight,
            bottomRight,
            bottomLeft,
            1,
            timestamp
        );


    } else if (
        geometryMode === 1
    ) {

        drawLandmarkDiamond(
            aThumb,
            bIndex,
            bThumb,
            aIndex,
            1,
            timestamp
        );


    } else if (
        geometryMode === 2
    ) {

        drawLandmarkShard(
            aThumb,
            bIndex,
            bThumb,
            bPinky,
            aPinky,
            aIndex,
            1,
            timestamp
        );


    } else {

        drawLandmarkFrame(
            aThumb,
            bThumb,
            bIndex,
            bPinky,
            aPinky,
            aIndex,
            1,
            timestamp
        );
    }


    /*
        Structural line between pinkies.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.25)";


    ctx.lineWidth = 1;


    ctx.beginPath();


    ctx.moveTo(
        aPinky.x,
        aPinky.y
    );


    ctx.lineTo(
        bPinky.x,
        bPinky.y
    );


    ctx.stroke();


    drawGeometryPoint(
        aIndex
    );


    drawGeometryPoint(
        aThumb
    );


    drawGeometryPoint(
        bIndex
    );


    drawGeometryPoint(
        bThumb
    );


    ctx.restore();
}


/* =========================================================
   GEOMETRY CONTROL POINT
========================================================= */

function drawGeometryPoint(
    p
) {

    ctx.save();


    ctx.strokeStyle =
        "rgba(255,255,255,0.75)";


    ctx.lineWidth = 1.5;


    ctx.beginPath();


    ctx.arc(
        p.x,
        p.y,
        3.5,
        0,
        Math.PI * 2
    );


    ctx.stroke();


    ctx.restore();
}


/* =========================================================
   POINT MATH
========================================================= */

function averagePoint(
    ...points
) {

    let x = 0;

    let y = 0;


    for (
        const p
        of points
    ) {

        x += p.x;

        y += p.y;
    }


    return {

        x:
            x /
            points.length,

        y:
            y /
            points.length
    };
}


/* =========================================================
   LINEAR INTERPOLATION
========================================================= */

function lerpPoint(
    a,
    b,
    amount
) {

    return {

        x:
            a.x +
            (
                b.x -
                a.x
            ) *
            amount,

        y:
            a.y +
            (
                b.y -
                a.y
            ) *
            amount
    };
}


/* =========================================================
   SCALE A POINT AROUND CENTER
========================================================= */

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
            ) *
            scale,

        y:
            center.y +
            (
                pointValue.y -
                center.y
            ) *
            scale
    };
}


/* =========================================================
   SMOOTH LANDMARK
========================================================= */

function smoothPoint(
    id,
    target
) {

    const previous =
        geometrySmooth.get(
            id
        );


    if (!previous) {

        geometrySmooth.set(
            id,
            {
                x: target.x,
                y: target.y
            }
        );


        return target;
    }


    previous.x +=
        (
            target.x -
            previous.x
        ) *
        GEOMETRY_SMOOTHING;


    previous.y +=
        (
            target.y -
            previous.y
        ) *
        GEOMETRY_SMOOTHING;


    return previous;
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
            const [a, b]
            of connections
        ) {

            const p1 =
                point(
                    hand[a]
                );


            const p2 =
                point(
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


        /*
            Joint dots.
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
            point(
                hand[8]
            );


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
                hand.map(
                    point
                );


            const bounds =
                getBounds(
                    points
                );


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


        const block =
            Math.max(
                12,
                Math.floor(
                    w / 8
                )
            );


        ctx.save();


        /*
            Base.
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
                            (
                                xx -
                                px
                            ) /
                            block
                        ) +
                        Math.floor(
                            (
                                yy -
                                py
                            ) /
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
            Label.
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
            Corners.
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
            `${
                category?.categoryName ||
                "OBJECT"
            } ${
                Math.round(
                    (
                        category?.score ||
                        0
                    ) *
                    100
                )
            }%`,
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

function getSign(
    hand
) {

    const index =
        fingerUp(
            hand,
            8
        );


    const middle =
        fingerUp(
            hand,
            12
        );


    const ring =
        fingerUp(
            hand,
            16
        );


    const pinky =
        fingerUp(
            hand,
            20
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

function getGesture(
    index
) {

    return (
        handResults
            ?.gestures
            ?.[index]
            ?.[0]
            ?.categoryName ||
        "None"
    );
}


/* =========================================================
   PINCH
========================================================= */

function isPinching(
    hand
) {

    const thumb =
        point(
            hand[4]
        );


    const index =
        point(
            hand[8]
        );


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

function getPalmCenter(
    hand
) {

    const a =
        point(
            hand[0]
        );


    const b =
        point(
            hand[5]
        );


    const c =
        point(
            hand[9]
        );


    const d =
        point(
            hand[17]
        );


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
   NORMALIZED LANDMARK → CANVAS
========================================================= */

function point(
    p
) {

    return {

        /*
            MediaPipe x is normalized:
            0 → 1

            Mirror because the webcam is mirrored.
        */

        x:
            (
                1 -
                p.x
            ) *
            canvas.width,


        /*
            MediaPipe y is normalized:
            0 → 1
        */

        y:
            p.y *
            canvas.height
    };
}


/* =========================================================
   DISTANCE
========================================================= */

function distance(
    a,
    b
) {

    return Math.hypot(
        a.x - b.x,
        a.y - b.y
    );
}


/* =========================================================
   BOUNDS
========================================================= */

function getBounds(
    points
) {

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


    /*
        Top left.
    */

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


    /*
        Top right.
    */

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


    /*
        Bottom left.
    */

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


    /*
        Bottom right.
    */

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

function setShapeStatus(
    text
) {

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
