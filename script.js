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


const signDisplay =
    createStatusElement(
        "signDisplay",
        "SIGN // WAITING"
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

let automaticTwistFilter = false;


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
        hands: 45,
        face: 150,
        objects: 350
    }
    : {
        hands: 24,
        face: 90,
        objects: 220
    };


const PERFORMANCE_BOOST = isMobile
    ? {
        hands: 55,
        face: 999999,
        objects: 999999
    }
    : {
        hands: 36,
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
   GEOMETRY
========================================================= */

let geometryMode = 0;


const GEOMETRY_MODES = [
    "QUAD WARP",
    "DIAMOND",
    "SHARD",
    "FRAME"
];


/*
    Much smoother than the previous 0.22 constant.

    The actual smoothing amount is adaptive:
    fast movements get less smoothing,
    tiny jitter gets more smoothing.
*/

const GEOMETRY_BASE_SMOOTHING = 0.24;

const GEOMETRY_FAST_SMOOTHING = 0.48;


const geometrySmooth =
    new Map();


let geometryLastCenter = null;

let geometryVelocity = 0;


/* =========================================================
   TWIST STATE
========================================================= */

let twistAngle = 0;

let twistNormalized = 0;

let twistTarget = 0;

let twistFilterName = "NORMAL";

let previousTwistAngle = 0;


/* =========================================================
   VISUAL STATE
========================================================= */

let visualEnergy = 0;

let visualEnergyTarget = 0;

let inversionAmount = 0;

let inversionTarget = 0;


/* =========================================================
   MEASUREMENT STATE
========================================================= */

let displayedWidth = 0;

let displayedHeight = 0;


/* =========================================================
   GESTURE SMOOTHING
========================================================= */

let lastGestureName = "NONE";

let gestureStableFrames = 0;

let lastSignName = "NONE";

let signStableFrames = 0;


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


                automaticTwistFilter =
                    false;


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


            if (!handTracking) {

                handResults =
                    null;


                geometrySmooth.clear();


                resetGeometryVisuals();
            }


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


            /*
                Manual button control disables
                automatic twist filtering until
                the user twists again.
            */

            automaticTwistFilter =
                false;


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


            if (!signLanguage) {

                setSign(
                    "OFF"
                );

            } else {

                setSign(
                    "WAITING"
                );
            }
        };
}


/* =========================================================
   SHAPE / GEOMETRY BUTTON
========================================================= */

if (shapeBtn) {

    shapeBtn.onclick =
        () => {

            if (!shapesEnabled) {

                shapesEnabled =
                    true;


                shapeBtn.textContent =
                    `SHAPE // ${GEOMETRY_MODES[geometryMode]}`;


                return;
            }


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


    drawCamera();


    if (handTracking) {

        runHands(
            timestamp
        );
    }


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


    if (
        !fpsBoost &&
        objectDetection
    ) {

        runObjects(
            timestamp
        );
    }


    if (!fpsBoost) {

        drawFacePrivacy();

        drawFaceFilter();

        drawObjects();
    }


    if (
        shapesEnabled &&
        handTracking
    ) {

        drawHandVFX(
            timestamp
        );
    }


    drawHands();


    /*
        Apply the twist inversion LAST.

        This means the geometry and camera receive
        the same visual treatment.
    */

    drawTwistInversion();


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

        updateSignLanguage();


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
        gesture
            .replaceAll(
                "_",
                " "
            )
            .toUpperCase();


    /*
        Small stability filter.

        This stops:
        VICTORY
        NONE
        VICTORY
        NONE

        from flickering every frame.
    */

    if (
        gesture ===
        lastGestureName
    ) {

        gestureStableFrames++;

    } else {

        gestureStableFrames = 0;

        lastGestureName =
            gesture;
    }


    if (
        gestureStableFrames >= 1 ||
        gesture === "NONE"
    ) {

        setGesture(
            gesture
        );
    }
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
   SIGN LANGUAGE DISPLAY
========================================================= */

function updateSignLanguage() {

    if (!signLanguage) return;


    const hands =
        handResults?.landmarks ||
        [];


    if (!hands.length) {

        setSign(
            "WAITING"
        );

        return;
    }


    const sign =
        getSign(
            hands[0]
        );


    if (
        sign ===
        lastSignName
    ) {

        signStableFrames++;

    } else {

        signStableFrames = 0;

        lastSignName =
            sign;
    }


    if (
        signStableFrames >= 1
    ) {

        setSign(
            sign
        );
    }
}


/* =========================================================
   SET SIGN
========================================================= */

function setSign(
    value
) {

    if (!signDisplay) return;


    signDisplay.textContent =
        `SIGN // ${value}`;
}


/* =========================================================
   LANDMARK GEOMETRY
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


        geometryVelocity *= 0.82;

        twistTarget *= 0.85;

        inversionTarget *= 0.85;

        visualEnergyTarget = 0;


        return;
    }


    visualEnergyTarget =
        0.5;


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


    drawTwoHandGeometry(
        hands[0],
        hands[1],
        timestamp
    );


    setShapeStatus(
        `GEOMETRY // ${GEOMETRY_MODES[geometryMode]} // TWIST ${Math.round(twistAngle)}°`
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


    /*
        A one-hand twist is based on the
        thumb → index direction.
    */

    const dx =
        index.x -
        thumb.x;


    const dy =
        index.y -
        thumb.y;


    const angle =
        Math.atan2(
            dy,
            dx
        ) *
        180 /
        Math.PI;


    updateTwist(
        angle
    );


    ctx.save();


    const palette =
        getVisualPalette();


    if (
        geometryMode === 0
    ) {

        drawQuadWarp(
            wrist,
            thumb,
            index,
            pinky,
            pinchAmount,
            timestamp,
            palette
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
            timestamp,
            palette
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
            timestamp,
            palette
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
            timestamp,
            palette
        );
    }


    ctx.restore();
}


/* =========================================================
   QUAD WARP
========================================================= */

function drawQuadWarp(
    p1,
    p2,
    p3,
    p4,
    pinchAmount,
    timestamp,
    palette
) {

    const pulse =
        1 +
        Math.sin(
            timestamp * 0.004
        ) *
        0.018;


    const center =
        averagePoint(
            p1,
            p2,
            p3,
            p4
        );


    const compression =
        0.86 +
        pinchAmount *
        0.14;


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
        Fill.
    */

    ctx.fillStyle =
        palette.fill;


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

    ctx.fill();


    /*
        Outer geometry.
    */

    ctx.strokeStyle =
        palette.primary;


    ctx.lineWidth =
        2.2 +
        twistNormalized *
        1.2;


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
        Inner quad.
    */

    const inner = [

        lerpPoint(
            a,
            center,
            0.20
        ),

        lerpPoint(
            b,
            center,
            0.20
        ),

        lerpPoint(
            c,
            center,
            0.20
        ),

        lerpPoint(
            d,
            center,
            0.20
        )
    ];


    ctx.strokeStyle =
        palette.secondary;


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


    drawQuadGrid(
        a,
        b,
        c,
        d,
        palette
    );


    /*
        Diagonal structure.
    */

    ctx.strokeStyle =
        palette.faint;


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


    drawGeometryPoint(
        a,
        palette
    );

    drawGeometryPoint(
        b,
        palette
    );

    drawGeometryPoint(
        c,
        palette
    );

    drawGeometryPoint(
        d,
        palette
    );


    /*
        Animated scan line.
    */

    drawQuadScanLine(
        a,
        b,
        c,
        d,
        timestamp,
        palette
    );


    /*
        Measurement HUD.
    */

    drawGeometryMeasurements(
        a,
        b,
        c,
        d,
        palette
    );
}


/* =========================================================
   QUAD GRID
========================================================= */

function drawQuadGrid(
    a,
    b,
    c,
    d,
    palette
) {

    ctx.save();


    ctx.strokeStyle =
        palette.faint;


    ctx.lineWidth = 0.8;


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
   QUAD SCAN LINE
========================================================= */

function drawQuadScanLine(
    a,
    b,
    c,
    d,
    timestamp,
    palette
) {

    const t =
        (
            timestamp *
            0.00045
        ) % 1;


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


    ctx.save();


    ctx.strokeStyle =
        palette.primary;


    ctx.globalAlpha =
        0.28;


    ctx.lineWidth =
        1 +
        twistNormalized;


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


/* =========================================================
   DIAMOND
========================================================= */

function drawLandmarkDiamond(
    thumb,
    index,
    middle,
    ring,
    pinchAmount,
    timestamp,
    palette
) {

    const points = [

        index,
        middle,
        ring,
        thumb
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


    ctx.fillStyle =
        palette.fill;


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

    ctx.fill();


    ctx.strokeStyle =
        palette.primary;


    ctx.lineWidth =
        2.2 +
        twistNormalized;


    ctx.stroke();


    ctx.strokeStyle =
        palette.secondary;


    ctx.lineWidth = 1;


    ctx.beginPath();


    ctx.moveTo(
        warped[0].x,
        warped[0].y
    );


    ctx.lineTo(
        warped[2].x,
        warped[2].y
    );


    ctx.moveTo(
        warped[1].x,
        warped[1].y
    );


    ctx.lineTo(
        warped[3].x,
        warped[3].y
    );


    ctx.stroke();


    for (
        const p
        of warped
    ) {

        drawGeometryPoint(
            p,
            palette
        );
    }


    drawGeometryMeasurements(
        warped[0],
        warped[1],
        warped[2],
        warped[3],
        palette
    );
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
    timestamp,
    palette
) {

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
        0.84 +
        pinchAmount *
        0.16;


    const warped =
        points.map(
            p =>
                scaleAround(
                    p,
                    center,
                    compression
                )
        );


    ctx.fillStyle =
        palette.fill;


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

    ctx.fill();


    ctx.strokeStyle =
        palette.primary;


    ctx.lineWidth =
        2.2 +
        twistNormalized;


    ctx.stroke();


    ctx.strokeStyle =
        palette.faint;


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


    for (
        const p
        of warped
    ) {

        drawGeometryPoint(
            p,
            palette
        );
    }


    const bounds =
        getBounds(
            warped
        );


    drawDimensionLabel(
        center.x,
        bounds.minY,
        bounds.width,
        bounds.height,
        0,
        palette
    );
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
    timestamp,
    palette
) {

    const points = [

        thumb,
        index,
        middle,
        wrist
    ];


    const center =
        averagePoint(
            ...points
        );


    const compression =
        0.84 +
        pinchAmount *
        0.16;


    const warped =
        points.map(
            p =>
                scaleAround(
                    p,
                    center,
                    compression
                )
        );


    ctx.fillStyle =
        palette.fill;


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

    ctx.fill();


    ctx.strokeStyle =
        palette.primary;


    ctx.lineWidth =
        2.2 +
        twistNormalized;


    ctx.stroke();


    ctx.strokeStyle =
        palette.secondary;


    ctx.lineWidth = 1;


    ctx.beginPath();


    ctx.moveTo(
        warped[0].x,
        warped[0].y
    );


    ctx.lineTo(
        warped[2].x,
        warped[2].y
    );


    ctx.moveTo(
        warped[1].x,
        warped[1].y
    );


    ctx.lineTo(
        warped[3].x,
        warped[3].y
    );


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
            ...warped,
            ring,
            pinky
        ]
    ) {

        drawGeometryPoint(
            p,
            palette
        );
    }


    drawGeometryMeasurements(
        warped[0],
        warped[1],
        warped[2],
        warped[3],
        palette
    );
}


/* =========================================================
   TWO HAND GEOMETRY
========================================================= */

function drawTwoHandGeometry(
    handA,
    handB,
    timestamp
) {

    const centerA =
        getPalmCenter(
            handA
        );


    const centerB =
        getPalmCenter(
            handB
        );


    /*
        Sort hands spatially.

        A is always left.
        B is always right.
    */

    let leftHand =
        handA;


    let rightHand =
        handB;


    if (
        centerA.x >
        centerB.x
    ) {

        leftHand =
            handB;


        rightHand =
            handA;
    }


    const leftIndex =
        smoothPoint(
            "LEFT_INDEX",
            point(
                leftHand[8]
            )
        );


    const leftThumb =
        smoothPoint(
            "LEFT_THUMB",
            point(
                leftHand[4]
            )
        );


    const leftPinky =
        smoothPoint(
            "LEFT_PINKY",
            point(
                leftHand[20]
            )
        );


    const rightIndex =
        smoothPoint(
            "RIGHT_INDEX",
            point(
                rightHand[8]
            )
        );


    const rightThumb =
        smoothPoint(
            "RIGHT_THUMB",
            point(
                rightHand[4]
            )
        );


    const rightPinky =
        smoothPoint(
            "RIGHT_PINKY",
            point(
                rightHand[20]
            )
        );


    /*
        Decide which finger is top/bottom
        for each hand.

        This makes the quad remain stable
        even when the hands rotate.
    */

    const leftPair =
        getVerticalPair(
            leftIndex,
            leftThumb
        );


    const rightPair =
        getVerticalPair(
            rightIndex,
            rightThumb
        );


    const topLeft =
        leftPair.top;


    const bottomLeft =
        leftPair.bottom;


    const topRight =
        rightPair.top;


    const bottomRight =
        rightPair.bottom;


    /*
        Calculate twist from the two
        vertical hand edges.

        This is much more stable than simply
        using screen rotation.
    */

    const leftAngle =
        Math.atan2(
            bottomLeft.y -
            topLeft.y,

            bottomLeft.x -
            topLeft.x
        ) *
        180 /
        Math.PI;


    const rightAngle =
        Math.atan2(
            bottomRight.y -
            topRight.y,

            bottomRight.x -
            topRight.x
        ) *
        180 /
        Math.PI;


    let combinedAngle =
        (
            leftAngle +
            rightAngle
        ) / 2;


    /*
        Also use the overall top edge angle.
    */

    const topEdgeAngle =
        Math.atan2(
            topRight.y -
            topLeft.y,

            topRight.x -
            topLeft.x
        ) *
        180 /
        Math.PI;


    combinedAngle =
        combinedAngle *
        0.7 +
        topEdgeAngle *
        0.3;


    updateTwist(
        combinedAngle
    );


    const palette =
        getVisualPalette();


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
            timestamp,
            palette
        );


    } else if (
        geometryMode === 1
    ) {

        drawLandmarkDiamond(
            leftThumb,
            rightIndex,
            rightThumb,
            leftIndex,
            1,
            timestamp,
            palette
        );


    } else if (
        geometryMode === 2
    ) {

        drawLandmarkShard(
            leftThumb,
            rightIndex,
            rightThumb,
            rightPinky,
            leftPinky,
            leftIndex,
            1,
            timestamp,
            palette
        );


    } else {

        drawLandmarkFrame(
            leftThumb,
            rightThumb,
            rightIndex,
            rightPinky,
            leftPinky,
            leftIndex,
            1,
            timestamp,
            palette
        );
    }


    /*
        Structural connection.
    */

    ctx.strokeStyle =
        palette.faint;


    ctx.lineWidth = 1;


    ctx.beginPath();


    ctx.moveTo(
        leftPinky.x,
        leftPinky.y
    );


    ctx.lineTo(
        rightPinky.x,
        rightPinky.y
    );


    ctx.stroke();


    ctx.restore();
}


/* =========================================================
   VERTICAL PAIR
========================================================= */

function getVerticalPair(
    a,
    b
) {

    if (
        a.y <
        b.y
    ) {

        return {
            top: a,
            bottom: b
        };
    }


    return {
        top: b,
        bottom: a
    };
}


/* =========================================================
   TWIST DETECTION
========================================================= */

function updateTwist(
    angle
) {

    /*
        Normalize angle to -180 → 180.
    */

    while (
        angle > 180
    ) {

        angle -= 360;
    }


    while (
        angle < -180
    ) {

        angle += 360;
    }


    /*
        Smooth the angle itself.

        Handles crossing -180 / 180.
    */

    let delta =
        angle -
        previousTwistAngle;


    if (
        delta > 180
    ) {

        delta -= 360;
    }


    if (
        delta < -180
    ) {

        delta += 360;
    }


    twistAngle +=
        delta *
        0.28;


    previousTwistAngle =
        angle;


    /*
        Twist intensity.

        0° = normal.
        Around 90° = strong.
        180° = maximum.
    */

    const absolute =
        Math.min(
            180,
            Math.abs(
                twistAngle
            )
        );


    twistTarget =
        absolute /
        180;


    twistNormalized +=
        (
            twistTarget -
            twistNormalized
        ) *
        0.16;


    /*
        Inversion begins around 35°.
    */

    inversionTarget =
        smoothStep(
            0.20,
            0.82,
            twistNormalized
        );


    inversionAmount +=
        (
            inversionTarget -
            inversionAmount
        ) *
        0.12;


    visualEnergyTarget =
        0.35 +
        twistNormalized *
        0.65;


    visualEnergy +=
        (
            visualEnergyTarget -
            visualEnergy
        ) *
        0.14;


    updateTwistFilter();
}


/* =========================================================
   TWIST FILTER
========================================================= */

function updateTwistFilter() {

    if (
        fpsBoost
    ) {

        return;
    }


    let nextFilter =
        "NORMAL";


    if (
        twistNormalized <
        0.20
    ) {

        nextFilter =
            "NORMAL";

    } else if (
        twistNormalized <
        0.42
    ) {

        nextFilter =
            "CYBER";

    } else if (
        twistNormalized <
        0.68
    ) {

        nextFilter =
            "SCAN";

    } else {

        nextFilter =
            "INVERT";
    }


    twistFilterName =
        nextFilter;


    /*
        Once the user starts twisting,
        geometry controls the face filter.

        When the twist comes back down,
        the manually selected filter remains.
    */

    if (
        twistNormalized >
        0.16
    ) {

        automaticTwistFilter =
            true;


        if (
            nextFilter ===
            "INVERT"
        ) {

            /*
                Keep the face filter itself on CYBER
                while the entire visual layer gets
                the stronger inversion.
            */

            faceFilter =
                "CYBER";

        } else if (
            nextFilter !==
            "NORMAL"
        ) {

            faceFilter =
                nextFilter;
        }


        if (filterBtn) {

            filterBtn.textContent =
                `FILTER // TWIST ${nextFilter}`;
        }


        if (
            (
                nextFilter ===
                "CYBER" ||
                nextFilter ===
                "SCAN"
            ) &&
            !faceDetector
        ) {

            loadFaceDetector();
        }


    } else if (
        automaticTwistFilter
    ) {

        automaticTwistFilter =
            false;


        if (filterBtn) {

            filterBtn.textContent =
                `FILTER ${faceFilter}`;
        }
    }
}


/* =========================================================
   VISUAL PALETTE
========================================================= */

function getVisualPalette() {

    const t =
        inversionAmount;


    /*
        Normal SPECTRA:
        white geometry on dark camera.

        Twist:
        moves toward a black/white difference
        treatment.
    */

    const primaryAlpha =
        0.78 +
        visualEnergy *
        0.22;


    const secondaryAlpha =
        0.28 +
        visualEnergy *
        0.32;


    const faintAlpha =
        0.10 +
        visualEnergy *
        0.18;


    /*
        Keep the actual geometry monochrome.

        The inversion effect is handled separately
        using difference blending.
    */

    return {

        primary:
            `rgba(255,255,255,${primaryAlpha})`,

        secondary:
            `rgba(255,255,255,${secondaryAlpha})`,

        faint:
            `rgba(255,255,255,${faintAlpha})`,

        fill:
            `rgba(255,255,255,${0.025 + visualEnergy * 0.035})`,

        measurement:
            `rgba(255,255,255,${0.72 + visualEnergy * 0.2})`,

        inversion:
            t
    };
}


/* =========================================================
   TWIST INVERSION
========================================================= */

function drawTwistInversion() {

    if (
        inversionAmount <
        0.01
    ) {

        return;
    }


    /*
        Canvas difference blending:

        white - pixel = inverted pixel.

        Lower alpha gives a softer transition
        instead of an instant hard inversion.
    */

    ctx.save();


    ctx.globalCompositeOperation =
        "difference";


    ctx.globalAlpha =
        inversionAmount *
        0.78;


    ctx.fillStyle =
        "#ffffff";


    ctx.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    ctx.restore();


    /*
        Small technical indicator.
    */

    drawTwistIndicator();
}


/* =========================================================
   TWIST INDICATOR
========================================================= */

function drawTwistIndicator() {

    const x =
        canvas.width -
        150;


    const y =
        24;


    ctx.save();


    ctx.font =
        "10px monospace";


    ctx.textAlign =
        "right";


    ctx.fillStyle =
        `rgba(255,255,255,${0.45 + inversionAmount * 0.4})`;


    ctx.fillText(
        `TWIST ${Math.round(Math.abs(twistAngle))}°`,
        x,
        y
    );


    ctx.fillText(
        `FILTER ${twistFilterName}`,
        x,
        y + 14
    );


    ctx.restore();
}


/* =========================================================
   GEOMETRY MEASUREMENTS
========================================================= */

function drawGeometryMeasurements(
    topLeft,
    topRight,
    bottomRight,
    bottomLeft,
    palette
) {

    /*
        Width:
        average of top and bottom edges.

        Height:
        average of left and right edges.
    */

    const topWidth =
        distance(
            topLeft,
            topRight
        );


    const bottomWidth =
        distance(
            bottomLeft,
            bottomRight
        );


    const leftHeight =
        distance(
            topLeft,
            bottomLeft
        );


    const rightHeight =
        distance(
            topRight,
            bottomRight
        );


    const width =
        (
            topWidth +
            bottomWidth
        ) / 2;


    const height =
        (
            leftHeight +
            rightHeight
        ) / 2;


    displayedWidth +=
        (
            width -
            displayedWidth
        ) *
        0.18;


    displayedHeight +=
        (
            height -
            displayedHeight
        ) *
        0.18;


    const topMid =
        lerpPoint(
            topLeft,
            topRight,
            0.5
        );


    const topDx =
        topRight.x -
        topLeft.x;


    const topDy =
        topRight.y -
        topLeft.y;


    const topLength =
        Math.hypot(
            topDx,
            topDy
        ) || 1;


    /*
        Normal pointing above the top edge.
    */

    let nx =
        topDy /
        topLength;


    let ny =
        -topDx /
        topLength;


    /*
        Push label away from the shape.
    */

    const offset =
        18 +
        Math.min(
            20,
            displayedHeight *
            0.05
        );


    const labelX =
        topMid.x +
        nx *
        offset;


    const labelY =
        topMid.y +
        ny *
        offset;


    const angle =
        Math.atan2(
            topDy,
            topDx
        );


    drawDimensionLabel(
        labelX,
        labelY,
        displayedWidth,
        displayedHeight,
        angle,
        palette
    );


    /*
        Measurement line over the top edge.
    */

    drawMeasurementTicks(
        topLeft,
        topRight,
        palette
    );
}


/* =========================================================
   DIMENSION LABEL
========================================================= */

function drawDimensionLabel(
    x,
    y,
    width,
    height,
    angle,
    palette
) {

    const safeWidth =
        Math.max(
            1,
            Math.round(
                width
            )
        );


    const safeHeight =
        Math.max(
            1,
            Math.round(
                height
            )
        );


    const text =
        `${safeWidth} PX  ×  ${safeHeight} PX`;


    const subText =
        "GEOMETRY DIMENSION";


    ctx.save();


    ctx.translate(
        x,
        y
    );


    /*
        Keep the HUD readable.

        Don't allow the label to become upside-down.
    */

    let displayAngle =
        angle;


    if (
        displayAngle >
        Math.PI / 2
    ) {

        displayAngle -=
            Math.PI;

    } else if (
        displayAngle <
        -Math.PI / 2
    ) {

        displayAngle +=
            Math.PI;
    }


    ctx.rotate(
        displayAngle
    );


    ctx.textAlign =
        "center";


    ctx.textBaseline =
        "middle";


    ctx.font =
        "bold 11px monospace";


    const mainWidth =
        ctx.measureText(
            text
        ).width;


    const boxWidth =
        mainWidth +
        24;


    const boxHeight =
        28;


    /*
        Small technical HUD plate.
    */

    ctx.fillStyle =
        "rgba(0,0,0,0.42)";


    ctx.fillRect(
        -boxWidth / 2,
        -boxHeight / 2,
        boxWidth,
        boxHeight
    );


    ctx.strokeStyle =
        palette.measurement;


    ctx.lineWidth = 1;


    ctx.strokeRect(
        -boxWidth / 2,
        -boxHeight / 2,
        boxWidth,
        boxHeight
    );


    /*
        Main dimensions.
    */

    ctx.fillStyle =
        palette.measurement;


    ctx.fillText(
        text,
        0,
        -2
    );


    /*
        Tiny subtitle.
    */

    ctx.globalAlpha =
        0.55;


    ctx.font =
        "7px monospace";


    ctx.fillText(
        subText,
        0,
        10
    );


    /*
        Corner marks.
    */

    ctx.globalAlpha =
        0.9;


    const s = 5;


    ctx.beginPath();


    ctx.moveTo(
        -boxWidth / 2,
        -boxHeight / 2 + s
    );


    ctx.lineTo(
        -boxWidth / 2,
        -boxHeight / 2
    );


    ctx.lineTo(
        -boxWidth / 2 + s,
        -boxHeight / 2
    );


    ctx.moveTo(
        boxWidth / 2 - s,
        -boxHeight / 2
    );


    ctx.lineTo(
        boxWidth / 2,
        -boxHeight / 2
    );


    ctx.lineTo(
        boxWidth / 2,
        -boxHeight / 2 + s
    );


    ctx.stroke();


    ctx.restore();
}


/* =========================================================
   MEASUREMENT TICKS
========================================================= */

function drawMeasurementTicks(
    a,
    b,
    palette
) {

    const dx =
        b.x -
        a.x;


    const dy =
        b.y -
        a.y;


    const len =
        Math.hypot(
            dx,
            dy
        ) || 1;


    const nx =
        dy /
        len;


    const ny =
        -dx /
        len;


    const tickLength =
        5;


    ctx.save();


    ctx.strokeStyle =
        palette.measurement;


    ctx.globalAlpha =
        0.5;


    ctx.lineWidth = 1;


    for (
        const p
        of [a, b]
    ) {

        ctx.beginPath();


        ctx.moveTo(
            p.x -
            nx *
            tickLength,

            p.y -
            ny *
            tickLength
        );


        ctx.lineTo(
            p.x +
            nx *
            tickLength,

            p.y +
            ny *
            tickLength
        );


        ctx.stroke();
    }


    ctx.restore();
}


/* =========================================================
   GEOMETRY CONTROL POINT
========================================================= */

function drawGeometryPoint(
    p,
    palette
) {

    ctx.save();


    const size =
        3.5 +
        twistNormalized *
        2;


    ctx.strokeStyle =
        palette.primary;


    ctx.lineWidth = 1.5;


    /*
        Square anchor instead of a circle.
    */

    ctx.strokeRect(
        p.x - size,
        p.y - size,
        size * 2,
        size * 2
    );


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


    /*
        Measure movement speed.
    */

    if (
        id ===
        "wrist"
    ) {

        if (
            geometryLastCenter
        ) {

            geometryVelocity =
                distance(
                    previous,
                    target
                );
        }


        geometryLastCenter =
            target;
    }


    /*
        Adaptive smoothing.

        Small movement:
        smoother.

        Fast movement:
        more responsive.
    */

    const speed =
        Math.min(
            1,
            geometryVelocity /
            30
        );


    const smoothing =
        GEOMETRY_BASE_SMOOTHING +
        (
            GEOMETRY_FAST_SMOOTHING -
            GEOMETRY_BASE_SMOOTHING
        ) *
        speed;


    previous.x +=
        (
            target.x -
            previous.x
        ) *
        smoothing;


    previous.y +=
        (
            target.y -
            previous.y
        ) *
        smoothing;


    return previous;
}


/* =========================================================
   RESET VISUALS
========================================================= */

function resetGeometryVisuals() {

    geometrySmooth.clear();

    geometryLastCenter =
        null;

    geometryVelocity =
        0;

    twistAngle =
        0;

    twistNormalized =
        0;

    twistTarget =
        0;

    inversionAmount =
        0;

    inversionTarget =
        0;

    automaticTwistFilter =
        false;

    if (shapeDisplay) {

        shapeDisplay.textContent =
            "GEOMETRY // WAITING";
    }
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


        const alpha =
            0.45 +
            visualEnergy *
            0.35;


        ctx.strokeStyle =
            `rgba(255,255,255,${alpha})`;


        ctx.lineWidth =
            fpsBoost
                ? 1
                : 1.2;


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


        if (!fpsBoost) {

            ctx.fillStyle =
                `rgba(255,255,255,${0.65 + visualEnergy * 0.25})`;


            for (
                const p
                of hand
            ) {

                const pos =
                    point(p);


                ctx.fillRect(
                    pos.x - 1,
                    pos.y - 1,
                    2,
                    2
                );
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


        ctx.lineWidth = 1.2;


        ctx.strokeRect(
            index.x - 5,
            index.y - 5,
            10,
            10
        );


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


        ctx.fillStyle =
            "rgba(0,0,0,0.22)";


        ctx.fillRect(
            px,
            py,
            pw,
            ph
        );


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


        ctx.fillStyle =
            "rgba(255,255,255,0.45)";


        ctx.font =
            "10px monospace";


        ctx.fillText(
            "SPECTRA VISION",
            px + 10,
            py + ph / 2
        );


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


        const energy =
            visualEnergy;


        if (
            faceFilter ===
            "CYBER"
        ) {

            ctx.strokeStyle =
                `rgba(255,255,255,${0.75 + energy * 0.2})`;


            ctx.lineWidth =
                1.5 +
                twistNormalized;


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
                `rgba(255,255,255,${0.55 + energy * 0.3})`;


            ctx.lineWidth =
                1.2 +
                twistNormalized;


            for (
                let i = 0;
                i < 6;
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
   SIGN LANGUAGE / HAND SHAPE
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


    /*
        V.
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
        Point.
    */

    if (
        index &&
        !middle &&
        !ring &&
        !pinky
    ) {

        return "POINT";
    }


    /*
        I.
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
        Open hand.
    */

    if (
        index &&
        middle &&
        ring &&
        pinky
    ) {

        return "OPEN";
    }


    /*
        Fist.
    */

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


/* =========================================================
   FINGER UP
========================================================= */

function fingerUp(
    hand,
    tip
) {

    if (
        !hand[tip] ||
        !hand[tip - 2]
    ) {

        return false;
    }


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
   SMOOTH STEP
========================================================= */

function smoothStep(
    edge0,
    edge1,
    value
) {

    const t =
        Math.max(
            0,
            Math.min(
                1,
                (
                    value -
                    edge0
                ) /
                (
                    edge1 -
                    edge0
                )
            )
        );


    return (
        t *
        t *
        (
            3 -
            2 * t
        )
    );
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

console.log(
    "SPECTRA // SYSTEM READY"
);

console.log(
    "SPECTRA // SMOOTH GEOMETRY"
);

console.log(
    "SPECTRA // TWIST FILTER SYSTEM"
);

console.log(
    "SPECTRA // GESTURE + SIGN DISPLAY"
);

console.log(
    "================================"
);
