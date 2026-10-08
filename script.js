import {
    FilesetResolver,
    GestureRecognizer,
    FaceDetector,
    ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


/* =========================================================
   SPECTRA
   FAST HAND TRACKING / LANDMARK GEOMETRY / FACE / FPS
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
   STATUS ELEMENT CREATOR
========================================================= */

function createStatusElement(id, text) {

    let el = document.getElementById(id);

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
        "SIGN // OFF"
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
        hands: 35,
        face: 180,
        objects: 400
    }
    : {
        hands: 20,
        face: 110,
        objects: 250
    };


const PERFORMANCE_BOOST = isMobile
    ? {
        hands: 40,
        face: 999999,
        objects: 999999
    }
    : {
        hands: 25,
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
    performance.now();


/* =========================================================
   GEOMETRY MODES
========================================================= */

let geometryMode = 0;


const GEOMETRY_MODES = [
    "QUAD WARP",
    "DIAMOND",
    "SHARD",
    "FRAME"
];


/* =========================================================
   GEOMETRY FILTERS
========================================================= */

let geometryFilter = 0;


const GEOMETRY_FILTERS = [
    "NORMAL",
    "INVERT",
    "CYBER",
    "BLUEPRINT",
    "CONTRAST"
];


/* =========================================================
   FAST GEOMETRY TRACKING
========================================================= */

/*
    Lower values = smoother.
    Higher values = more responsive.

    The adaptive system below automatically becomes
    more responsive when the hand starts moving quickly.
*/

const GEOMETRY_SMOOTHING_SLOW = 0.42;

const GEOMETRY_SMOOTHING_FAST = 0.78;


/*
    Stores previous positions and velocity.
*/

const geometryPoints =
    new Map();


/*
    Temporary predicted positions.
*/

const geometryVelocity =
    new Map();


/*
    How much we predict forward.
*/

const PREDICTION_TIME =
    0.055;


/*
    Maximum prediction so fast movement
    does not launch the geometry off-screen.
*/

const MAX_PREDICTION =
    55;


/* =========================================================
   DOUBLE PINCH
========================================================= */

let bothHandsPinching = false;

let pinchSequenceCount = 0;

let lastPinchSequence =
    0;


const DOUBLE_PINCH_WINDOW =
    700;


function checkDoublePinch(
    hands,
    timestamp
) {

    if (hands.length < 2) {

        bothHandsPinching = false;
        pinchSequenceCount = 0;

        return;
    }


    const first =
        isPinchingFast(
            hands[0]
        );


    const second =
        isPinchingFast(
            hands[1]
        );


    const both =
        first &&
        second;


    /*
        Only register the moment
        when both fingers become pinched.

        Holding the pinch does NOT repeatedly
        change the filter.
    */

    if (
        both &&
        !bothHandsPinching
    ) {

        if (
            timestamp -
            lastPinchSequence <=
            DOUBLE_PINCH_WINDOW
        ) {

            pinchSequenceCount++;

        } else {

            pinchSequenceCount = 1;
        }


        lastPinchSequence =
            timestamp;


        if (
            pinchSequenceCount >= 2
        ) {

            cycleGeometryFilter();

            pinchSequenceCount = 0;
        }
    }


    bothHandsPinching =
        both;
}


/* =========================================================
   CHANGE GEOMETRY FILTER
========================================================= */

function cycleGeometryFilter() {

    geometryFilter =
        (
            geometryFilter + 1
        ) %
        GEOMETRY_FILTERS.length;


    const filter =
        GEOMETRY_FILTERS[
            geometryFilter
        ];


    setShapeStatus(
        `GEOMETRY // ${GEOMETRY_MODES[geometryMode]} // ${filter}`
    );


    console.log(
        "SPECTRA // GEOMETRY FILTER:",
        filter
    );
}


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
   LOAD VISION
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
                            0.40,

                        minHandPresenceConfidence:
                            0.40,

                        minTrackingConfidence:
                            0.35
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
                            0.45
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


            if (!handTracking) {

                setGesture(
                    "OFF"
                );


                if (signDisplay) {

                    signDisplay.textContent =
                        "SIGN // OFF";
                }
            }
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


            if (!signLanguage) {

                if (signDisplay) {

                    signDisplay.textContent =
                        "SIGN // OFF";
                }

            } else {

                updateSignLanguage();
            }
        };
}


/* =========================================================
   SHAPE BUTTON
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


    updateFPS();


    requestAnimationFrame(
        render
    );
}


/* =========================================================
   CAMERA
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

        updateSignLanguage();


    } catch (error) {

        console.error(
            "HAND DETECTION:",
            error
        );
    }
}


/* =========================================================
   FACE
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
   OBJECTS
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
            "I LOVE YOU",

        None:
            "NONE"
    };


    const detected =
        hands
            .map(
                hand =>
                    hand?.[0]?.categoryName ||
                    "TRACKING"
            )
            .map(
                name =>
                    names[name] ||
                    name
            );


    setGesture(
        detected.join(
            " + "
        )
    );
}


/* =========================================================
   SIGN LANGUAGE DISPLAY
========================================================= */

function updateSignLanguage() {

    if (!signDisplay) return;


    if (!signLanguage) {

        signDisplay.textContent =
            "SIGN // OFF";

        return;
    }


    const hands =
        handResults?.landmarks ||
        [];


    if (!hands.length) {

        signDisplay.textContent =
            "SIGN // WAITING";

        return;
    }


    const signs =
        hands.map(
            hand =>
                getSign(
                    hand
                )
        );


    signDisplay.textContent =
        `SIGN // ${signs.join(" + ")}`;
}


/* =========================================================
   SET GESTURE
========================================================= */

function setGesture(value) {

    if (!gestureDisplay) return;


    gestureDisplay.textContent =
        `GESTURE // ${value}`;
}


/* =========================================================
   NEW GEOMETRY
========================================================= */

function drawHandVFX(timestamp) {

    const hands =
        handResults?.landmarks ||
        [];


    if (!hands.length) {

        setShapeStatus(
            "GEOMETRY // WAITING"
        );

        return;
    }


    checkDoublePinch(
        hands,
        timestamp
    );


    /*
        TWO HANDS
    */

    if (hands.length >= 2) {

        drawTwoHandGeometry(
            hands[0],
            hands[1],
            timestamp
        );


        return;
    }


    /*
        ONE HAND
    */

    drawLandmarkGeometry(
        hands[0],
        timestamp
    );
}


/* =========================================================
   ONE HAND
========================================================= */

function drawLandmarkGeometry(
    hand,
    timestamp
) {

    const wrist =
        trackedPoint(
            "wrist",
            point(hand[0])
        );


    const thumb =
        trackedPoint(
            "thumb",
            point(hand[4])
        );


    const index =
        trackedPoint(
            "index",
            point(hand[8])
        );


    const middle =
        trackedPoint(
            "middle",
            point(hand[12])
        );


    const ring =
        trackedPoint(
            "ring",
            point(hand[16])
        );


    const pinky =
        trackedPoint(
            "pinky",
            point(hand[20])
        );


    const pinch =
        isPinchingFast(
            hand
        );


    if (
        geometryMode === 0
    ) {

        drawQuadWarp(
            wrist,
            thumb,
            index,
            pinky,
            pinch,
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
            pinch,
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
            pinch,
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
            pinch,
            timestamp
        );
    }


    setShapeStatus(
        `GEOMETRY // ${GEOMETRY_MODES[geometryMode]} // ${GEOMETRY_FILTERS[geometryFilter]}`
    );
}


/* =========================================================
   TWO HAND QUAD
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
        Make A the left hand in screen space.
    */

    if (
        centerA.x >
        centerB.x
    ) {

        const temp =
            handA;

        handA =
            handB;

        handB =
            temp;
    }


    const leftIndex =
        trackedPoint(
            "leftIndex",
            point(handA[8])
        );


    const leftThumb =
        trackedPoint(
            "leftThumb",
            point(handA[4])
        );


    const rightIndex =
        trackedPoint(
            "rightIndex",
            point(handB[8])
        );


    const rightThumb =
        trackedPoint(
            "rightThumb",
            point(handB[4])
        );


    const leftTop =
        leftIndex.y <
        leftThumb.y
            ? leftIndex
            : leftThumb;


    const leftBottom =
        leftIndex.y <
        leftThumb.y
            ? leftThumb
            : leftIndex;


    const rightTop =
        rightIndex.y <
        rightThumb.y
            ? rightIndex
            : rightThumb;


    const rightBottom =
        rightIndex.y <
        rightThumb.y
            ? rightThumb
            : rightIndex;


    /*
        This is the main corner-pin surface.
    */

    const corners = {

        tl: leftTop,

        tr: rightTop,

        br: rightBottom,

        bl: leftBottom
    };


    drawGeometrySurface(
        corners,
        timestamp
    );


    setShapeStatus(
        `GEOMETRY // ${GEOMETRY_MODES[geometryMode]} // ${GEOMETRY_FILTERS[geometryFilter]}`
    );
}


/* =========================================================
   MAIN GEOMETRY SURFACE
========================================================= */

function drawGeometrySurface(
    q,
    timestamp
) {

    const a = q.tl;
    const b = q.tr;
    const c = q.br;
    const d = q.bl;


    const center =
        averagePoint(
            a,
            b,
            c,
            d
        );


    /*
        LENGTH

        Average of top and bottom edges.
    */

    const topLength =
        distance(
            a,
            b
        );


    const bottomLength =
        distance(
            d,
            c
        );


    const length =
        (
            topLength +
            bottomLength
        ) / 2;


    /*
        BREADTH

        Average of left and right edges.
    */

    const leftBreadth =
        distance(
            a,
            d
        );


    const rightBreadth =
        distance(
            b,
            c
        );


    const breadth =
        (
            leftBreadth +
            rightBreadth
        ) / 2;


    /*
        Draw the filter first so the geometry
        appears to contain its own visual treatment.
    */

    drawGeometryFilter(
        a,
        b,
        c,
        d,
        center,
        timestamp
    );


    /*
        OUTER SURFACE
    */

    drawQuadOutline(
        a,
        b,
        c,
        d
    );


    /*
        INNER SURFACE
    */

    const inner = [

        lerpPoint(
            a,
            center,
            0.16
        ),

        lerpPoint(
            b,
            center,
            0.16
        ),

        lerpPoint(
            c,
            center,
            0.16
        ),

        lerpPoint(
            d,
            center,
            0.16
        )
    ];


    drawQuadOutline(
        inner[0],
        inner[1],
        inner[2],
        inner[3],
        0.35
    );


    /*
        DIAGONALS
    */

    ctx.save();

    ctx.strokeStyle =
        getGeometryLineColor(
            0.22
        );

    ctx.lineWidth = 1;

    ctx.setLineDash([
        5,
        7
    ]);


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

    ctx.setLineDash([]);

    ctx.restore();


    /*
        GRID
    */

    drawQuadGrid(
        a,
        b,
        c,
        d
    );


    /*
        CORNER ANCHORS
    */

    drawGeometryAnchor(
        a
    );

    drawGeometryAnchor(
        b
    );

    drawGeometryAnchor(
        c
    );

    drawGeometryAnchor(
        d
    );


    /*
        MEASUREMENTS
    */

    drawGeometryMeasurements(
        a,
        b,
        c,
        d,
        length,
        breadth
    );
}


/* =========================================================
   QUAD OUTLINE
========================================================= */

function drawQuadOutline(
    a,
    b,
    c,
    d,
    alpha = 0.95
) {

    ctx.save();


    ctx.strokeStyle =
        getGeometryLineColor(
            alpha
        );


    ctx.lineWidth =
        geometryFilter === 1
            ? 2.8
            : 2.2;


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


    ctx.restore();
}


/* =========================================================
   GEOMETRY FILTER
========================================================= */

function drawGeometryFilter(
    a,
    b,
    c,
    d,
    center,
    timestamp
) {

    ctx.save();


    /*
        Clip everything to the geometry.
    */

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

    ctx.clip();


    if (
        geometryFilter === 0
    ) {

        /*
            Normal subtle glass surface.
        */

        ctx.fillStyle =
            "rgba(255,255,255,0.035)";

        ctx.fill();


    } else if (
        geometryFilter === 1
    ) {

        /*
            INVERT

            Difference mode makes the
            geometry invert the camera beneath it.
        */

        ctx.globalCompositeOperation =
            "difference";


        ctx.fillStyle =
            "rgba(255,255,255,0.95)";


        ctx.fillRect(
            0,
            0,
            canvas.width,
            canvas.height
        );


    } else if (
        geometryFilter === 2
    ) {

        /*
            CYBER
        */

        ctx.fillStyle =
            "rgba(255,255,255,0.055)";

        ctx.fill();


        ctx.strokeStyle =
            "rgba(255,255,255,0.25)";

        ctx.lineWidth = 1;


        for (
            let i = -10;
            i < canvas.width + 100;
            i += 18
        ) {

            ctx.beginPath();

            ctx.moveTo(
                i,
                0
            );

            ctx.lineTo(
                i + canvas.height,
                canvas.height
            );

            ctx.stroke();
        }


    } else if (
        geometryFilter === 3
    ) {

        /*
            BLUEPRINT / TECH GRID
        */

        ctx.fillStyle =
            "rgba(255,255,255,0.025)";

        ctx.fill();


        ctx.strokeStyle =
            "rgba(255,255,255,0.18)";

        ctx.lineWidth = 1;


        for (
            let x = 0;
            x < canvas.width;
            x += 20
        ) {

            ctx.beginPath();

            ctx.moveTo(
                x,
                0
            );

            ctx.lineTo(
                x,
                canvas.height
            );

            ctx.stroke();
        }


        for (
            let y = 0;
            y < canvas.height;
            y += 20
        ) {

            ctx.beginPath();

            ctx.moveTo(
                0,
                y
            );

            ctx.lineTo(
                canvas.width,
                y
            );

            ctx.stroke();
        }


    } else {

        /*
            HIGH CONTRAST
        */

        ctx.fillStyle =
            "rgba(255,255,255,0.11)";

        ctx.fill();


        ctx.strokeStyle =
            "rgba(0,0,0,0.7)";

        ctx.lineWidth = 3;


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
    }


    /*
        Moving scan line.
    */

    const scan =
        (
            timestamp * 0.00025
        ) % 1;


    const left =
        lerpPoint(
            a,
            d,
            scan
        );


    const right =
        lerpPoint(
            b,
            c,
            scan
        );


    ctx.globalCompositeOperation =
        geometryFilter === 1
            ? "difference"
            : "source-over";


    ctx.strokeStyle =
        geometryFilter === 1
            ? "rgba(255,255,255,0.75)"
            : "rgba(255,255,255,0.32)";


    ctx.lineWidth = 1;


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
   MEASUREMENTS
========================================================= */

function drawGeometryMeasurements(
    a,
    b,
    c,
    d,
    length,
    breadth
) {

    /*
        Put the measurement above the top edge.

        This is intentionally not just a tiny number.
        It has a proper technical HUD treatment.
    */

    const topMid =
        lerpPoint(
            a,
            b,
            0.5
        );


    const topAngle =
        Math.atan2(
            b.y - a.y,
            b.x - a.x
        );


    /*
        Perpendicular vector pointing upward.
    */

    let nx =
        Math.sin(
            topAngle
        );


    let ny =
        -Math.cos(
            topAngle
        );


    /*
        Make sure it generally points upward.
    */

    if (ny > 0) {

        nx *= -1;
        ny *= -1;
    }


    const offset = 26;


    const labelX =
        topMid.x +
        nx *
        offset;


    const labelY =
        topMid.y +
        ny *
        offset;


    ctx.save();


    /*
        Measurement line.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.58)";


    ctx.lineWidth = 1;


    ctx.setLineDash([
        4,
        5
    ]);


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


    ctx.setLineDash([]);


    /*
        Tiny end caps.
    */

    drawMeasurementTick(
        a,
        topAngle
    );


    drawMeasurementTick(
        b,
        topAngle
    );


    /*
        Dimension text.
    */

    const lengthText =
        `L ${formatDimension(length)}`;


    const breadthText =
        `B ${formatDimension(breadth)}`;


    const text =
        `${lengthText}  ×  ${breadthText}`;


    ctx.font =
        "bold 11px monospace";


    const metrics =
        ctx.measureText(
            text
        );


    const paddingX = 10;

    const paddingY = 6;


    const boxWidth =
        metrics.width +
        paddingX * 2;


    const boxHeight =
        20 +
        paddingY;


    const boxX =
        labelX -
        boxWidth / 2;


    const boxY =
        labelY -
        boxHeight;


    /*
        Small technical connector.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.35)";


    ctx.beginPath();


    ctx.moveTo(
        topMid.x,
        topMid.y
    );


    ctx.lineTo(
        labelX,
        labelY
    );


    ctx.stroke();


    /*
        HUD background.
    */

    ctx.fillStyle =
        "rgba(0,0,0,0.72)";


    ctx.fillRect(
        boxX,
        boxY,
        boxWidth,
        boxHeight
    );


    /*
        HUD border.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.75)";


    ctx.lineWidth = 1;


    ctx.strokeRect(
        boxX,
        boxY,
        boxWidth,
        boxHeight
    );


    /*
        Small side marks.
    */

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";


    ctx.beginPath();


    ctx.moveTo(
        boxX,
        boxY
    );


    ctx.lineTo(
        boxX + 7,
        boxY
    );


    ctx.moveTo(
        boxX,
        boxY
    );


    ctx.lineTo(
        boxX,
        boxY + 7
    );


    ctx.moveTo(
        boxX + boxWidth,
        boxY + boxHeight
    );


    ctx.lineTo(
        boxX + boxWidth - 7,
        boxY + boxHeight
    );


    ctx.moveTo(
        boxX + boxWidth,
        boxY + boxHeight
    );


    ctx.lineTo(
        boxX + boxWidth,
        boxY + boxHeight - 7
    );


    ctx.stroke();


    /*
        Main text.
    */

    ctx.fillStyle =
        "#ffffff";


    ctx.textAlign =
        "center";


    ctx.textBaseline =
        "middle";


    ctx.fillText(
        text,
        labelX,
        boxY +
        boxHeight / 2
    );


    ctx.restore();
}


/* =========================================================
   MEASUREMENT TICK
========================================================= */

function drawMeasurementTick(
    p,
    angle
) {

    const size = 5;


    const nx =
        Math.sin(
            angle
        );


    const ny =
        -Math.cos(
            angle
        );


    ctx.beginPath();


    ctx.moveTo(
        p.x - nx * size,
        p.y - ny * size
    );


    ctx.lineTo(
        p.x + nx * size,
        p.y + ny * size
    );


    ctx.stroke();
}


/* =========================================================
   DIMENSION FORMAT
========================================================= */

function formatDimension(
    value
) {

    /*
        Convert pixels into a clean HUD
        value instead of dumping decimals.
    */

    if (
        value < 10
    ) {

        return value.toFixed(1);
    }


    return Math.round(
        value
    );
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
        getGeometryLineColor(
            0.16
        );


    ctx.lineWidth = 1;


    for (
        let i = 1;
        i < 5;
        i++
    ) {

        const t =
            i / 5;


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
        i < 5;
        i++
    ) {

        const t =
            i / 5;


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
   GEOMETRY COLOR
========================================================= */

function getGeometryLineColor(
    alpha
) {

    if (
        geometryFilter === 1
    ) {

        return `rgba(255,255,255,${alpha})`;
    }


    if (
        geometryFilter === 2
    ) {

        return `rgba(150,220,255,${alpha})`;
    }


    if (
        geometryFilter === 3
    ) {

        return `rgba(190,220,255,${alpha})`;
    }


    if (
        geometryFilter === 4
    ) {

        return `rgba(255,255,255,${alpha})`;
    }


    return `rgba(255,255,255,${alpha})`;
}


/* =========================================================
   GEOMETRY ANCHOR
========================================================= */

function drawGeometryAnchor(
    p
) {

    const s = 5;


    ctx.save();


    ctx.strokeStyle =
        getGeometryLineColor(
            0.9
        );


    ctx.lineWidth = 1.5;


    /*
        Square anchor instead of a circle.
    */

    ctx.strokeRect(
        p.x - s,
        p.y - s,
        s * 2,
        s * 2
    );


    /*
        Crosshair.
    */

    ctx.beginPath();


    ctx.moveTo(
        p.x - 9,
        p.y
    );


    ctx.lineTo(
        p.x - 6,
        p.y
    );


    ctx.moveTo(
        p.x + 6,
        p.y
    );


    ctx.lineTo(
        p.x + 9,
        p.y
    );


    ctx.moveTo(
        p.x,
        p.y - 9
    );


    ctx.lineTo(
        p.x,
        p.y - 6
    );


    ctx.moveTo(
        p.x,
        p.y + 6
    );


    ctx.lineTo(
        p.x,
        p.y + 9
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
    pinch,
    timestamp
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
        pinch
            ? 0.88
            : 1;


    const warped =
        points.map(
            p =>
                scaleAround(
                    p,
                    center,
                    compression
                )
        );


    drawGeometryFilter(
        warped[0],
        warped[1],
        warped[2],
        warped[3],
        center,
        timestamp
    );


    drawQuadOutline(
        warped[0],
        warped[1],
        warped[2],
        warped[3]
    );


    ctx.save();


    ctx.strokeStyle =
        getGeometryLineColor(
            0.3
        );


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


    ctx.restore();


    for (
        const p
        of warped
    ) {

        drawGeometryAnchor(
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
    pinch,
    timestamp
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
        pinch
            ? 0.88
            : 1;


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
        Draw connected angular surface.
    */

    ctx.save();


    ctx.strokeStyle =
        getGeometryLineColor(
            0.9
        );


    ctx.lineWidth = 2;


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
        Internal structure.
    */

    ctx.strokeStyle =
        getGeometryLineColor(
            0.25
        );


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


    ctx.restore();


    for (
        const p
        of warped
    ) {

        drawGeometryAnchor(
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
    pinch,
    timestamp
) {

    const a =
        thumb;

    const b =
        index;

    const c =
        middle;

    const d =
        wrist;


    const center =
        averagePoint(
            a,
            b,
            c,
            d
        );


    drawGeometryFilter(
        a,
        b,
        c,
        d,
        center,
        timestamp
    );


    drawQuadOutline(
        a,
        b,
        c,
        d
    );


    ctx.save();


    ctx.strokeStyle =
        getGeometryLineColor(
            0.3
        );


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


    ctx.restore();


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

        drawGeometryAnchor(
            p
        );
    }
}


/* =========================================================
   FAST TRACKED POINT
========================================================= */

function trackedPoint(
    id,
    target
) {

    const previous =
        geometryPoints.get(
            id
        );


    if (!previous) {

        const initial = {

            x: target.x,

            y: target.y,

            lastTime:
                performance.now(),

            vx: 0,

            vy: 0
        };


        geometryPoints.set(
            id,
            initial
        );


        return {
            x: target.x,
            y: target.y
        };
    }


    const now =
        performance.now();


    const dt =
        Math.max(
            8,
            Math.min(
                80,
                now -
                previous.lastTime
            )
        );


    /*
        Current velocity.
    */

    const vx =
        (
            target.x -
            previous.x
        ) /
        dt;


    const vy =
        (
            target.y -
            previous.y
        ) /
        dt;


    /*
        Smooth velocity.
    */

    previous.vx =
        previous.vx * 0.55 +
        vx * 0.45;


    previous.vy =
        previous.vy * 0.55 +
        vy * 0.45;


    /*
        Speed in pixels/ms.
    */

    const speed =
        Math.hypot(
            previous.vx,
            previous.vy
        );


    /*
        Faster hand =
        more aggressive smoothing.

        This is intentionally adaptive:
        slow movements stay smooth,
        fast movements catch up immediately.
    */

    const speedFactor =
        Math.min(
            1,
            speed / 1.2
        );


    const smoothing =
        GEOMETRY_SMOOTHING_SLOW +
        (
            GEOMETRY_SMOOTHING_FAST -
            GEOMETRY_SMOOTHING_SLOW
        ) *
        speedFactor;


    /*
        Predict slightly ahead.
    */

    let predictedX =
        target.x +
        previous.vx *
        PREDICTION_TIME *
        1000;


    let predictedY =
        target.y +
        previous.vy *
        PREDICTION_TIME *
        1000;


    /*
        Limit prediction.
    */

    const predictionDX =
        predictedX -
        target.x;


    const predictionDY =
        predictedY -
        target.y;


    const predictionLength =
        Math.hypot(
            predictionDX,
            predictionDY
        );


    if (
        predictionLength >
        MAX_PREDICTION
    ) {

        const scale =
            MAX_PREDICTION /
            predictionLength;


        predictedX =
            target.x +
            predictionDX *
            scale;


        predictedY =
            target.y +
            predictionDY *
            scale;
    }


    /*
        Smooth toward predicted position.
    */

    previous.x +=
        (
            predictedX -
            previous.x
        ) *
        smoothing;


    previous.y +=
        (
            predictedY -
            previous.y
        ) *
        smoothing;


    previous.lastTime =
        now;


    return {

        x:
            previous.x,

        y:
            previous.y
    };
}


/* =========================================================
   FAST PINCH
========================================================= */

function isPinchingFast(
    hand
) {

    if (!hand) return false;


    const thumb =
        hand[4];


    const index =
        hand[8];


    const distance =
        Math.hypot(
            thumb.x -
            index.x,

            thumb.y -
            index.y
        );


    /*
        Normalized threshold.

        Larger threshold makes pinch easier
        to trigger during quick movements.
    */

    return distance <
        0.085;
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
            "rgba(255,255,255,0.58)";


        ctx.lineWidth =
            fpsBoost
                ? 1
                : 1.25;


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
                "rgba(255,255,255,0.8)";


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
                    1.6,
                    0,
                    Math.PI * 2
                );


                ctx.fill();
            }
        }


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
            6,
            0,
            Math.PI * 2
        );


        ctx.stroke();


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
                "rgba(255,255,255,0.25)";


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
   OBJECT DETECTION DRAW
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
   POINT
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
        performance.now();


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
    "SPECTRA // FAST GEOMETRY READY"
);

console.log(
    "SPECTRA // DOUBLE PINCH FILTERS READY"
);

console.log(
    "SPECTRA // MEASUREMENT HUD READY"
);

console.log(
    "================================"
);
