import {
    FilesetResolver,
    GestureRecognizer,
    FaceLandmarker,
    ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


/* =========================================================
   SPECTRA
   PERFORMANCE + HAND TRACKING BUILD
========================================================= */

const video =
    document.getElementById("camera");

const canvas =
    document.getElementById("overlay");

if (!video || !canvas) {
    throw new Error(
        "SPECTRA: camera or overlay canvas not found."
    );
}

const ctx =
    canvas.getContext("2d");


/* =========================================================
   OPTIONAL UI
   These won't crash the program if an element is missing.
========================================================= */

function get(...ids) {

    for (const id of ids) {

        const element =
            document.getElementById(id);

        if (element) {
            return element;
        }
    }

    return null;
}


const startBtn =
    get("startBtn", "startCamera");

const trackingBtn =
    get("trackingBtn", "handTrackingBtn");

const targetBtn =
    get("targetBtn", "targetBoxBtn");

const objectBtn =
    get("objectBtn", "objectDetectionBtn");

const privacyBtn =
    get("privacyBtn", "facePrivacyBtn");

const signBtn =
    get("signBtn", "signLanguageBtn");

const filterBtn =
    get("filterBtn", "faceFilterBtn");

const shapeBtn =
    get("shapeBtn", "shapesBtn");

const fpsDisplay =
    get("fps", "fpsText");


/* =========================================================
   AUTOMATIC FPS BOOST BUTTON
========================================================= */

let fpsBoostBtn =
    get(
        "fpsBoostBtn",
        "performanceBtn"
    );


if (!fpsBoostBtn) {

    const controls =
        document.querySelector(".controls");

    if (controls) {

        fpsBoostBtn =
            document.createElement("button");

        fpsBoostBtn.id =
            "fpsBoostBtn";

        fpsBoostBtn.textContent =
            "FPS BOOST OFF";

        controls.appendChild(
            fpsBoostBtn
        );
    }
}


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
   DEVICE
========================================================= */

const mobile =
    /Android|iPhone|iPad|iPod/i.test(
        navigator.userAgent
    ) ||
    window.innerWidth < 800;


/* =========================================================
   PERFORMANCE MODES
========================================================= */

let fpsBoost = false;


/*
    Normal mode:
    Good visuals + normal AI.

    Boost mode:
    Hand tracking gets priority.
*/

const NORMAL = mobile
    ? {
        hand: 55,
        face: 160,
        object: 350
    }
    : {
        hand: 35,
        face: 110,
        object: 250
    };


const BOOST = mobile
    ? {
        hand: 70,
        face: 999999,
        object: 999999
    }
    : {
        hand: 50,
        face: 999999,
        object: 999999
    };


let performanceSettings =
    NORMAL;


/* =========================================================
   STATE
========================================================= */

let cameraRunning = false;

let handTracking = true;

let targetVisible = true;

let objectDetection = false;

let signLanguage = false;

let shapesEnabled = true;


/* =========================================================
   FACE
========================================================= */

let facePrivacy = false;

let faceFilter = "OFF";

let privacyMode = "OFF";


/* =========================================================
   MEDIAPIPE
========================================================= */

let vision = null;

let gestureRecognizer = null;

let faceLandmarker = null;

let objectDetector = null;


/* =========================================================
   RESULTS
========================================================= */

let handResults = null;

let faceResults = null;

let objectResults = null;


/* =========================================================
   TIMERS
========================================================= */

let lastHandDetection = 0;

let lastFaceDetection = 0;

let lastObjectDetection = 0;


/* =========================================================
   FPS
========================================================= */

let renderFrames = 0;

let fpsCounterTime =
    performance.now();

let currentFPS = 0;


/* =========================================================
   SHAPES
========================================================= */

let shapes = [];

let drawingShape = false;

let currentPath = [];

let grabbedShape = null;

let pinchActive = false;

let lastHandPoint = null;


/* =========================================================
   START CAMERA
========================================================= */

async function startCamera() {

    if (cameraRunning) {
        return;
    }


    try {

        const stream =
            await navigator.mediaDevices
                .getUserMedia({

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


        video.playsInline =
            true;

        video.muted =
            true;


        await video.play();


        canvas.width =
            video.videoWidth ||
            640;

        canvas.height =
            video.videoHeight ||
            360;


        cameraRunning =
            true;


        if (startBtn) {

            startBtn.textContent =
                "CAMERA RUNNING";
        }


        /*
            IMPORTANT:

            Camera starts immediately.
            AI loads separately.
        */

        requestAnimationFrame(
            render
        );


        loadHandModel();


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );


        if (startBtn) {

            startBtn.textContent =
                "CAMERA ERROR";
        }


        alert(
            "SPECTRA could not access the camera.\n\n" +
            "Allow camera permission and use the HTTPS GitHub Pages version."
        );
    }
}


/* =========================================================
   LOAD VISION CORE
========================================================= */

async function loadHandModel() {

    if (gestureRecognizer) {
        return;
    }


    try {

        vision =
            await FilesetResolver
                .forVisionTasks(
                    WASM_URL
                );


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
            "SPECTRA hand tracking READY"
        );


    } catch (error) {

        console.error(
            "HAND MODEL ERROR:",
            error
        );
    }
}


/* =========================================================
   FACE MODEL
   Lazy loaded ONLY when required.
========================================================= */

async function loadFaceModel() {

    if (faceLandmarker) {
        return;
    }


    if (!vision) {

        await loadHandModel();
    }


    try {

        faceLandmarker =
            await FaceLandmarker
                .createFromOptions(
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
                            0.5,

                        minFacePresenceConfidence:
                            0.5,

                        minTrackingConfidence:
                            0.5
                    }
                );


        console.log(
            "SPECTRA face tracking READY"
        );


    } catch (error) {

        console.error(
            "FACE MODEL ERROR:",
            error
        );
    }
}


/* =========================================================
   OBJECT MODEL
========================================================= */

async function loadObjectModel() {

    if (objectDetector) {
        return;
    }


    if (!vision) {

        await loadHandModel();
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
                            mobile
                                ? 3
                                : 5,

                        scoreThreshold:
                            0.5
                    }
                );


        console.log(
            "SPECTRA object detection READY"
        );


    } catch (error) {

        console.error(
            "OBJECT MODEL ERROR:",
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


            performanceSettings =
                fpsBoost
                    ? BOOST
                    : NORMAL;


            if (fpsBoost) {

                fpsBoostBtn.textContent =
                    "FPS BOOST ON";


                /*
                    Stop expensive systems.
                */

                objectDetection =
                    false;

                facePrivacy =
                    false;

                faceFilter =
                    "OFF";


                /*
                    Remove unnecessary visual
                    target box.
                */

                targetVisible =
                    false;


                /*
                    IMPORTANT:

                    We DO NOT disable:

                    - hand tracking
                    - gestures
                    - pinch
                    - shapes
                    - sign detection
                */


                if (objectBtn) {

                    objectBtn.textContent =
                        "OBJECT DETECTION OFF";
                }


                if (privacyBtn) {

                    privacyBtn.textContent =
                        "FACE PRIVACY OFF";
                }


                if (filterBtn) {

                    filterBtn.textContent =
                        "FACE FILTER OFF";
                }


                if (targetBtn) {

                    targetBtn.textContent =
                        "TARGET BOX OFF";
                }


                console.log(
                    "SPECTRA FPS BOOST ENABLED"
                );


            } else {

                fpsBoostBtn.textContent =
                    "FPS BOOST OFF";


                targetVisible =
                    true;


                if (targetBtn) {

                    targetBtn.textContent =
                        "TARGET BOX ON";
                }


                console.log(
                    "SPECTRA FPS BOOST DISABLED"
                );
            }
        };
}


/* =========================================================
   BUTTONS
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


if (shapeBtn) {

    shapeBtn.onclick =
        () => {

            shapesEnabled =
                !shapesEnabled;


            shapeBtn.textContent =
                shapesEnabled
                    ? "SHAPES ON"
                    : "SHAPES OFF";


            if (!shapesEnabled) {

                drawingShape =
                    false;

                currentPath =
                    [];

                grabbedShape =
                    null;
            }
        };
}


/* =========================================================
   FACE PRIVACY BUTTON
========================================================= */

if (privacyBtn) {

    privacyBtn.onclick =
        async () => {

            if (fpsBoost) {
                return;
            }


            const modes = [
                "OFF",
                "BLUR",
                "PIXEL",
                "REDACT",
                "SCAN",
                "BOX"
            ];


            const index =
                modes.indexOf(
                    privacyMode
                );


            const next =
                modes[
                    (index + 1) %
                    modes.length
                ];


            privacyMode =
                next;


            facePrivacy =
                next !== "OFF";


            privacyBtn.textContent =
                `PRIVACY ${next}`;


            if (facePrivacy) {

                await loadFaceModel();
            }
        };
}


/* =========================================================
   FACE FILTER
========================================================= */

if (filterBtn) {

    filterBtn.onclick =
        async () => {

            if (fpsBoost) {
                return;
            }


            const filters = [
                "OFF",
                "CYBER",
                "CROWN",
                "NEON"
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

                await loadFaceModel();
            }
        };
}


/* =========================================================
   OBJECT DETECTION
========================================================= */

if (objectBtn) {

    objectBtn.onclick =
        async () => {

            if (fpsBoost) {
                return;
            }


            objectDetection =
                !objectDetection;


            if (
                objectDetection
            ) {

                objectBtn.textContent =
                    "LOADING OBJECT AI...";


                await loadObjectModel();


                if (
                    objectDetector
                ) {

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
   MAIN RENDER LOOP
========================================================= */

function render(timestamp) {

    if (!cameraRunning) {
        return;
    }


    /*
        ALWAYS draw camera first.
    */

    drawCamera();


    /*
        HAND TRACKING HAS HIGHEST PRIORITY.
    */

    runHands(timestamp);


    /*
        Only run face when actually required.
    */

    if (
        !fpsBoost &&
        (
            facePrivacy ||
            faceFilter !== "OFF"
        )
    ) {

        runFace(timestamp);
    }


    /*
        Object detector only when enabled.
    */

    if (
        !fpsBoost &&
        objectDetection
    ) {

        runObjects(timestamp);
    }


    /*
        Visuals.
    */

    if (!fpsBoost) {

        drawFacePrivacy();

        drawFaceFilter();

        drawObjects();
    }


    /*
        HANDS + SHAPES ALWAYS DRAW.
    */

    drawShapes();

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
   HAND AI
========================================================= */

function runHands(timestamp) {

    if (
        !handTracking ||
        !gestureRecognizer
    ) {
        return;
    }


    if (
        timestamp -
        lastHandDetection <
        performanceSettings.hand
    ) {
        return;
    }


    lastHandDetection =
        timestamp;


    try {

        handResults =
            gestureRecognizer
                .recognizeForVideo(
                    video,
                    timestamp
                );


        processHands();


    } catch (error) {

        console.error(
            "HAND DETECTION:",
            error
        );
    }
}


/* =========================================================
   FACE AI
========================================================= */

function runFace(timestamp) {

    if (!faceLandmarker) {
        return;
    }


    if (
        timestamp -
        lastFaceDetection <
        performanceSettings.face
    ) {
        return;
    }


    lastFaceDetection =
        timestamp;


    try {

        faceResults =
            faceLandmarker
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
   OBJECT AI
========================================================= */

function runObjects(timestamp) {

    if (!objectDetector) {
        return;
    }


    if (
        timestamp -
        lastObjectDetection <
        performanceSettings.object
    ) {
        return;
    }


    lastObjectDetection =
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
   PROCESS HANDS
========================================================= */

function processHands() {

    const hands =
        handResults?.landmarks ||
        [];


    if (!hands.length) {

        if (drawingShape) {

            finishShape();
        }


        pinchActive =
            false;

        return;
    }


    /*
        First hand.
    */

    const hand =
        hands[0];


    const index =
        getPoint(
            hand[8]
        );


    const thumb =
        getPoint(
            hand[4]
        );


    /*
        Pinch detection.
    */

    const pinchDistance =
        Math.hypot(
            hand[4].x -
            hand[8].x,

            hand[4].y -
            hand[8].y
        );


    const pinch =
        pinchDistance <
        0.075;


    /*
        Gesture Recognizer result.
    */

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

        gesture =
            "PINCH";
    }


    /*
        Update existing UI if available.
    */

    const gestureElement =
        get(
            "gesture",
            "gestureText"
        );


    if (gestureElement) {

        gestureElement.textContent =
            gesture;
    }


    /*
        Sign language.
    */

    if (signLanguage) {

        const sign =
            detectSign(
                hand
            );


        const signElement =
            get(
                "sign",
                "signText"
            );


        if (signElement) {

            signElement.textContent =
                sign;
        }
    }


    /*
        SHAPE SYSTEM
    */

    if (shapesEnabled) {

        handleShape(
            index,
            pinch
        );
    }


    lastHandPoint =
        index;
}


/* =========================================================
   SHAPE CREATION
========================================================= */

function handleShape(
    point,
    pinch
) {

    /*
        PINCH START
    */

    if (
        pinch &&
        !pinchActive
    ) {

        const existing =
            findShape(
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


        } else {

            /*
                Start drawing a NEW
                shape with the finger.
            */

            drawingShape =
                true;


            currentPath = [
                {
                    x: point.x,
                    y: point.y
                }
            ];
        }
    }


    /*
        WHILE PINCHING
    */

    if (pinch) {

        /*
            Move existing shape.
        */

        if (grabbedShape) {

            grabbedShape.x =
                point.x -
                grabbedShape.offsetX;


            grabbedShape.y =
                point.y -
                grabbedShape.offsetY;


        }

        /*
            OR draw a new shape.
        */

        else if (drawingShape) {

            if (
                !lastHandPoint ||
                Math.hypot(
                    point.x -
                    lastHandPoint.x,

                    point.y -
                    lastHandPoint.y
                ) > 3
            ) {

                currentPath.push({
                    x: point.x,
                    y: point.y
                });
            }
        }
    }


    /*
        PINCH RELEASE
    */

    if (
        !pinch &&
        pinchActive
    ) {

        if (grabbedShape) {

            grabbedShape =
                null;

        } else if (drawingShape) {

            finishShape();
        }
    }


    pinchActive =
        pinch;
}


/* =========================================================
   FINISH SHAPE
========================================================= */

function finishShape() {

    if (
        currentPath.length <
        5
    ) {

        currentPath =
            [];

        drawingShape =
            false;

        return;
    }


    const bounds =
        getBounds(
            currentPath
        );


    /*
        Very small gesture =
        circle.
    */

    if (
        bounds.width < 40 &&
        bounds.height < 40
    ) {

        shapes.push({

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

        /*
            Save the ACTUAL drawn shape.
        */

        shapes.push({

            type:
                "polygon",

            points:
                simplifyPath(
                    currentPath,
                    8
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


    currentPath =
        [];

    drawingShape =
        false;
}


/* =========================================================
   DRAW SHAPES
========================================================= */

function drawShapes() {

    /*
        Currently drawing.
    */

    if (
        drawingShape &&
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

        ctx.lineWidth =
            fpsBoost
                ? 2
                : 3;


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
            "rgba(255,255,255,0.85)";

        ctx.fillStyle =
            "rgba(255,255,255,0.04)";

        ctx.lineWidth = 2;


        if (
            shape.type ===
            "circle"
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


                const x =
                    shape.x +
                    (
                        p.x -
                        shape.centerX
                    );


                const y =
                    shape.y +
                    (
                        p.y -
                        shape.centerY
                    );


                if (i === 0) {

                    ctx.moveTo(
                        x,
                        y
                    );

                } else {

                    ctx.lineTo(
                        x,
                        y
                    );
                }
            }


            ctx.closePath();

            ctx.fill();

            ctx.stroke();
        }


        ctx.restore();
    }
}


/* =========================================================
   HAND DRAWING
========================================================= */

function drawHands() {

    if (!handTracking) {
        return;
    }


    const hands =
        handResults?.landmarks ||
        [];


    for (
        const hand
        of hands
    ) {

        const connections = [

            [0,1],[1,2],[2,3],[3,4],

            [0,5],[5,6],[6,7],[7,8],

            [5,9],[9,10],[10,11],[11,12],

            [9,13],[13,14],[14,15],[15,16],

            [13,17],[17,18],[18,19],[19,20],

            [0,17]
        ];


        ctx.save();


        /*
            In FPS BOOST mode use a simpler
            skeleton.
        */

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
                getPoint(
                    hand[a]
                );

            const p2 =
                getPoint(
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
            Don't draw every joint in boost mode.
        */

        if (!fpsBoost) {

            ctx.fillStyle =
                "rgba(255,255,255,0.9)";


            for (
                const p
                of hand
            ) {

                const point =
                    getPoint(p);


                ctx.beginPath();

                ctx.arc(
                    point.x,
                    point.y,
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
            getPoint(
                hand[8]
            );


        ctx.strokeStyle =
            "rgba(255,255,255,0.95)";

        ctx.lineWidth = 2;


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
            Pinch indicator.
        */

        const pinch =
            Math.hypot(
                hand[4].x -
                hand[8].x,

                hand[4].y -
                hand[8].y
            ) < 0.075;


        if (pinch) {

            const thumb =
                getPoint(
                    hand[4]
                );


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

                10,
                0,
                Math.PI * 2
            );

            ctx.stroke();
        }


        /*
            Target box only when not using
            FPS boost.
        */

        if (
            targetVisible &&
            !fpsBoost
        ) {

            const points =
                hand.map(
                    p =>
                        getPoint(p)
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
                "rgba(255,255,255,0.35)";


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
   FACE PRIVACY
========================================================= */

function drawFacePrivacy() {

    if (
        !facePrivacy ||
        !faceResults
    ) {
        return;
    }


    const faces =
        faceResults.faceLandmarks ||
        [];


    for (
        const face
        of faces
    ) {

        const points =
            face.map(
                p =>
                    getPoint(p)
            );


        const bounds =
            getBounds(
                points
            );


        const padX =
            bounds.width *
            0.15;


        const padY =
            bounds.height *
            0.20;


        const x =
            bounds.minX -
            padX;


        const y =
            bounds.minY -
            padY;


        const width =
            bounds.width +
            padX * 2;


        const height =
            bounds.height +
            padY * 2;


        if (
            privacyMode ===
            "BLUR"
        ) {

            /*
                Lightweight privacy effect.
            */

            ctx.save();

            ctx.fillStyle =
                "rgba(0,0,0,0.45)";

            ctx.fillRect(
                x,
                y,
                width,
                height
            );


            ctx.restore();
        }


        if (
            privacyMode ===
            "PIXEL"
        ) {

            ctx.save();

            ctx.fillStyle =
                "rgba(0,0,0,0.75)";

            ctx.fillRect(
                x,
                y,
                width,
                height
            );


            /*
                Pixel-style grid.
            */

            ctx.strokeStyle =
                "rgba(255,255,255,0.25)";


            for (
                let xx = x;
                xx < x + width;
                xx += 8
            ) {

                ctx.beginPath();

                ctx.moveTo(
                    xx,
                    y
                );

                ctx.lineTo(
                    xx,
                    y + height
                );

                ctx.stroke();
            }


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


            ctx.restore();
        }


        if (
            privacyMode ===
            "REDACT"
        ) {

            ctx.save();

            ctx.fillStyle =
                "rgba(0,0,0,0.90)";

            ctx.fillRect(
                x,
                y,
                width,
                height
            );


            ctx.fillStyle =
                "#ffffff";

            ctx.font =
                "bold 12px monospace";

            ctx.fillText(
                "FACE HIDDEN",
                x + 10,
                y + 20
            );


            ctx.restore();
        }


        if (
            privacyMode ===
            "SCAN"
        ) {

            ctx.save();

            ctx.fillStyle =
                "rgba(0,0,0,0.55)";

            ctx.fillRect(
                x,
                y,
                width,
                height
            );


            ctx.strokeStyle =
                "rgba(255,255,255,0.9)";

            ctx.lineWidth = 2;


            drawCorners(
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
                "FACE // SCANNING",
                x,
                y - 6
            );


            ctx.restore();
        }


        if (
            privacyMode ===
            "BOX"
        ) {

            ctx.save();

            ctx.strokeStyle =
                "rgba(255,255,255,0.9)";

            ctx.lineWidth = 2;


            drawCorners(
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
                "FACE TRACK",
                x,
                y - 6
            );


            ctx.restore();
        }
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


    if (!faceResults) {
        return;
    }


    const faces =
        faceResults.faceLandmarks ||
        [];


    for (
        const face
        of faces
    ) {

        const top =
            getPoint(
                face[10]
            );


        const left =
            getPoint(
                face[234]
            );


        const right =
            getPoint(
                face[454]
            );


        const width =
            Math.abs(
                right.x -
                left.x
            );


        if (
            faceFilter ===
            "CROWN"
        ) {

            ctx.save();

            ctx.strokeStyle =
                "rgba(255,255,255,0.9)";

            ctx.lineWidth = 2;


            ctx.beginPath();

            ctx.moveTo(
                top.x -
                width * 0.45,

                top.y
            );

            ctx.lineTo(
                top.x -
                width * 0.25,

                top.y -
                width * 0.45
            );

            ctx.lineTo(
                top.x -
                width * 0.05,

                top.y -
                width * 0.15
            );

            ctx.lineTo(
                top.x,

                top.y -
                width * 0.55
            );

            ctx.lineTo(
                top.x +
                width * 0.05,

                top.y -
                width * 0.15
            );

            ctx.lineTo(
                top.x +
                width * 0.25,

                top.y -
                width * 0.45
            );

            ctx.lineTo(
                top.x +
                width * 0.45,

                top.y
            );

            ctx.stroke();

            ctx.restore();
        }


        if (
            faceFilter ===
            "CYBER"
        ) {

            ctx.save();

            ctx.strokeStyle =
                "rgba(255,255,255,0.9)";

            ctx.lineWidth = 2;


            ctx.strokeRect(
                left.x,
                top.y -
                width * 0.1,
                width * 0.42,
                width * 0.18
            );


            ctx.strokeRect(
                right.x -
                width * 0.42,
                top.y -
                width * 0.1,
                width * 0.42,
                width * 0.18
            );


            ctx.restore();
        }


        if (
            faceFilter ===
            "NEON"
        ) {

            ctx.save();

            ctx.strokeStyle =
                "rgba(255,255,255,0.8)";

            ctx.lineWidth = 2;

            ctx.shadowBlur = 18;

            ctx.shadowColor =
                "rgba(255,255,255,0.9)";


            ctx.strokeRect(
                left.x -
                width * 0.1,

                top.y -
                width * 0.15,

                width * 1.2,

                width * 1.35
            );


            ctx.restore();
        }
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


    const detections =
        objectResults.detections ||
        [];


    for (
        const detection
        of detections
    ) {

        const box =
            detection.boundingBox;


        const category =
            detection.categories?.[0];


        if (!box) {
            continue;
        }


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

function detectSign(hand) {

    const index =
        fingerExtended(
            hand,
            8
        );


    const middle =
        fingerExtended(
            hand,
            12
        );


    const ring =
        fingerExtended(
            hand,
            16
        );


    const pinky =
        fingerExtended(
            hand,
            20
        );


    const count =
        [
            index,
            middle,
            ring,
            pinky
        ].filter(Boolean)
        .length;


    if (
        index &&
        !middle &&
        !ring &&
        !pinky
    ) {

        return "D";
    }


    if (
        index &&
        middle &&
        !ring &&
        !pinky
    ) {

        return "V";
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
        count === 0
    ) {

        return "FIST";
    }


    if (
        count === 4
    ) {

        return "OPEN";
    }


    return "UNKNOWN";
}


function fingerExtended(
    hand,
    tip
) {

    return (
        hand[tip].y <
        hand[tip - 2].y
    );
}


/* =========================================================
   FIND SHAPE
========================================================= */

function findShape(point) {

    for (
        let i =
            shapes.length - 1;

        i >= 0;

        i--
    ) {

        const shape =
            shapes[i];


        if (
            shape.type ===
            "circle"
        ) {

            const d =
                Math.hypot(
                    point.x -
                    shape.x,

                    point.y -
                    shape.y
                );


            if (
                d <
                shape.radius + 25
            ) {

                return shape;
            }
        }


        if (
            shape.type ===
            "polygon"
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
   GET POINT
========================================================= */

function getPoint(p) {

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
            maxX -
            minX,

        height:
            maxY -
            minY,

        cx:
            (minX +
            maxX) / 2,

        cy:
            (minY +
            maxY) / 2
    };
}


/* =========================================================
   SIMPLIFY DRAWING
========================================================= */

function simplifyPath(
    points,
    spacing
) {

    const result =
        [points[0]];


    let last =
        points[0];


    for (
        let i = 1;
        i < points.length;
        i++
    ) {

        const d =
            Math.hypot(
                points[i].x -
                last.x,

                points[i].y -
                last.y
            );


        if (
            d >= spacing
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
   FACE CORNERS
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
   FPS COUNTER
========================================================= */

function updateFPS() {

    renderFrames++;


    const now =
        performance.now();


    if (
        now -
        fpsCounterTime >=
        1000
    ) {

        currentFPS =
            renderFrames;


        renderFrames =
            0;


        fpsCounterTime =
            now;


        if (fpsDisplay) {

            fpsDisplay.textContent =
                currentFPS;
        }
    }
}


/* =========================================================
   STARTUP
========================================================= */

console.log(
    "SPECTRA loaded."
);

console.log(
    mobile
        ? "Mobile performance detected."
        : "Desktop performance detected."
);

console.log(
    "Hand tracking has highest priority."
);
