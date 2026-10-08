import {
    FilesetResolver,
    GestureRecognizer,
    FaceLandmarker,
    ObjectDetector
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


/* =========================================================
   SPECTRA
   VISION / FACE / HAND / GESTURE / SHAPE SYSTEM
========================================================= */

const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

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
   CONFIG
========================================================= */

const VERSION = "SPECTRA v0.6";

const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";

const GESTURE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

const FACE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const OBJECT_MODEL =
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite";


/* =========================================================
   MOBILE PERFORMANCE
========================================================= */

const isMobile =
    /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
    window.innerWidth < 800;

const PERFORMANCE = isMobile
    ? {
        hand: 45,
        face: 140,
        object: 250
      }
    : {
        hand: 30,
        face: 90,
        object: 125
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
let facePrivacy = false;
let signLanguage = false;
let shapesEnabled = true;

let lastHandTime = 0;
let lastFaceTime = 0;
let lastObjectTime = 0;

let handResults = null;
let faceResults = null;
let objectResults = null;


/* =========================================================
   FILTERS
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
   SHAPE SYSTEM
========================================================= */

const shapes = [];

let drawing = false;
let currentPath = [];

let grabbedShape = null;

let shapeId = 0;

let pinchWasActive = false;

let lastPoint = null;


/* =========================================================
   FPS
========================================================= */

let frames = 0;
let lastFpsTime = performance.now();
let currentFPS = 0;


/* =========================================================
   CAMERA
========================================================= */

async function startCamera() {

    try {

        const stream = await navigator.mediaDevices.getUserMedia({
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

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 360;

        cameraRunning = true;

        startBtn.textContent = "CAMERA RUNNING";

        await loadVision();

        requestAnimationFrame(loop);

    } catch (error) {

        console.error(error);

        startBtn.textContent = "CAMERA ERROR";

        alert(
            "SPECTRA could not access the camera.\n\n" +
            "Make sure you are using HTTPS and allowed camera permission."
        );
    }
}


/* =========================================================
   LOAD VISION
========================================================= */

async function loadVision() {

    if (vision) return;

    try {

        vision = await FilesetResolver.forVisionTasks(WASM_URL);

        gestureRecognizer =
            await GestureRecognizer.createFromOptions(
                vision,
                {
                    baseOptions: {
                        modelAssetPath: GESTURE_MODEL
                    },

                    runningMode: "VIDEO",

                    numHands: 2,

                    minHandDetectionConfidence: 0.45,
                    minHandPresenceConfidence: 0.45,
                    minTrackingConfidence: 0.45
                }
            );

        console.log("SPECTRA: hand system ready");

    } catch (error) {

        console.error("Gesture system failed:", error);

        alert(
            "SPECTRA could not load the hand AI model.\n" +
            "Check the browser console for details."
        );
    }
}


/* =========================================================
   FACE MODEL
========================================================= */

async function loadFaceModel() {

    if (faceLandmarker) return true;

    try {

        faceLandmarker =
            await FaceLandmarker.createFromOptions(
                vision,
                {
                    baseOptions: {
                        modelAssetPath: FACE_MODEL
                    },

                    runningMode: "VIDEO",

                    numFaces: 2,

                    minFaceDetectionConfidence: 0.45,
                    minFacePresenceConfidence: 0.45,
                    minTrackingConfidence: 0.45,

                    outputFaceBlendshapes: true
                }
            );

        console.log("SPECTRA: face tracker ready");

        return true;

    } catch (error) {

        console.error("Face tracker failed:", error);

        faceLandmarker = null;

        return false;
    }
}


/* =========================================================
   OBJECT MODEL
========================================================= */

async function loadObjectModel() {

    if (objectDetector) return true;

    try {

        objectDetector =
            await ObjectDetector.createFromOptions(
                vision,
                {
                    baseOptions: {
                        modelAssetPath: OBJECT_MODEL
                    },

                    runningMode: "VIDEO",

                    maxResults: 5,

                    scoreThreshold: 0.45
                }
            );

        console.log("SPECTRA: object detector ready");

        return true;

    } catch (error) {

        console.error("Object detector failed:", error);

        objectDetector = null;

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

    handTracking = !handTracking;

    trackingBtn.textContent =
        handTracking
            ? "HAND TRACKING ON"
            : "HAND TRACKING OFF";
};


targetBtn.onclick = () => {

    targetVisible = !targetVisible;

    targetBtn.textContent =
        targetVisible
            ? "TARGET BOX ON"
            : "TARGET BOX OFF";
};


objectBtn.onclick = async () => {

    objectDetection = !objectDetection;

    if (objectDetection) {

        const loaded = await loadObjectModel();

        if (!loaded) {

            objectDetection = false;

            objectBtn.textContent =
                "OBJECT DETECTION ERROR";

            return;
        }
    }

    objectBtn.textContent =
        objectDetection
            ? "OBJECT DETECTION ON"
            : "OBJECT DETECTION OFF";
};


privacyBtn.onclick = async () => {

    facePrivacy = !facePrivacy;

    if (facePrivacy) {
        await loadFaceModel();
    }

    privacyBtn.textContent =
        facePrivacy
            ? "FACE PRIVACY ON"
            : "FACE PRIVACY OFF";
};


signBtn.onclick = () => {

    signLanguage = !signLanguage;

    signBtn.textContent =
        signLanguage
            ? "SIGN LANGUAGE ON"
            : "SIGN LANGUAGE OFF";
};


filterBtn.onclick = async () => {

    filterIndex++;

    if (filterIndex >= filters.length) {
        filterIndex = 0;
    }

    const current = filters[filterIndex];

    if (current !== "OFF") {
        await loadFaceModel();
    }

    filterBtn.textContent =
        `FILTER ${current}`;

    filterStatus.textContent = current;
};


shapeBtn.onclick = () => {

    shapesEnabled = !shapesEnabled;

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
   MAIN LOOP
========================================================= */

function loop(timestamp) {

    if (!cameraRunning) return;

    renderCamera();

    runHandTracking(timestamp);
    runFaceTracking(timestamp);
    runObjectDetection(timestamp);

    drawFacePrivacy();
    drawFaceTracker();
    drawFaceFilter();

    drawObjects();

    drawShapes();

    drawHands();

    updateFPS();

    requestAnimationFrame(loop);
}


/* =========================================================
   CAMERA RENDER
========================================================= */

function renderCamera() {

    const width = canvas.width;
    const height = canvas.height;

    ctx.save();

    ctx.clearRect(0, 0, width, height);

    /*
        Mirror the camera like a normal selfie camera.
    */

    ctx.translate(width, 0);
    ctx.scale(-1, 1);

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
   HAND TRACKING
========================================================= */

function runHandTracking(timestamp) {

    if (!handTracking || !gestureRecognizer) {
        return;
    }

    if (
        timestamp - lastHandTime <
        PERFORMANCE.hand
    ) {
        return;
    }

    lastHandTime = timestamp;

    try {

        handResults =
            gestureRecognizer.recognizeForVideo(
                video,
                timestamp
            );

        processHands();

    } catch (error) {

        console.error("Hand tracking error:", error);
    }
}


/* =========================================================
   PROCESS HANDS
========================================================= */

function processHands() {

    const hands =
        handResults?.handLandmarks || [];

    handsInfo.textContent = hands.length;

    handStatus.textContent =
        hands.length
            ? `${hands.length} DETECTED`
            : "WAITING";

    if (!hands.length) {

        gestureText.textContent = "NONE";
        gestureInfo.textContent = "NONE";

        if (signLanguage) {
            signText.textContent = "NONE";
            signInfo.textContent = "NONE";
        }

        if (drawing) {
            finishShape();
        }

        pinchWasActive = false;

        return;
    }


    const primaryHand = hands[0];

    const index =
        primaryHand[8];

    const thumb =
        primaryHand[4];

    const indexPoint = {
        x: (1 - index.x) * canvas.width,
        y: index.y * canvas.height
    };


    const pinch =
        distance(
            thumb,
            index
        ) < 0.055;


    let gesture = "TRACKING";

    if (
        handResults.gestures &&
        handResults.gestures[0] &&
        handResults.gestures[0][0]
    ) {

        gesture =
            handResults.gestures[0][0].categoryName ||
            "TRACKING";
    }

    /*
        Better manual gesture detection for interaction.
    */

    if (pinch) {
        gesture = "PINCH";
    }

    gestureText.textContent = gesture;
    gestureInfo.textContent = gesture;


    /* SIGN LANGUAGE */

    if (signLanguage) {

        const sign =
            detectSign(primaryHand);

        signText.textContent = sign;
        signInfo.textContent = sign;
    }


    /* SHAPE SYSTEM */

    if (shapesEnabled) {

        handleShapeInteraction(
            indexPoint,
            pinch,
            gesture
        );
    }


    /* TWO HAND SCALING */

    if (hands.length >= 2) {

        handleTwoHandScaling(
            hands[0],
            hands[1]
        );
    }


    /* DELETE */

    if (
        gesture === "Thumb_Down" ||
        gesture === "THUMBS_DOWN"
    ) {

        deleteNearestShape(
            indexPoint
        );
    }


    lastPoint = indexPoint;
}


/* =========================================================
   SHAPE INTERACTION
========================================================= */

function handleShapeInteraction(
    point,
    pinch,
    gesture
) {

    /*
        PINCH START
    */

    if (pinch && !pinchWasActive) {

        const existing =
            findShapeAtPoint(point);

        if (existing) {

            grabbedShape = existing;

            grabbedShape.offsetX =
                point.x - grabbedShape.x;

            grabbedShape.offsetY =
                point.y - grabbedShape.y;

            shapeStatus.textContent =
                "GRABBING";

        } else {

            /*
                Nothing under the finger:
                start creating a brand-new shape.
            */

            drawing = true;

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


    /*
        PINCH MOVEMENT
    */

    if (pinch) {

        if (grabbedShape) {

            grabbedShape.x =
                point.x - grabbedShape.offsetX;

            grabbedShape.y =
                point.y - grabbedShape.offsetY;

        } else if (drawing) {

            if (!lastPoint ||
                distance2D(
                    point,
                    lastPoint
                ) > 4) {

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

    if (!pinch && pinchWasActive) {

        if (grabbedShape) {

            shapeStatus.textContent =
                "RELEASED";

            grabbedShape = null;

        } else if (drawing) {

            finishShape();
        }
    }


    pinchWasActive = pinch;
}


/* =========================================================
   CREATE SHAPE
========================================================= */

function finishShape() {

    if (!currentPath.length) {
        drawing = false;
        return;
    }

    if (currentPath.length < 4) {

        drawing = false;
        currentPath = [];

        shapeStatus.textContent =
            "READY";

        return;
    }


    const bounds =
        getBounds(currentPath);

    const width = bounds.width;
    const height = bounds.height;


    /*
        Tiny movement = create a circle.
    */

    if (
        width < 35 &&
        height < 35
    ) {

        shapes.push({
            id: ++shapeId,

            type: "circle",

            x: bounds.cx,
            y: bounds.cy,

            radius: 25,

            rotation: 0
        });

    } else {

        /*
            Large movement =
            convert hand drawing into
            a polygon shape.
        */

        const points =
            simplifyPath(
                currentPath,
                10
            );

        shapes.push({

            id: ++shapeId,

            type: "polygon",

            points,

            x: 0,
            y: 0,

            rotation: 0,

            centerX: bounds.cx,
            centerY: bounds.cy
        });
    }


    drawing = false;

    currentPath = [];

    shapeStatus.textContent =
        "CREATED";
}


/* =========================================================
   DRAW SHAPES
========================================================= */

function drawShapes() {

    /*
        Draw current hand-created path.
    */

    if (drawing && currentPath.length > 1) {

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

        ctx.lineWidth = 3;

        ctx.strokeStyle =
            "rgba(255,255,255,0.9)";

        ctx.shadowBlur = 15;

        ctx.shadowColor =
            "rgba(255,255,255,0.7)";

        ctx.stroke();

        ctx.restore();
    }


    /*
        Draw finished shapes.
    */

    for (const shape of shapes) {

        ctx.save();

        ctx.lineWidth = 2;

        ctx.strokeStyle =
            shape === grabbedShape
                ? "#ffffff"
                : "rgba(255,255,255,0.75)";

        ctx.fillStyle =
            "rgba(255,255,255,0.04)";

        ctx.shadowBlur =
            shape === grabbedShape
                ? 25
                : 10;

        ctx.shadowColor =
            "rgba(255,255,255,0.5)";


        if (shape.type === "circle") {

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

        }


        else if (
            shape.type === "polygon"
        ) {

            const centerX =
                shape.centerX;

            const centerY =
                shape.centerY;

            ctx.beginPath();

            for (
                let i = 0;
                i < shape.points.length;
                i++
            ) {

                const p =
                    shape.points[i];

                const x =
                    centerX +
                    (p.x - centerX);

                const y =
                    centerY +
                    (p.y - centerY);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
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
   TWO HAND SCALING
========================================================= */

function handleTwoHandScaling(
    hand1,
    hand2
) {

    if (!grabbedShape) {
        return;
    }

    const p1 = hand1[8];
    const p2 = hand2[8];

    const d =
        distance(
            p1,
            p2
        );

    const scale =
        Math.max(
            0.5,
            Math.min(
                2.5,
                d * 5
            )
        );

    if (
        grabbedShape.type === "circle"
    ) {

        grabbedShape.radius =
            25 * scale;
    }
}


/* =========================================================
   FIND SHAPE
========================================================= */

function findShapeAtPoint(point) {

    for (
        let i = shapes.length - 1;
        i >= 0;
        i--
    ) {

        const shape =
            shapes[i];

        if (
            shape.type === "circle"
        ) {

            if (
                distance2D(
                    point,
                    {
                        x: shape.x,
                        y: shape.y
                    }
                ) <
                shape.radius + 20
            ) {

                return shape;
            }
        }


        if (
            shape.type === "polygon"
        ) {

            const b =
                getBounds(
                    shape.points
                );

            if (
                point.x >
                    b.minX - 20 &&
                point.x <
                    b.maxX + 20 &&
                point.y >
                    b.minY - 20 &&
                point.y <
                    b.maxY + 20
            ) {

                return shape;
            }
        }
    }

    return null;
}


/* =========================================================
   DELETE SHAPE
========================================================= */

function deleteNearestShape(point) {

    let nearest = null;
    let nearestIndex = -1;
    let smallest = Infinity;

    shapes.forEach(
        (shape, index) => {

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

            if (d < smallest) {

                smallest = d;

                nearest = shape;
                nearestIndex = index;
            }
        }
    );


    if (
        nearest &&
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
   FACE TRACKING
========================================================= */

async function runFaceTracking(timestamp) {

    if (
        !faceLandmarker &&
        !facePrivacy &&
        filterIndex === 0
    ) {
        return;
    }

    if (
        timestamp - lastFaceTime <
        PERFORMANCE.face
    ) {
        return;
    }

    lastFaceTime = timestamp;

    if (!faceLandmarker) {

        const loaded =
            await loadFaceModel();

        if (!loaded) return;
    }


    try {

        faceResults =
            faceLandmarker.detectForVideo(
                video,
                timestamp
            );

    } catch (error) {

        console.error(
            "Face tracking error:",
            error
        );
    }
}


/* =========================================================
   FACE PRIVACY
========================================================= */

function drawFacePrivacy() {

    if (!facePrivacy) return;

    const faces =
        faceResults?.faceLandmarks || [];

    if (!faces.length) return;


    for (const face of faces) {

        const bounds =
            getBounds(
                face.map(
                    p => ({
                        x: (1 - p.x) * canvas.width,
                        y: p.y * canvas.height
                    })
                )
            );

        const padding = 20;

        const x =
            Math.max(
                0,
                bounds.minX - padding
            );

        const y =
            Math.max(
                0,
                bounds.minY - padding
            );

        const width =
            Math.min(
                canvas.width - x,
                bounds.width + padding * 2
            );

        const height =
            Math.min(
                canvas.height - y,
                bounds.height + padding * 2
            );


        /*
            Pixelated privacy effect.

            The camera has already been drawn
            into the canvas, so this modifies
            what the user actually sees.
        */

        ctx.save();

        ctx.beginPath();

        ctx.ellipse(
            x + width / 2,
            y + height / 2,
            width / 2,
            height / 2,
            0,
            0,
            Math.PI * 2
        );

        ctx.clip();

        ctx.filter = "blur(24px)";

        ctx.drawImage(
            video,
            canvas.width - (x + width),
            y,
            width,
            height,
            x,
            y,
            width,
            height
        );

        ctx.restore();
    }
}


/* =========================================================
   FACE TRACKER
========================================================= */

function drawFaceTracker() {

    const faces =
        faceResults?.faceLandmarks || [];

    facesInfo.textContent =
        faces.length;

    faceStatus.textContent =
        faces.length
            ? `${faces.length} DETECTED`
            : "WAITING";


    if (!faces.length) {
        return;
    }


    /*
        Only draw face landmarks when a filter
        is active. This keeps normal mode clean.
    */

    if (filterIndex === 0) {
        return;
    }


    for (const face of faces) {

        ctx.save();

        ctx.fillStyle =
            "rgba(255,255,255,0.45)";

        for (let i = 0; i < face.length; i += 4) {

            const p = face[i];

            const x =
                (1 - p.x) * canvas.width;

            const y =
                p.y * canvas.height;

            ctx.beginPath();

            ctx.arc(
                x,
                y,
                1.2,
                0,
                Math.PI * 2
            );

            ctx.fill();
        }

        ctx.restore();
    }
}


/* =========================================================
   FACE FILTERS
========================================================= */

function drawFaceFilter() {

    const filter =
        filters[filterIndex];

    if (filter === "OFF") return;

    const faces =
        faceResults?.faceLandmarks || [];

    if (!faces.length) return;


    for (const face of faces) {

        if (filter === "CYBER") {
            drawCyberFilter(face);
        }

        if (filter === "DOG") {
            drawDogFilter(face);
        }

        if (filter === "CROWN") {
            drawCrownFilter(face);
        }

        if (filter === "NEON") {
            drawNeonFilter(face);
        }
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

    const width =
        Math.abs(
            leftEye.x -
            rightEye.x
        ) * 2.4;


    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";

    ctx.lineWidth = 2;

    ctx.shadowBlur = 15;

    ctx.shadowColor =
        "rgba(255,255,255,0.8)";


    /*
        Cyber glasses.
    */

    ctx.beginPath();

    ctx.roundRect(
        leftEye.x - width * 0.45,
        leftEye.y - width * 0.18,
        width * 0.38,
        width * 0.25,
        8
    );

    ctx.roundRect(
        rightEye.x + width * 0.07,
        rightEye.y - width * 0.18,
        width * 0.38,
        width * 0.25,
        8
    );

    ctx.stroke();


    /*
        Nose HUD marker.
    */

    ctx.beginPath();

    ctx.moveTo(
        nose.x - 8,
        nose.y
    );

    ctx.lineTo(
        nose.x + 8,
        nose.y
    );

    ctx.stroke();

    ctx.restore();
}


/* =========================================================
   DOG FILTER
========================================================= */

function drawDogFilter(face) {

    const leftEye =
        point(face, 33);

    const rightEye =
        point(face, 263);

    const nose =
        point(face, 1);

    const top =
        point(face, 10);

    const mouth =
        point(face, 13);

    const eyeDistance =
        Math.abs(
            leftEye.x -
            rightEye.x
        );


    ctx.save();

    ctx.lineWidth = 2;

    ctx.strokeStyle =
        "rgba(255,255,255,0.85)";

    ctx.fillStyle =
        "rgba(255,255,255,0.10)";


    /*
        Ears.
    */

    ctx.beginPath();

    ctx.moveTo(
        top.x - eyeDistance * 0.8,
        top.y + 10
    );

    ctx.lineTo(
        top.x - eyeDistance * 1.1,
        top.y - eyeDistance * 0.9
    );

    ctx.lineTo(
        top.x - eyeDistance * 0.35,
        top.y - eyeDistance * 0.35
    );

    ctx.closePath();

    ctx.fill();
    ctx.stroke();


    ctx.beginPath();

    ctx.moveTo(
        top.x + eyeDistance * 0.8,
        top.y + 10
    );

    ctx.lineTo(
        top.x + eyeDistance * 1.1,
        top.y - eyeDistance * 0.9
    );

    ctx.lineTo(
        top.x + eyeDistance * 0.35,
        top.y - eyeDistance * 0.35
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
        eyeDistance * 0.09,
        0,
        Math.PI * 2
    );

    ctx.fill();


    /*
        Whiskers.
    */

    for (
        const side of [-1, 1]
    ) {

        for (
            let i = 1;
            i <= 3;
            i++
        ) {

            ctx.beginPath();

            ctx.moveTo(
                nose.x,
                mouth.y
            );

            ctx.lineTo(
                nose.x +
                side *
                eyeDistance *
                (0.55 + i * 0.15),

                mouth.y +
                (i - 2) * 9
            );

            ctx.stroke();
        }
    }

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


    ctx.save();

    ctx.strokeStyle =
        "rgba(255,255,255,0.9)";

    ctx.fillStyle =
        "rgba(255,255,255,0.08)";

    ctx.lineWidth = 2;

    ctx.shadowBlur = 18;

    ctx.shadowColor =
        "rgba(255,255,255,0.8)";


    const baseY =
        top.y -
        width * 0.25;

    const crownWidth =
        width * 0.8;

    const x =
        top.x -
        crownWidth / 2;


    ctx.beginPath();

    ctx.moveTo(
        x,
        baseY
    );

    ctx.lineTo(
        x + crownWidth * 0.15,
        baseY - crownWidth * 0.5
    );

    ctx.lineTo(
        x + crownWidth * 0.35,
        baseY - crownWidth * 0.2
    );

    ctx.lineTo(
        x + crownWidth * 0.5,
        baseY - crownWidth * 0.6
    );

    ctx.lineTo(
        x + crownWidth * 0.65,
        baseY - crownWidth * 0.2
    );

    ctx.lineTo(
        x + crownWidth * 0.85,
        baseY - crownWidth * 0.5
    );

    ctx.lineTo(
        x + crownWidth,
        baseY
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

    const jaw =
        face.map(
            p => ({
                x: (1 - p.x) * canvas.width,
                y: p.y * canvas.height
            })
        );


    ctx.save();

    ctx.beginPath();

    ctx.moveTo(
        jaw[0].x,
        jaw[0].y
    );

    for (
        let i = 1;
        i < jaw.length;
        i += 4
    ) {

        ctx.lineTo(
            jaw[i].x,
            jaw[i].y
        );
    }

    ctx.strokeStyle =
        "rgba(255,255,255,0.8)";

    ctx.lineWidth = 2;

    ctx.shadowBlur = 20;

    ctx.shadowColor =
        "rgba(255,255,255,0.8)";

    ctx.stroke();

    ctx.restore();
}


/* =========================================================
   SIGN LANGUAGE
========================================================= */

function detectSign(hand) {

    const extended =
        countExtendedFingers(hand);


    /*
        These are FUN approximations.
        This is NOT a complete ASL translator.
    */


    if (
        isFingerExtended(hand, 8) &&
        !isFingerExtended(hand, 12) &&
        !isFingerExtended(hand, 16) &&
        !isFingerExtended(hand, 20)
    ) {

        return "D";
    }


    if (
        isFingerExtended(hand, 8) &&
        isFingerExtended(hand, 12) &&
        !isFingerExtended(hand, 16) &&
        !isFingerExtended(hand, 20)
    ) {

        return "V";
    }


    if (
        !isFingerExtended(hand, 8) &&
        !isFingerExtended(hand, 12) &&
        !isFingerExtended(hand, 16) &&
        isFingerExtended(hand, 20)
    ) {

        return "I";
    }


    if (
        isFingerExtended(hand, 8) &&
        isFingerExtended(hand, 12) &&
        isFingerExtended(hand, 16) &&
        isFingerExtended(hand, 20)
    ) {

        if (
            isThumbExtended(hand)
        ) {

            return "I LOVE YOU";
        }

        return "B / OPEN";
    }


    if (extended === 0) {

        return "A / FIST";
    }


    if (
        extended === 1 &&
        isThumbExtended(hand)
    ) {

        return "THUMBS";
    }


    return "UNKNOWN";
}


/* =========================================================
   GESTURE HELPERS
========================================================= */

function isFingerExtended(
    hand,
    tipIndex
) {

    const tip =
        hand[tipIndex];

    const pip =
        hand[tipIndex - 2];

    return tip.y < pip.y;
}


function isThumbExtended(hand) {

    const thumb =
        hand[4];

    const joint =
        hand[3];

    return Math.abs(
        thumb.x - joint.x
    ) > 0.045;
}


function countExtendedFingers(hand) {

    let count = 0;

    if (
        isFingerExtended(
            hand,
            8
        )
    ) count++;

    if (
        isFingerExtended(
            hand,
            12
        )
    ) count++;

    if (
        isFingerExtended(
            hand,
            16
        )
    ) count++;

    if (
        isFingerExtended(
            hand,
            20
        )
    ) count++;

    return count;
}


/* =========================================================
   HAND DRAWING
========================================================= */

function drawHands() {

    if (!handTracking) return;

    const hands =
        handResults?.handLandmarks || [];

    for (
        const hand of hands
    ) {

        drawHandSkeleton(hand);

        const index =
            point(
                hand,
                8
            );

        const thumb =
            point(
                hand,
                4
            );


        /*
            Cursor.
        */

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


        /*
            Pinch indicator.
        */

        if (
            distance(
                hand[4],
                hand[8]
            ) < 0.055
        ) {

            ctx.beginPath();

            ctx.arc(
                (index.x + thumb.x) / 2,
                (index.y + thumb.y) / 2,
                12,
                0,
                Math.PI * 2
            );

            ctx.stroke();

        }

        ctx.restore();


        /*
            Target box.
        */

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
                "rgba(255,255,255,0.45)";

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
        "rgba(255,255,255,0.55)";

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
        timestamp - lastObjectTime <
        PERFORMANCE.object
    ) {
        return;
    }

    lastObjectTime = timestamp;


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
   DRAW OBJECTS
========================================================= */

function drawObjects() {

    if (!objectDetection) {

        objectStatus.textContent =
            "OFF";

        return;
    }


    const detections =
        objectResults?.detections || [];


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

        if (!box) continue;


        const x =
            canvas.width -
            box.originX -
            box.width;

        const y =
            box.originY;

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


        const category =
            detection.categories?.[0];

        const label =
            category?.categoryName ||
            "OBJECT";

        const score =
            category?.score || 0;


        ctx.font =
            "11px monospace";

        ctx.fillStyle =
            "rgba(0,0,0,0.65)";

        ctx.fillRect(
            x,
            Math.max(
                0,
                y - 18
            ),
            130,
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
   MATH
========================================================= */

function distance(a,b) {

    return Math.sqrt(
        Math.pow(a.x - b.x, 2) +
        Math.pow(a.y - b.y, 2) +
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
    handOrFace,
    index
) {

    const p =
        handOrFace[index];

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

    let minX = Infinity;
    let minY = Infinity;

    let maxX = -Infinity;
    let maxY = -Infinity;


    for (
        const p of points
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


    const result = [
        points[0]
    ];

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


    /*
        Close the shape visually.
    */

    if (
        result.length > 2
    ) {

        const first =
            result[0];

        const final =
            result[result.length - 1];

        if (
            distance2D(
                first,
                final
            ) > spacing
        ) {

            result.push(
                first
            );
        }
    }


    return result;
}


/* =========================================================
   FPS
========================================================= */

function updateFPS() {

    frames++;

    const now =
        performance.now();


    if (
        now - lastFpsTime >= 1000
    ) {

        currentFPS =
            frames;

        frames = 0;

        lastFpsTime =
            now;

        fpsText.textContent =
            currentFPS;
    }
}


/* =========================================================
   INITIAL STATUS
========================================================= */

console.log(
    `${VERSION} loaded`
);

if (isMobile) {

    console.log(
        "SPECTRA: mobile performance mode"
    );
}
