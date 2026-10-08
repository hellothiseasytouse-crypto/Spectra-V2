const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");

const ctx = canvas.getContext("2d");

const startButton =
    document.getElementById("startButton");

const targetButton =
    document.getElementById("targetButton");

const statusText =
    document.getElementById("status");

const cameraMessage =
    document.getElementById("cameraMessage");

const systemStatus =
    document.getElementById("systemStatus");

const trackingState =
    document.getElementById("trackingState");

const targetState =
    document.getElementById("targetState");

const posX =
    document.getElementById("posX");

const posY =
    document.getElementById("posY");

const deltaX =
    document.getElementById("deltaX");

const deltaY =
    document.getElementById("deltaY");

const fpsText =
    document.getElementById("fps");


let cameraRunning = false;
let trackingEnabled = false;

let previousX = null;
let previousY = null;

let currentX = 0;
let currentY = 0;

let lastFrameTime = performance.now();

let frameCounter = 0;
let lastFPSUpdate = performance.now();

let fps = 0;


/* =========================
   CAMERA
========================= */

startButton.addEventListener(
    "click",
    startCamera
);


async function startCamera() {

    if (cameraRunning) {
        return;
    }


    try {

        statusText.textContent =
            "Requesting camera access...";


        const stream =
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


        video.srcObject = stream;


        cameraMessage.style.display =
            "none";


        cameraRunning = true;


        systemStatus.textContent =
            "ONLINE";


        statusText.textContent =
            "Camera online.";


        startButton.textContent =
            "SYSTEM ONLINE";


        startButton.disabled = true;


        targetButton.disabled = false;


        video.addEventListener(
            "loadedmetadata",
            setupCanvas,
            { once: true }
        );


    } catch (error) {

        console.error(error);


        statusText.textContent =
            "Camera access failed.";


        cameraMessage.querySelector(
            "strong"
        ).textContent =
            "CAMERA ERROR";

    }

}


/* =========================
   CANVAS
========================= */

function setupCanvas() {

    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;


    render();
}


/* =========================
   TRACKING TOGGLE
========================= */

targetButton.addEventListener(
    "click",
    toggleTracking
);


function toggleTracking() {

    trackingEnabled =
        !trackingEnabled;


    if (trackingEnabled) {

        targetButton.textContent =
            "DISABLE TRACKING";


        trackingState.textContent =
            "TRACKING";


        targetState.textContent =
            "ACTIVE";


        statusText.textContent =
            "Tracking system active.";


        previousX = null;
        previousY = null;

    } else {

        targetButton.textContent =
            "ENABLE TRACKING";


        trackingState.textContent =
            "NO TARGET";


        targetState.textContent =
            "STANDBY";


        statusText.textContent =
            "Tracking disabled.";


        clearData();
    }

}


/* =========================
   MAIN RENDER LOOP
========================= */

function render() {

    if (!cameraRunning) {
        return;
    }


    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    updateFPS();


    if (trackingEnabled) {

        /*
         * Temporary tracking target.
         *
         * For v0.2 we create a central
         * tracking area. The next vision
         * system will replace this with
         * actual computer vision.
         */

        updateTracking();

        drawTrackingOverlay();

    }


    requestAnimationFrame(render);
}


/* =========================
   TRACKING
========================= */

function updateTracking() {

    /*
     * Current temporary target:
     * center of camera frame.
     *
     * This gives us the complete
     * tracking/HUD system first.
     */

    currentX =
        canvas.width / 2;

    currentY =
        canvas.height / 2;


    if (
        previousX !== null &&
        previousY !== null
    ) {

        const dx =
            currentX - previousX;

        const dy =
            currentY - previousY;


        deltaX.textContent =
            formatNumber(dx);


        deltaY.textContent =
            formatNumber(dy);

    }


    previousX = currentX;
    previousY = currentY;


    posX.textContent =
        Math.round(currentX);


    posY.textContent =
        Math.round(currentY);

}


/* =========================
   TRACKING VISUAL
========================= */

function drawTrackingOverlay() {

    const x = currentX;
    const y = currentY;


    const boxSize =
        Math.min(
            canvas.width,
            canvas.height
        ) * 0.22;


    const half =
        boxSize / 2;


    /*
     * Tracking box
     */

    ctx.strokeStyle =
        "rgba(255,255,255,0.85)";

    ctx.lineWidth = 2;


    ctx.strokeRect(
        x - half,
        y - half,
        boxSize,
        boxSize
    );


    /*
     * Crosshair
     */

    ctx.beginPath();

    ctx.moveTo(
        x - 35,
        y
    );

    ctx.lineTo(
        x + 35,
        y
    );

    ctx.moveTo(
        x,
        y - 35
    );

    ctx.lineTo(
        x,
        y + 35
    );

    ctx.stroke();


    /*
     * Center point
     */

    ctx.beginPath();

    ctx.arc(
        x,
        y,
        5,
        0,
        Math.PI * 2
    );

    ctx.fillStyle =
        "white";

    ctx.fill();


    /*
     * Extended tracking lines
     */

    ctx.setLineDash([
        8,
        8
    ]);

    ctx.lineWidth = 1;

    ctx.strokeStyle =
        "rgba(255,255,255,0.35)";


    ctx.beginPath();

    ctx.moveTo(
        x,
        0
    );

    ctx.lineTo(
        x,
        y - half
    );

    ctx.moveTo(
        x,
        y + half
    );

    ctx.lineTo(
        x,
        canvas.height
    );

    ctx.moveTo(
        0,
        y
    );

    ctx.lineTo(
        x - half,
        y
    );

    ctx.moveTo(
        x + half,
        y
    );

    ctx.lineTo(
        canvas.width,
        y
    );

    ctx.stroke();


    ctx.setLineDash([]);


    /*
     * Target label
     */

    ctx.font =
        "12px monospace";

    ctx.fillStyle =
        "white";

    ctx.fillText(
        "TARGET // 01",
        x + half + 10,
        y - half
    );


    /*
     * Coordinate label
     */

    ctx.fillStyle =
        "rgba(255,255,255,0.65)";

    ctx.fillText(
        `X:${Math.round(x)} Y:${Math.round(y)}`,
        x + half + 10,
        y - half + 18
    );

}


/* =========================
   FPS
========================= */

function updateFPS() {

    frameCounter++;


    const now =
        performance.now();


    if (
        now - lastFPSUpdate >= 1000
    ) {

        fps =
            frameCounter;


        frameCounter = 0;

        lastFPSUpdate = now;


        fpsText.textContent =
            fps.toString()
                .padStart(2, "0");
    }

}


/* =========================
   CLEAR DATA
========================= */

function clearData() {

    posX.textContent =
        "---";

    posY.textContent =
        "---";

    deltaX.textContent =
        "---";

    deltaY.textContent =
        "---";
}


/* =========================
   FORMAT NUMBERS
========================= */

function formatNumber(value) {

    const rounded =
        Math.round(value);


    if (rounded > 0) {
        return "+" + rounded;
    }


    return rounded.toString();
}
