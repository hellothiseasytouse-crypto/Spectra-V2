const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");

const ctx = canvas.getContext("2d");

const startButton = document.getElementById("startButton");
const statusText = document.getElementById("status");

const cameraMessage =
    document.getElementById("cameraMessage");


startButton.addEventListener("click", startCamera);


async function startCamera() {

    try {

        statusText.textContent =
            "Requesting camera access...";

        const stream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    facingMode: {
                        ideal: "environment"
                    }
                },

                audio: false

            });

        video.srcObject = stream;

        cameraMessage.style.display = "none";

        statusText.textContent =
            "Camera online.";

        video.addEventListener(
            "loadedmetadata",
            setupCanvas,
            { once: true }
        );

    } catch (error) {

        console.error(error);

        statusText.textContent =
            "Camera access failed.";

        cameraMessage.textContent =
            "CAMERA ERROR";
    }
}


function setupCanvas() {

    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;

    render();
}


function render() {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    requestAnimationFrame(render);
}
