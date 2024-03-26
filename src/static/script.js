var video = document.querySelector("#videoElement");
var uploadedImage = document.querySelector("#uploadedImage");
var canvas = document.createElement('canvas');
var context = canvas.getContext('2d');
var intervalId;
var capturing = false;
var lastRequest = null;

var plotClassification = document.getElementById('plotClassificationCheckbox')
var plotObjectConfidence = document.getElementById('plotObjectConfidenceCheckbox')

var detectFacesCheckbox = document.getElementById('detectFacesCheckbox')
var plotFaceDistanceCheckbox = document.getElementById('plotFaceDistanceCheckbox')
var stopFirstMatchCheckbox = document.getElementById('stopFirstMatchCheckbox')
var showOnlyBestMatchCheckbox = document.getElementById('showOnlyBestMatchCheckbox')

if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            video.srcObject = stream;
        })
        .catch(function(error) {
            console.log("Something went wrong!", error);
        });
}

function toggleCapture() {
    capturing = !capturing;
    var button = document.getElementById("toggleButton");

    if (capturing) {
        button.textContent = "Parar Captura";
        startCapturing();
    } else {
        button.textContent = "Iniciar Captura";
        stopCapturing();
    }
}

function saveFace() {
    var nameInput = document.getElementById('nameInput').value

    if (nameInput == '')
    {
        return
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    var imageDataURL = canvas.toDataURL('image/jpeg');

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/save_face', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.send(JSON.stringify({ face: imageDataURL, name: nameInput }));
}

function startCapturing() {
    var fpsInput = document.getElementById("fpsInput").value;
    var intervalTime = 1000 / fpsInput;

    clearInterval(intervalId);

    intervalId = setInterval(function() {
        captureFrame();
    }, intervalTime);
}

function stopCapturing() {
    clearInterval(intervalId);
}

function captureFrame() {
    if (!capturing) return; 

    if (lastRequest !== null && lastRequest.readyState !== XMLHttpRequest.DONE) {
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    var imageDataURL = canvas.toDataURL('image/jpeg');

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/predict_frame', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            uploadedImage.src = response.processed_frame;
            lastRequest = null;
        }
    };

    xhr.send(JSON.stringify({ 
        frame: imageDataURL,
        
        plot_classification: plotClassification.checked,
        plot_object_confidence: plotObjectConfidence.checked,
        
        detect_faces: detectFacesCheckbox.checked,
        plot_face_distance: plotFaceDistanceCheckbox.checked,
        stop_first_match: stopFirstMatchCheckbox.checked,
        showOnlyBestMatchCheckbox: showOnlyBestMatchCheckbox.checked,
        
    }));

    lastRequest = xhr;
}