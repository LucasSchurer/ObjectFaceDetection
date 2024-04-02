document.addEventListener('DOMContentLoaded', function() {
    var sliders = document.querySelectorAll('.slider');
    var saveFaceInput = document.getElementById('save-face-input');
    var saveFaceButton = document.getElementById('save-face-button');
    var popup = document.getElementById('popup');
    var videoFeed = document.getElementById('video-feed');
    var capturedImage = document.getElementById('captured-image');
    
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');

    var lastRequest = null;

    var detectObjects = document.getElementById('detect-objects')
    var objectDetectionThreshold = document.getElementById('object-detection-threshold')
    var objectDrawClassification = document.getElementById('object-draw-classification')
    var objectDrawConfidence = document.getElementById('object-draw-confidence')

    var detectFaces = document.getElementById('detect-faces')
    var faceDetectionThreshold = document.getElementById('face-detection-threshold')
    var faceDetectionDrawConfidence = document.getElementById('face-detection-draw-confidence')

    var recognizeFaces = document.getElementById('recognize-faces')
    var faceRecognitionMatchType = document.getElementById('face-recognition-match-type')
    var faceRecognitionThreshold = document.getElementById('face-recognition-threshold')

    var capturing = false

    animate()

    sliders.forEach(function(slider) {
        slider.addEventListener('input', function() {
            var sliderValue = this.parentElement.querySelector('.slider-value');
            sliderValue.textContent = this.value;
        });
    });
        
    saveFaceInput.addEventListener('input', function() {
        if (saveFaceInput.value.trim() !== '') {
            saveFaceButton.disabled = false;
        } else {
            saveFaceButton.disabled = true;
        }
    });
    
    saveFaceButton.addEventListener('click', function() {
        saveFace()
    });

    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {        
        var videoFeed = document.getElementById('video-feed');
        videoFeed.srcObject = stream;
        videoFeed.play();
        capturing = true
    })
    .catch(function (error) {
        console.error('Cant connect to the camera: ', error);
        capturedImage.src = webcamErrorImagePath;
    });

    function animate() {
        captureFrame()
        requestAnimationFrame(animate)
    }

    function showSaveFacePopup(success) {
        popup.className = 'popup ' + (success ? 'success' : 'failure');
        popup.textContent = success ? "Successfully saved face" : "Failed to save face"

        popup.classList.add('active');
        setTimeout(function() {
            popup.classList.remove('active');
        }, 3000);
    }

    function saveFace() {
        if (saveFaceInput.value == '')
        {
            showSaveFacePopup(false)    
            return
        }

        canvas.width = videoFeed.videoWidth;
        canvas.height = videoFeed.videoHeight;
        context.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
        var dataURL = canvas.toDataURL('image/png');
        
        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/save_face', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
                var response = xhr.responseText
                showSaveFacePopup(response.success)
            } else
            {
                showSaveFacePopup(false)
            }
        };

        xhr.send(JSON.stringify({ face: dataURL, name: saveFaceInput.value }));
    }

    function captureFrame() {
        if (!capturing) {
            return;
        }

        if (lastRequest !== null && lastRequest.readyState !== XMLHttpRequest.DONE) {
            return;
        }        
        
        canvas.width = videoFeed.videoWidth;
        canvas.height = videoFeed.videoHeight;
        context.scale(-1, 1);
        context.drawImage(videoFeed, 0, 0, -canvas.width, canvas.height);
        context.scale(1, 1);
        var dataURL = canvas.toDataURL('image/png');

        if (!detectFaces.checked && !detectObjects.checked & !recognizeFaces.checked) {
            capturedImage.src = dataURL;
            return;
        }

        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/predict_frame', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.onreadystatechange = function () {
            if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
                var response = JSON.parse(xhr.responseText);
                capturedImage.src = response.processed_frame;
                lastRequest = null;
            }
        };

        xhr.send(JSON.stringify({ 
            frame: dataURL,
            detect_objects: detectObjects.checked,
            object_detection_threshold: objectDetectionThreshold.value,
            object_draw_classification: objectDrawClassification.checked,
            object_draw_confidence: objectDrawConfidence.checked,

            detect_faces: detectFaces.checked,
            face_detection_threshold: faceDetectionThreshold.value,
            face_detection_draw_confidence: faceDetectionDrawConfidence.checked,
            
            recognize_faces: recognizeFaces.checked,
            face_recognition_match_type: faceRecognitionMatchType.value,
            face_recognition_threshold: faceRecognitionThreshold.value
        }));

        lastRequest = xhr;
    }
});
