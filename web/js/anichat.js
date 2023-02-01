// const e = require("cors");

URL = window.URL || window.webkitURL;

var gumStream;              //stream from getUserMedia()
var rec;                    //Recorder.js object
var input;                  //MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //new audio context to help us record


$(document).ready(function () {

    $("#btnSend").click(function () {
        var message = $("#exampleFormControlInput1").val();
        sendMyMessage(message)
    });

    $("#recordButton").click(function() {
        clickRecord();
    });
});

function validationCheck(message) {
    // 이상한 문자 검열하기.
    var regex = new RegExp('[^가-힣ㄱ-ㅎ0-9\s\t\n.,;?!]+');

    if (regex.test(message)) {

        $("#alertExample").fadeIn("fast", function () {
            setTimeout(() => $("#alertExample").fadeOut("slow", function () { }), 3000);
        });

        return false;
    }
    return true;
}

function sendMyMessage(message) {
    // validation check
    if (message == '') {
        return false;
    }

    if (validationCheck(message)) {
        let today = new Date();
        var hours = ('0' + today.getHours()).slice(-2);
        var minutes = ('0' + today.getMinutes()).slice(-2);
        var timeString = hours + ':' + minutes;

        // add message

        htmlTags = $(".card-body").html();
        htmlTags += "<div class='d-flex flex-row justify-content-end'>";
        htmlTags += "<div>";
        htmlTags += "  <p class='small p-2 me-3 mb-1 text-white rounded-3 bg-primary'>";
        htmlTags += message + "</p>";
        htmlTags += "  <p class='small me-3 mb-3 rounded-3 text-muted d-flex justify-content-end'>"
        htmlTags += timeString + "</p>";
        htmlTags += "</div>";
        htmlTags += "<img src='https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-chat/ava4-bg.webp'";
        htmlTags += "  alt='avatar 1' style='width: 45px; height: 100%;'>";
        htmlTags += "</div>";

        $(".card-body").html(htmlTags);

        // scroll down
        $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');

        // send message
    }
}

function sendAnichatMessage() {

}


function addRecordMessage(record) {
    let today = new Date();
    var hours = ('0' + today.getHours()).slice(-2);
    var minutes = ('0' + today.getMinutes()).slice(-2);
    var timeString = hours + ':' + minutes;

    // add message

    htmlTags = $(".card-body").html();
    htmlTags += "<div class='d-flex flex-row justify-content-end'>";
    htmlTags += "<div>";
    htmlTags += "  <p class='small p-2 me-3 mb-1 text-white rounded-3 bg-primary'>";
    htmlTags += message + "</p>";
    htmlTags += "  <p class='small me-3 mb-3 rounded-3 text-muted d-flex justify-content-end'>"
    htmlTags += timeString + "</p>";
    htmlTags += "</div>";
    htmlTags += "<img src='https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-chat/ava4-bg.webp'";
    htmlTags += "  alt='avatar 1' style='width: 45px; height: 100%;'>";
    htmlTags += "</div>";

    $(".card-body").html(htmlTags);

    // scroll down
    $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');

    // send message
}

function clickRecord() {
    if ($("#recordFlag").val() == '0') {
        $("#recordButton").addClass('Blink'); 
        $("#recordButton > i").addClass('red');
        // start record
        startRecording();
        $("#recordFlag").val('1');
    } else {
        $("#recordButton").removeClass('Blink'); 
        $("#recordButton > i").removeClass('red');
        stopRecording();
        $("#recordFlag").val('0');
    }

}

function startRecording() {
    console.log("recordButton clicked");

    // Disable the record button until we get a success or fail from getUserMedia()
    // recordButton.disabled = true;
    // stopButton.disabled = false;

    
    navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        audioContext = new AudioContext({ sampleRate: 16000 });

        // assign to gumStream for later use
        gumStream = stream;

        // use the stream
        input = audioContext.createMediaStreamSource(stream);

        // Create the Recorder object and configure to record mono sound (1 channel) Recording 2 channels will double the file size
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton.disabled = false;
        stopButton.disabled = true;
    });

}
function stopRecording() {
    console.log("stopButton clicked");

    //disable the stop button, enable the record too allow for new recordings
    // stopButton.disabled = true;
    // recordButton.disabled = false;

    //tell the recorder to stop the recording
    rec.stop(); //stop microphone access
    gumStream.getAudioTracks()[0].stop();

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createMessageLink);

}

function createMessageLink(blob) {
    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');
    var recordingsList = document.getElementById("recordingsList");
    //name of .wav file to use during upload and download (without extension)
    var filename = new Date().toISOString();

    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    /*
    //save to disk link
    link.href = url;
    link.download = filename + ".wav"; //download forces the browser to download the file using the  filename
    link.innerHTML = "Save to disk";

    //add the new audio element to li
    li.appendChild(au);

    //add the filename to the li
    li.appendChild(document.createTextNode(filename + ".wav "))

    //add the save to disk link to li
    li.appendChild(link);

    //add the li element to the ol
    recordingsList.appendChild(li);
    */

    let today = new Date();
    var hours = ('0' + today.getHours()).slice(-2);
    var minutes = ('0' + today.getMinutes()).slice(-2);
    var timeString = hours + ':' + minutes;

    var cardBody = document.getElementsByClassName("card-body")[0];

    // add message

    var divFlex = document.createElement('div');
    divFlex.className = "d-flex flex-row justify-content-end";

    var divTemp = document.createElement('div');

    var pSmall2 = document.createElement('p');
    pSmall2.className = "p class='small me-3 mb-3 rounded-3 text-muted d-flex justify-content-end";
    pSmall2.innerText = timeString;

    var imgAvatar1 = document.createElement('img');
    imgAvatar1.style.width = '45px';
    imgAvatar1.style.height = '100%';
    imgAvatar1.src = 'https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-chat/ava4-bg.webp';
    var div = document.createElement('div');
    div.style.marginRight= '17px';
    div.appendChild(au);
    divTemp.appendChild(div);
    divTemp.appendChild(pSmall2);
    divFlex.appendChild(divTemp);
    divFlex.appendChild(imgAvatar1);

    cardBody.appendChild(divFlex);

    // scroll down
    $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');

}
