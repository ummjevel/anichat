// const e = require("cors");

URL = window.URL || window.webkitURL;

var gumStream;              //stream from getUserMedia()
var rec;                    //Recorder.js object
var input;                  //MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //new audio context to help us record

var current_profile;
var current_blob, current_record_url;

$(document).ready(function () {

    $("#btnSend").click(function () {
        var message = $("#exampleFormControlInput1").val();
        var use_tts = false;
        if ($("#flexSwitchCheckDefault").is(":checked") == true) {
            use_tts = true;
        }

        sendMyMessage(message, use_tts)
    });

    // press enter then send message
    $('#exampleFormControlInput1').keypress(function (e) {
        var key = e.which;
        if(key == 13) { // the enter key code
            $('#btnSend').click();
        }
    });  

    $('#flexSwitchCheckDefaultMori').change(function() {
        if (this.checked) {
            // turn on you
            character = 'you';
        } else {
            // turn on conan
            character = 'conan';
        }    
        console.log(character);
        $.ajax({
            url: "turnTTS",
            type: "post",
            accept: "application/json",
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify({'character': character}),
            dataType: "json",
            success: function(data) {
                console.log(data);
            },
            error: function(jqXHR, textStatus, errorThrown) {
                console.log(jqXHR);
            }
        });   
    });

    // When the user clicks the button, open the modal
    $("#btnSelectCharacter").click(function () {
        $("#modalCharacter").css('display', 'block');
    });

    // When the user clicks on <span> (x), close the modal
    $(".characterClose").click(function () {
        $("#modalCharacter").css('display', 'none');
    });

    // When the user clicks the button, change the background img
    function replaceClass(id, oldClass, newClass) {
        var elem = $(`#${id}`);
        if (elem.hasClass(oldClass)) {
            elem.removeClass(oldClass);
        }
        elem.addClass(newClass);
    }
     
    $("#character_change_conan").click(function() {
        replaceClass("chat2", "you_card", "conan_card");
        replaceClass("chat2", "nam_card", "conan_card");
        $('h5').text('CONAN');
        $("#modalCharacter").css('display', 'none');
    });
    
    $("#character_change_you").click(function() {
        replaceClass("chat2", "conan_card", "you_card");
        replaceClass("chat2", "nam_card", "you_card");
        $('h5').text('KOGORO');
        $("#modalCharacter").css('display', 'none');
    });

    $("#character_change_nam").click(function() {
        replaceClass("chat2", "conan_card", "nam_card");
        replaceClass("chat2", "you_card", "nam_card");
        $('h5').text('KUDO');
        $("#modalCharacter").css('display', 'none');
    });

    $("#recordButton").click(function() {
        // clickRecord();
        $("#modalRecorder").css('display', 'block');
    });

    // When the user clicks on <span> (x), close the modal
    $(".recorderClose").click(function () {
        $("#modalRecorder").css('display', 'none');
    });

    $("#btnRecord").click(function() {
        $("#btnRecordReload").click();
        clickRecord();
    })

    $("#btnRecordReload").click(function() {
        $("#divPreRecord > img").css('display', 'block');
        $("#divPreRecord > audio").remove();
        $("#btnRecordReload").css('display', 'none');
        $("#btnRecordUpload").css('display', 'none');
    });

    $("#btnRecordUpload").click(function() {
        uploadRecord();

    });

    var modal = document.getElementById("modalCharacter");
    var modalRecorder = document.getElementById("modalRecorder");

    window.onclick = function (event) {
        if (event.target == modal) {
            $("#modalCharacter").css('display', 'none');
        }
        if (event.target == modalRecorder) {
            $("#modalRecorder").css('display', 'none');
        }
    };

    var conan_talk_image = './static/img/profile_conan.png';
    var you_talk_image = './static/img/profile_you.png';
    var nam_talk_image = './static/img/profile_nam.png';

    current_profile = conan_talk_image;
    var current_profile_value = $("#currentProfile").val();
    if (current_profile_value == 0) {
        current_profile = conan_talk_image;
    } else if (current_profile_value == 1) {
        current_profile = you_talk_image;
    } else if (current_profile_value == 2) {
        current_profile = nam_talk_image;
    }

});

function validationCheck(message) {
    // 이상한 문자 검열하기.
    var regex = new RegExp('[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ0-9\\s.,;?!]+');

    if (regex.test(message)) {
        $("#alertExample").fadeIn('fast');
        setTimeout(() => $("#alertExample").fadeOut("slow"), 2000);
        
        return false;
    }
    return true;
}

function sendMyMessage(message, use_tts) {
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
        htmlTags += "<div class='d-flex flex-row justify-content-end mb-4'>";
        htmlTags += "<div>";
        htmlTags += "  <p class='small p-2 me-3 mb-1 text-black rounded-3 text-back balloon'>";
        htmlTags += message + "</p>";
        htmlTags += "  <p class='small me-3 mb-3 rounded-3 text-white d-flex justify-content-end'>"
        htmlTags += timeString + "</p>";
        htmlTags += "</div>";
        // htmlTags += "<img src='https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-chat/ava4-bg.webp'";
        // htmlTags += "  alt='avatar 1' style='width: 45px; height: 100%;'>";
        htmlTags += "</div>";

        $(".card-body").html(htmlTags);

        // scroll down
        $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');

        $("#exampleFormControlInput1").val("");

        // send message
        $.ajax({
            url: "sendChat",
            type: "post",
            accept: "application/json",
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify({'use_tts': use_tts, 'message': message}),
            dataType: "json",
            success: function(data) {
                console.log(data)
                sendAnichatMessage(data);
            },
            error: function(jqXHR, textStatus, errorThrown) {
                console.log(jqXHR);
            }
        });
    }
}

function sendAnichatMessage(data) {

    console.log(data);

    let today = new Date();
    var hours = ('0' + today.getHours()).slice(-2);
    var minutes = ('0' + today.getMinutes()).slice(-2);
    var timeString = hours + ':' + minutes;

    // add stt message
    if (data.hasOwnProperty('use_stt')) {
        console.log('this use stt');
        console.log(data.question);
        question_message = "<p class='small p-2 me-3 mb-1 text-black rounded-3 text-back balloon'>";
        question_message += data.question + "</p>";
        // var ttsP = document.createElement("p");
        // ttsP.classList.add('small', 'p-2', 'me-3', 'mb-1', 'text-black', 'rounded-3', 'text-back');
        $(question_message).insertBefore($($("audio")[$("audio").length - 1]));
    }

    // add message
    htmlTags = $(".card-body").html();
    htmlTags += "<div class='d-flex flex-row justify-content-start mb-4' style='margin-bottom:-24px;'>";
    htmlTags += "<img src='" + current_profile + "'";
    htmlTags += "  alt='avatar 1' style='' class='img_profile_character'>";
    htmlTags += "<div>";
    htmlTags += "  <p class='small p-2 ms-3 mb-1 rounded-3 balloon-ai' style='background-color: #F5F6F7; '>";
    htmlTags += data.message + "</p>";
    // if use tts, add audio.
    if (data.use_tts == true || data.use_tts == "true") {
        htmlTags += "<div style='margin-right: 17px;'><audio controls='' src='" + data.wav_file + "'></audio></div>"
    }
    htmlTags += "  <p class='small ms-3 mb-3 rounded-3 text-white'>"
    htmlTags += timeString + "</p>";
    htmlTags += "</div>";
    htmlTags += "</div>";

    $(".card-body").html(htmlTags);

    // scroll down
    $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');

    if (data.use_tts == true || data.use_tts == "true") {
        $("audio")[$("audio").length - 1].play();
    }
}


function addRecordMessage(record) {
    let today = new Date();
    var hours = ('0' + today.getHours()).slice(-2);
    var minutes = ('0' + today.getMinutes()).slice(-2);
    var timeString = hours + ':' + minutes;

    // add message

    htmlTags = $(".card-body").html();
    htmlTags += "<div class='d-flex flex-row justify-content-end mb-4'>";
    htmlTags += "<div>";
    htmlTags += "  <p class='small p-2 me-3 mb-1 text-black rounded-3 text-back balloon '>";
    htmlTags += message + "</p>";
    htmlTags += "  <p class='small me-3 mb-3 rounded-3 text-white'>"
    htmlTags += timeString + "</p>";
    htmlTags += "</div>";
    // htmlTags += "<img src='https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-chat/ava4-bg.webp'";
    // htmlTags += "  alt='avatar 1' style='width: 45px; height: 100%;'>";
    htmlTags += "</div>";

    $(".card-body").html(htmlTags);

    // scroll down
    $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');

    // send message

    $("#exampleFormControlInput1").val("");
}


function clickRecord() {
    
    if ($("#recordFlag").val() == '0') {
        $("#btnRecord").addClass('Blink'); 
        // $("#btnRecord > i").addClass('red');
        // start record
        startRecording();
        $("#recordFlag").val('1');
        // limit record time
        setTimeout(() => {
            if ($("#btnRecord").hasClass('Blink')) {
                $("#btnRecord").removeClass('Blink'); 
                // $("#btnRecord > i").removeClass('red');
                stopRecording();
                $("#recordFlag").val('0');
            }
        }, 10000);
    } else {
        $("#btnRecord").removeClass('Blink'); 
        // $("#btnRecord > i").removeClass('red');
        stopRecording();
        $("#recordFlag").val('0');
    
    }

}


function startRecording() {
    console.log("recordButton clicked");

    // Disable the record button until we get a success or fail from getUserMedia()
    // recordButton.disabled = true;
    // stopButton.disabled = false;
    if (navigator.mediaDevices === undefined) {
        navigator.mediaDevices = {};
    }

    if (navigator.mediaDevices.getUserMedia === undefined) {
        navigator.mediaDevices.getUserMedia = function(constraints) {
    
            // First get ahold of the legacy getUserMedia, if present
            var getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    
            // Some browsers just don't implement it - return a rejected promise with an error
            // to keep a consistent interface
            if (!getUserMedia) {
                return Promise.reject(new Error('getUserMedia is not implemented in this browser'));
            }
    
            // Otherwise, wrap the call to the old navigator.getUserMedia with a Promise
            return new Promise(function(resolve, reject) {
                getUserMedia.call(navigator, constraints, resolve, reject);
            });
        }
    }

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
        // recordButton.disabled = false;
        // stopButton.disabled = true;
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
    au.id = "audioPreRecorded";
    current_blob = blob;
    current_record_url = url;

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

    /*
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
    pSmall2.className = "small me-3 mb-3 rounded-3 text-white d-flex justify-content-end mb-4";
    pSmall2.innerText = timeString;

    // var imgAvatar1 = document.createElement('img');
    // imgAvatar1.style.width = '45px';
    // imgAvatar1.style.height = '100%';
    // imgAvatar1.src = 'https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-chat/ava4-bg.webp';
    var div = document.createElement('div');
    div.style.marginRight= '17px';
    div.appendChild(au);
    divTemp.appendChild(div);
    divTemp.appendChild(pSmall2);
    divFlex.appendChild(divTemp);
    // divFlex.appendChild(imgAvatar1);

    cardBody.appendChild(divFlex);

    // scroll down
    $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');

    // send message
    var message = $("#exampleFormControlInput1").val();
    var use_tts = "false";
    if ($("#flexSwitchCheckDefault").is(":checked") == true) {
        use_tts = "true";
    }
    var use_stt = "false";
    if ($("#flexSwitchCheckDefaultSTT").is(":checked") == true) {
        use_stt = "true";
    }

    let formData = new FormData();
    formData.append('data', blob);

    var data = {   
        "use_tts" : use_tts,
        "user_stt" : use_stt
    }
    formData.append('key', new Blob([ JSON.stringify(data) ], {type : "application/json"}));
    

    addLoading();
    $.ajax({
        type: 'POST',
        url: 'sendSTT',
        data: formData,
        contentType: false,
        processData: false,
        success: function(data) {
            console.log('success', data);
            deleteLoading();
            sendAnichatMessage(data);
        },
        error: function(result) {
            deleteLoading();
            alert('sorry an error occured');
        }
    });
    
*/
    // var div = document.getElementById('divPreRecord');
    // div.appendChild(au);
    $("#divPreRecord > img").css('display', 'none');
    $("#divPreRecord").append(au);
    $("#btnRecordReload").css('display', 'block');
    $("#btnRecordUpload").css('display', 'block');

}

function uploadRecord() {
    // close modal
    $("#modalRecorder").css('display', 'none');
    // move audio
    var au = document.createElement('audio');
    au.controls = true;
    au.src = current_record_url;

    var blob = current_blob;
    // delete prerecord audio
    $("#btnRecordReload").click();

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
    pSmall2.className = "small me-3 mb-3 rounded-3 text-white d-flex justify-content-end mb-4";
    pSmall2.innerText = timeString;

    // var imgAvatar1 = document.createElement('img');
    // imgAvatar1.style.width = '45px';
    // imgAvatar1.style.height = '100%';
    // imgAvatar1.src = 'https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-chat/ava4-bg.webp';
    var div = document.createElement('div');
    div.style.marginRight= '17px';
    div.appendChild(au);
    divTemp.appendChild(div);
    divTemp.appendChild(pSmall2);
    divFlex.appendChild(divTemp);
    // divFlex.appendChild(imgAvatar1);

    cardBody.appendChild(divFlex);

    // scroll down
    $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');

    // send message
    var message = $("#exampleFormControlInput1").val();
    var use_tts = "false";
    if ($("#flexSwitchCheckDefault").is(":checked") == true) {
        use_tts = "true";
    }
    var use_stt = "false";
    if ($("#flexSwitchCheckDefaultSTT").is(":checked") == true) {
        use_stt = "true";
    }

    let formData = new FormData();
    formData.append('data', blob);

    var data = {   
        "use_tts" : use_tts,
        "user_stt" : use_stt
    }
    formData.append('key', new Blob([ JSON.stringify(data) ], {type : "application/json"}));
    

    addLoading();
    $.ajax({
        type: 'POST',
        url: 'sendSTT',
        data: formData,
        contentType: false,
        processData: false,
        success: function(data) {
            console.log('success', data);
            deleteLoading();
            sendAnichatMessage(data);
        },
        error: function(result) {
            deleteLoading();
            alert('sorry an error occured');
        }
    });
}

function btclick() {

    // loading 2 seconds
    let today = new Date();
    var hours = ('0' + today.getHours()).slice(-2);
    var minutes = ('0' + today.getMinutes()).slice(-2);
    var timeString = hours + ':' + minutes;

    // add message

    htmlTags = $(".card-body").html();
    htmlTags += "<div class='d-flex flex-row justify-content-start mb-4'>";
    htmlTags += "<img src='" + current_profile + "'";
    htmlTags += "  alt='avatar 1' style='' class='img_profile_character'>";
    htmlTags += "<div>";
    htmlTags += "  <p class='small p-2 ms-3 mb-1 rounded-3 balloon-ai' style='background-color: #F5F6F7;position: absolute;width: 100px;height: 37px;'>";
    htmlTags += "<div class='dot-flashing'></div>" + "</p>";
    htmlTags += "  <p class='small ms-3 mb-3 rounded-3 text-white'>"
    htmlTags += "</div>";
    htmlTags += "</div>";
    $(".card-body").html(htmlTags);
    $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');
    setTimeout(() => {
        var divCardbody = $(".card-body > div");
        divCardbody[divCardbody.length - 1].remove();
        htmlTags = $(".card-body").html();
        htmlTags += "<div class='d-flex flex-row justify-content-start mb-4' style='margin-bottom:-24px;'>";
        htmlTags += "<img src='" + current_profile +"'";
        htmlTags += "  alt='avatar 1' style='' class='img_profile_character'>";
        htmlTags += "<div>";
        htmlTags += "  <p class='small p-2 ms-3 mb-1 rounded-3 balloon-ai' style='background-color: #F5F6F7; '>";
        htmlTags += "ㅇㅇ" + "</p>";
        htmlTags += "  <p class='small ms-3 mb-3 rounded-3 text-white'>"
        htmlTags += timeString + "</p>";
        htmlTags += "</div>";
        htmlTags += "</div>";
        $(".card-body").html(htmlTags);
        $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');
    }, 2000);

}

function addLoading() {
    htmlTags = $(".card-body").html();
    htmlTags += "<div class='d-flex flex-row justify-content-start' id='dot-flashing'>";
    htmlTags += "<img src='" + current_profile + "'";
    htmlTags += "  alt='avatar 1' style='' class='img_profile_character'>";
    htmlTags += "<div>";
    htmlTags += "  <p class='small p-2 ms-3 mb-1 rounded-3 balloon-ai' style='background-color: #F5F6F7;position: absolute;width: 100px;height: 37px;'>";
    htmlTags += "<div class='dot-flashing'></div>" + "</p>";
    htmlTags += "  <p class='small ms-3 mb-3 rounded-3 text-white'>"
    htmlTags += "</div>";
    htmlTags += "</div>";
    $(".card-body").html(htmlTags);
    $('.card-body').animate({ scrollTop: document.getElementsByClassName("card-body")[0].scrollHeight }, 'fast');
}

function deleteLoading() {
    $("#dot-flashing").remove();
    
}