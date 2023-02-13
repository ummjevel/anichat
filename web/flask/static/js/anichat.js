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

var recordTimeout;
var conan_talk_image = './static/img/profile_conan.png';
var you_talk_image = './static/img/profile_you.png';
var nam_talk_image = './static/img/profile_nam.png';

$(document).ready(function () {

    var modal = document.getElementById("modalCharacter");
    var modalRecorder = document.getElementById("modalRecorder");
    var modalHelper = document.getElementById("modalHelper");

    window.onclick = function (event) {
        if (event.target == modal) {
            $("#modalCharacter").css('display', 'none');
        }
        /*if (event.target == modalRecorder) {
            $("#modalRecorder").css('display', 'none');
        }*/
        if(event.target == document.getElementsByClassName("modalRecorderContent")[0]) {
            $("#modalRecorder").css('display', 'none');
        }
        if (event.target == modalHelper) {
            $("#modalHelper").css('display', 'none');
        }
    };


    current_profile = conan_talk_image;
    changeProfile();

    // if url is select and hidden value is true then click help


    $("#btnSend").click(function () {
        var message = $("#exampleFormControlInput1").val();
        var use_tts = false;
        if ($("#btnTTS").val() == "on" || $("#btnTTS").val() == "") {
            use_tts = true;
        }
        var choose = 'conan';
        if (current_profile == you_talk_image) {
            choose = 'you';
        }

        sendMyMessage(message, use_tts, choose);
    });

    // press enter then send message
    $('#exampleFormControlInput1').keypress(function (e) {
        var key = e.which;
        if(key == 13) { // the enter key code
            $('#btnSend').click();
        }
    });  

    // When the user clicks the button, open the modal
    $(".div_character").click(function () {
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
     
    $("#imgConanProfile").click(function() {
        replaceClass("chat2", "you_card", "conan_card");
        replaceClass("chat2", "nam_card", "conan_card");
        $('h5').text('CONAN');
        $("#modalCharacter").css('display', 'none');
        $("#currentProfile").val(0);
        changeProfile();
        // changeCharacter('conan');
        $(this).attr('src', "./static/img/change_to_conan.png");
        $("#imgYouProfile").attr('src', './static/img/before_select_you.png');
    });
    
    $("#imgYouProfile").click(function() {
        replaceClass("chat2", "conan_card", "you_card");
        replaceClass("chat2", "nam_card", "you_card");
        $('h5').text('KOGORO');
        $("#modalCharacter").css('display', 'none');
        $("#currentProfile").val(1);
        changeProfile();
        // changeCharacter('you');  
        $(this).attr('src', "./static/img/change_to_you.png");
        $("#imgConanProfile").attr('src', './static/img/before_select_conan.png');
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

    $(".btnHelp").click(function() {
        $("#modalHelper").css('display', 'block');
    });

    $(".btnPrevious").click(function() {
        location.href = "/select"
    });

    $("#btnTTS").click(function() {
        var previousClass = 'fa-volume-high';
        var nextClass = 'fa-volume-xmark';
        if ($("#btnTTS").val() == "off") {
            previousClass = 'fa-volume-xmark';
            nextClass = 'fa-volume-high';
            $("#btnTTS").val("on");
        } else {
            $("#btnTTS").val("off");
        }
        $("#btnTTS > i").removeClass(previousClass);
        $("#btnTTS > i").addClass(nextClass)
    });

    $("#btnMimic").click(function() {
        var message = $("#exampleFormControlInput1").val();
        sendMimic(message);
        $("#exampleFormControlInput1").val("");
    });

    $("#divDiveToConan").click(function() {
        if (navigator.userAgent.match(/Android/i)
         || navigator.userAgent.match(/webOS/i)
         || navigator.userAgent.match(/iPhone/i)
         || navigator.userAgent.match(/iPad/i)
         || navigator.userAgent.match(/iPod/i)
         || navigator.userAgent.match(/BlackBerry/i)
         || navigator.userAgent.match(/Windows Phone/i)) {
            location.href = "/webchat";
         } else {
            
            location.href = "/chatbot";
         }
    });

    $("#imgConanProfile").hover(function() {
        if (current_profile != conan_talk_image) {
            $(this).attr('src', './static/img/change_to_gray_conan.png');
        } else {
            $(this).attr('src', './static/img/change_to_conan.png');
        }
        $($(".div-profile-img-parent > p")[0]).addClass('Blink');
        $($(".div-profile-img-parent > p")[0]).append("<img src='./static/img/character_select_button.png' class='arrow_btn'/>");

    }, function() {
        if (current_profile != conan_talk_image) {
            $(this).attr('src', './static/img/before_select_conan.png');
        } else {
            $(this).attr('src', './static/img/change_to_conan.png');
        }
        $($(".div-profile-img-parent > p")[0]).removeClass('Blink');
        $($(".div-profile-img-parent > p > img")[0]).remove();
    });

    $("#imgYouProfile").hover(function() {
        if (current_profile != you_talk_image) {
            $(this).attr('src', './static/img/change_to_gray_you.png');
        } else {
            $(this).attr('src', './static/img/change_to_you.png');
        }
        $($(".div-profile-img-parent > p")[1]).addClass('Blink');
        $($(".div-profile-img-parent > p")[1]).append("<img src='./static/img/character_select_button.png' class='arrow_btn'/>");
    }, function() {
        if (current_profile != you_talk_image) {
            $(this).attr('src', './static/img/before_select_you.png');
        } else {
            $(this).attr('src', './static/img/change_to_you.png');
        }
        $($(".div-profile-img-parent > p")[1]).removeClass('Blink');
        $($(".div-profile-img-parent > p > img")[0]).remove();
    });

    $(".divPosterConan").hover(function() {
        $(".divPosterConan > img").attr('src', './static/img/conan_poster_hover.png');
    }, function() {
        $(".divPosterConan > img").attr('src', './static/img/conan_poster.png');
    });

    $(".divPosterOnePiece").hover(function() {
        $(".divPosterOnePiece > img").attr('src', './static/img/onepiece_poster_hover.png');
    }, function() {
        $(".divPosterOnePiece > img").attr('src', './static/img/onepiece_poster.png');
    });

    $(".divPosterNaruto").hover(function() {
        $(".divPosterNaruto > img").attr('src', './static/img/naruto_poster_hover.png');
    }, function() {
        $(".divPosterNaruto > img").attr('src', './static/img/naruto_poster.png');
    });

    $(".divPosterBleach").hover(function() {
        $(".divPosterBleach > img").attr('src', './static/img/bleach_poster_hover.png');
    }, function() {
        $(".divPosterBleach > img").attr('src', './static/img/bleach_poster.jpg');
    });

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

function sendMyMessage(message, use_tts, choose) {
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

        htmlTags = $("#cardBody").html();
        htmlTags += "<div class='d-flex flex-row justify-content-end '>";
        htmlTags += "<div style='display: inline-grid;'>";
        htmlTags += "  <p class='small p-2 me-3 mb-1 text-black rounded-3 text-back balloon'>";
        htmlTags += message + "</p>";
        htmlTags += "  <p class='small me-3 mb-3 rounded-3 text-white d-flex justify-content-end'>"
        htmlTags += timeString + "</p>";
        htmlTags += "</div>";
        // htmlTags += "<img src='https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-chat/ava4-bg.webp'";
        // htmlTags += "  alt='avatar 1' style='width: 45px; height: 100%;'>";
        htmlTags += "</div>";

        $("#cardBody").html(htmlTags);

        // scroll down
        $('#cardBody').animate({ scrollTop: document.getElementById("cardBody").scrollHeight }, 'fast');

        $("#exampleFormControlInput1").val("");

        // send message
        $.ajax({
            url: "sendChat",
            type: "post",
            accept: "application/json",
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify({'use_tts': use_tts, 'message': message, "choose": choose}),
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
        $(question_message).insertBefore($($($("audio")[$("audio").length - 1]).parent()[$($("audio")[$("audio").length - 1]).parent().length - 1]));
    }

    // add message
    htmlTags = $("#cardBody").html();
    htmlTags += "<div class='d-flex flex-row justify-content-start mb-4' style=''>";
    htmlTags += "<img src='" + current_profile + "'";
    htmlTags += "  alt='avatar 1' style='' class='img_profile_character'>";
    htmlTags += "<div>";
    htmlTags += "  <p class='small p-2 ms-3 mb-1 rounded-3 balloon-ai' style=''>";
    htmlTags += data.message + "</p>";
    // if use tts, add audio.
    if (data.use_tts == true || data.use_tts == "true") {
        if (location.pathname == "/webchat") {
            htmlTags += "<div class='div-audio' ><audio controls='' src='" + data.wav_file + "' class='div-in-audio'></audio>";
        } else {
            htmlTags += "<div style='margin-right: 17px;' ><audio controls='' src='" + data.wav_file + "'></audio>";
        }
        htmlTags += "<div style='display:inline'><a href='#' class='audio-a'></a></div><p class='audio-p'>";
        htmlTags += "<a href='" + '/download/' + data.wav_file.replace('/static/record/', '') + "' class='audioDown'>";
        htmlTags += "<i class='fa-solid fa-arrow-down audio-i'></i></a></p>";
        htmlTags += "</div>";
    }
    
    if (location.pathname == "/webchat" && ((data.use_tts == true || data.use_tts == "true"))) {
        htmlTags += "  <p class='small ms-3 mb-3 rounded-3 text-white text-p'>";
    } else {
        htmlTags += "  <p class='small ms-3 mb-3 rounded-3 text-white'>";
    }
    htmlTags += timeString + "</p>";
    htmlTags += "</div>";
    htmlTags += "</div>";

    $("#cardBody").html(htmlTags);

    // scroll down
    $('#cardBody').animate({ scrollTop: document.getElementById("cardBody").scrollHeight }, 'fast');

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

    htmlTags = $("#cardBody").html();
    htmlTags += "<div class='d-flex flex-row justify-content-end '>";
    htmlTags += "<div>";
    htmlTags += "  <p class='small p-2 me-3 mb-1 text-black rounded-3 text-back balloon '>";
    htmlTags += message + "</p>";
    htmlTags += "  <p class='small me-3 mb-3 rounded-3 text-white'>"
    htmlTags += timeString + "</p>";
    htmlTags += "</div>";
    // htmlTags += "<img src='https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-chat/ava4-bg.webp'";
    // htmlTags += "  alt='avatar 1' style='width: 45px; height: 100%;'>";
    htmlTags += "</div>";

    $("#cardBody").html(htmlTags);

    // scroll down
    $('#cardBody').animate({ scrollTop: document.getElementById("cardBody").scrollHeight }, 'fast');

    // send message

    $("#exampleFormControlInput1").val("");
}


function clickRecord() {
    
    if ($("#recordFlag").val() == '0') {
        $("#btnRecord").addClass('Blink'); 
        $(".gifVoice").attr('src', './static/img/voice.gif');
        // $("#btnRecord > i").addClass('red');
        // start record
        startRecording();
        $("#recordFlag").val('1');
        // limit record time
        recordTimeout = setTimeout(() => {
            if ($("#btnRecord").hasClass('Blink')) {
                $("#btnRecord").removeClass('Blink'); 
                $(".gifVoice").attr('src', './static/img/voice.png');
                // $("#btnRecord > i").removeClass('red');
                clearTimeout(recordTimeout);
                $("#recordFlag").val('0');
                stopRecording();
            }
        }, 12000);
    } else {
        $("#btnRecord").removeClass('Blink'); 
        $(".gifVoice").attr('src', './static/img/voice.png');
        // $("#btnRecord > i").removeClass('red');
        clearTimeout(recordTimeout);
        $("#recordFlag").val('0');
        stopRecording();
    
    }
}


function startRecording() {
    console.log("recordButton clicked");

    // Disable the record button until we get a success or fail from getUserMedia()
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

    //tell the recorder to stop the recording
    rec.stop(); //stop microphone access
    gumStream.getAudioTracks()[0].stop();

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createMessageLink);

    // rec.clear();
}

function createMessageLink(blob) {
    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    //name of .wav file to use during upload and download (without extension)
    var filename = new Date().toISOString();

    //add controls to the <audio> element
    au.controls = true;
    au.src = url;
    au.id = "audioPreRecorded";
    current_blob = blob;
    current_record_url = url;
    console.log(current_record_url);

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

    var cardBody = document.getElementById("cardBody");

    // add message

    var divFlex = document.createElement('div');
    divFlex.className = "d-flex flex-row justify-content-end";

    var divTemp = document.createElement('div');

    var pSmall2 = document.createElement('p');
    pSmall2.className = "small me-3 mb-3 rounded-3 text-white d-flex justify-content-end ";
    pSmall2.innerText = timeString;

    // var imgAvatar1 = document.createElement('img');
    // imgAvatar1.style.width = '45px';
    // imgAvatar1.style.height = '100%';
    // imgAvatar1.src = 'https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-chat/ava4-bg.webp';
    var div = document.createElement('div');
    // div.style.marginRight= '17px';
    div.appendChild(au);
    divTemp.appendChild(div);
    divTemp.appendChild(pSmall2);
    divFlex.appendChild(divTemp);
    // divFlex.appendChild(imgAvatar1);

    cardBody.appendChild(divFlex);

    // scroll down
    $('#cardBody').animate({ scrollTop: document.getElementById("cardBody").scrollHeight }, 'fast');

    // send message
    var message = $("#exampleFormControlInput1").val();
    var use_tts = "false";
    if ($("#btnTTS").val() == "on" || $("#btnTTS").val() == "") {
        use_tts = "true";
    }
    var choose = 'conan';
    if (current_profile == you_talk_image) {
        choose = 'you';
    }

    let formData = new FormData();
    formData.append('data', blob);

    var data = {   
        "use_tts" : use_tts,
        "user_stt" : true,
        "choose" : choose
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

    htmlTags = $("#cardBody").html();
    htmlTags += "<div class='d-flex flex-row justify-content-start '>";
    htmlTags += "<img src='" + current_profile + "'";
    htmlTags += "  alt='avatar 1' style='' class='img_profile_character'>";
    htmlTags += "<div>";
    htmlTags += "  <p class='small p-2 ms-3 mb-1 rounded-3 balloon-ai' style='position: absolute;width: 100px;height: 37px;'>";
    htmlTags += "<div class='dot-flashing'></div>" + "</p>";
    htmlTags += "  <p class='small ms-3 mb-3 rounded-3 text-white'>"
    htmlTags += "</div>";
    htmlTags += "</div>";
    $("#cardBody").html(htmlTags);
    $('#cardBody').animate({ scrollTop: document.getElementById("cardBody").scrollHeight }, 'fast');
    setTimeout(() => {
        var divCardbody = $("#cardBody > div");
        divCardbody[divCardbody.length - 1].remove();
        htmlTags = $("#cardBody").html();
        htmlTags += "<div class='d-flex flex-row justify-content-start ' style=''>";
        htmlTags += "<img src='" + current_profile +"'";
        htmlTags += "  alt='avatar 1' style='' class='img_profile_character'>";
        htmlTags += "<div>";
        htmlTags += "  <p class='small p-2 ms-3 mb-1 rounded-3 balloon-ai' style=''>";
        htmlTags += "ㅇㅇ" + "</p>";
        htmlTags += "  <p class='small ms-3 mb-3 rounded-3 text-white'>"
        htmlTags += timeString + "</p>";
        htmlTags += "</div>";
        htmlTags += "</div>";
        $("#cardBody").html(htmlTags);
        $('#cardBody').animate({ scrollTop: document.getElementById("cardBody").scrollHeight }, 'fast');
    }, 2000);

}

function addLoading() {
    htmlTags = $("#cardBody").html();
    htmlTags += "<div class='d-flex flex-row justify-content-start mb-4' id='dot-flashing'>";
    htmlTags += "<img src='" + current_profile + "'";
    htmlTags += "  alt='avatar 1' style='' class='img_profile_character'>";
    htmlTags += "<div>";
    htmlTags += "  <p class='small p-2 ms-3 mb-1 rounded-3 balloon-ai' style='position: absolute;width: 100px;height: 37px;'>";
    htmlTags += "<div class='dot-flashing'></div>" + "</p>";
    htmlTags += "  <p class='small ms-3 mb-3 rounded-3 text-white'>"
    htmlTags += "</div>";
    htmlTags += "</div>";
    $("#cardBody").html(htmlTags);
    $('#cardBody').animate({ scrollTop: document.getElementById("cardBody").scrollHeight }, 'fast');
}

function deleteLoading() {
    $("#dot-flashing").remove();
    
}

function changeCharacter(character) {
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
}

function changeProfile() {
    var current_profile_value = $("#currentProfile").val();
    if (current_profile_value == 0) {
        current_profile = conan_talk_image;
    } else if (current_profile_value == 1) {
        current_profile = you_talk_image;
    } else if (current_profile_value == 2) {
        current_profile = nam_talk_image;
    }
}

function sendMimic(message) {

    if (message == '') {
        return false;
    }
    var choose = 'conan';
    if (current_profile == you_talk_image) {
        choose = 'you';
    }

    if (validationCheck(message)) {
        $.ajax({
            url: "sendMimic",
            type: "post",
            accept: "application/json",
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify({'message': message, 'choose': choose}),
            dataType: "json",
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
}