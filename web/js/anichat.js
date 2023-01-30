
$(document).ready(function() {

    $( "#btnSend" ).click(function() {
        var message = $("#exampleFormControlInput1").val();
        sendMyMessage(message)
    });

});


function validationCheck(message) {
    // 이상한 문자 검열하기.
    var regex = new RegExp('[^가-힣ㄱ-ㅎ0-9\s\t\n.,;?!]+');

    if (regex.test(message)){

        $( "#alertExample" ).fadeIn( "fast", function() {
            setTimeout(() => $( "#alertExample" ).fadeOut( "slow", function() {}), 3000);
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
    if(validationCheck(message)) {
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
        // $('.card-body').animate({ scrollTop:  $(".card-body").offset().top + 400 }, 'fast');
  

        // send message
    }

    
}

function sendAnichatMessage() {

}