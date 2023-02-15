$(document).ready(function () {
    $(".img-login").hover(function() {
        $(".img-login").attr('src', './static/img/login_hover.png');
    }, function() {
        $(".img-login").attr('src', './static/img/login.png');
    });

    $(".img-register").hover(function() {
        $(".img-register").attr('src', './static/img/register_hover.png');
    }, function() {
        $(".img-register").attr('src', './static/img/register.png');
    });
});