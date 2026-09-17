$(document).ready(function() {
    $("#prediction-form").on("submit", function() {
        $("button[type='submit']").prop("disabled", true).html(
            '<span class="spinner-border spinner-border-sm" role="status"></span> Predicting...'
        );
    });

    $(".form-select, .form-control").on("change", function() {
        $(this).removeClass("is-invalid").addClass("is-valid");
    });
});
