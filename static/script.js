function navigateTo(sectionId) {
    // Remove active class from all links
    document.querySelectorAll('.header-link').forEach(link => link.classList.remove('active'));

    // Add active class to the clicked link
    document.querySelector(`[href="#${sectionId}"]`).classList.add('active');

    // Hide all sections
    document.querySelectorAll('.content').forEach(section => section.style.display = 'none');

    // Show the selected section
    document.getElementById(sectionId).style.display = 'block';

    console.log(`Navigating to ${sectionId}`);
}

document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.header-link').forEach(link => {
        link.addEventListener('click', function (event) {
            event.preventDefault(); // Prevent default behavior of anchor tag
            var sectionId = this.getAttribute('href').substring(1);
            navigateTo(sectionId);
        });
    });

    // Set initial active link to Home
    navigateTo('home');

    // Add an event listener for the "Start here" link
    document.querySelector('.v44_7').addEventListener('click', function () {
        navigateTo('foundyou');
    });
    // Add an event listener for the "Start here" link
    document.querySelector('.v44_5').addEventListener('click', function () {
        navigateTo('foundyou');
    });
});