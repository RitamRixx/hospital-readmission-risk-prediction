// Optional: Add any client-side JavaScript enhancements here
// Since we're using form submission with Jinja2, most interactivity is server-side

document.addEventListener('DOMContentLoaded', function() {
    console.log('Hospital Readmission Risk Predictor loaded');
    
    const resultContainer = document.getElementById('resultContainer');
    if (resultContainer && resultContainer.classList.contains('show')) {
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});