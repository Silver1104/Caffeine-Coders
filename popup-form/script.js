// Delay in milliseconds (e.g., 3000 for 3 seconds)
const delay = 3000;

// Open the modal after a delay
setTimeout(function() {
    openModal();
}, delay);

// Display the modal
function openModal() {
    document.getElementById('myModal').style.display = 'block';
}

// Close the modal
function closeModal() {
    document.getElementById('myModal').style.display = 'none';
}