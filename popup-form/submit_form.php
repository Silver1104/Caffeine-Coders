<?php
    error_reporting(E_ALL);
    ini_set('display_errors', '1');

    $servername = "127.0.0.1";
    $username = "root";
    $password = "";
    $dbname = "datas";

    // Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);

    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    // Retrieve form data
    $name = $_POST['name'];
    $email = $_POST['email'];
    $phone = $_POST['phone'];
    $npsRating = $_POST['nps-rating'];
    $visitFrequency = $_POST['visit-frequency'];
    $comments = $_POST['comments'];

    // Log the received data
    error_log("Received form data: Name=$name, Email=$email, Phone=$phone, NPS Rating=$npsRating, Visit Frequency=$visitFrequency, Comments=$comments");

    // Prepare and execute SQL INSERT statement
    $sql = "INSERT INTO form_data (name, email, phone, nps_rating, visit_frequency, comments)
        VALUES ('$name', '$email', '$phone', '$npsRating', '$visitFrequency', '$comments')";

    if ($conn->query($sql) === TRUE) {
        echo "Form data submitted successfully";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }

    // Close connection
    $conn->close();
?>
