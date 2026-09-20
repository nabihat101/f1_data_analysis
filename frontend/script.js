
/**
 * predicts race by connecting the frontend to the backend and fetching the prediction data + waiting for it
*/

async function predictRace() {

    // find the track input and results container in the DOM
    const track = document.getElementById("track").value.trim();
    const results = document.getElementById("results");

    // ensures user enters race
    if (!track) {
        results.innerHTML = '<p class="error">Please enter a race.</p>';
        return;
    }

    // while waiting for the prediction, show a loading message
    results.innerHTML = "<p>Loading prediction...</p>";

    // error handling for the fetch request to the backend
    try {

        // waiting for python to send the prediction data back to the frontend
        const response = await fetch(`http://127.0.0.1:8000/predict?track=${encodeURIComponent(track)}`);
        
        // waiting for the response to be converted to JSON format
        const data = await response.json();

        // if the backend returns an error, display it to the user
        if (data.error) {
            results.innerHTML = `<p class="error">${data.error}</p>`;
            return;
        }

        // creates row to be displayed in frontend
        let html = `
            <h2>Predicted ${data.track} ${data.year} Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Position</th>
                        <th>Driver</th>
                        <th>Predicted Finish</th>
                    </tr>
                </thead>
                <tbody>
        `;

        // the columns to be displayed 
        data.predictions.forEach(prediction => {
            html += `
                <tr>
                    <td>${prediction.predicted_position}</td>
                    <td>${prediction.driver}</td>
                    <td>${prediction.predicted_finish.toFixed(2)}</td>
                </tr>
            `;
        });

        // add to table
        html += "</tbody></table>";

        // add to html
        results.innerHTML = html;
    } 
    
    // error handling for if the fetch request fails
    catch (error) {
        results.innerHTML = '<p class="error">Could not connect to the prediction server.</p>';
    }
}