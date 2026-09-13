async function predictRace() {
    const track = document.getElementById("track").value.trim();
    const results = document.getElementById("results");

    if (!track) {
        results.innerHTML = '<p class="error">Please enter a race.</p>';
        return;
    }

    results.innerHTML = "<p>Loading prediction...</p>";

    try {
        const response = await fetch(`http://127.0.0.1:8000/predict?track=${encodeURIComponent(track)}`);
        const data = await response.json();

        if (data.error) {
            results.innerHTML = `<p class="error">${data.error}</p>`;
            return;
        }

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

        data.predictions.forEach(prediction => {
            html += `
                <tr>
                    <td>${prediction.predicted_position}</td>
                    <td>${prediction.driver}</td>
                    <td>${prediction.predicted_finish.toFixed(2)}</td>
                </tr>
            `;
        });

        html += "</tbody></table>";
        results.innerHTML = html;
    } catch (error) {
        results.innerHTML = '<p class="error">Could not connect to the prediction server.</p>';
    }
}