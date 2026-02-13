async function runPrediction() {
    const name = document.getElementById('name').value;
    
    const formData = {
        gender: document.getElementById('gender').value,
        age: document.getElementById('age').value,
        tc: document.getElementById('tc').value,
        hdl: document.getElementById('hdl').value,
        smoke: document.getElementById('smoke').value,
        bpm: document.getElementById('bpm').value,
        diab: document.getElementById('diab').value
    };

    if(name.length < 2 || !formData.age || !formData.tc || !formData.hdl) {
        alert("Please fill in all fields correctly.");
        return;
    }

    document.getElementById('form-section').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');

    try {
        // api calling
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (result.error) {
            
            console.error("Server Error:", result.error);
            alert("Error: " + result.error);
            location.reload(); 
        } else {
       
            showResult(name, result.score);
        }
        

    } catch (error) {
        console.error("Connection Error:", error);
        alert("Connection Failed! Make sure Docker is running on Port 5000.");
        
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('form-section').classList.remove('hidden');
    }
}

function showResult(name, value) {
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('result-section').classList.remove('hidden');

    document.getElementById('res-name').innerText = name;
    
    if (value === undefined || value === null) {
        alert("Prediction failed. Please try again.");
        location.reload();
        return;
    }

    const scoreElement = document.getElementById('res-score');
    scoreElement.innerText = value.toFixed(2); 

    if (value < 0.3) {
        scoreElement.style.color = "#10b981"; 
    } else if (value < 0.7) {
        scoreElement.style.color = "#f59e0b"; 
    } else {
        scoreElement.style.color = "#ef4444"; 
    }
}