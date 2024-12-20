document.addEventListener("DOMContentLoaded", function() {
    // Example Data (Replace these with dynamic values fetched from your backend)
    const reportData = {
        name: "John Doe",
        age: 45,
        sex: "Male",
        precautions: "Regular health check-ups, balanced diet, adequate rest",
        metrics: {
            tp: 80, // True Positives
            tn: 90, // True Negatives
            fp: 10, // False Positives
            fn: 20  // False Negatives
        }
    };

    // Compute Metrics
    const accuracy = ((reportData.metrics.tp + reportData.metrics.tn) / 
                      (reportData.metrics.tp + reportData.metrics.tn + reportData.metrics.fp + reportData.metrics.fn) * 100).toFixed(2) + "%";

    const precision = (reportData.metrics.tp / 
                      (reportData.metrics.tp + reportData.metrics.fp) * 100).toFixed(2) + "%";

    const recall = (reportData.metrics.tp / 
                   (reportData.metrics.tp + reportData.metrics.fn) * 100).toFixed(2) + "%";

    const f1Score = (2 * (precision.replace('%', '') * recall.replace('%', '')) / 
                    (parseFloat(precision) + parseFloat(recall))).toFixed(2) + "%";

    const specificity = (reportData.metrics.tn / 
                        (reportData.metrics.tn + reportData.metrics.fp) * 100).toFixed(2) + "%";

    const sensitivity = recall; // Recall is equivalent to Sensitivity

    // Update the HTML with data
    document.getElementById("patient-name").innerText = reportData.name;
    document.getElementById("patient-age").innerText = reportData.age;
    document.getElementById("patient-sex").innerText = reportData.sex;
    document.getElementById("precautions").innerText = reportData.precautions;

    document.getElementById("accuracy").innerText = accuracy;
    document.getElementById("confidence").innerText = "Confidence computation is model-dependent";
    document.getElementById("precision").innerText = precision;
    document.getElementById("recall").innerText = recall;
    document.getElementById("f1-score").innerText = f1Score;
    document.getElementById("specificity").innerText = specificity;
    document.getElementById("sensitivity").innerText = sensitivity;
});
