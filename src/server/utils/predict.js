const { PythonShell } = require("python-shell");
const path = require("path");

const MODEL_DIR = path.join(__dirname, "../../models");
const DATA_PATH = path.join(__dirname, "../../data/processed/reddit_features_clean.csv");

function predict(samples, modelName = "LogisticRegression_model.joblib") {
    return new Promise((resolve, reject) => {
        let options = {
            mode: "json",
            pythonOptions: ["-u"], // unbuffered stdout
            scriptPath: path.join(__dirname, "../../"),
            args: [MODEL_DIR, DATA_PATH, modelName, JSON.stringify(samples)]
        };

        PythonShell.run("predict_node.py", options, (err, results) => {
            if (err) reject(err);
            else resolve(results[0]);  // Python returns JSON
        });
    });
}

module.exports = { predict };
