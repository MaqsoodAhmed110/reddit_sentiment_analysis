const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const { predict } = require("./utils/predict");

const app = express();
app.use(cors());
app.use(bodyParser.json());

app.post("/predict", async (req, res) => {
    try {
        const { samples, model } = req.body;
        const predictions = await predict(samples, model);
        res.json({ predictions });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

const PORT = 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
