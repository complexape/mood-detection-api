const loadTf = require('tensorflow-lambda');
const sharp = require('sharp');
const { join } = require('path');
const fs = require('fs');

const modelPath = join(process.cwd(), 'models', 'tired_model');

// Let vercel know we're using shard files!
const shard1 = fs.readFileSync(join(modelPath, 'group1-shard1of3.bin'));
const shard2 = fs.readFileSync(join(modelPath, 'group1-shard2of3.bin'));
const shard3 = fs.readFileSync(join(modelPath, 'group1-shard3of3.bin'));

module.exports = async (req, res) => {
    if (req.method === 'POST') {
        const { imgData } = req.body;

        const tf = await loadTf();

        // Convert base64 to tensor of shape (1, 64, 64, 1)
        const buffer = Buffer.from(imgData, 'base64');
        const { data, info } = await sharp(buffer)
            .raw()
            .toBuffer({ resolveWithObject: true });
        const tensor = tf.tensor(data, [info.height, info.width, info.channels]);
        const normalizedTensor = tensor
            .mean(2)
            .toFloat()
            .div(255)
            .expandDims(0)
            .expandDims(-1);

        // Load model from files
        const tiredModel = await tf.loadLayersModel(`file://${join(modelPath, 'model.json')}`);

        const prediction = tiredModel.predict(normalizedTensor).dataSync();
        res.status(200).json({"closed_eye": prediction[0]});
    }
    else {
        res.status(200).json({
            hello: 'world!'    
        })
    }
}