import fs from 'fs';
import path from 'path';
import readline from 'readline';
import simplify from 'simplify-js';
import stream from 'stream';
import exportExampleToSVG from './utils/exportExampleToSVG';

const INPUT_DATA_PATH = path.join('e:', 'quickdraw');

export interface RawDrawingExample {
  'key_id': number,
  word: string,
  recognized: boolean,
  timestamp: string,
  countrycode: string,
  drawing: number[][][],
}

// StrokeData: index as strokeData[pointIdx] = [x, y]
export type StrokeData = [number, number][]

export interface DrawingExample {
  strokes: StrokeData[],
}

function processExample(example: RawDrawingExample): DrawingExample {
  // Normalize coordinates to range [0, 255 on each axis]
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity
  example.drawing.forEach(([xs, ys]) => {
    xs.forEach((x) => {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
    });
    ys.forEach((y) => {
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    });
  });

  // Determine axis that has a larger range, make x and y values range that much to preserve scale
  const range = Math.max(maxX - minX, maxY - minY);
  maxX = minX + range;
  maxY = minY + range;

  const strokes = example.drawing.map(([xs, ys]) => {
    const points: StrokeData = [];
    for (let i = 0; i < xs.length; i++) {
      // Interpolate to range [0, 255]
      const x = (xs[i] - minX) / (maxX - minX) * 255;
      const y = (ys[i] - minY) / (maxY - minY) * 255;
      points.push([x, y]);
    }
    return points;
  });

  return { strokes };
}

async function getExamplesFromFile(filename: string, numExamples: number): Promise<DrawingExample[]> {
  // Create streams to read lines from file
  const input = fs.createReadStream(filename, 'utf-8');
  const output = new stream.Writable();
  const rl = readline.createInterface({
    input,
    output,
    terminal: false,
  });

  const examples: DrawingExample[] = [];
  let remainingExamples = numExamples;

  // Read lines and put into examples
  return new Promise((resolve) => {
    rl.on('line', (line) => {
      examples.push(processExample(JSON.parse(line) as RawDrawingExample));

      remainingExamples -= 1;
      if (remainingExamples == 0) {
        rl.close();
        resolve(examples);
      }
    });
  });
}

// Saves a few examples to the examples directory to test getting examples works
const createTestExamples = async () => {
  const examples = await getExamplesFromFile(path.join(INPUT_DATA_PATH, 'truck.ndjson'), 10);
  examples.forEach((example, idx) => {
    exportExampleToSVG(example, path.join('examples', `example ${idx}.svg`));
  });
};

// createTestExamples();

// Create files with path data for each example file
function createTrainingDataFile(filename: string, examples: DrawingExample[]) {
  const trainingDataPath = path.join('training_data', `${path.basename(filename)}.csv`);
  const fstream = fs.createWriteStream(trainingDataPath);
  examples.forEach((example) => {
    example.strokes.forEach((stroke) => {
      const originalStroke = stroke.flat().join(',');
      const simplifiedStroke = simplify(stroke, 5, true).flat().join(',');
      fstream.write(originalStroke + ';');
      fstream.write(simplifiedStroke + '\r\n');
    });
  });
}

async function createTrainingDataFiles() {
  const files = fs.readdirSync(INPUT_DATA_PATH);
  for (let i = 0; i < files.length; i++) {
    const filename = files[i];
    console.log(`(${i + 1}/${files.length}) Generating training data from ${filename}...`)
    const pathToFile = path.join(INPUT_DATA_PATH, filename);
    await getExamplesFromFile(pathToFile, 1000).then((examples) => {
      createTrainingDataFile(filename, examples);
    });
  }
}

createTrainingDataFiles();
