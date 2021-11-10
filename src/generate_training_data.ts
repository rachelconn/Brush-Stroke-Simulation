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

const cache = new Map<string, DrawingExample[]>();

/**
 * Adds examplesPerFile examples from each file in the dataset to the cache.
 * @param datasetPath Path containing raw quick draw dataset files
 * @param examplesPerFile Number of examples per file to cache
 */
async function addExamplesToCache(datasetPath: string, examplesPerFile: number) {
  await Promise.all(fs.readdirSync(datasetPath).map(async (filename) => {
    const pathToFile = path.join(datasetPath, filename);
    console.log(`Reading from ${pathToFile}`);
    cache.set(filename, await getExamplesFromFile(pathToFile, examplesPerFile));
  }));
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
function createTrainingDataFile(filename: string) {
  const trainingDataPath = path.join('training_data', `${path.basename(filename)}.csv`);
  const fstream = fs.createWriteStream(trainingDataPath);
  cache.get(filename).forEach((example) => {
    example.strokes.forEach((stroke) => {
      const originalStroke = stroke.flat().join(',');
      const simplifiedStroke = simplify(stroke, 5, true).flat().join(',');
      fstream.write(originalStroke);
      fstream.write(simplifiedStroke);
    });
  });
}

addExamplesToCache(INPUT_DATA_PATH, 1000).then(() => {
  cache.forEach((_, filename) => {
    createTrainingDataFile(filename);
  });
});
