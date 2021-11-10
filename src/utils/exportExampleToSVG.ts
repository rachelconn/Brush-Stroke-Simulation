import fs from 'fs';
import paper from 'paper';
import simplify from 'simplify-js';
import { DrawingExample, StrokeData } from '../generate_training_data';

if (!paper.project) paper.setup(new paper.Size(255, 255));

function strokeDataToPath(strokeData: StrokeData): paper.Path {
    return new paper.Path(strokeData.map((point) => new paper.Point(...point)));
}

export default function exportExampleToSVG(example: DrawingExample, filename: string) {
  paper.project.clear();
  example.strokes.forEach((stroke) => {
    const path = strokeDataToPath(stroke);
    path.strokeColor = new paper.Color('black');
    path.strokeWidth = 1;

    // const simplified = simplifyPath(path, 0.95);
    const simplified = strokeDataToPath(simplify(stroke, 5, true));
    simplified.strokeColor = new paper.Color('#ff00ff');
    simplified.strokeWidth = 1;
    simplified.insertAbove(path);
  });
  const exported = paper.project.exportSVG({ asString: true }) as string;

  fs.writeFile(filename, exported, (e) => {
    if (e) throw e;
  });
}
