import fs from 'fs';
import paper from 'paper';
import { DrawingExample } from '../generate_training_data';

if (!paper.project) paper.setup(new paper.Size(255, 255));

export default function exportExampleToSVG(example: DrawingExample, filename: string) {
  paper.project.clear();
  example.strokes.forEach((stroke) => {
    const path = new paper.Path(stroke.map((point) => new paper.Point(...point)));
    path.strokeColor = new paper.Color('black');
    path.strokeWidth = 1;
  });
  const exported = paper.project.exportSVG({ asString: true }) as string;

  fs.writeFile(filename, exported, (e) => {
    if (e) throw e;
  });
}
