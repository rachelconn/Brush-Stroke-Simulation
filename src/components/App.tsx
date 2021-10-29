import paper from 'paper';
import React from 'react';

const App: React.FC = () => {
  const canvasElement = React.useRef<HTMLCanvasElement>();

  React.useEffect(() => {
    if (canvasElement.current) paper.setup(canvasElement.current);
  }, [canvasElement]);

  return (
    <canvas style={{ width: 1000, height: 1000 }} ref={canvasElement} />
  );
};

export default App;
