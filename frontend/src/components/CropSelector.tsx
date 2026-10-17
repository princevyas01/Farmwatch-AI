import React, { useState, useRef, useEffect } from 'react';
import type { MouseEvent } from 'react';
import { RotateCcw, Crosshair } from 'lucide-react';

interface CropSelectorProps {
  imageSrc: string;
  onBboxChange: (bbox: [number, number, number, number] | null) => void;
  bbox: [number, number, number, number] | null;
}

export const CropSelector: React.FC<CropSelectorProps> = ({
  imageSrc,
  onBboxChange,
  bbox,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);

  // Sync internal state if external bbox is reset
  useEffect(() => {
    if (!bbox) {
      setCurrentBox(null);
    }
  }, [bbox]);

  const getNormalizedCoords = (e: MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return { x: 0, y: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return { x, y };
  };

  const handleMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    const coords = getNormalizedCoords(e);
    setIsDrawing(true);
    setStartPoint(coords);
    setCurrentBox({ x: coords.x, y: coords.y, width: 0, height: 0 });
  };

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!isDrawing || !startPoint) return;
    const coords = getNormalizedCoords(e);

    const xMin = Math.min(startPoint.x, coords.x);
    const yMin = Math.min(startPoint.y, coords.y);
    const width = Math.abs(coords.x - startPoint.x);
    const height = Math.abs(coords.y - startPoint.y);

    setCurrentBox({ x: xMin, y: yMin, width, height });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentBox) return;
    setIsDrawing(false);

    // If box has significant area (> 3% in width and height), register it
    if (currentBox.width > 0.03 && currentBox.height > 0.03) {
      const xMax = currentBox.x + currentBox.width;
      const yMax = currentBox.y + currentBox.height;
      onBboxChange([currentBox.x, currentBox.y, xMax, yMax]);
    } else {
      // Too small, discard
      setCurrentBox(null);
      onBboxChange(null);
    }
  };

  const handleResetCrop = () => {
    setCurrentBox(null);
    onBboxChange(null);
  };

  return (
    <div className="space-y-3">
      {/* Controls Bar */}
      <div className="flex items-center justify-between text-xs text-slate-600 bg-slate-100 p-2.5 rounded-lg border border-slate-200">
        <div className="flex items-center space-x-1.5 font-medium text-slate-700">
          <Crosshair className="w-3.5 h-3.5 text-emerald-600" />
          <span>Interactive Leaf ROI Selector:</span>
          <span className="text-slate-500 font-normal">
            {bbox
              ? `Crop selected (${(bbox[2] - bbox[0]).toFixed(2)} × ${(bbox[3] - bbox[1]).toFixed(2)})`
              : 'Drag across image to isolate a single leaf'}
          </span>
        </div>

        {bbox && (
          <button
            type="button"
            onClick={handleResetCrop}
            className="flex items-center space-x-1 px-2.5 py-1 rounded bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 shadow-xs transition-colors font-medium"
          >
            <RotateCcw className="w-3 h-3 text-slate-500" />
            <span>Reset to Full Image</span>
          </button>
        )}
      </div>

      {/* Interactive Image Container */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        className="relative select-none rounded-xl overflow-hidden border border-slate-200 bg-slate-950 flex items-center justify-center cursor-crosshair group shadow-inner"
        style={{ minHeight: '340px', maxHeight: '480px' }}
      >
        <img
          src={imageSrc}
          alt="Target Field Crop"
          className="max-h-[480px] w-auto object-contain pointer-events-none select-none"
        />

        {/* Drawn Bounding Box Highlight */}
        {currentBox && (
          <div
            className="absolute border-2 border-emerald-400 bg-emerald-500/25 pointer-events-none rounded-xs shadow-sm transition-none"
            style={{
              left: `${currentBox.x * 100}%`,
              top: `${currentBox.y * 100}%`,
              width: `${currentBox.width * 100}%`,
              height: `${currentBox.height * 100}%`,
            }}
          >
            <div className="absolute top-1 left-1 bg-emerald-800/90 text-white text-[10px] px-1.5 py-0.5 rounded-xs font-mono font-semibold tracking-wider uppercase">
              Leaf ROI
            </div>
            {/* Corner Markers */}
            <div className="absolute -top-1 -left-1 w-2 h-2 bg-emerald-400 border border-white" />
            <div className="absolute -top-1 -right-1 w-2 h-2 bg-emerald-400 border border-white" />
            <div className="absolute -bottom-1 -left-1 w-2 h-2 bg-emerald-400 border border-white" />
            <div className="absolute -bottom-1 -right-1 w-2 h-2 bg-emerald-400 border border-white" />
          </div>
        )}
      </div>
    </div>
  );
};
