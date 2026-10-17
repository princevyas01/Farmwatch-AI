import React, { useState } from 'react';
import { Eye, Sparkles, Sliders, AlertCircle } from 'lucide-react';

interface GradCamViewerProps {
  originalImageSrc: string;
  gradCamBase64: string | null;
  predictedClass: string;
}

export const GradCamViewer: React.FC<GradCamViewerProps> = ({
  originalImageSrc,
  gradCamBase64,
  predictedClass,
}) => {
  const [viewMode, setViewMode] = useState<'overlay' | 'split'>('overlay');
  const [overlayAlpha, setOverlayAlpha] = useState<number>(85); // 0 to 100

  if (!gradCamBase64) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-6 text-center text-slate-500 shadow-sm">
        <AlertCircle className="w-8 h-8 text-slate-400 mx-auto mb-2" />
        <p className="text-sm">Grad-CAM explanation was not generated for this prediction.</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden p-6 space-y-4">
      {/* Title & View Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-emerald-600" />
            <h3 className="text-base font-bold text-slate-900 m-0">
              Grad-CAM Visual Feature Attention
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Backpropagated class gradients on final layer <code className="bg-slate-100 text-slate-800 text-[11px] px-1 py-0.5 rounded font-mono">block4[3]</code> (256 channels)
          </p>
        </div>

        {/* View mode toggle */}
        <div className="flex items-center space-x-1 bg-slate-100 p-1 rounded-lg border border-slate-200 self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setViewMode('overlay')}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              viewMode === 'overlay'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Overlay Mode
          </button>
          <button
            type="button"
            onClick={() => setViewMode('split')}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              viewMode === 'split'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Side-by-Side
          </button>
        </div>
      </div>

      {/* Main Display Area */}
      {viewMode === 'overlay' ? (
        <div className="space-y-3">
          <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-950 flex items-center justify-center max-h-[380px]">
            {/* Base Image */}
            <img
              src={originalImageSrc}
              alt="Leaf crop"
              className="max-h-[380px] w-auto object-contain"
            />
            {/* Heatmap Overlay with adjustable opacity */}
            <img
              src={gradCamBase64}
              alt="Grad-CAM Overlay"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-150"
              style={{ opacity: overlayAlpha / 100 }}
            />
          </div>

          {/* Opacity Slider */}
          <div className="flex items-center justify-between text-xs text-slate-600 bg-slate-50 p-2.5 rounded-lg border border-slate-200">
            <div className="flex items-center space-x-2">
              <Sliders className="w-3.5 h-3.5 text-slate-500" />
              <span>Heatmap Blend Intensity:</span>
            </div>
            <div className="flex items-center space-x-3">
              <input
                type="range"
                min="0"
                max="100"
                value={overlayAlpha}
                onChange={(e) => setOverlayAlpha(Number(e.target.value))}
                className="w-32 accent-emerald-600 cursor-pointer"
              />
              <span className="font-mono text-[11px] w-8 text-right font-medium">
                {overlayAlpha}%
              </span>
            </div>
          </div>
        </div>
      ) : (
        /* Side by Side Split */
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-1.5 text-center">
            <span className="text-xs font-semibold text-slate-600 block">Input Leaf Crop</span>
            <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-950 flex items-center justify-center h-64">
              <img
                src={originalImageSrc}
                alt="Input crop"
                className="max-h-64 w-auto object-contain"
              />
            </div>
          </div>

          <div className="space-y-1.5 text-center">
            <span className="text-xs font-semibold text-emerald-800 block">
              Grad-CAM Attention ({predictedClass})
            </span>
            <div className="rounded-xl overflow-hidden border border-emerald-300 bg-slate-950 flex items-center justify-center h-64">
              <img
                src={gradCamBase64}
                alt="Grad-CAM activation"
                className="max-h-64 w-auto object-contain"
              />
            </div>
          </div>
        </div>
      )}

      {/* Explanatory Caption & Academic Disclaimer */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-600 space-y-1">
        <div className="font-semibold text-slate-800 flex items-center space-x-1.5">
          <Eye className="w-3.5 h-3.5 text-emerald-600" />
          <span>Interpreting Activation Regions:</span>
        </div>
        <p className="leading-relaxed">
          Warm colors (red/yellow) indicate pixels whose spatial activations exerted the strongest positive
          gradient pull toward the prediction <span className="font-semibold text-slate-800 font-mono">"{predictedClass}"</span>.
          Cool regions (blue/dark) had minimal influence on the classification head.
        </p>
        <p className="text-[11px] text-slate-400 italic pt-1 border-t border-slate-200/60">
          Visual explanation shows neural network feature sensitivity; does not constitute biological or causal diagnosis.
        </p>
      </div>
    </div>
  );
};
