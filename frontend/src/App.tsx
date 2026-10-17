import React, { useState, useEffect } from 'react';
import {
  Sprout,
  AlertCircle,
  Play,
  CheckCircle,
  Eye,
  Microscope,
} from 'lucide-react';

import type {
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
  MetricsResponse,
  SampleItem,
} from './types/api';
import {
  fetchHealth,
  fetchModelInfo,
  fetchMetrics,
  fetchSamples,
  predictImage,
} from './lib/api';

import { Header } from './components/Header';
import { ImageUploader } from './components/ImageUploader';
import { CropSelector } from './components/CropSelector';
import { PredictionResult } from './components/PredictionResult';
import { GradCamViewer } from './components/GradCamViewer';
import { MetricsDashboard } from './components/MetricsDashboard';
import { DatasetLimitations } from './components/DatasetLimitations';

export const App: React.FC = () => {
  // Navigation
  const [activeTab, setActiveTab] = useState<'diagnosis' | 'metrics'>('diagnosis');

  // Backend state
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  // Diagnosis state
  const [selectedFile, setSelectedFile] = useState<File | Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedSampleClass, setSelectedSampleClass] = useState<string | undefined>(undefined);
  const [bbox, setBbox] = useState<[number, number, number, number] | null>(null);
  const [generateCam, setGenerateCam] = useState<boolean>(true);

  // Prediction state
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // Load initial backend data
  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((err) => console.warn('Backend not yet reachable', err));

    fetchModelInfo()
      .then(setModelInfo)
      .catch((err) => console.warn('Model info not yet reachable', err));

    fetchSamples()
      .then(setSamples)
      .catch((err) => console.warn('Samples not yet reachable', err));

    setMetricsLoading(true);
    fetchMetrics()
      .then((data) => {
        setMetrics(data);
        setMetricsLoading(false);
      })
      .catch((err) => {
        setMetricsError(err.message || 'Could not load metrics');
        setMetricsLoading(false);
      });
  }, []);

  const handleImageSelected = (file: File | Blob, url: string, sampleClass?: string) => {
    setSelectedFile(file);
    setPreviewUrl(url);
    setSelectedSampleClass(sampleClass);
    setBbox(null);
    setPrediction(null);
    setPredictionError(null);
  };

  const handleClearImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setSelectedSampleClass(undefined);
    setBbox(null);
    setPrediction(null);
    setPredictionError(null);
  };

  const handleRunPrediction = async () => {
    if (!selectedFile) return;

    setIsPredicting(true);
    setPredictionError(null);

    try {
      const result = await predictImage(selectedFile, bbox, generateCam);
      setPrediction(result);
    } catch (err: any) {
      setPredictionError(err.message || 'Inference failed. Check backend connection.');
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans text-slate-900">
      {/* Navigation Header */}
      <Header
        health={health}
        modelInfo={modelInfo}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'diagnosis' ? (
          <div className="space-y-8">
            {/* Mission & Feature Scope Banner */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center space-x-2 text-emerald-700 text-xs font-bold uppercase tracking-wider">
                  <Microscope className="w-4 h-4" />
                  <span>Field Leaf Pathology Assistant</span>
                </div>
                <h2 className="text-xl font-bold text-slate-900 tracking-tight m-0">
                  Visible Crop Condition Classification & Feature Attention
                </h2>
                <p className="text-xs text-slate-600 max-w-3xl leading-relaxed mt-1">
                  Upload an unconstrained field leaf photo or select from genuine held-out benchmark samples.
                  Inference executes locally on your scratch-trained PyTorch CNN with Grad-CAM visual attention
                  and heuristic low-score advisories (&lt; 60%).
                </p>
              </div>

              <div className="text-right shrink-0">
                <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-800 border border-emerald-200">
                  <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Custom CNN — trained from random initialization</span>
                </span>
              </div>
            </div>

            {/* If no image selected: Image Uploader */}
            {!previewUrl ? (
              <div className="max-w-3xl mx-auto">
                <ImageUploader
                  onImageSelected={handleImageSelected}
                  samples={samples}
                  selectedSampleClass={selectedSampleClass}
                />
              </div>
            ) : (
              /* If image selected: 2-column Analysis Layout */
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                {/* Left Column: Image & ROI Selector */}
                <div className="lg:col-span-7 space-y-4">
                  <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
                    <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                      <div>
                        <h3 className="text-base font-bold text-slate-900 m-0">
                          Target Field Image
                        </h3>
                        {selectedSampleClass && (
                          <p className="text-xs text-emerald-700 font-medium mt-0.5">
                            Sample: {selectedSampleClass}
                          </p>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={handleClearImage}
                        className="text-xs font-medium text-slate-600 hover:text-slate-900 px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors"
                      >
                        Change Image
                      </button>
                    </div>

                    {/* Interactive Canvas */}
                    <CropSelector
                      imageSrc={previewUrl}
                      onBboxChange={setBbox}
                      bbox={bbox}
                    />

                    {/* Execution Controls */}
                    <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                      {/* Grad-CAM toggle */}
                      <label className="flex items-center space-x-2 text-xs text-slate-700 cursor-pointer select-none">
                        <input
                          type="checkbox"
                          checked={generateCam}
                          onChange={(e) => setGenerateCam(e.target.checked)}
                          className="w-4 h-4 rounded text-emerald-600 focus:ring-emerald-500 accent-emerald-600 cursor-pointer"
                        />
                        <span className="font-medium">Generate Grad-CAM Attention Map</span>
                      </label>

                      {/* Run Button */}
                      <button
                        type="button"
                        onClick={handleRunPrediction}
                        disabled={isPredicting}
                        className="flex items-center justify-center space-x-2 px-6 py-2.5 rounded-xl bg-emerald-700 hover:bg-emerald-800 disabled:bg-slate-400 text-white font-semibold text-sm shadow-sm transition-all cursor-pointer"
                      >
                        {isPredicting ? (
                          <>
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            <span>Processing CNN Inference...</span>
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4 fill-white" />
                            <span>Run Leaf Diagnosis</span>
                          </>
                        )}
                      </button>
                    </div>

                    {/* Error Banner */}
                    {predictionError && (
                      <div className="flex items-start space-x-2.5 text-xs text-rose-800 bg-rose-50 border border-rose-200 p-3.5 rounded-xl">
                        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-600" />
                        <div>
                          <div className="font-semibold">Prediction Failed</div>
                          <div>{predictionError}</div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Column: Prediction Results & Grad-CAM */}
                <div className="lg:col-span-5 space-y-6">
                  {prediction ? (
                    <>
                      <PredictionResult
                        prediction={prediction}
                        groundTruthClass={selectedSampleClass}
                      />

                      {prediction.gradcam_base64 && (
                        <GradCamViewer
                          originalImageSrc={previewUrl}
                          gradCamBase64={prediction.gradcam_base64}
                          predictedClass={prediction.predicted_class}
                        />
                      )}
                    </>
                  ) : (
                    <div className="bg-white rounded-2xl border border-slate-200 p-10 text-center text-slate-400 shadow-xs space-y-3">
                      <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 mx-auto">
                        <Eye className="w-6 h-6" />
                      </div>
                      <div className="font-semibold text-slate-700 text-sm">
                        Awaiting Diagnosis
                      </div>
                      <p className="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">
                        Draw an optional bounding box around a specific leaf to isolate symptoms,
                        then click <strong className="text-slate-700">"Run Leaf Diagnosis"</strong>.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Metrics & Model Transparency Tab */
          <div className="space-y-8">
            <MetricsDashboard
              metrics={metrics}
              modelInfo={modelInfo}
              loading={metricsLoading}
              error={metricsError}
            />

            <DatasetLimitations />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 mt-auto py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
          <div className="flex items-center space-x-2">
            <Sprout className="w-4 h-4 text-emerald-600" />
            <span className="font-semibold text-slate-700">FarmWatch AI</span>
            <span>·</span>
            <span>GenHack V1.0 PS 12</span>
          </div>

          <div className="text-slate-400 text-center sm:text-right">
            PyTorch Scratch CNN · Intel Core Ultra 5 (CPU) · PlantDoc Benchmark (CC BY 4.0)
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
