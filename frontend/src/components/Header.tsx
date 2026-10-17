import React from 'react';
import { Sprout, Cpu, Layers, ShieldCheck, Microscope } from 'lucide-react';
import type { HealthResponse, ModelInfoResponse } from '../types/api';

interface HeaderProps {
  health: HealthResponse | null;
  modelInfo: ModelInfoResponse | null;
  activeTab: 'diagnosis' | 'metrics';
  setActiveTab: (tab: 'diagnosis' | 'metrics') => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  modelInfo,
  activeTab,
  setActiveTab,
}) => {
  return (
    <header className="bg-slate-900 text-white border-b border-slate-800 shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between py-4 gap-4">
          {/* Brand & Mission */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-600 flex items-center justify-center shadow-lg shadow-emerald-900/30">
              <Sprout className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold tracking-tight text-white m-0">FarmWatch AI</h1>
                <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-medium">
                  GenHack V1.0 · PS 12
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Visible Crop/Leaf Condition Classification · Intel Arc XPU Ready Baseline
              </p>
            </div>
          </div>

          {/* System & Hardware Status */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {/* Device Badge */}
            <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700/60 text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-sky-400" />
              <span>
                {health?.device_name
                  ? health.device_name.replace("CPU (", "").replace(")", "")
                  : "Intel Core Ultra 5"}
              </span>
              <span className="text-slate-500">|</span>
              <span className="uppercase text-sky-300 font-semibold">{health?.device || 'CPU'}</span>
            </div>

            {/* Model Architecture Badge */}
            <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700/60 text-slate-300">
              <Layers className="w-3.5 h-3.5 text-emerald-400" />
              <span className="font-medium text-emerald-300">
                {modelInfo?.model_architecture || 'EnhancedCropCNN'}
              </span>
              <span className="text-slate-500 font-mono text-[11px]">
                ({modelInfo?.parameters ? `${(modelInfo.parameters / 1_000_000).toFixed(2)}M` : '1.5M'} params · {modelInfo?.is_pretrained ? 'ImageNet Transfer' : 'Random Init'})
              </span>
            </div>

            {/* Health Status */}
            <div className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800 border border-slate-700/60">
              <span
                className={`w-2 h-2 rounded-full ${
                  health?.status === 'healthy' ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
                }`}
              />
              <span className="text-slate-300 text-[11px]">
                {health?.status === 'healthy' ? 'API Online' : 'Connecting...'}
              </span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 border-t border-slate-800/80 pt-1 -mb-px">
          <button
            onClick={() => setActiveTab('diagnosis')}
            className={`flex items-center space-x-2 py-3 px-4 text-sm font-medium border-b-2 transition-colors cursor-pointer ${
              activeTab === 'diagnosis'
                ? 'border-emerald-500 text-emerald-400 bg-slate-800/40'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
            }`}
          >
            <Microscope className="w-4 h-4" />
            <span>Leaf Diagnosis & Grad-CAM</span>
          </button>

          <button
            onClick={() => setActiveTab('metrics')}
            className={`flex items-center space-x-2 py-3 px-4 text-sm font-medium border-b-2 transition-colors cursor-pointer ${
              activeTab === 'metrics'
                ? 'border-emerald-500 text-emerald-400 bg-slate-800/40'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            <span>Benchmark Metrics & Model Transparency</span>
          </button>
        </div>
      </div>
    </header>
  );
};
