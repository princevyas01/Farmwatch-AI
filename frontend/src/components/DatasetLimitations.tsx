import React from 'react';
import { BookOpen, AlertOctagon, TrendingUp, Check } from 'lucide-react';

export const DatasetLimitations: React.FC = () => {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
      {/* Title */}
      <div className="flex items-center space-x-2 pb-3 border-b border-slate-100">
        <BookOpen className="w-5 h-5 text-emerald-700" />
        <div>
          <h3 className="text-base font-bold text-slate-900 m-0">
            Dataset Attribution, Field Realities & Scientific Scope
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Transparent disclosure of research sources, engineering iterations, and deployment boundaries
          </p>
        </div>
      </div>

      {/* Grid: Provenance & Iteration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Source & Benchmark */}
        <div className="bg-slate-50 rounded-xl p-4 border border-slate-200/80 space-y-2.5">
          <div className="text-xs font-bold text-slate-900 flex items-center justify-between">
            <span>Research Dataset Provenance</span>
            <span className="text-[10px] bg-slate-200/80 text-slate-700 px-2 py-0.5 rounded font-mono">
              CC BY 4.0 License
            </span>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            Trained on the <strong className="text-slate-800">PlantDoc Benchmark</strong>, a verified field-image
            dataset curated specifically for natural agricultural settings.
          </p>

          <div className="text-xs bg-white p-3 rounded-lg border border-slate-200 space-y-1 font-mono text-[11px] text-slate-700">
            <div><strong>Citation:</strong> Kayal et al. (IEEE Access 2023)</div>
            <div><strong>DOI:</strong> 10.1109/ACCESS.2023.3263042</div>
            <div><strong>Total Audited Images:</strong> 2,560 unique (21 duplicates deduplicated)</div>
          </div>

          <p className="text-[11px] text-slate-500 italic">
            Unlike lab datasets (e.g. PlantVillage with plain black/grey backgrounds), PlantDoc features
            uncontrolled sunlight, multiple leaves, wind motion, and soil noise.
          </p>
        </div>

        {/* The Two-Experiment Engineering Journey */}
        <div className="bg-slate-50 rounded-xl p-4 border border-slate-200/80 space-y-2.5">
          <div className="text-xs font-bold text-slate-900 flex items-center space-x-1.5">
            <TrendingUp className="w-4 h-4 text-emerald-600" />
            <span>Measured Model Evolution (Audited Experiments)</span>
          </div>

          <div className="space-y-2 text-xs">
            <div className="p-2.5 rounded-lg bg-white border border-slate-200">
              <div className="font-semibold text-slate-800 flex justify-between">
                <span>Experiment 1: Baseline 28-Class CNN</span>
                <span className="text-rose-600 font-mono">18.29% Test Acc</span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1">
                Trained on raw 28 classes. Audit identified severe class starvation (classes with only 2–5 images),
                resulting in 13 classes having 0.00% F1-score (Macro F1: 11.17%).
              </p>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
              <div className="font-semibold text-slate-900 flex justify-between">
                <span>Experiment 2: Enhanced 11-Class Scratch CNN</span>
                <span className="text-slate-700 font-mono font-bold">27.10% Test Acc</span>
              </div>
              <p className="text-[11px] text-slate-600 mt-1">
                Subset filtered to classes with ≥ 100 samples. Deeper 4-block scratch CNN improved Macro F1 to 21.74% (reduced zero-F1 classes from 13 to 3).
              </p>
            </div>

            <div className="p-2.5 rounded-lg bg-emerald-50/80 border border-emerald-200">
              <div className="font-semibold text-emerald-950 flex justify-between">
                <span>Experiment 3: MobileNetV3-Small Transfer Learning</span>
                <span className="text-emerald-700 font-mono font-bold">67.29% Test Acc</span>
              </div>
              <p className="text-[11px] text-emerald-900 mt-1">
                ImageNet-1K pretrained feature backbone with replacement 11-class classification head. Macro F1 improved to 66.52% (+44.78 pp) and eliminated all zero-F1 classes (0 zero-F1 classes).
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Scope Boundaries & Ethical Constraints */}
      <div className="rounded-xl bg-slate-900 text-white p-4 space-y-2">
        <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-amber-400">
          <AlertOctagon className="w-4 h-4" />
          <span>Operational Boundaries & Integrity Constraints</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-xs text-slate-300 pt-1">
          <div className="flex items-start space-x-2">
            <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
            <span>
              <strong>Visible Symptoms Only:</strong> Feature is strictly named "Visible Condition Classification". Never claims drought, heat, or nitrogen stress.
            </span>
          </div>
          <div className="flex items-start space-x-2">
            <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
            <span>
              <strong>No Chemical Prescriptions:</strong> Only cultural and physical sanitation advice is provided; no chemical pesticide dosages.
            </span>
          </div>
          <div className="flex items-start space-x-2">
            <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
            <span>
              <strong>Model Lineage Transparency:</strong> Baseline CNNs were scratch-trained (zero ImageNet weights). MobileNetV3 uses explicitly labeled ImageNet-1K transfer learning.
            </span>
          </div>
          <div className="flex items-start space-x-2">
            <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
            <span>
              <strong>Foliage Class Taxonomy:</strong> Categories like Blueberry, Peach, and Raspberry leaf represent general foliage, not certified pathogen-free status.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
