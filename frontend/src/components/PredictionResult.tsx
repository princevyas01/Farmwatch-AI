import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Info,
  ShieldAlert,
  ChevronRight,
  ClipboardList,
} from 'lucide-react';
import type { PredictionResponse } from '../types/api';

interface PredictionResultProps {
  prediction: PredictionResponse;
  groundTruthClass?: string;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({
  prediction,
  groundTruthClass,
}) => {
  const {
    predicted_class,
    model_score,
    confidence_level,
    is_uncertain,
    score_interpretation,
    top_predictions,
    recommendations,
    crop_applied,
    disclaimer,
  } = prediction;

  const scorePct = (model_score * 100).toFixed(1);

  // Confidence pill color
  const getConfidenceBadge = () => {
    if (confidence_level === 'high') {
      return {
        bg: 'bg-emerald-100 text-emerald-800 border-emerald-300',
        text: 'High Confidence',
      };
    }
    if (confidence_level === 'moderate') {
      return {
        bg: 'bg-amber-100 text-amber-800 border-amber-300',
        text: 'Moderate Confidence',
      };
    }
    return {
      bg: 'bg-rose-100 text-rose-800 border-rose-300',
      text: 'Low Confidence',
    };
  };

  const badge = getConfidenceBadge();

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden space-y-6 p-6">
      {/* Header Condition & Score */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <span className="text-xs font-semibold tracking-wider uppercase text-slate-500">
            Visible Condition Classification
          </span>
          <div className="flex items-center space-x-2">
            {crop_applied && (
              <span className="text-[11px] px-2 py-0.5 rounded-md bg-sky-50 text-sky-700 border border-sky-200 font-medium">
                Leaf ROI Crop Applied
              </span>
            )}
            <span
              className={`text-xs px-2.5 py-0.5 rounded-full border font-semibold ${badge.bg}`}
            >
              {scorePct}% · {badge.text}
            </span>
          </div>
        </div>

        <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
          {predicted_class}
        </h2>

        {groundTruthClass && (
          <div className="mt-1 text-xs text-slate-600 flex items-center space-x-1">
            <span className="font-semibold text-slate-700">Benchmark Label:</span>
            <span className="font-mono bg-slate-100 px-1.5 py-0.5 rounded text-slate-800">
              {groundTruthClass}
            </span>
            {groundTruthClass.toLowerCase() === predicted_class.toLowerCase() ? (
              <span className="text-emerald-600 font-medium flex items-center space-x-0.5">
                <CheckCircle2 className="w-3.5 h-3.5 inline" />
                <span>Exact Match</span>
              </span>
            ) : (
              <span className="text-slate-500 italic">
                (Empirical field test sample)
              </span>
            )}
          </div>
        )}

        <p className="text-xs text-slate-600 mt-2 bg-slate-50 p-2.5 rounded-lg border border-slate-200/80">
          <Info className="w-3.5 h-3.5 inline text-slate-400 mr-1.5 -mt-0.5" />
          {score_interpretation}
        </p>
      </div>

      {/* Low-Score Advisory Banner (<60%) */}
      {is_uncertain && (
        <div className="rounded-xl bg-amber-50 border border-amber-200/90 p-4 text-amber-900 flex items-start space-x-3 shadow-xs">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1 text-xs">
            <div className="font-bold text-amber-950 text-sm">
              Heuristic Low-Score Advisory (&lt;60%)
            </div>
            <p className="text-amber-800 leading-relaxed">
              Top candidate score is below the heuristic 60% threshold. Softmax outputs
              are uncalibrated empirical scores rather than true posterior probabilities. Under
              uncontrolled field lighting, overlapping foliage, or soil noise, lower scores reflect model ambiguity.
              Verify with an agronomist before taking management action.
            </p>
          </div>
        </div>
      )}

      {/* Top 3 Ranked Candidates */}
      {top_predictions && top_predictions.length > 0 && (
        <div className="space-y-2.5 pt-2 border-t border-slate-100">
          <div className="text-xs font-semibold text-slate-700 flex items-center justify-between">
            <span>Top Candidate Hypotheses</span>
            <span className="text-slate-400 font-normal">Softmax Probability</span>
          </div>

          <div className="space-y-2">
            {top_predictions.map((candidate, idx) => {
              const pct = (candidate.score * 100).toFixed(1);
              const isTop = idx === 0;
              return (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span
                      className={`truncate max-w-[280px] ${
                        isTop ? 'font-semibold text-slate-900' : 'text-slate-600'
                      }`}
                    >
                      {idx + 1}. {candidate.class}
                    </span>
                    <span className="font-mono text-slate-700 font-medium">{pct}%</span>
                  </div>
                  <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isTop
                          ? is_uncertain
                            ? 'bg-amber-500'
                            : 'bg-emerald-600'
                          : 'bg-slate-300'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(2, candidate.score * 100))}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Low-Risk Agronomic Guidance */}
      <div className="pt-2 border-t border-slate-100 space-y-3">
        <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-800">
          <ClipboardList className="w-4 h-4 text-emerald-700" />
          <span>Low-Risk Cultural & Sanitation Actions</span>
        </div>

        <ul className="space-y-2 text-xs text-slate-700">
          {recommendations.map((rec, i) => (
            <li key={i} className="flex items-start space-x-2">
              <ChevronRight className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
              <span>{rec}</span>
            </li>
          ))}
        </ul>

        <div className="flex items-center space-x-1.5 text-[11px] text-slate-500 bg-slate-50 p-2 rounded-lg border border-slate-200">
          <ShieldAlert className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          <span>
            Strict Safety Policy: No chemical pesticide dosages provided. Cultural and physical practices only.
          </span>
        </div>
      </div>

      {/* Academic / Scientific Disclaimer */}
      <div className="text-[11px] text-slate-400 border-t border-slate-100 pt-3 italic leading-relaxed">
        {disclaimer}
      </div>
    </div>
  );
};
