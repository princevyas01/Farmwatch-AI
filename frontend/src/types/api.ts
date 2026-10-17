export interface HealthResponse {
  status: string;
  device: string;
  device_name: string;
  pytorch_version: string;
  model_loaded: boolean;
}

export interface ModelInfoResponse {
  model_architecture: string;
  num_classes: number;
  class_names: string[];
  input_resolution: [number, number];
  active_device: string;
  dataset_source: string;
  license: string;
  citation: string;
  parameters: number;
  is_pretrained: boolean;
}

export interface CandidatePrediction {
  class: string;
  score: number;
}

export interface PredictionResponse {
  predicted_class: string;
  model_score: number;
  confidence_level: 'high' | 'moderate' | 'low';
  is_uncertain: boolean;
  score_interpretation: string;
  top_predictions: CandidatePrediction[];
  recommendations: string[];
  gradcam_base64: string | null;
  explanation_available?: boolean;
  crop_applied: boolean;
  disclaimer: string;
}

export interface MetricsResponse {
  model_name?: string | null;
  checkpoint_name?: string | null;
  test_accuracy: number | null;
  macro_precision: number | null;
  macro_recall: number | null;
  macro_f1: number | null;
  test_samples_count: number | null;
  num_classes: number;
  class_names: string[];
  per_class_f1: Record<string, number>;
  average_latency_ms: number | null;
  evaluation_timestamp: string | null;
  confusion_matrix_base64: string | null;
}

export interface SampleItem {
  id: number;
  class_name: string;
  filename: string;
  url: string;
}
