import type {
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
  MetricsResponse,
  SampleItem,
} from '../types/api';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || '/api';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Failed to fetch health status: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/model-info`);
  if (!res.ok) {
    throw new Error(`Failed to fetch model info: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) {
    throw new Error(`Failed to fetch metrics: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchSamples(): Promise<SampleItem[]> {
  try {
    const res = await fetch(`${API_BASE}/samples`);
    if (!res.ok) return [];
    return res.json();
  } catch (err) {
    console.warn('Could not load samples', err);
    return [];
  }
}

export async function predictImage(
  imageFile: File | Blob,
  bbox?: [number, number, number, number] | null,
  generateCam: boolean = true,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', imageFile, 'leaf_photo.jpg');
  formData.append('generate_cam', generateCam ? 'true' : 'false');

  if (bbox && bbox.length === 4) {
    formData.append('bbox', JSON.stringify(bbox));
  }

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let errorDetail = res.statusText;
    try {
      const errJson = await res.json();
      if (errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // Use statusText fallback
    }
    throw new Error(errorDetail);
  }

  return res.json();
}
