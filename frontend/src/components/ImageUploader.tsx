import React, { useRef, useState } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import { Upload, Sparkles, AlertCircle } from 'lucide-react';
import type { SampleItem } from '../types/api';

interface ImageUploaderProps {
  onImageSelected: (file: File | Blob, previewUrl: string, sampleClass?: string) => void;
  samples: SampleItem[];
  selectedSampleClass?: string;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageSelected,
  samples,
  selectedSampleClass,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const processFile = (file: File) => {
    setUploadError(null);
    if (!file.type.startsWith('image/')) {
      setUploadError('Please select a valid image file (JPEG, PNG, or WebP).');
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setUploadError('Image size exceeds 15MB limit.');
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    onImageSelected(file, previewUrl);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleSelectSample = async (sample: SampleItem) => {
    try {
      setUploadError(null);
      const res = await fetch(sample.url);
      if (!res.ok) throw new Error('Could not fetch sample image');
      const blob = await res.blob();
      const previewUrl = URL.createObjectURL(blob);
      onImageSelected(blob, previewUrl, sample.class_name);
    } catch {
      setUploadError('Failed to load sample image.');
    }
  };

  return (
    <div className="space-y-4">
      {/* Upload Dropzone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-emerald-500 bg-emerald-50/50'
            : 'border-slate-300 hover:border-emerald-400 bg-white hover:bg-slate-50/80 shadow-xs'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileChange}
          className="hidden"
        />

        <div className="flex flex-col items-center space-y-2">
          <div className="w-12 h-12 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-700">
            <Upload className="w-6 h-6" />
          </div>
          <div className="text-sm font-semibold text-slate-800">
            Click to upload field photo or drag & drop here
          </div>
          <p className="text-xs text-slate-500 max-w-sm">
            High-resolution leaf or field crop images (JPEG, PNG, WebP up to 15MB).
          </p>
        </div>
      </div>

      {uploadError && (
        <div className="flex items-center space-x-2 text-xs text-rose-700 bg-rose-50 border border-rose-200 p-2.5 rounded-lg">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Held-Out Benchmark Samples Gallery */}
      {samples.length > 0 && (
        <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs">
          <div className="flex items-center space-x-1.5 mb-2.5 text-xs font-semibold text-slate-700">
            <Sparkles className="w-3.5 h-3.5 text-amber-500" />
            <span>Try Held-Out Test Samples (No Manual Upload Needed):</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {samples.map((sample) => {
              const isSelected = selectedSampleClass === sample.class_name;
              return (
                <button
                  key={sample.id}
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSelectSample(sample);
                  }}
                  className={`text-left p-2 rounded-lg border text-xs transition-all flex flex-col justify-between cursor-pointer ${
                    isSelected
                      ? 'border-emerald-600 bg-emerald-50/80 font-semibold text-emerald-900 shadow-xs'
                      : 'border-slate-200 hover:border-slate-300 bg-slate-50/60 hover:bg-slate-100/80 text-slate-700'
                  }`}
                >
                  <span className="font-medium truncate block w-full">{sample.class_name}</span>
                  <span className="text-[10px] text-slate-500 block truncate mt-1">
                    {sample.filename}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
