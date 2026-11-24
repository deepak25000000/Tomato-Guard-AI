import { InferenceSession, Tensor } from "onnxruntime-web";

export interface ModelMeta {
  labels: string[];
  input_size: number;
  mean: number[];
  std: number[];
  uncertain_threshold: number;
  model_version: string;
  sha256?: Record<string, string>;
}

export const UI = {
  DISCLAIMER: "Screening aid for research. Not a diagnosis. Confirm with an agronomist or laboratory test.",
  SCOPE: "Trained on one public dataset of lab-style leaf photos. Accuracy on farm photos is unknown.",
  TOBRFV: "This leaf model does not detect ToBRFV.",
  UNCERTAIN: "Not confident. Re-photograph one leaf, filling the frame, in good light.",
  RESEARCH_BANNER: "Research demo on recorded greenhouse data. Not for field use.",
  MODEL_SCORE_LABEL: "model score",
} as const;

// IMPORTANT: This file is the web copy of the fixed strings; tests scan it.
// The authoritative source is build/ui_strings.json. Keep identical.

export async function loadMeta(): Promise<ModelMeta> {
  const r = await fetch("./assets/model_meta.json");
  if (!r.ok) throw new Error(`model_meta.json: ${r.status}`);
  const m = await r.json();
  // Normalize build/model_meta.json schema to web runtime schema.
  if (Array.isArray(m.input_size)) m.input_size = m.input_size[0];
  if (!m.labels) m.labels = m.display_labels || m.classes;
  if (m.uncertain_threshold == null && m.threshold != null) m.uncertain_threshold = m.threshold;
  if (!m.model_version) m.model_version = m.model_version || "unknown";
  return m as ModelMeta;
}

export async function createSession(modelUrl: string): Promise<InferenceSession> {
  // WASM backend; single-thread by default for broad compatibility.
  (InferenceSession as any).env = (InferenceSession as any).env || {};
  return InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
}

export function preprocessToNchw(imageData: ImageData, size: number, mean: number[], std: number[]): Float32Array {
  // Contract: imageData is already RGB at `size x size` (direct bilinear resize, no crop).
  // Scale [0,255]->[0,1], ImageNet normalize, NCHW.
  const { data } = imageData;
  const hw = size * size;
  const out = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    out[i] = (r - mean[0]) / std[0];
    out[hw + i] = (g - mean[1]) / std[1];
    out[2 * hw + i] = (b - mean[2]) / std[2];
  }
  return out;
}

export function softmax(logits: Float32Array): Float32Array {
  let m = -Infinity;
  for (const v of logits) if (v > m) m = v;
  const e = new Float32Array(logits.length);
  let s = 0;
  for (let i = 0; i < logits.length; i++) { e[i] = Math.exp(logits[i] - m); s += e[i]; }
  for (let i = 0; i < e.length; i++) e[i] /= s;
  return e;
}

export async function predict(session: InferenceSession, nchw: Float32Array, size: number): Promise<Float32Array> {
  const input = new Tensor("float32", nchw, [1, 3, size, size]);
  const feeds: Record<string, Tensor> = {};
  feeds[session.inputNames[0]] = input;
  const out = await session.run(feeds);
  const logits = out[session.outputNames[0]].data as Float32Array;
  return logits;
}
