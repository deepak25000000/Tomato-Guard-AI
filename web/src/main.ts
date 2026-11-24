import "./style.css";
import { createSession, loadMeta, predict, preprocessToNchw, softmax, UI, type ModelMeta } from "./infer";
import { InferenceSession } from "onnxruntime-web";

type Hist = { ts: string; top1: string; score: number; uncertain: boolean; version: string };

const DB = "tomatoguard-web";
function openDb(): Promise<IDBDatabase> {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore("history", { keyPath: "ts" });
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
async function addHist(h: Hist) { const db = await openDb(); const tx = db.transaction("history", "readwrite"); tx.objectStore("history").put(h); }
async function allHist(): Promise<Hist[]> { const db = await openDb(); return new Promise((res, rej)=>{const tx=db.transaction("history"); const req=tx.objectStore("history").getAll(); req.onsuccess=()=>res((req.result as Hist[]).sort((a,b)=>b.ts.localeCompare(a.ts))); req.onerror=()=>rej(req.error);}); }
async function clearHist() { const db = await openDb(); db.transaction("history","readwrite").objectStore("history").clear(); }

let session: InferenceSession | null = null;
let meta: ModelMeta | null = null;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function bilinearResize(src: ImageData, dstW: number, dstH: number): ImageData {
  const sw = src.width, sh = src.height, sd = src.data;
  const out = new ImageData(dstW, dstH);
  const od = out.data;
  const xRatio = sw / dstW;
  const yRatio = sh / dstH;
  for (let y = 0; y < dstH; y++) {
    const sy = (y + 0.5) * yRatio - 0.5;
    const y0 = Math.max(0, Math.floor(sy));
    const y1 = Math.min(sh - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < dstW; x++) {
      const sx = (x + 0.5) * xRatio - 0.5;
      const x0 = Math.max(0, Math.floor(sx));
      const x1 = Math.min(sw - 1, x0 + 1);
      const fx = sx - x0;
      const i00 = (y0 * sw + x0) * 4;
      const i10 = (y0 * sw + x1) * 4;
      const i01 = (y1 * sw + x0) * 4;
      const i11 = (y1 * sw + x1) * 4;
      const oi = (y * dstW + x) * 4;
      for (let c = 0; c < 3; c++) {
        const v00 = sd[i00 + c], v10 = sd[i10 + c], v01 = sd[i01 + c], v11 = sd[i11 + c];
        od[oi + c] = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
      }
      od[oi + 3] = 255;
    }
  }
  return out;
}

async function resizeToCanvas(img: HTMLImageElement, size: number): Promise<ImageData> {
  // Contract: direct resize to 224x224, bilinear, no crop. EXIF should be applied by decode;
  // we draw the decoded image at natural size then bilinear-resize with half-pixel centers.
  const nat = document.createElement("canvas");
  nat.width = img.naturalWidth; nat.height = img.naturalHeight;
  const nctx = nat.getContext("2d")!;
  nctx.drawImage(img, 0, 0);
  const src = nctx.getImageData(0, 0, nat.width, nat.height);
  return bilinearResize(src, size, size);
}

async function runFile(file: File, out: HTMLElement) {
  if (!session || !meta) { out.textContent = "Model not loaded."; return; }
  out.textContent = "Running...";
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url; await img.decode();
  const id = await resizeToCanvas(img, meta.input_size);
  const nchw = preprocessToNchw(id, meta.input_size, meta.mean, meta.std);
  const logits = await predict(session, nchw, meta.input_size);
  const probs = softmax(logits);
  const idx = Array.from(probs).map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p);
  const top1 = idx[0]; const uncertain = top1.p < meta.uncertain_threshold;
  const h: Hist = { ts: new Date().toISOString(), top1: meta.labels[top1.i], score: top1.p, uncertain, version: meta.model_version };
  await addHist(h);
  out.innerHTML = "";
  const banner = el("div","warn", uncertain ? UI.UNCERTAIN : `${UI.MODEL_SCORE_LABEL}: ${(top1.p*100).toFixed(1)}%`);
  out.appendChild(banner);
  const list = el("ul");
  for (const t of idx.slice(0,3)) { const li = el("li", "", `${meta.labels[t.i]} — ${UI.MODEL_SCORE_LABEL} ${(t.p*100).toFixed(1)}%`); list.appendChild(li); }
  out.appendChild(list);
  out.appendChild(el("p","muted", `Model ${meta.model_version} • ${UI.DISCLAIMER}`));
  out.appendChild(el("p","muted", UI.SCOPE));
  out.appendChild(el("p","muted", UI.TOBRFV));
  URL.revokeObjectURL(url);
  renderHistory();
}

async function renderHistory() {
  const box = document.getElementById("history")!; box.innerHTML = "";
  const rows = await allHist();
  if (!rows.length) { box.textContent = "No scans yet."; return; }
  for (const r of rows.slice(0,30)) {
    box.appendChild(el("div","hist", `${r.ts.slice(0,19)}Z — ${r.top1} (${UI.MODEL_SCORE_LABEL} ${(r.score*100).toFixed(1)}%${r.uncertain? ", uncertain":""})`));
  }
}

async function boot() {
  const app = document.getElementById("app")!;
  app.innerHTML = "";
  app.appendChild(el("div","banner", UI.RESEARCH_BANNER));
  app.appendChild(el("h1","", "TomatoGuard AI — Leaf screening (research)"));
  app.appendChild(el("p","muted", UI.DISCLAIMER));
  app.appendChild(el("p","muted", UI.SCOPE));
  app.appendChild(el("p","muted", UI.TOBRFV));

  const input = document.createElement("input");
  input.type = "file"; input.accept = "image/*"; input.setAttribute("capture","environment");
  const out = el("div","result","Loading model...");
  input.onchange = () => { const f = input.files?.[0]; if (f) runFile(f, out).catch(e=> out.textContent=String(e)); };
  app.appendChild(input); app.appendChild(out);
  app.appendChild(el("h2","","History (local only)"));
  const clear = el("button","", "Delete all data");
  clear.onclick = async () => { await clearHist(); renderHistory(); };
  app.appendChild(clear);
  const hist = el("div","history"); hist.id = "history"; app.appendChild(hist);

  try {
    meta = await loadMeta();
    session = await createSession("./assets/leaf_mobilenet_v3_large_seed42.onnx");
    (window as any).__tg = { session, meta, predict, preprocessToNchw, softmax, resizeToCanvas };
    out.textContent = "Model loaded. Choose a leaf photo.";
  } catch (e) {
    out.textContent = `Model load failed: ${e}`;
  }
  renderHistory();
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(()=>{});
}
boot();
