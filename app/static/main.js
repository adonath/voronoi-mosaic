"use strict";

// a module worker, so that it can import Pyodide from the CDN, see worker.js
const worker = new Worker("worker.js", { type: "module" });

const form = document.getElementById("controls");
const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const dropLabel = document.getElementById("drop-label");
const runButton = document.getElementById("run");
const statusBox = document.getElementById("status");
const initMethod = document.getElementById("init-method");
const previewInput = document.getElementById("preview-input");
const previewOutput = document.getElementById("preview-output");
const placeholderInput = document.getElementById("placeholder-input");
const placeholderOutput = document.getElementById("placeholder-output");
const download = document.getElementById("download");

// the parameters of `make_mosaic`, with the parser that puts them back into
// the type the Python side expects
const PARAMETERS = {
  "init-method": String,
  cellsize: Number,
  jitter: Number,
  npoints: parseInt,
  niter: parseInt,
  seed: parseInt,
  "outline-color": String,
  "background-color": String,
  pad: Number,
  radius: Number,
  dpi: parseInt,
};

let selectedFile = null;
let isReady = false;
let isRunning = false;
const objectUrls = [];

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
}

function updateRunButton() {
  runButton.disabled = !isReady || isRunning || selectedFile === null;
}

function toObjectUrl(blob) {
  const url = URL.createObjectURL(blob);
  objectUrls.push(url);
  return url;
}

/** Show only the parameters that apply to the selected placement method. */
function updateVisibleParameters() {
  for (const label of document.querySelectorAll("label[data-method]")) {
    label.hidden = label.dataset.method !== initMethod.value;
  }
}

function selectFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Please choose an image file.", true);
    return;
  }

  selectedFile = file;
  dropLabel.textContent = file.name;

  previewInput.src = toObjectUrl(file);
  previewInput.hidden = false;
  placeholderInput.hidden = true;

  updateRunButton();
}

/**
 * Decode an image file into raw RGBA pixels, scaled down so that its long edge
 * is at most `maxSize`. The optimizer touches every pixel on every iteration,
 * so the size of the image dominates the runtime.
 */
async function decodeImage(file, maxSize) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));

  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();

  return context.getImageData(0, 0, width, height);
}

function collectParameters() {
  const params = {};

  for (const [id, parse] of Object.entries(PARAMETERS)) {
    // the form uses the option names of the command line tool
    params[id.replace(/-/g, "_")] = parse(document.getElementById(id).value);
  }

  return params;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!isReady || isRunning || selectedFile === null) {
    return;
  }

  isRunning = true;
  updateRunButton();
  setStatus("Decoding the image");

  try {
    const maxSize = parseInt(document.getElementById("maxsize").value);
    const image = await decodeImage(selectedFile, maxSize);
    const pixels = new Uint8Array(image.data.buffer.slice(0));

    setStatus(`Rendering a ${image.width} by ${image.height} mosaic`);

    worker.postMessage(
      {
        type: "run",
        pixels,
        width: image.width,
        height: image.height,
        params: collectParameters(),
      },
      [pixels.buffer],
    );
  } catch (error) {
    isRunning = false;
    updateRunButton();
    setStatus(`Could not read the image: ${error.message}`, true);
  }
});

worker.addEventListener("message", ({ data }) => {
  if (data.type === "status" || data.type === "progress") {
    setStatus(data.message);
    return;
  }

  if (data.type === "ready") {
    isReady = true;
    updateRunButton();
    setStatus("Ready.");
    return;
  }

  if (data.type === "result") {
    isRunning = false;
    updateRunButton();

    const url = toObjectUrl(new Blob([data.png], { type: "image/png" }));

    previewOutput.src = url;
    previewOutput.hidden = false;
    placeholderOutput.hidden = true;

    download.href = url;
    download.download = selectedFile.name.replace(/\.[^.]+$/, "") + "-mosaic.png";
    download.hidden = false;

    setStatus(`Done in ${data.seconds.toFixed(1)} s.`);
    return;
  }

  if (data.type === "error") {
    isRunning = false;
    updateRunButton();
    setStatus(data.message, true);
  }
});

worker.addEventListener("error", (event) => {
  isRunning = false;
  updateRunButton();
  // a worker that fails to load at all reports no message of its own
  setStatus(`The worker failed: ${event.message || "see the browser console"}`, true);
});

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
initMethod.addEventListener("change", updateVisibleParameters);

for (const name of ["dragenter", "dragover"]) {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.add("dragover");
  });
}

for (const name of ["dragleave", "drop"]) {
  drop.addEventListener(name, () => drop.classList.remove("dragover"));
}

drop.addEventListener("drop", (event) => {
  event.preventDefault();
  selectFile(event.dataTransfer.files[0]);
});

window.addEventListener("pagehide", () => {
  objectUrls.forEach(URL.revokeObjectURL);
});

updateVisibleParameters();
