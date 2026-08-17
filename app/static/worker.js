"use strict";

// The mosaic is rendered off the main thread, so that the page stays
// responsive while the optimizer runs. Pyodide and its packages are fetched
// from the CDN once and are then served from the browser cache.
//
// This is a module worker, because Pyodide is loaded from a different origin:
// `importScripts` fetches without CORS and browsers reject the opaque response
// that comes back, while a module import is fetched with CORS and succeeds.

const PYODIDE_VERSION = "314.0.4";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
// numpy, scipy and matplotlib carry the algorithm, and click comes along
// because `voronoi_mosaic` imports it for its command line interface
const PACKAGES = ["numpy", "scipy", "matplotlib", "click"];

// the algorithm, and the bridge that hands the pixels over to it
const MODULES = ["voronoi_mosaic.py", "driver.py"];
const MODULE_DIR = "/home/pyodide";

let pyodide = null;
let driver = null;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

/** Copy a Python module of the app into the Pyodide file system. */
async function writeModule(name) {
  const url = new URL(name, self.location.href).href;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`could not fetch ${url}, status ${response.status}`);
  }

  pyodide.FS.writeFile(`${MODULE_DIR}/${name}`, await response.text());
}

async function init() {
  post("status", { message: `Loading Pyodide ${PYODIDE_VERSION}, this happens once.` });

  // imported dynamically, so that the version above stays the only place it is named
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);

  pyodide = await loadPyodide({
    indexURL: PYODIDE_URL,
    stderr: (line) => console.warn(line),
  });

  post("status", { message: `Loading ${PACKAGES.join(", ")}, about 27 MB.` });
  await pyodide.loadPackage(PACKAGES);

  post("status", { message: "Loading the mosaic algorithm." });
  await Promise.all(MODULES.map(writeModule));

  pyodide.runPython(`import sys; sys.path.insert(0, "${MODULE_DIR}")`);
  driver = pyodide.pyimport("driver");

  driver.install_progress_handler((message) => post("progress", { message }));
  driver.warm_up();

  post("ready");
}

function run({ pixels, width, height, params }) {
  const started = performance.now();

  pyodide.FS.writeFile(driver.PIXELS_PATH, pixels);
  driver.run_mosaic(width, height, JSON.stringify(params));

  // copied out of the Pyodide file system, so that handing the buffer to the
  // page cannot detach anything the WebAssembly heap still points at
  const png = new Uint8Array(pyodide.FS.readFile(driver.MOSAIC_PATH));
  const seconds = (performance.now() - started) / 1000;

  self.postMessage({ type: "result", png, seconds }, [png.buffer]);
}

self.addEventListener("message", ({ data }) => {
  if (data.type !== "run") {
    return;
  }

  try {
    run(data);
  } catch (error) {
    post("error", { message: `Rendering failed: ${error.message}` });
  }
});

init().catch((error) => post("error", { message: `Startup failed: ${error.message}` }));
