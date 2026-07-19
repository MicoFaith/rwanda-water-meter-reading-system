const dropzone = document.getElementById("dropzone");
const dropzoneContent = document.getElementById("dropzoneContent");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const loading = document.getElementById("loading");
const result = document.getElementById("result");

let selectedFile = null;

function setSelectedFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please select a valid image file.");
    return;
  }

  selectedFile = file;
  predictBtn.disabled = false;
  hideResult();

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropzoneContent.hidden = true;
  };
  reader.readAsDataURL(file);
}

browseBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) {
    setSelectedFile(e.target.files[0]);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.add("dropzone--active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dropzone--active");
  });
});

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    setSelectedFile(file);
  }
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideResult();
  loading.hidden = false;
  predictBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("Unexpected response from server.");
    }

    if (!response.ok && !data.status) {
      throw new Error("Something went wrong. Please try again.");
    }

    renderResult(data);
  } catch (err) {
    showError(err.message || "Could not reach the server. Please try again.");
  } finally {
    loading.hidden = true;
    predictBtn.disabled = false;
  }
});

function renderResult(data) {
  result.hidden = false;

  if (data.status === "success") {
    result.className = "result result--success";
    result.innerHTML = `
      <div class="result__row"><span class="result__label">Status</span><span>Success</span></div>
      <div class="result__row"><span class="result__label">Reading</span><span>${escapeHtml(data.reading)}</span></div>
      <div class="result__row"><span class="result__label">Confidence</span><span>${(data.confidence * 100).toFixed(0)}%</span></div>
    `;
  } else if (data.status === "uncertain") {
    result.className = "result result--uncertain";
    result.innerHTML = `
      <div class="result__row"><span class="result__label">Status</span><span>Uncertain</span></div>
      <div class="result__row"><span class="result__label">Reading</span><span>${escapeHtml(data.reading)}</span></div>
      <div class="result__warning">${escapeHtml(data.warning || "Please recapture the image.")}</div>
    `;
  } else {
    showError(data.message || "Prediction failed. Please try a different image.");
  }
}

function showError(message) {
  result.hidden = false;
  result.className = "result result--error";
  result.innerHTML = `<div class="result__row"><span class="result__label">Error</span><span>${escapeHtml(message)}</span></div>`;
}

function hideResult() {
  result.hidden = true;
  result.innerHTML = "";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
}
