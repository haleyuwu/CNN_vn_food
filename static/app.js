// ================== Helpers ==================
async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  const text = await res.text(); // đọc text để debug nếu không phải JSON
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    console.error("Server did not return JSON. Raw response:", text);
    throw new Error(text.substring(0, 300) || "Non-JSON response");
  }
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

// ================== Upload 1 ảnh ==================
const fileInput  = document.getElementById("file-input");
const btnPredict = document.getElementById("btn-predict");
const preview    = document.getElementById("preview");
const resultCard = document.getElementById("result");
const predLabel  = document.getElementById("pred-label");
const predConf   = document.getElementById("pred-conf");
const top3List   = document.getElementById("top3");

let selectedFile = null;

if (fileInput) {
  fileInput.addEventListener("change", (e) => {
    preview.innerHTML = "";
    selectedFile = e.target.files[0] || null;
    if (selectedFile) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(selectedFile);
      preview.appendChild(img);
    }
  });
}

if (btnPredict) {
  btnPredict.addEventListener("click", async () => {
    if (!selectedFile) {
      alert("Chọn 1 ảnh trước đã nha!");
      return;
    }
    const formData = new FormData();
    formData.append("image", selectedFile);

    btnPredict.disabled = true;
    btnPredict.textContent = "Đang dự đoán...";

    try {
      const data = await fetchJSON("/predict", { method: "POST", body: formData });
      predLabel.textContent = data.prediction;
      predConf.textContent = (data.confidence * 100).toFixed(2) + "%";
      top3List.innerHTML = "";
      data.top3.forEach(item => {
        const li = document.createElement("li");
        li.textContent = `${item.label}: ${(item.prob * 100).toFixed(2)}%`;
        top3List.appendChild(li);
      });
      resultCard.classList.remove("hidden");
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      btnPredict.disabled = false;
      btnPredict.textContent = "Dự đoán";
    }
  });
}

// ================== Realtime webcam ==================
const video     = document.getElementById("video");
const canvas    = document.getElementById("canvas");
const btnStart  = document.getElementById("btn-start");
const btnStop   = document.getElementById("btn-stop");
const liveLabel = document.getElementById("live-label");
const liveBar   = document.getElementById("live-bar");
const liveBadge = document.getElementById("live-badge");

let stream = null;
let loopId = null;
let sending = false;
const INTERVAL_MS = 1000; // ~1 fps để nhẹ máy
const CNN_INPUT_SIZE = 64; // khớp backend

async function startCam() {
  try {
    // Lưu ý: camera chỉ hoạt động ở https hoặc http://localhost
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    btnStart.disabled = true;
    btnStop.disabled  = false;
    if (liveBadge) liveBadge.textContent = "LIVE";
    startLoop();
  } catch (e) {
    alert("Không bật được camera: " + e.message);
  }
}

function stopCam() {
  if (loopId) { clearInterval(loopId); loopId = null; }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  btnStart.disabled = false;
  btnStop.disabled  = true;
  if (liveBadge) liveBadge.textContent = "OFF";
}

function startLoop() {
  const ctx = canvas.getContext("2d");
  loopId = setInterval(async () => {
    if (sending) return;
    if (!video.videoWidth) return;

    // Resize frame về kích thước mô hình (giảm băng thông)
    canvas.width  = CNN_INPUT_SIZE;
    canvas.height = CNN_INPUT_SIZE;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    sending = true;
    try {
      const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.9));
      const form = new FormData();
      form.append("image", blob, "frame.jpg"); // có đuôi jpg để pass allowed_file

      const data = await fetchJSON("/predict", { method: "POST", body: form });
      liveLabel.textContent = data.prediction;
      const pct = Math.max(0, Math.min(100, Math.round(data.confidence * 100)));
      liveBar.style.width = pct + "%";
    } catch (err) {
      console.warn("Realtime error:", err.message);
    } finally {
      sending = false;
    }
  }, INTERVAL_MS);
}

if (btnStart) btnStart.addEventListener("click", startCam);
if (btnStop)  btnStop .addEventListener("click", stopCam);
window.addEventListener("beforeunload", stopCam);
