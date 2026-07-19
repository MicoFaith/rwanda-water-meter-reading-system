const historyLoading = document.getElementById("historyLoading");
const historyEmpty = document.getElementById("historyEmpty");
const historyError = document.getElementById("historyError");
const historyTable = document.getElementById("historyTable");
const historyBody = document.getElementById("historyBody");

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
}

function formatTimestamp(ts) {
  if (!ts) return "—";
  const d = new Date(ts.replace(" ", "T") + "Z");
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleString();
}

async function loadHistory() {
  try {
    const response = await fetch("/api/readings?limit=50");
    if (!response.ok) {
      throw new Error("Server returned an error while loading history.");
    }
    const readings = await response.json();

    historyLoading.hidden = true;

    if (!readings.length) {
      historyEmpty.hidden = false;
      return;
    }

    historyBody.innerHTML = readings.map((r) => `
      <tr>
        <td>${escapeHtml(formatTimestamp(r.timestamp))}</td>
        <td>${escapeHtml(r.image_name || "—")}</td>
        <td>${escapeHtml(r.reading || "—")}</td>
        <td>${r.confidence != null ? (r.confidence * 100).toFixed(0) + "%" : "—"}</td>
        <td><span class="badge badge--${escapeHtml(r.status)}">${escapeHtml(r.status)}</span></td>
      </tr>
    `).join("");

    historyTable.hidden = false;
  } catch (err) {
    historyLoading.hidden = true;
    historyError.hidden = false;
    historyError.textContent = err.message || "Could not load reading history.";
  }
}

loadHistory();
