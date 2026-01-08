const API_BASE = ''; // Relative path since we'll serve it from main.py

// DOM Elements
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const controls = document.getElementById('controls');
const checkQualityBtn = document.getElementById('check-quality-btn');
const qualityResult = document.getElementById('quality-result');
const extractBtn = document.getElementById('extract-btn');

const extractedDataTab = document.querySelector('[data-tab="extracted-data"]');
const ocrTextTab = document.querySelector('[data-tab="ocr-text"]');
const tabContentData = document.getElementById('extracted-data');
const tabContentOcr = document.getElementById('ocr-text');
const loadingIndicator = document.getElementById('loading-indicator');
const resultsTableContainer = document.getElementById('results-table-container');
const ocrRawText = document.getElementById('ocr-raw-text');

const metricsTableHead = document.querySelector('#metrics-table thead');
const metricsTableBody = document.querySelector('#metrics-table tbody');
const refreshMetricsBtn = document.getElementById('refresh-metrics-btn');

// New Elements
const historyList = document.getElementById('history-list');
const historyTab = document.querySelector('[data-tab="history"]');
const tabContentHistory = document.getElementById('history');
const saveChangesBtn = document.getElementById('save-changes-btn');
const editActions = document.getElementById('edit-actions');

// State
// State
let currentFile = null;
let currentQualityMetrics = null;
let currentQualityPass = null;
let currentRunId = null; // Track current run ID for saving
let currentRawJson = null; // Track raw JSON for merging edits

// Event Listeners
fileInput.addEventListener('change', handleFileSelect);
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--primary-color)';
});
dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border-color)';
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border-color)';
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

checkQualityBtn.addEventListener('click', checkQuality);
extractBtn.addEventListener('click', extractData);
refreshMetricsBtn.addEventListener('click', loadMetrics);

extractedDataTab.addEventListener('click', () => switchTab('data'));
ocrTextTab.addEventListener('click', () => switchTab('ocr'));
historyTab.addEventListener('click', () => {
    switchTab('history');
    loadHistory();
});
saveChangesBtn.addEventListener('click', saveEdits);

// Init
loadMetrics();

// Functions

function handleFileSelect(e) {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }
    currentFile = file;

    // Reset State
    currentQualityMetrics = null;
    currentQualityPass = null;
    qualityResult.style.display = 'none';
    extractBtn.disabled = true;
    resultsTableContainer.innerHTML = '<p class="placeholder-text">Click "Extract Data" to process.</p>';
    editActions.style.display = 'none';
    ocrRawText.textContent = '';

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        controls.style.display = 'block';

        // Auto-trigger quality check
        checkQuality();
    };
    reader.readAsDataURL(file);
}

async function checkQuality() {
    if (!currentFile) return;

    // UI Loading State (custom for quality)
    qualityResult.style.display = 'block';
    qualityResult.className = 'quality-status';
    qualityResult.style.backgroundColor = '#f8f9fa';
    qualityResult.style.color = '#666';
    qualityResult.style.border = '1px solid #ddd';
    qualityResult.textContent = 'Checking blur, brightness, contrast...';

    // checkQualityBtn.disabled = true; // Button is hidden now

    const formData = new FormData();
    formData.append('image', currentFile);

    try {
        const res = await fetch(`${API_BASE}/api/quality`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error('Quality check failed');

        const data = await res.json();
        currentQualityMetrics = data;
        currentQualityPass = data.quality_pass;

        // Show Result
        qualityResult.textContent = currentQualityPass ? 'Good Quality' : 'Not Good Quality';
        qualityResult.className = 'quality-status ' + (currentQualityPass ? 'pass' : 'fail');
        qualityResult.style.display = 'block';

        extractBtn.disabled = false;

    } catch (err) {
        console.error(err);
        alert('Error checking quality: ' + err.message);
    } finally {
        // checkQualityBtn.disabled = false;
        // checkQualityBtn.textContent = 'Check Quality';
    }
}

async function extractData() {
    if (!currentFile) return;

    // Switch to data tab and show loading
    switchTab('data');
    loadingIndicator.classList.remove('hidden');
    resultsTableContainer.innerHTML = '';
    extractBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', currentFile);
    if (currentQualityPass !== null) {
        formData.append('quality_pass', currentQualityPass);
        formData.append('quality_metrics_json', JSON.stringify(currentQualityMetrics));
    }

    try {
        const res = await fetch(`${API_BASE}/api/extract`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error('Extraction failed');

        const data = await res.json();

        currentRunId = data.run_id; // Store run ID
        currentRawJson = data.raw_json; // Store raw JSON

        renderResults(data);
        loadMetrics(); // Refresh metrics after extraction
        editActions.style.display = 'block'; // Show save button

    } catch (err) {
        console.error(err);
        resultsTableContainer.innerHTML = `<p class="error-text">Error: ${err.message}</p>`;
    } finally {
        loadingIndicator.classList.add('hidden');
        extractBtn.disabled = false;
    }
}

function renderResults(data) {
    // Render Fields Table
    if (data.fields_table && data.fields_table.length > 0) {
        let html = '<table id="results-table"><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>';
        data.fields_table.forEach(row => {
            html += `
                <tr>
                    <td><strong>${row.field}</strong></td>
                    <td contenteditable="true" class="editable-cell" data-field="${row.field}">${row.value || ''}</td>
                </tr>
            `;
        });
        html += '</tbody></table>';
        resultsTableContainer.innerHTML = html;
    } else {
        resultsTableContainer.innerHTML = '<p>No fields extracted.</p>';
        editActions.style.display = 'none';
    }

    // Render OCR Text
    const ocrText = data.ocr?.raw_text || '';
    ocrRawText.textContent = ocrText;
}

// History Functions
async function loadHistory() {
    historyList.innerHTML = '<p class="placeholder-text">Loading...</p>';
    try {
        const res = await fetch(`${API_BASE}/api/history`);
        if (!res.ok) throw new Error('Failed to load history');

        let data = await res.json();

        // Handle potential object wrapper (e.g. {"runs": [...]})
        if (!Array.isArray(data) && data.runs && Array.isArray(data.runs)) {
            data = data.runs;
        }

        renderHistoryList(data);
    } catch (err) {
        historyList.innerHTML = `<p class="error-text">Error: ${err.message}</p>`;
    }
}

function renderHistoryList(runs) {
    if (!Array.isArray(runs)) {
        console.error('Expected array for history, got:', runs);
        historyList.innerHTML = `<p class="error-text" style="word-break: break-all;">Server Error: Expected list, got ${typeof runs}: ${JSON.stringify(runs)}</p>`;
        return;
    }

    if (runs.length === 0) {
        historyList.innerHTML = '<p class="placeholder-text">No history found.</p>';
        return;
    }

    let html = `
        <table class="history-table">
            <thead>
                <tr>
                    <th style="width: 50px;">#</th>
                    <th>Date & Time</th>
                </tr>
            </thead>
            <tbody>
    `;

    runs.forEach((run, index) => {
        // Handle different schema variations
        const runId = run.run_id || run.id;
        const ts = run.timestamp || run.created_at;

        if (!runId) return; // Skip invalid items

        const dateObj = ts ? new Date(ts) : null;
        const dateStr = dateObj ? dateObj.toLocaleDateString() + ' ' + dateObj.toLocaleTimeString() : 'Unknown Date';

        html += `
            <tr onclick="loadHistoryItem('${runId}')" class="history-row" title="Run ID: ${runId}">
                <td>${runs.length - index}</td>
                <td>${dateStr}</td>
            </tr>
         `;
    });
    html += '</tbody></table>';
    historyList.innerHTML = html;
}

async function loadHistoryItem(runId) {
    switchTab('data');
    loadingIndicator.classList.remove('hidden');
    resultsTableContainer.innerHTML = '';

    try {
        const res = await fetch(`${API_BASE}/api/history/${runId}`);
        if (!res.ok) throw new Error('Failed to load run details');
        const data = await res.json();

        currentRunId = data.run_id;
        currentRawJson = data.raw_json;

        renderResults(data);
        editActions.style.display = 'block';

    } catch (err) {
        resultsTableContainer.innerHTML = `<p class="error-text">Error: ${err.message}</p>`;
    } finally {
        loadingIndicator.classList.add('hidden');
    }
}

async function saveEdits() {
    if (!currentRunId) return;

    // Get status element
    const statusSpan = document.getElementById('save-status');

    saveChangesBtn.disabled = true;
    saveChangesBtn.textContent = 'Saving...';

    if (statusSpan) {
        statusSpan.textContent = '';
        statusSpan.style.color = 'var(--text-color)';
    }

    // Gather edits from table
    const table = document.getElementById('results-table');
    if (!table) return;

    const rows = table.querySelectorAll('tbody tr');
    const updates = {};

    rows.forEach(row => {
        const cell = row.querySelector('.editable-cell');
        const fieldName = cell.dataset.field;
        const newValue = cell.innerText.trim();
        updates[fieldName] = newValue;
    });

    try {
        const res = await fetch(`${API_BASE}/api/save_edits`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                run_id: currentRunId,
                updates: updates
            })
        });

        if (!res.ok) throw new Error('Failed to save');

        console.log('Save successful'); // Debug

        // Visual feedback
        saveChangesBtn.textContent = 'Saved!';
        saveChangesBtn.style.backgroundColor = 'var(--success-color)';

        if (statusSpan) {
            console.log('Updating status text');
            statusSpan.textContent = 'Changes saved successfully!';
            statusSpan.style.color = 'var(--success-color)';
            statusSpan.style.display = 'inline-block'; // Force visibility
        } else {
            console.error('Status span element not found');
        }

        setTimeout(() => {
            saveChangesBtn.textContent = 'Save Changes';
            saveChangesBtn.disabled = false;
            saveChangesBtn.style.backgroundColor = '';

            // Clear message after 3 seconds
            setTimeout(() => {
                if (statusSpan) statusSpan.textContent = '';
            }, 3000);

        }, 1500);

    } catch (err) {
        console.error(err);
        saveChangesBtn.disabled = false;
        saveChangesBtn.textContent = 'Save Changes';

        if (statusSpan) {
            statusSpan.textContent = 'Error: ' + err.message;
            statusSpan.style.color = 'var(--danger-color)';
        } else {
            alert('Error saving edits: ' + err.message);
        }
    }
}

async function loadMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/metrics`);
        if (!res.ok) throw new Error('Failed to load metrics');

        const data = await res.json();
        renderMetrics(data);
    } catch (err) {
        console.error(err);
    }
}

function renderMetrics(data) {
    if (!data.header || !data.rows) return;

    // Header
    let theadHtml = '<tr>';
    data.header.forEach(h => {
        theadHtml += `<th>${h}</th>`;
    });
    theadHtml += '</tr>';
    metricsTableHead.innerHTML = theadHtml;

    // Rows (Limit to last 50 for performance if huge, reversed)
    let tbodyHtml = '';
    const rows = [...data.rows].reverse().slice(0, 50);

    rows.forEach(row => {
        tbodyHtml += '<tr>';
        row.forEach(cell => {
            tbodyHtml += `<td>${cell}</td>`;
        });
        tbodyHtml += '</tr>';
    });
    metricsTableBody.innerHTML = tbodyHtml;
}

function switchTab(tabName) {
    if (tabName === 'data') {
        extractedDataTab.classList.add('active');
        ocrTextTab.classList.remove('active');
        historyTab.classList.remove('active');

        tabContentData.classList.remove('hidden');
        tabContentOcr.classList.add('hidden');
        tabContentHistory.classList.add('hidden');
    } else if (tabName === 'history') {
        extractedDataTab.classList.remove('active');
        ocrTextTab.classList.remove('active');
        historyTab.classList.add('active');

        tabContentData.classList.add('hidden');
        tabContentOcr.classList.add('hidden');
        tabContentHistory.classList.remove('hidden');
    } else {
        extractedDataTab.classList.remove('active');
        ocrTextTab.classList.add('active');
        historyTab.classList.remove('active');

        tabContentData.classList.add('hidden');
        tabContentOcr.classList.remove('hidden');
        tabContentHistory.classList.add('hidden');
    }
}
