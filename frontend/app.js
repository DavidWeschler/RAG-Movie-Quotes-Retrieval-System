/**
 * RAG Movie Quotes Search - Frontend Application
 *
 * This JavaScript file handles:
 * - API communication with the FastAPI backend
 * - Search functionality
 * - Result rendering
 * - UI state management
 */

const API_BASE_URL = "http://localhost:8000";

const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const topKInput = document.getElementById("topK");
const thresholdInput = document.getElementById("threshold");
const thresholdValue = document.getElementById("thresholdValue");
const statusMessage = document.getElementById("statusMessage");
const resultsSection = document.getElementById("resultsSection");
const resultsContainer = document.getElementById("resultsContainer");
const resultsCount = document.getElementById("resultsCount");
const emptyState = document.getElementById("emptyState");
const loadingOverlay = document.getElementById("loadingOverlay");
const dbStatus = document.getElementById("dbStatus");

// Initialize the application
document.addEventListener("DOMContentLoaded", () => {
  initializeApp();
  setupEventListeners();
});

async function initializeApp() {
  await checkDatabaseStatus();
  thresholdValue.textContent = thresholdInput.value;
}

// Set up event listeners
function setupEventListeners() {
  // Enter key to search
  searchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      performSearch();
    }
  });

  // Update threshold display on change
  thresholdInput.addEventListener("input", () => {
    thresholdValue.textContent = thresholdInput.value;
  });
}

// Check if the database is connected and initialized
async function checkDatabaseStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();

    if (data.status === "healthy" && data.documents_loaded > 0) {
      dbStatus.textContent = `Database connected (${data.documents_loaded} quotes loaded)`;
      dbStatus.className = "db-status connected";
    } else if (data.status === "healthy") {
      dbStatus.textContent = "Database empty - Click to initialize";
      dbStatus.className = "db-status warning";
      dbStatus.style.cursor = "pointer";
      dbStatus.onclick = initializeDatabase;
    } else {
      throw new Error(data.error || "Unknown error");
    }
  } catch (error) {
    dbStatus.textContent = "Cannot connect to API server";
    dbStatus.className = "db-status error";
    showStatus("Cannot connect to API server. Make sure the backend is running on port 8000.", "error");
  }
}

// Initialize the database
async function initializeDatabase() {
  showLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/initialize`, {
      method: "POST",
    });
    const data = await response.json();

    if (response.ok) {
      showStatus(data.message, "success");
      await checkDatabaseStatus();
    } else {
      throw new Error(data.detail || "Failed to initialize");
    }
  } catch (error) {
    showStatus(`Initialization failed: ${error.message}`, "error");
  } finally {
    showLoading(false);
  }
}

// Perform a search query
async function performSearch() {
  const query = searchInput.value.trim();

  if (!query) {
    showStatus("Please enter a search query", "warning");
    return;
  }

  showLoading(true);
  hideStatus();

  try {
    const params = new URLSearchParams({
      query: query,
      top_k: topKInput.value,
      similarity_threshold: thresholdInput.value,
    });

    const response = await fetch(`${API_BASE_URL}/search?${params}`);
    const data = await response.json();

    if (response.ok) {
      displayResults(data);
    } else {
      throw new Error(data.detail || "Search failed");
    }
  } catch (error) {
    showStatus(`Search failed: ${error.message}`, "error");
    hideResults();
  } finally {
    showLoading(false);
  }
}

// Display search results
function displayResults(data) {
  const { query, results, total_results, parameters } = data;

  // Show results section, hide empty state
  emptyState.classList.add("hidden");
  resultsSection.classList.remove("hidden");

  // Update results count
  resultsCount.textContent = `${total_results} result${total_results !== 1 ? "s" : ""} for "${query}"`;

  // Clear previous results
  resultsContainer.innerHTML = "";

  if (results.length === 0) {
    resultsContainer.innerHTML = `
            <div class="empty-state">
                <h3>No Matches Found</h3>
                <p>Try adjusting your query or lowering the similarity threshold.</p>
            </div>
        `;
    return;
  }

  // Render each result
  results.forEach((result, index) => {
    const card = createResultCard(result, index + 1);
    resultsContainer.appendChild(card);
  });
}

// Create a result card element
function createResultCard(result, rank) {
  const { id, document: docText, metadata, similarity_score } = result;

  // Determine similarity class
  let similarityClass = "low";
  if (similarity_score >= 0.7) similarityClass = "high";
  else if (similarity_score >= 0.5) similarityClass = "medium";

  // Parse themes into array (may not exist in CSV data)
  const themes = metadata.theme ? metadata.theme.split(", ") : [];

  // Create card element
  const card = window.document.createElement("div");
  card.className = "result-card";
  card.innerHTML = `
        <div class="result-header">
            <span class="result-id">Chunk #${id} (Rank: ${rank})</span>
            <div class="similarity-badge">
                <span class="similarity-score ${similarityClass}">${(similarity_score * 100).toFixed(1)}%</span>
                <span class="similarity-label">similarity</span>
            </div>
        </div>
        <div class="quote-text">"${metadata.original_quote}"</div>
        <div class="result-metadata">
            <div class="metadata-item">
                <span class="metadata-icon">Movie:</span>
                <span>${metadata.movie}</span>
            </div>
            ${
              metadata.character
                ? `<div class="metadata-item">
                <span class="metadata-icon">Character:</span>
                <span>${metadata.character}</span>
            </div>`
                : ""
            }
            <div class="metadata-item">
                <span class="metadata-icon">Year:</span>
                <span>${metadata.year}</span>
            </div>
            ${
              metadata.type
                ? `<div class="metadata-item">
                <span class="metadata-icon">Type:</span>
                <span>${metadata.type}</span>
            </div>`
                : ""
            }
        </div>
        ${
          themes.length > 0
            ? `<div class="theme-tags">
            ${themes.map((theme) => `<span class="theme-tag">${theme.trim()}</span>`).join("")}
        </div>`
            : ""
        }
    `;

  return card;
}

function hideResults() {
  resultsSection.classList.add("hidden");
  emptyState.classList.remove("hidden");
}

function showStatus(message, type = "info") {
  statusMessage.textContent = message;
  statusMessage.className = `status-message ${type}`;
  statusMessage.classList.remove("hidden");
}

function hideStatus() {
  statusMessage.classList.add("hidden");
}

function showLoading(show) {
  if (show) {
    loadingOverlay.classList.remove("hidden");
  } else {
    loadingOverlay.classList.add("hidden");
  }
}

// Set a query in the search input (for example buttons)
function setQuery(query) {
  searchInput.value = query;
  performSearch();
}
