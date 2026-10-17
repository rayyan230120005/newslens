const DEFAULT_API_URL = "http://127.0.0.1:8000/analyze";
const REQUEST_TIMEOUT_MS = 15000;

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({
    apiUrl: DEFAULT_API_URL
  });
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "NEWSLENS_ANALYZE") {
    return false;
  }

  analyzeText(message.payload)
    .then((data) => {
      sendResponse({ ok: true, data });
    })
    .catch((error) => {
      sendResponse({
        ok: false,
        error: error?.message || "Analysis request failed"
      });
    });

  return true;
});

async function analyzeText(payload) {
  const { apiUrl = DEFAULT_API_URL } = await chrome.storage.local.get({
    apiUrl: DEFAULT_API_URL
  });
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    const responseText = await response.text();
    const data = parseJson(responseText);

    if (!response.ok) {
      throw new Error(getHttpErrorMessage(response.status, data, responseText));
    }

    return data;
  } finally {
    clearTimeout(timeoutId);
  }
}

function parseJson(responseText) {
  if (!responseText) {
    return null;
  }

  try {
    return JSON.parse(responseText);
  } catch {
    return null;
  }
}

function getHttpErrorMessage(status, data, responseText) {
  if (data?.detail) {
    return `HTTP ${status}: ${JSON.stringify(data.detail)}`;
  }

  if (responseText) {
    return `HTTP ${status}: ${responseText}`;
  }

  return `HTTP ${status}`;
}
