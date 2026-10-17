(() => {
  const selectors = window.NewsLensSelectors;
  const processedAttr = "data-newslens-processed";
  const panelClass = "newslens-panel";
  const cachePrefix = "newslens:analysis:";
  const minTextLength = 12;

  let scanTimer = null;

  function scheduleScan() {
    window.clearTimeout(scanTimer);
    scanTimer = window.setTimeout(scanTweets, 300);
  }

  function scanTweets() {
    document.querySelectorAll(selectors.tweet).forEach((tweet) => {
      if (tweet.getAttribute(processedAttr) === "true") {
        return;
      }

      const text = extractTweetText(tweet);
      if (text.length < minTextLength) {
        return;
      }

      tweet.setAttribute(processedAttr, "true");
      void analyzeTweet(tweet, text);
    });
  }

  function extractTweetText(tweet) {
    return Array.from(tweet.querySelectorAll(selectors.tweetText))
      .map((node) => node.innerText.trim())
      .filter(Boolean)
      .join("\n")
      .trim();
  }

  function extractTweetId(tweet) {
    const link = tweet.querySelector(selectors.tweetLink);
    const href = link?.getAttribute("href") || "";
    const match = href.match(/\/status\/(\d+)/);
    return match ? match[1] : null;
  }

  function getCacheKey(tweetId, text) {
    return `${cachePrefix}${tweetId || hashText(text)}`;
  }

  function hashText(text) {
    let hash = 0;
    for (let index = 0; index < text.length; index += 1) {
      hash = (hash << 5) - hash + text.charCodeAt(index);
      hash |= 0;
    }
    return Math.abs(hash).toString(36);
  }

  async function analyzeTweet(tweet, text) {
    const tweetId = extractTweetId(tweet);
    const cacheKey = getCacheKey(tweetId, text);
    const cached = readCache(cacheKey);

    if (cached) {
      renderResult(tweet, cached);
      return;
    }

    renderPending(tweet);

    try {
      const result = await requestAnalysis({
        text,
        tweet_id: tweetId,
        source: "x"
      });
      writeCache(cacheKey, result);
      renderResult(tweet, result);
    } catch (error) {
      renderError(tweet, error);
      console.warn("NewsLens analysis failed", error);
    }
  }

  function requestAnalysis(payload) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        {
          type: "NEWSLENS_ANALYZE",
          payload
        },
        (response) => {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
            return;
          }

          if (!response?.ok) {
            reject(new Error(response?.error || "Analysis request failed"));
            return;
          }

          resolve(response.data);
        }
      );
    });
  }

  function readCache(cacheKey) {
    try {
      const rawValue = window.sessionStorage.getItem(cacheKey);
      return rawValue ? JSON.parse(rawValue) : null;
    } catch {
      return null;
    }
  }

  function writeCache(cacheKey, value) {
    try {
      window.sessionStorage.setItem(cacheKey, JSON.stringify(value));
    } catch {
      // Session storage can be unavailable in strict browser modes.
    }
  }

  function ensurePanel(tweet) {
    let panel = tweet.querySelector(`.${panelClass}`);
    if (!panel) {
      panel = document.createElement("div");
      panel.className = panelClass;
    }

    const anchor = getPanelAnchor(tweet);
    if (anchor && panel.parentElement !== anchor.parentElement) {
      anchor.insertAdjacentElement("afterend", panel);
    } else if (!panel.parentElement) {
      tweet.append(panel);
    }

    panel.replaceChildren();
    return panel;
  }

  function getPanelAnchor(tweet) {
    const textBlock = tweet.querySelector(selectors.tweetText);
    return textBlock || null;
  }

  function renderPending(tweet) {
    tweet.classList.add("newslens-tweet");
    const panel = ensurePanel(tweet);
    panel.classList.add("newslens-panel--pending");
    panel.append(createBadge("Checking", "neutral"));
  }

  function renderError(tweet, error) {
    tweet.classList.remove("newslens-is-blurred");
    const panel = ensurePanel(tweet);
    panel.className = `${panelClass} newslens-panel--error`;
    panel.append(createBadge("NewsLens offline", "neutral"));

    const message = document.createElement("span");
    message.className = "newslens-explanation";
    message.textContent = error?.message || "Backend request failed.";
    panel.append(message);
  }

  function renderResult(tweet, result) {
    tweet.classList.add("newslens-tweet");
    tweet.classList.toggle("newslens-is-blurred", Boolean(result.should_blur));

    const panel = ensurePanel(tweet);
    panel.className = panelClass;

    const severity = getSeverity(result);
    panel.append(createBadge(getLabelText(result), severity));

    if (typeof result.bias_score === "number") {
      const score = document.createElement("span");
      score.className = "newslens-score";
      score.textContent = `Score ${result.bias_score.toFixed(2)}`;
      panel.append(score);
    }

    if (result.explanation) {
      const explanation = document.createElement("span");
      explanation.className = "newslens-explanation";
      explanation.textContent = result.explanation;
      panel.append(explanation);
    }

    if (result.should_blur) {
      const revealButton = document.createElement("button");
      revealButton.className = "newslens-reveal";
      revealButton.type = "button";
      revealButton.textContent = "Reveal";
      revealButton.addEventListener("click", () => {
        tweet.classList.remove("newslens-is-blurred");
        revealButton.remove();
      });
      panel.append(revealButton);
    }
  }

  function createBadge(text, severity) {
    const badge = document.createElement("span");
    badge.className = `newslens-badge newslens-badge--${severity}`;
    badge.textContent = text;
    return badge;
  }

  function getSeverity(result) {
    if (result.should_blur) {
      return "high";
    }
    if (result.label === "review_framing") {
      return "medium";
    }
    return "neutral";
  }

  function getLabelText(result) {
    switch (result.label) {
      case "possible_framing_bias":
        return "Possible framing bias";
      case "review_framing":
        return "Review framing";
      case "low_framing_signal":
        return "Low framing signal";
      case "not_checkable":
        return "Not checkable";
      default:
        return "NewsLens checked";
    }
  }

  const observer = new MutationObserver(scheduleScan);

  if (document.body) {
    observer.observe(document.body, { childList: true, subtree: true });
    scanTweets();
  } else {
    window.addEventListener("DOMContentLoaded", () => {
      observer.observe(document.body, { childList: true, subtree: true });
      scanTweets();
    });
  }
})();
