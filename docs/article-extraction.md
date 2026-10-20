# Backend article extraction

The FastAPI backend exposes `POST /articles/extract` for extracting a bounded
article excerpt from a public HTTP(S) URL. The extension should send the URL to
the backend; it should not fetch publisher pages or hold scraping credentials.

Example request:

```json
{
  "url": "https://example.com/news/story",
  "max_words": 1500
}
```

`max_words` defaults to 1,500 and is limited to 50–5,000. A successful response
includes the final/canonical URL, title and other available metadata, the text
excerpt, its word count, the full extracted word count, and whether it was
truncated. The article body is returned to the caller and is not written to a
database or cache by this endpoint.

The fetcher follows a small number of redirects, validates every redirect
destination, rejects non-public network addresses, limits response size and
duration, and accepts only HTML or plain text pages. Extraction uses
Trafilatura. Pages that block requests, require client-side rendering, or
contain no extractable article text can fail; the endpoint returns an HTTP error
in those cases. This endpoint does not bypass publisher access controls or
replace the separate news search needed to find coverage from multiple outlets.
