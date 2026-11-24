# Privacy

## Data handling
- **On-device by default.** The mobile app and PWA process images locally. No image, scan, or metadata is uploaded by default.
- **Local storage only.** Scan history is stored locally (SQLite on mobile, IndexedDB in the PWA). Users can delete individual records and use "Delete all data".
- **Optional export is user-initiated.** CSV export is created locally; sharing is initiated by the user via the OS share sheet / browser download.
- **Research viewer API is local/read-only.** The FastAPI viewer serves precomputed plant-split replay data from local disk. It exposes no file-upload endpoints and is intended for local research use only. Do not expose it to the public internet without an explicit security review.

## Permissions
- **Camera/Photos**: used only when the user chooses to capture or pick an image.
- **No INTERNET permission in the mobile release manifest** (verified by inspecting the merged manifest during build; see `build/BUILD_REPORT.md`).

## Not collected
No accounts, no analytics, no ad identifiers, no background uploads, no third-party tracking SDKs are added by this prototype.
