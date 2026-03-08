# Copilot Instructions for DotArtGenerator

## Project Overview
- **DotArtGenerator** is a full-stack project for generating dot art from images.
- The backend is a Django app (`backend/`) with a sub-app (`app/`) for core logic and endpoints.
- The frontend (`frontend/`) is a React app using Tailwind CSS for UI.
- Artifacts (outputs) are stored in the `outputs/` directory.

## Key Components
- **Backend**: Django project in `backend/` with main app in `backend/app/`.
  - `HEDToAscii.py`: Core logic for converting images to ASCII/dot art.
  - `views.py`, `viewsHED.py`: API endpoints for image processing and serving results.
  - `models.py`: Django models for storing relevant data.
- **Frontend**: React app in `frontend/`.
  - `src/components/`: Contains UI components for uploading, resizing, and viewing dot art.
  - `public/`: Static assets and HTML entry point.

## Developer Workflows
- **Backend**:
  - Run server: `python backend/manage.py runserver`
  - Apply migrations: `python backend/manage.py migrate`
  - Run tests: `python backend/manage.py test`
- **Frontend**:
  - Install deps: `cd frontend && npm install`
  - Start dev server: `npm start`
  - Run tests: `npm test`

## Project-Specific Patterns
- Image processing logic is centralized in `backend/app/HEDToAscii.py`.
- API endpoints for image upload and processing are in `views.py` and `viewsHED.py`.
- Frontend communicates with backend via REST API (see `frontend/src/components/ImageUploader.js`).
- Outputs are saved to `outputs/` and may be referenced by both backend and frontend.

## Conventions
- Use Django's app structure for backend logic; keep processing code in `HEDToAscii.py`.
- Place new React components in `frontend/src/components/`.
- Static files (images, results) should be stored in `outputs/` or `frontend/public/` as appropriate.
- Prefer functional React components and hooks.

## Integration Points
- Backend and frontend are decoupled; communicate via HTTP API.
- Update CORS settings in `backend/backend/settings.py` if changing frontend/backend ports.

## Examples
- To add a new image processing feature, implement logic in `HEDToAscii.py`, expose via a new view in `viewsHED.py`, and update frontend components to call the new endpoint.

---

For questions or unclear patterns, review `README.md` files or ask for clarification.
