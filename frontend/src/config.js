const DEFAULT_API_URL = "http://localhost:8000";

/** Base URL of the generator API, without a trailing slash. */
export const API_URL = (process.env.REACT_APP_API_URL || DEFAULT_API_URL).replace(/\/+$/, "");

/** Where the bundled sample images are served from. */
export const PUBLIC_URL = process.env.PUBLIC_URL || "";
