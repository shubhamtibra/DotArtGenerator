import { useEffect, useState } from "react";
import { readStorage, writeStorage } from "../utils/storage";

/** useState persisted to localStorage; `sanitize` turns stored (untrusted) data into a valid value. */
export function useStoredState(key, sanitize) {
  const [value, setValue] = useState(() => sanitize(readStorage(key)));
  useEffect(() => {
    writeStorage(key, value);
  }, [key, value]);
  return [value, setValue];
}
