import "@testing-library/jest-dom";

const consoleError = console.error;

beforeEach(() => {
  // @testing-library/react 13 calls react-dom/test-utils' act, which React 18.3
  // flags as deprecated. That warning is about the library, not this code;
  // everything else is passed through. (Installed per test: CRA resets mocks.)
  jest.spyOn(console, "error").mockImplementation((message, ...rest) => {
    if (typeof message === "string" && message.includes("ReactDOMTestUtils.act` is deprecated")) return;
    consoleError(message, ...rest);
  });

  // jsdom has no canvas or object URLs. Model a browser without canvas support
  // (getContext → null) so paths that must degrade gracefully are exercised,
  // and stub object URLs so previews can be created.
  HTMLCanvasElement.prototype.getContext = () => null;
  URL.createObjectURL = jest.fn(() => "blob:mock-preview");
  URL.revokeObjectURL = jest.fn();
  window.localStorage.clear();
});
