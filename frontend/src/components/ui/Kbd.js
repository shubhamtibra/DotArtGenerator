export function Kbd({ children }) {
  return (
    <kbd className="inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded border border-zinc-700 bg-zinc-800/80 px-1 font-sans text-[11px] font-medium text-zinc-300">
      {children}
    </kbd>
  );
}
