import { Logo } from "./Logo";
import { Button } from "./ui/Button";
import { ImagePlusIcon } from "./ui/icons";

export function Header({ hasSource, onNewImage }) {
  return (
    <header className="sticky top-0 z-30 h-12 shrink-0 border-b border-zinc-800/80 bg-zinc-950">
      <div className="flex h-full items-center justify-between gap-4 px-4 lg:px-5">
        <div className="flex min-w-0 items-center gap-2.5">
          <Logo className="h-5 w-5 shrink-0 text-violet-400" />
          <span className="truncate text-sm font-semibold tracking-tight text-zinc-100">Dot Art Generator</span>
        </div>
        {hasSource && (
          <Button size="sm" icon={ImagePlusIcon} onClick={onNewImage}>
            New image
          </Button>
        )}
      </div>
    </header>
  );
}
