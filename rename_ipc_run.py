from pathlib import Path
import re
import sys

PATTERN = re.compile(
    r"^IPC_(\d{4})-(\d{2})-(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d+)\.jpg$",
    re.IGNORECASE,
)

def main(folder: str, do_it: bool = False) -> None:
    p = Path(folder)
    files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"])

    frame = 1
    planned = []

    for f in files:
        m = PATTERN.match(f.name)
        if not m:
            continue

        yyyy, mm, dd, HH, MM, SS, frac = m.groups()
        new_name = f"{yyyy}{mm}{dd}T{HH}{MM}{SS}_{frac}__f-{frame:06d}.jpg"
        planned.append((f, f.with_name(new_name)))
        frame += 1

    # Preview
    for old_path, new_path in planned:
        print(f"{old_path.name}  ->  {new_path.name}")

    if not do_it:
        print("\nDry run only. Re-run with --do to actually rename.")
        return

    # Safety: ensure no collisions
    new_names = [np.name for _, np in planned]
    if len(new_names) != len(set(new_names)):
        raise RuntimeError("Name collision detected in planned renames.")

    for old_path, new_path in planned:
        old_path.rename(new_path)  # rename in-place
    print("\nDone.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rename_ipc_run.py <folder> [--do]")
        raise SystemExit(2)
    folder = sys.argv[1]
    do_it = "--do" in sys.argv[2:]
    main(folder, do_it=do_it)
