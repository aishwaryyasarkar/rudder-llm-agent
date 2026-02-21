#!/usr/bin/env python3
import os
import re
import argparse

def clean_agent_files(base_dir):
    """
    Walks through base_dir, finds all .txt files that contain your model logs,
    and in-place replaces:
      "Decision:" -> "Decision: <agent>"
      "LLM returned message:" -> "<agent> LLM returned message:"
    """

    # Potentially refine this pattern if you want to filter only certain files (like "gemma" or "llama" in the filename).
    # For now, we apply to any .txt file we find.
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".txt"):
                fullpath = os.path.join(root, fname)

                # We'll read the file and do line-by-line replacements in memory, then overwrite the file.
                try:
                    with open(fullpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except (UnicodeDecodeError, OSError) as e:
                    print(f"[WARNING] Unable to read {fullpath}: {e}")
                    continue

                changed = False
                new_lines = []
                for line in lines:
                    # Replace "Decision:" with "Decision: <agent>"
                    # Do a strict left-anchored replace or a broader search?
                    # If you want to be sure it's at the beginning, use '^Decision:\s*'
                    # Or do a simpler approach:
                    new_line = re.sub(r"^Decision:\s*", "Decision: <agent> ", line)

                    # Then replace "LLM returned message:" with "<agent> LLM returned message:"
                    # Similarly:
                    new_line = re.sub(r"^LLM returned message:\s*", "<agent> LLM returned message: ", new_line)

                    if new_line != line:
                        changed = True
                    new_lines.append(new_line)

                if changed:
                    # Overwrite the file with the updated lines
                    try:
                        with open(fullpath, "w", encoding="utf-8") as f:
                            f.writelines(new_lines)
                        print(f"[INFO] Cleaned {fullpath}.")
                    except OSError as e:
                        print(f"[WARNING] Could not overwrite {fullpath}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean agent logs so that 'Decision:' lines become 'Decision: <agent>' and 'LLM returned message:' becomes '<agent> LLM returned message:'."
    )
    parser.add_argument("--base_dir", help="The base directory to walk, searching for .txt files.")
    args = parser.parse_args()

    clean_agent_files(args.base_dir)

if __name__ == "__main__":
    main()
