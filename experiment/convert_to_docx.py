"""Convert main-submit.tex to DOCX via pandoc, replacing TikZ figures with PNG."""
import subprocess, os

BASE_DIR = "/home/sundana/projects/Corn_Diseases_for_SIET_Conference_IEEE"
TEX_FILE = os.path.join(BASE_DIR, "main-submit.tex")
OUT_DOCX = os.path.join(BASE_DIR, "main-submit.docx")
TEX_CONV = os.path.join(BASE_DIR, "main-submit-conv.tex")

PANDOC = "/mnt/c/Users/USER/anaconda3/Scripts/pandoc.exe"
BIB_FILE = os.path.join(BASE_DIR, "references.bib")

with open(TEX_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False
tikz_count = 0

for line in lines:
    if line.strip().startswith("\\usepackage{tikz}") or \
       line.strip().startswith("\\usetikzlibrary") or \
       line.strip().startswith("\\usepackage{pgfplots}") or \
       line.strip().startswith("\\pgfplotsset"):
        continue

    if "\\begin{tikzpicture}" in line:
        skip = True
        tikz_count += 1
        if tikz_count == 1:
            new_lines.append("    \\includegraphics[width=0.95\\linewidth]{images/exported/fig1_pipeline.png}\n")
        elif tikz_count == 2:
            new_lines.append("    \\includegraphics[width=0.85\\linewidth]{images/exported/fig5_time.png}\n")
        continue

    if skip and "\\end{tikzpicture}" in line:
        skip = False
        continue

    if not skip:
        new_lines.append(line)

content = "".join(new_lines)
content = content.replace("\\textbar{}", "|")

with open(TEX_CONV, "w", encoding="utf-8") as f:
    f.write(content)

# Run pandoc
os.chdir(BASE_DIR)
cmd = [
    PANDOC, TEX_CONV,
    "-o", OUT_DOCX,
    "--from", "latex",
    "--to", "docx",
    "--citeproc",
    "--bibliography", BIB_FILE,
    "--resource-path", f"{BASE_DIR};{BASE_DIR}/images;{BASE_DIR}/images/samples;{BASE_DIR}/images/seg_results;{BASE_DIR}/images/exported;.",
]
print(f"Running pandoc...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

if result.returncode == 0:
    size_kb = os.path.getsize(OUT_DOCX) / 1024
    print(f"SUCCESS: {OUT_DOCX} ({size_kb:.0f} KB)")
else:
    err_lines = result.stderr.strip().split('\n')
    print("STDERR (last 15 lines):")
    for l in err_lines[-15:]:
        print(l[:250])
