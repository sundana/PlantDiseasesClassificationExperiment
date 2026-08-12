"""Export TikZ figures from main.tex to standalone PNG images."""
import subprocess, os, sys

TEX_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TEX_DIR)

PDLATEX = "/mnt/c/Users/USER/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe"
OUT_DIR = "images/exported"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Figure 1: Pipeline Flowchart (TikZ) ---
fig1_tex = r"""\documentclass[border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds}
\begin{document}
\begin{tikzpicture}[
    node distance=0.55cm,
    block/.style={rectangle, draw=black, line width=0.5pt,
                  rounded corners=3pt, minimum width=3.8cm, minimum height=0.75cm,
                  align=center, font=\footnotesize, inner sep=4pt},
    mainblock/.style={block, minimum height=0.8cm, font=\small\bfseries, fill=black!5},
    arr/.style={-{Stealth[length=2.5mm, width=2.5mm]}, line width=0.6pt},
    inter/.style={arr, shorten >=1pt, shorten <=1pt},
    group/.style={draw=black!35, dashed, rounded corners=5pt, inner xsep=14pt, inner ysep=10pt},
]
\node[mainblock] (input) {Input Corn Leaf};
\node[block, below=1.0cm of input] (unet) {U-Net Segmentation};
\draw[inter] (input) -- (unet);
\node[block] (gray) at (unet.south) [below=0.6cm of unet] {Grayscale Conversion};
\draw[arr] (unet) -- (gray);
\begin{scope}[on background layer]
    \node[group, fit=(unet)(gray),
          label={[font=\footnotesize\bfseries, yshift=5pt]north:Preprocessing}] {};
\end{scope}
\node[block, below=1.0cm of gray] (glcm) {GLCM Feature Extraction\\
    \scriptsize ($d = 5$ jarak, $\theta = 4$ sudut, 6 metrik)};
\draw[inter] (gray) -- (glcm);
\node[block] (pca) at (glcm.south) [below=0.6cm of glcm] {PCA Feature Selection\\
    \scriptsize $k \in \{10, 30, 60, 90, 120\}$};
\draw[arr] (glcm) -- (pca);
\begin{scope}[on background layer]
    \node[group, fit=(glcm)(pca),
          label={[font=\footnotesize\bfseries, yshift=5pt]north:Feature Engineering}] {};
\end{scope}
\node[block, below=1.0cm of pca] (rf) {Random Forest Classification};
\draw[inter] (pca) -- (rf);
\node[mainblock, below=1.0cm of rf] (out) {Output: 4 Disease Classes};
\draw[inter] (rf) -- (out);
\node[below=0.15cm of out, font=\footnotesize] {Healthy \textbar{} Common Rust \textbar{} Northern Leaf Blight \textbar{} Gray Leaf Spot};
\end{tikzpicture}
\end{document}
"""

# --- Figure 5: Time Computation Bar Chart (pgfplots) ---
fig5_tex = r"""\documentclass[border=5pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.9}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    symbolic x coords={120, 90, 60, 30, 10},
    xtick=data,
    ymin=0.0,
    ymax=0.3,
    ylabel={Time computation (s)},
    xlabel={Num. of PCA dimensionality of features},
    nodes near coords,
    bar width=1cm,
    width=8cm,
    height=6cm,
    enlarge x limits={abs=0.75cm}
]
\addplot coordinates {(120, 0.11) (90, 0.16) (60, 0.17) (30, 0.16) (10, 0.15)};
\end{axis}
\end{tikzpicture}
\end{document}
"""

figures = [
    ("fig1_pipeline", fig1_tex),
    ("fig5_time", fig5_tex),
]

for name, tex_content in figures:
    tex_path = os.path.join(OUT_DIR, f"{name}.tex")
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    
    with open(tex_path, "w") as f:
        f.write(tex_content)
    
    # Compile twice for proper rendering
    for _ in range(2):
        subprocess.run([PDLATEX, "-interaction=nonstopmode", "-output-directory", OUT_DIR, tex_path],
                       capture_output=True)
    
    # Convert PDF to PNG using Python
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(4.0, 4.0)  # 4x resolution = ~1200 DPI equivalent
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        pix.save(png_path)
        doc.close()
        print(f"OK: {png_path} ({pix.width}x{pix.height})")
    except ImportError:
        print(f"PyMuPDF not installed. PDF saved at {pdf_path}")
        print("Install with: pip install PyMuPDF")

# Clean up aux/log files
for f in os.listdir(OUT_DIR):
    if f.endswith(('.aux', '.log', '.tex', '.pdf')):
        pass  # keep source + pdf for reference
