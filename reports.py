import pypandoc
from pathlib import Path
import re

def sanitize_text(text):
    """Odstranění znaků, které mohou rozbít LaTeX při převodu na PDF."""
    text = re.sub(r'[\\{}]', '', text)
    text = text.replace('_', '\\_')  # podtržítka escapujeme
    
    # Odstranění emojis (přes Unicode rozsahy)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # smajlíci
        "\U0001F300-\U0001F5FF"  # symboly & piktogramy
        "\U0001F680-\U0001F6FF"  # doprava a mapy
        "\U0001F1E0-\U0001F1FF"  # vlajky
        "\U00002700-\U000027BF"  # různé symboly
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # Escapování LaTeX speciálních znaků
    specials = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    for char, replacement in specials.items():
        text = text.replace(char, replacement)

    return text.strip()


def generate_markdown_report(init_prompt, agents_history, summary_history, total_rounds):
    def insert_image_nofloat(path: str) -> str:
        return f"\\noindent\\includegraphics[width=\\textwidth]{{{path}}}"

    report_lines = ["# SocratisAI report\n"]
    report_lines.append(sanitize_text(init_prompt))

    for round_idx in range(total_rounds):
        report_lines.append(f"\n---\n\n## Kolo {round_idx + 1}\n")

        # Výstupy jednotlivých agentů
        report_lines.append("### Příspěvky agentů:\n")
        for agent, messages in agents_history.items():
            if round_idx < len(messages):
                content = sanitize_text(messages[round_idx])
                report_lines.append(f"{content}")

        # Shrnutí moderátorem
        report_lines.append("\n### Shrnutí moderátorem:\n")
        if round_idx < len(summary_history):
            summary = sanitize_text(summary_history[round_idx].content)
            report_lines.append(summary)

        # Obrázek: podobnost mezi agenty
        img_path = f"Interagentni podobnost kolo {round_idx+1}.png"
        if Path(img_path).exists():
            report_lines.append(insert_image_nofloat(img_path))

    # Celkový vývoj podobnosti
    img_path = "Vyvoj podobnosti nazoru agentu.png"
    if Path(img_path).exists():
        report_lines.append("\n---\n\n## Vývoj stability odpovědí agentů")
        report_lines.append(insert_image_nofloat(img_path))

    md_report = "\n\n".join(report_lines)
    output_path = Path("report.md")
    output_path.write_text(md_report, encoding="utf-8")
    return output_path.name


def ensure_preamble_file():
    preamble_path = Path("preamble.tex")
    if not preamble_path.exists():
        preamble_path.write_text("\\usepackage{graphicx}\n", encoding="utf-8")


def convert_markdown_to_pdf(md_file_path, output_pdf_path):
    ensure_preamble_file()
    try:
        pypandoc.get_pandoc_path()
    except OSError:
        print("🔽 Pandoc není nainstalován. Stahuji ho...")
        pypandoc.download_pandoc()
    
    try:
        pypandoc.convert_text(
            Path(md_file_path).read_text(encoding="utf-8"),
            'pdf',
            format='md',
            outputfile=output_pdf_path,
            extra_args=[
                '--standalone',
                '--pdf-engine=pdflatex',
                '-V', 'geometry:margin=2.5cm',
                '-V', 'documentclass:article',
                '-H', 'preamble.tex'  # 👈 přidej tento řádek
            ]
        )
        print(f"✅ PDF report byl úspěšně vytvořen: {output_pdf_path}")
    except Exception as e:
        print(f"❌ Chyba při převodu do PDF: {e}")