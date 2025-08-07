import pypandoc
from pathlib import Path

def generate_markdown_report(init_prompt, agents_history, summary_history, total_rounds):
    report_lines = ["# SocratisAI report\n"]
    report_lines.append(f"{init_prompt}")

    for round_idx in range(total_rounds):
        report_lines.append(f"\n---\n\n## Kolo {round_idx + 1}\n")

        # Výstupy jednotlivých agentů
        report_lines.append("### Příspěvky agentů:\n")
        for agent, messages in agents_history.items():
            if round_idx < len(messages):
                content = messages[round_idx].strip()
                report_lines.append(f"{content}")

        # Shrnutí moderátorem
        report_lines.append("\n### Shrnutí moderátorem:\n")
        if round_idx < len(summary_history):
            summary = summary_history[round_idx].content.strip()
            report_lines.append(summary)

        # Obrázek: podobnost mezi agenty
        img_path = f"Interagentni podobnost kolo {round_idx+1}.png"
        if Path(img_path).exists():
            report_lines.append(f"\n![Kosinová podobnost mezi agenty – Kolo {round_idx + 1}]({img_path})")

    # Celkový vývoj podobnosti
    img_path = "Vyvoj podobnosti nazoru agentu.png"
    if Path(img_path).exists():
        report_lines.append("\n---\n\n## Vývoj stability odpovědí agentů")
        report_lines.append(f"\n![Vývoj stability]({img_path})")

    # Uložení do souboru
    md_report = "\n\n".join(report_lines)
    output_path = Path("report.md")
    output_path.write_text(md_report, encoding="utf-8")
    return output_path.name


def convert_markdown_to_pdf(md_file_path, output_pdf_path):
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
            extra_args=['--standalone']
        )
        print(f"✅ PDF report byl úspěšně vytvořen: {output_pdf_path}")
    except Exception as e:
        print(f"❌ Chyba při převodu do PDF: {e}")