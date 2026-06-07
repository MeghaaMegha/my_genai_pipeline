import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

def create_project_pdf(filename="pipeline_blueprint.pdf"):
    if os.path.exists(filename):
        os.remove(filename)

    # Establish clean 0.5-inch safety margins
    doc = SimpleDocTemplate(
        filename, 
        pagesize=letter,
        rightMargin=36, leftMargin=36, 
        topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # -------------------------------------------------------------------------
    # PRODUCTION GRAPHICAL COMPONENT TYLES
    # -------------------------------------------------------------------------
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=24, leading=28, 
        textColor='#1A365D', alignment=TA_CENTER, spaceAfter=15
    )
    h1_style = ParagraphStyle(
        'SecHeading', parent=styles['Heading2'], fontSize=15, leading=19, 
        textColor='#1A365D', spaceBefore=22, spaceAfter=10, keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SubHeading', parent=styles['Heading3'], fontSize=11, leading=14, 
        textColor='#2C5282', spaceBefore=18, spaceAfter=10, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['BodyText'], fontSize=9.5, leading=13.5, 
        textColor='#2D3748', spaceAfter=6
    )
    info_box_style = ParagraphStyle(
        'InfoBox', parent=styles['Normal'], fontSize=9, leading=13,
        textColor='#2C5282', backColor='#EBF8FF', borderColor='#BEE3F8',
        borderWidth=1, borderPadding=8, spaceBefore=6, spaceAfter=10, keepWithNext=True
    )
    # Swapped to a wrapped Courier style that keeps the box format but safely wraps text
    code_box_style = ParagraphStyle(
        'WrappedCodeBox', fontName='Courier', fontSize=7.5, leading=10,
        textColor='#1A202C', backColor='#F7FAFC', borderColor='#E2E8F0',
        borderWidth=0.5, borderPadding=8, spaceBefore=6, spaceAfter=16
    )

    story = []
    
    # Document Header
    story.append(Paragraph("<b>GenAI Pipeline Project Blueprint</b>", title_style))
    story.append(Paragraph("<i>Complete Enterprise System Layout, Configurations, and Verified Source Repositories</i>", ParagraphStyle('Sub', alignment=TA_CENTER, fontSize=10, spaceAfter=15)))
    story.append(Spacer(1, 10))

    # Helper function to convert raw lines into safely wrapped paragraph elements
    def format_code_text(raw_text):
        return raw_text.replace('\n', '<br/>').replace(' ', '&nbsp;')

    # =========================================================================
    # SECTION 1: GLOBAL ENVIRONMENT ENGINE
    # =========================================================================
    story.append(Paragraph("<b>1. Environment & Global Envs</b>", h1_style))
    story.append(Spacer(1, 8)) # Explicit separation gap added here
    story.append(Paragraph("<b>Context Commentary:</b> These variables establish cross-platform alignment across your WSL terminal. Registering these overrides forces Python to resolve directory module lookups natively and prevents third-party binary version mismatches within serialization libraries.", info_box_style))
    story.append(Spacer(1, 6))
    
    sys_cmd = (
        "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64\n"
        "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python\n"
        "export PYTHONPATH=\"${PYTHONPATH}:${PWD}\""
    )
    story.append(Paragraph(format_code_text(sys_cmd), code_box_style))

    # =========================================================================
    # SECTION 2: ARCHITECTURAL CONFIGURATION PROFILES
    # =========================================================================
    story.append(Paragraph("<b>2. Configuration Profiles</b>", h1_style))
    
    # 2.1 pyproject.toml
    story.append(Paragraph("<b>2.1 pyproject.toml</b>", h2_style))
    story.append(Spacer(1, 8)) # Explicit separation gap added here
    story.append(Paragraph("<b>Context Commentary:</b> This file explicitly structures the Poetry application deployment manifest. It down-selects dependencies to strict target ranges, tracks isolated packages, disables library packaging mode, and defines development environment boundary groups.", info_box_style))
    story.append(Spacer(1, 6))
    
    content_py = ""
    for p in ["pyproject.toml", os.path.join("..", "pyproject.toml")]:
        if os.path.exists(p):
            with open(p, "r") as f: content_py = f.read()
            break
    if not content_py: content_py = "# File pyproject.toml could not be resolved."
    story.append(Paragraph(format_code_text(content_py), code_box_style))

    # 2.2 logging.conf
    story.append(Paragraph("<b>2.2 logging.conf</b>", h2_style))
    story.append(Spacer(1, 8)) # Explicit separation gap added here
    story.append(Paragraph("<b>Context Commentary:</b> Configures a standard configuration mapping for internal platform logs. It explicitly ensures that critical operational errors stream asynchronously to file handlers to maintain long-term production auditable systems.", info_box_style))
    story.append(Spacer(1, 6))
    
    content_log = ""
    for p in ["logging.conf", "src/logging.conf", os.path.join("..", "logging.conf")]:
        if os.path.exists(p):
            with open(p, "r") as f: content_log = f.read()
            break
    if not content_log: content_log = "[loggers]\nkeys=root\n\n[handlers]\nkeys=fileHandler\n\n[formatters]\nkeys=sampleFormatter\n\n[logger_root]\nlevel=ERROR\nhandlers=fileHandler\n\n[handler_fileHandler]\nclass=FileHandler\nlevel=ERROR\nformatter=sampleFormatter\nargs=('app.log',)\n\n[formatter_sampleFormatter]\nformat=%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    story.append(Paragraph(format_code_text(content_log), code_box_style))

    # =========================================================================
    # SECTION 3: CORE APPLICATION DEPLOYMENT SOURCE CODE
    # =========================================================================
    story.append(Paragraph("<b>3. Core Source Repositories</b>", h1_style))
    
    modules = [
        ("src/utils.py", "3.1 src/utils.py (Logging Utility)", "Exposes a centralized instantiation utility module. This script abstracts logger generation parameters to keep warning outputs completely standardized across separate, decoupled pipeline folders."),
        ("src/data_ingest.py", "3.2 src/data_ingest.py (PySpark Engine)", "Implements your Days 3–7 PySpark cluster dataframe manipulation engine. It performs automated high-throughput string cleaning conversions, normalizes casing rules, strips punctuation, and builds array tokenization arrays."),
        ("src/bert_prep.py", "3.3 src/bert_prep.py (BERT Tokenization)", "Establishes your Days 8–14 Deep Learning preprocessing architecture. It mounts a pre-trained HuggingFace tokenizer dictionary to encode text sequences into numeric tensor arrays matching strict sequence dimension constraints."),
        ("src/train_track.py", "3.4 src/train_track.py (MLflow MLOps)", "Drives modern lifecycle tracking integrations. It spins up an MLflow workspace tracking session loop to index system configurations and loss metrics parameters directly down into an active queryable evaluation matrix.")
    ]
    
    for filepath, subtitle, commentary in modules:
        story.append(Paragraph(f"<b>{subtitle}</b>", h2_style))
        story.append(Spacer(1, 8)) # Explicit separation gap added here
        story.append(Paragraph(f"<b>Context Commentary:</b> {commentary}", info_box_style))
        story.append(Spacer(1, 6))
        
        content = ""
        for p in [filepath, os.path.join("..", filepath)]:
            if os.path.exists(p):
                with open(p, "r") as f: content = f.read()
                break
        if not content: content = f"# File {filepath} missing or unreadable within workspace directory."
        story.append(Paragraph(format_code_text(content), code_box_style))

    # =========================================================================
    # SECTION 4: AUTOMATED TEST VALIDATION ENGINE
    # =========================================================================
    story.append(Paragraph("<b>4. Test Engine Harness</b>", h1_style))
    story.append(Paragraph("<b>4.1 tests/test_pipeline.py</b>", h2_style))
    story.append(Spacer(1, 8)) # Explicit separation gap added here
    story.append(Paragraph("<b>Context Commentary:</b> Ensures continuous integration confidence. This file initializes isolated local mock clusters to execute and assert correctness rules for all logging, PySpark, token arrays, and tensor formats automatically.", info_box_style))
    story.append(Spacer(1, 6))
    
    content_tp = ""
    for p in ["tests/test_pipeline.py", os.path.join("..", "tests/test_pipeline.py")]:
        if os.path.exists(p):
            with open(p, "r") as f: content_tp = f.read()
            break
    if not content_tp: content_tp = "# File tests/test_pipeline.py missing or unreadable within workspace."
    story.append(Paragraph(format_code_text(content_tp), code_box_style))

    # Compile layout
    doc.build(story)
    print(f"[SUCCESS] Blueprint compiled with clean header offsets into: '{filename}'")

if __name__ == "__main__":
    create_project_pdf()
