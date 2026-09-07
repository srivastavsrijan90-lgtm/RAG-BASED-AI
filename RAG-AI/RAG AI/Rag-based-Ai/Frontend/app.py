import re
import streamlit as st
import sys
import os
import html

from io import BytesIO
from docx import Document

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, BASE_DIR)


# ============================================================
# IMPORT RAG
# ============================================================

from process_incoming import run_rag


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Video RAG AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CLEAN AI RESPONSE
# ============================================================

def clean_ai_response(response):
    """
    Advanced AI response cleaner.

    Features:
    - HTML tags remove
    - HTML entities decode
    - Markdown code fences remove
    - Duplicate lines remove
    - Duplicate paragraphs remove
    - Duplicate sentences remove
    - Repeated phrases remove
    - Topic / Answer formatting improve
    - Bullets preserve
    - Line breaks preserve
    - Video / Timestamp / Duration metadata preserve
    """

    if response is None:
        return ""

    response = str(response).strip()

    if not response:
        return ""

    # ========================================================
    # 1. NORMALIZE NEWLINES
    # ========================================================

    response = response.replace("\r\n", "\n")
    response = response.replace("\r", "\n")

    # ========================================================
    # 2. REMOVE MARKDOWN CODE FENCES
    # ========================================================

    response = re.sub(
        r"```(?:html|HTML|python|Python|text|Text|markdown|Markdown)?",
        "",
        response
    )

    response = response.replace(
        "```",
        ""
    )

    # ========================================================
    # 3. REMOVE HTML TAGS
    # ========================================================

    response = re.sub(
        r"<[^>]+>",
        "",
        response
    )

    # ========================================================
    # 4. DECODE HTML ENTITIES
    # ========================================================

    response = html.unescape(
        response
    )

    # ========================================================
    # 5. NORMALIZE SPECIAL SPACES
    # ========================================================

    response = response.replace(
        "\u00a0",
        " "
    )

    response = response.replace(
        "\u200b",
        ""
    )

    response = response.replace(
        "\ufeff",
        ""
    )

    # ========================================================
    # 6. REMOVE EXCESSIVE SPACES
    # ========================================================

    lines = response.split("\n")

    normalized_lines = []

    for line in lines:

        line = re.sub(
            r"[ \t]+",
            " ",
            line
        ).strip()

        normalized_lines.append(
            line
        )

    response = "\n".join(
        normalized_lines
    )

    # ========================================================
    # 7. DUPLICATE NORMALIZATION FUNCTION
    # ========================================================

    def normalize_text(text):

        text = text.lower()

        # Remove markdown bullets
        text = re.sub(
            r"^[\-\*\•\▪\◦]+\s*",
            "",
            text
        )

        # Remove numbering
        text = re.sub(
            r"^\d+[\.\)]\s*",
            "",
            text
        )

        # Remove markdown formatting
        text = text.replace(
            "**",
            ""
        )

        text = text.replace(
            "__",
            ""
        )

        text = text.replace(
            "`",
            ""
        )

        # Normalize punctuation
        text = re.sub(
            r"[^\w\s]",
            "",
            text
        )

        # Normalize spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ========================================================
    # 8. REMOVE EXACT DUPLICATE LINES
    # ========================================================

    lines = response.split("\n")

    cleaned_lines = []

    seen_lines = set()

    for line in lines:

        stripped = line.strip()

        if not stripped:

            if (
                cleaned_lines
                and cleaned_lines[-1] != ""
            ):
                cleaned_lines.append("")

            continue

        normalized = normalize_text(
            stripped
        )

        if not normalized:
            continue

        if normalized in seen_lines:
            continue

        seen_lines.add(
            normalized
        )

        cleaned_lines.append(
            stripped
        )

    response = "\n".join(
        cleaned_lines
    )

    # ========================================================
    # 9. REMOVE DUPLICATE PARAGRAPHS
    # ========================================================

    paragraphs = re.split(
        r"\n\s*\n",
        response
    )

    cleaned_paragraphs = []

    seen_paragraphs = set()

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        normalized = normalize_text(
            paragraph
        )

        if not normalized:
            continue

        if normalized in seen_paragraphs:
            continue

        seen_paragraphs.add(
            normalized
        )

        cleaned_paragraphs.append(
            paragraph
        )

    response = "\n\n".join(
        cleaned_paragraphs
    )

    # ========================================================
    # 10. REMOVE DUPLICATE SENTENCES
    # ========================================================

    final_blocks = []

    for block in response.split("\n\n"):

        block = block.strip()

        if not block:
            continue

        # Do not break metadata lines
        if re.match(
            r"^(Video|Timestamp|Duration|Topic|Answer|What is taught)\s*:",
            block,
            re.IGNORECASE
        ):
            final_blocks.append(
                block
            )
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            block
        )

        cleaned_sentences = []

        seen_sentences = set()

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            normalized = normalize_text(
                sentence
            )

            if not normalized:
                continue

            if normalized in seen_sentences:
                continue

            seen_sentences.add(
                normalized
            )

            cleaned_sentences.append(
                sentence
            )

        if cleaned_sentences:

            final_blocks.append(
                " ".join(
                    cleaned_sentences
                )
            )

    response = "\n\n".join(
        final_blocks
    )

    # ========================================================
    # 11. REMOVE REPEATED PHRASES
    # ========================================================

    words = response.split()

    if len(words) >= 20:

        result = []

        i = 0

        while i < len(words):

            found_duplicate = False

            # Check repeated phrase sizes
            for size in range(
                min(12, (len(words) - i) // 2),
                4,
                -1
            ):

                first = words[
                    i:i + size
                ]

                second = words[
                    i + size:
                    i + (size * 2)
                ]

                first_normalized = [
                    re.sub(
                        r"[^\w]",
                        "",
                        word.lower()
                    )
                    for word in first
                ]

                second_normalized = [
                    re.sub(
                        r"[^\w]",
                        "",
                        word.lower()
                    )
                    for word in second
                ]

                if (
                    first_normalized
                    and first_normalized == second_normalized
                ):

                    result.extend(
                        first
                    )

                    i += size * 2

                    found_duplicate = True

                    break

            if found_duplicate:
                continue

            result.append(
                words[i]
            )

            i += 1

        response = " ".join(
            result
        )

    # ========================================================
    # 12. RESTORE IMPORTANT LINE BREAKS
    # ========================================================

    # Metadata ko separate lines mein rakho
    metadata_patterns = [
        r"\s+(Topic:)",
        r"\s+(Answer:)",
        r"\s+(Video:)",
        r"\s+(Timestamp:)",
        r"\s+(Duration:)",
        r"\s+(What is taught:)"
    ]

    for pattern in metadata_patterns:

        response = re.sub(
            pattern,
            r"\n\n\1",
            response,
            flags=re.IGNORECASE
        )

    # ========================================================
    # 13. FORMAT BULLET POINTS
    # ========================================================

    response = re.sub(
        r"\s+([•▪◦])\s*",
        r"\n\1 ",
        response
    )

    response = re.sub(
        r"\s+(-)\s+",
        r"\n- ",
        response
    )

    response = re.sub(
        r"\s+(\d+[\.\)])\s+",
        r"\n\1 ",
        response
    )

    # ========================================================
    # 14. FIX MULTIPLE BLANK LINES
    # ========================================================

    response = re.sub(
        r"\n[ \t]*\n[ \t]*\n+",
        "\n\n",
        response
    )

    # ========================================================
    # 15. REMOVE DUPLICATE METADATA
    # ========================================================

    metadata_seen = set()

    final_lines = []

    for line in response.split("\n"):

        line = line.strip()

        if not line:
            if (
                final_lines
                and final_lines[-1] != ""
            ):
                final_lines.append("")

            continue

        metadata_match = re.match(
            r"^(Topic|Answer|Video|Timestamp|Duration|What is taught)\s*:\s*(.*)$",
            line,
            re.IGNORECASE
        )

        if metadata_match:

            key = metadata_match.group(1).lower()

            value = normalize_text(
                metadata_match.group(2)
            )

            metadata_key = (
                key,
                value
            )

            if metadata_key in metadata_seen:
                continue

            metadata_seen.add(
                metadata_key
            )

        final_lines.append(
            line
        )

    response = "\n".join(
        final_lines
    )

    # ========================================================
    # 16. FINAL WHITESPACE CLEANUP
    # ========================================================

    response = re.sub(
        r"[ \t]+",
        " ",
        response
    )

    response = re.sub(
        r"\n{3,}",
        "\n\n",
        response
    )

    return response.strip()
    
    # --------------------------------------------------------
    # REMOVE DUPLICATE CONSECUTIVE LINES
    # --------------------------------------------------------

    lines = response.splitlines()

    cleaned_lines = []
    previous_line = None

    for line in lines:

        line = line.strip()

        normalized = re.sub(
            r"\s+",
            " ",
            line
        ).lower()

        # Keep only one blank line
        if not normalized:

            if (
                cleaned_lines
                and cleaned_lines[-1] != ""
            ):
                cleaned_lines.append("")

            continue

        # Remove immediately repeated line
        if normalized == previous_line:
            continue

        cleaned_lines.append(
            line
        )

        previous_line = normalized

    response = "\n".join(
        cleaned_lines
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE SENTENCES
    # --------------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        response
    )

    seen_sentences = set()
    cleaned_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        normalized = re.sub(
            r"\s+",
            " ",
            sentence
        ).lower().strip()

        if normalized in seen_sentences:
            continue

        seen_sentences.add(
            normalized
        )

        cleaned_sentences.append(
            sentence
        )

    response = " ".join(
        cleaned_sentences
    )

    # --------------------------------------------------------
    # REMOVE IMMEDIATELY REPEATED LONG PHRASES
    # --------------------------------------------------------

    words = response.split()

    if len(words) > 30:

        final_words = []

        i = 0

        while i < len(words):

            if i + 16 <= len(words):

                first = words[
                    i:i + 8
                ]

                second = words[
                    i + 8:i + 16
                ]

                first_normalized = [
                    re.sub(
                        r"[^\w]",
                        "",
                        word.lower()
                    )
                    for word in first
                ]

                second_normalized = [
                    re.sub(
                        r"[^\w]",
                        "",
                        word.lower()
                    )
                    for word in second
                ]

                if (
                    first_normalized
                    == second_normalized
                ):

                    final_words.extend(
                        first
                    )

                    i += 16

                    continue

            final_words.append(
                words[i]
            )

            i += 1

        response = " ".join(
            final_words
        )

    # --------------------------------------------------------
    # FINAL CLEANUP
    # --------------------------------------------------------

    response = re.sub(
        r"[ ]{2,}",
        " ",
        response
    )

    return response.strip()


# ============================================================
# EXTRACT VIDEO INFORMATION
# ============================================================

def extract_video_info(response):

    response = clean_ai_response(
        response
    )

    # --------------------------------------------------------
    # VIDEO NAME
    # --------------------------------------------------------

    video_match = re.search(
        r"Video:\s*(.+?)(?:\n|$)",
        response,
        re.IGNORECASE
    )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp_match = re.search(
        r"Timestamp:\s*"
        r"(\d{1,2}:\d{2}(?:\.\d{1,2})?)"
        r"\s*-\s*"
        r"(\d{1,2}:\d{2}(?:\.\d{1,2})?)",
        response,
        re.IGNORECASE
    )

    if not video_match:

        return (
            None,
            None,
            None
        )

    video_name = (
        video_match
        .group(1)
        .strip()
    )

    if timestamp_match:

        start_time = (
            timestamp_match
            .group(1)
            .strip()
        )

        end_time = (
            timestamp_match
            .group(2)
            .strip()
        )

    else:

        start_time = None
        end_time = None

    return (
        video_name,
        start_time,
        end_time
    )


# ============================================================
# TIMESTAMP TO SECONDS
# ============================================================

def timestamp_to_seconds(timestamp):

    if not timestamp:
        return 0

    try:

        parts = timestamp.split(":")

        if len(parts) != 2:
            return 0

        minutes = int(
            parts[0]
        )

        seconds = float(
            parts[1]
        )

        return int(
            minutes * 60
            + seconds
        )

    except Exception:

        return 0


# ============================================================
# FIND VIDEO FILE
# ============================================================

def find_video_file(video_name):

    """
    RAG ke video title ko Videos folder
    ke actual MP4 filename se match karta hai.
    """

    if not video_name:
        return None

    video_dir = os.path.join(
        BASE_DIR,
        "Videos"
    )

    if not os.path.exists(
        video_dir
    ):
        return None

    # --------------------------------------------------------
    # CLEAN VIDEO NAME
    # --------------------------------------------------------

    video_name = video_name.strip()

    video_name = video_name.strip(
        '"'
    )

    video_name = video_name.strip(
        "'"
    )

    if video_name.lower().endswith(
        ".mp4"
    ):

        search_name = (
            video_name[:-4]
        )

    else:

        search_name = video_name

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    def normalize(text):

        text = text.lower()

        text = text.replace(
            ".mp4",
            ""
        )

        text = re.sub(
            r"[^a-z0-9]+",
            " ",
            text
        )

        return " ".join(
            text.split()
        )

    target = normalize(
        search_name
    )

    if not target:
        return None

    # --------------------------------------------------------
    # GET MP4 FILES
    # --------------------------------------------------------

    video_files = [
        filename
        for filename in os.listdir(
            video_dir
        )
        if filename.lower().endswith(
            ".mp4"
        )
    ]

    if not video_files:
        return None

    # --------------------------------------------------------
    # EXACT MATCH
    # --------------------------------------------------------

    for filename in video_files:

        if (
            normalize(filename)
            == target
        ):

            return os.path.join(
                video_dir,
                filename
            )

    # --------------------------------------------------------
    # TARGET INSIDE FILENAME
    # --------------------------------------------------------

    for filename in video_files:

        normalized_filename = normalize(
            filename
        )

        if (
            target
            in normalized_filename
        ):

            return os.path.join(
                video_dir,
                filename
            )

    # --------------------------------------------------------
    # FILENAME INSIDE TARGET
    # --------------------------------------------------------

    for filename in video_files:

        normalized_filename = normalize(
            filename
        )

        if (
            normalized_filename
            in target
        ):

            return os.path.join(
                video_dir,
                filename
            )

    # --------------------------------------------------------
    # COMMON WORD MATCH
    # --------------------------------------------------------

    target_words = set(
        target.split()
    )

    best_file = None
    best_score = 0

    for filename in video_files:

        normalized_filename = normalize(
            filename
        )

        file_words = set(
            normalized_filename.split()
        )

        common_words = (
            target_words
            & file_words
        )

        score = len(
            common_words
        )

        if score > best_score:

            best_score = score

            best_file = filename

    if (
        best_file
        and best_score >= 2
    ):

        return os.path.join(
            video_dir,
            best_file
        )

    return None


# ============================================================
# CREATE WORD DOCUMENT
# ============================================================

def create_answer_document(
    question,
    answer,
    video_name=None,
    start_time=None,
    end_time=None
):

    doc = Document()

    doc.add_heading(
        "🎓 Video RAG AI",
        level=0
    )

    doc.add_paragraph(
        "Course Video Search Result"
    )

    # Question
    doc.add_heading(
        "Question",
        level=1
    )

    doc.add_paragraph(
        str(question)
    )

    # Answer
    doc.add_heading(
        "AI Answer",
        level=1
    )

    doc.add_paragraph(
        clean_ai_response(
            answer
        )
    )

    # Video
    if video_name:

        doc.add_heading(
            "Recommended Course Video",
            level=1
        )

        doc.add_paragraph(
            f"Video: {video_name}"
        )

        if start_time:

            timestamp = start_time

            if end_time:

                timestamp += (
                    f" - {end_time}"
                )

            doc.add_paragraph(
                f"Timestamp: {timestamp}"
            )

    file = BytesIO()

    doc.save(
        file
    )

    file.seek(0)

    return file


# ============================================================
# CREATE PDF DOCUMENT
# ============================================================

def create_answer_pdf(
    question,
    answer,
    video_name=None,
    start_time=None,
    end_time=None
):

    file = BytesIO()

    pdf = SimpleDocTemplate(
        file,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles[
        "Title"
    ]

    title_style.alignment = (
        TA_CENTER
    )

    story = []

    # Title
    story.append(
        Paragraph(
            "Video RAG AI",
            title_style
        )
    )

    story.append(
        Spacer(1, 20)
    )

    # Question
    story.append(
        Paragraph(
            "Question",
            styles["Heading2"]
        )
    )

    question_text = html.escape(
        str(question)
    )

    story.append(
        Paragraph(
            question_text,
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # Answer
    story.append(
        Paragraph(
            "AI Answer",
            styles["Heading2"]
        )
    )

    answer_text = clean_ai_response(
        answer
    )

    answer_text = html.escape(
        answer_text
    )

    answer_text = answer_text.replace(
        "\n",
        "<br/>"
    )

    story.append(
        Paragraph(
            answer_text,
            styles["BodyText"]
        )
    )

    # Video
    if video_name:

        story.append(
            Spacer(1, 15)
        )

        story.append(
            Paragraph(
                "Recommended Course Video",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                "Video: "
                + html.escape(
                    str(video_name)
                ),
                styles["BodyText"]
            )
        )

        if start_time:

            timestamp = start_time

            if end_time:

                timestamp += (
                    f" - {end_time}"
                )

            story.append(
                Paragraph(
                    "Timestamp: "
                    + html.escape(
                        timestamp
                    ),
                    styles["BodyText"]
                )
            )

    pdf.build(
        story
    )

    file.seek(0)

    return file


# ============================================================
# LOAD CSS
# ============================================================

css_path = os.path.join(
    os.path.dirname(__file__),
    "style.css"
)

if os.path.exists(
    css_path
):

    with open(
        css_path,
        "r",
        encoding="utf-8"
    ) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns(
    [5, 1],
    vertical_alignment="center"
)

with header_col1:

    st.markdown(
        """
        <div class="top-title">
            🎓 Video + Document RAG AI
        </div>

        <div class="top-subtitle">
            Intelligent course video search
        </div>
        """,
        unsafe_allow_html=True
    )

with header_col2:

    st.markdown(
        """
        <div class="online-status">
            🟢 AI Online
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# ============================================================
# WELCOME
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="examples-title">
            TRY ASKING
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message(
            "user",
            avatar="👤"
        ):

            st.markdown(
                message["content"]
            )

    else:

        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            st.markdown(
                """
                <div class="ai-label">
                    🤖 AI RESPONSE
                </div>
                """,
                unsafe_allow_html=True
            )

            clean_response = clean_ai_response(
                message["content"]
            )

            st.markdown(
                clean_response
            )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask about your course videos..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        st.markdown(
            """
            <div class="ai-label">
                🤖 AI RESPONSE
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.spinner(
            "🔎 Searching your course videos..."
        ):

            try:

                # ============================================
                # RUN RAG
                # ============================================

                raw_response = st.write_stream(
                    run_rag(question)
                )

                # ============================================
                # CLEAN RESPONSE
                # ============================================

                response = clean_ai_response(
                    raw_response
                )

                # ============================================
                # SHOW CLEAN ANSWER
                # ============================================

                st.markdown(
                    response
                )

                # ============================================
                # EXTRACT VIDEO
                # ============================================

                (
                    video_name,
                    start_time,
                    end_time
                ) = extract_video_info(
                    response
                )

                actual_video_name = (
                    video_name
                )

                video_path = None

                # ============================================
                # FIND VIDEO FILE
                # ============================================

                if video_name:

                    video_path = find_video_file(
                        video_name
                    )

                # ============================================
                # VIDEO FOUND
                # ============================================

                if video_path:

                    actual_video_name = (
                        os.path.basename(
                            video_path
                        )
                    )

                    st.markdown(
                        "### 🎥 Recommended Course Video"
                    )

                    st.write(
                        f"🎬 **Video:** "
                        f"{actual_video_name}"
                    )

                    # ----------------------------------------
                    # TIMESTAMP
                    # ----------------------------------------

                    if start_time:

                        timestamp_text = (
                            start_time
                        )

                        if end_time:

                            timestamp_text += (
                                f" - {end_time}"
                            )

                        st.write(
                            f"⏱ **Timestamp:** "
                            f"`{timestamp_text}`"
                        )

                        seconds = (
                            timestamp_to_seconds(
                                start_time
                            )
                        )

                    else:

                        seconds = 0

                    # ----------------------------------------
                    # VIDEO PLAYER
                    # ----------------------------------------

                    st.video(
                        video_path,
                        start_time=seconds
                    )

                    st.success(
                        "▶ Video loaded successfully"
                    )

                # ============================================
                # VIDEO NAME BUT FILE NOT FOUND
                # ============================================

                elif video_name:

                    st.warning(
                        "⚠️ Video file not found.\n\n"
                        f"RAG returned: {video_name}"
                    )

                # ============================================
                # NO VIDEO INFO
                # ============================================

                else:

                    st.info(
                        "🎥 No video information was found "
                        "in the RAG response."
                    )

                # ============================================
                # DOWNLOAD NOTES
                # ============================================

                st.markdown(
                    "### 📚 Download Notes"
                )

                # ============================================
                # WORD
                # ============================================

                document_file = (
                    create_answer_document(
                        question=question,
                        answer=response,
                        video_name=actual_video_name,
                        start_time=start_time,
                        end_time=end_time
                    )
                )

                st.download_button(
                    label="📄 Download Notes as Word",
                    data=document_file,
                    file_name="video_rag_notes.docx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    use_container_width=True,
                    key=f"download_word_{len(st.session_state.messages)}"
                )

                # ============================================
                # PDF
                # ============================================

                pdf_file = (
                    create_answer_pdf(
                        question=question,
                        answer=response,
                        video_name=actual_video_name,
                        start_time=start_time,
                        end_time=end_time
                    )
                )

                st.download_button(
                    label="📕 Download Notes as PDF",
                    data=pdf_file,
                    file_name="video_rag_notes.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"download_pdf_{len(st.session_state.messages)}"
                )

            # =================================================
            # ERROR
            # =================================================

            except Exception as e:

                response = (
                    "❌ Something went wrong.\n\n"
                    f"{str(e)}"
                )

                st.error(
                    response
                )

    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

