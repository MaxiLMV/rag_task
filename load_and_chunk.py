import pdfplumber
import spacy

# Lightweight (12 MB) spaCy model used for splitting sentences during chunking
nlp = spacy.load("en_core_web_sm")


# Main PDF processing function
def load_pdf(file_path: str) -> list:
    text_chunks = []
    table_chunks = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract any found text from the page
            page_text = page.extract_text()
            if page_text:
                text_chunks.extend(chunk_text(page_text))

            # Extract all tables from the page
            tables = page.extract_tables()
            for table in tables:
                formatted_table = format_table(table)
                if formatted_table:
                    table_chunks.append(formatted_table)

    return text_chunks + table_chunks


# Converts extracted tables into more readable information
def format_table(table) -> str:
    # A skip for empty spaces or header-only tables
    if not table or len(table) < 2:
        return ""

    # Separates the heads for tables from the rest of data
    header = table[0]
    rows = table[1:]

    # A prefix to label the chunk as a table
    lines = ["[TABLE CHUNK]"]

    for row in rows:
        # Skip rows that don’t match the header length
        # No longer occurs in practice and was used only in early development for chunk testing
        if len(row) != len(header):
            continue

        entries = []
        # Zips header and row together to pair column names with their corresponding cell values
        for col_name, cell in zip(header, row):
            # Cleans up whitespaces
            col_name = col_name.strip() if col_name else ""
            cell = cell.strip() if cell else ""
            if col_name and cell:
                entries.append(f"{col_name} = {cell}")

        # Joins the row’s entries into a single line and adds to the chunk
        if entries:
            lines.append("; ".join(entries))

    return "\n".join(lines)


# Breaks up text chunks into pieces with full sentences
def chunk_text(text: str, max_chars=800) -> list:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk = ""

    # Concatenates sentences until the character limit is reached, then starts a new chunk
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# A main function to test chunk creation
if __name__ == "__main__":
    PDF_PATH = "data/NVIDIAa.pdf"

    chunks = load_pdf(PDF_PATH)
    print(f"Total chunks created: {len(chunks)}\n")

    # Prints the first 5 chunks
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i + 1} ---\n{chunk}\n")
