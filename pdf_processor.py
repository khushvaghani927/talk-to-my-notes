import io
import re
from typing import List, Dict, Any, Union
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pages_from_pdf(
    pdf_source: Union[str, io.BytesIO, bytes],
    file_name: str = "document.pdf"
) -> List[Document]:
    """
    Extracts text page by page from a PDF file using pypdf.
    Returns a list of LangChain Document objects with page number metadata (1-indexed).
    """
    if isinstance(pdf_source, bytes):
        reader = pypdf.PdfReader(io.BytesIO(pdf_source))
    elif isinstance(pdf_source, io.BytesIO):
        pdf_source.seek(0)
        reader = pypdf.PdfReader(pdf_source)
    else:
        reader = pypdf.PdfReader(pdf_source)

    documents = []
    total_pages = len(reader.pages)

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        # Clean basic whitespace issues while preserving line structure
        cleaned_text = re.sub(r'[ \t]+', ' ', raw_text).strip()
        
        if cleaned_text:
            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "source": file_name,
                    "page": page_idx + 1,
                    "total_pages": total_pages,
                    "char_count": len(cleaned_text),
                }
            )
            documents.append(doc)
            
    return documents


def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Splits page-level documents into smaller semantic chunks with overlapping boundaries.
    Preserves page number and source metadata for each chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = []
    chunk_counter = 0

    for doc in documents:
        page_chunks = text_splitter.split_text(doc.page_content)
        for i, chunk_text in enumerate(page_chunks):
            if chunk_text.strip():
                chunk_doc = Document(
                    page_content=chunk_text.strip(),
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata.get('source', 'doc')}_p{doc.metadata.get('page', 1)}_c{i+1}",
                        "chunk_index": chunk_counter,
                    }
                )
                chunks.append(chunk_doc)
                chunk_counter += 1

    return chunks


def process_uploaded_pdfs(
    uploaded_files: List[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> Dict[str, Any]:
    """
    Processes multiple uploaded PDF files (e.g. from Streamlit file uploader).
    Returns extracted page documents, chunks, and metadata statistics.
    """
    all_pages: List[Document] = []
    files_summary = []

    for uploaded_file in uploaded_files:
        file_name = getattr(uploaded_file, "name", "uploaded_document.pdf")
        
        if hasattr(uploaded_file, "getvalue"):
            file_bytes = uploaded_file.getvalue()
            pages = extract_pages_from_pdf(io.BytesIO(file_bytes), file_name=file_name)
        elif hasattr(uploaded_file, "read"):
            pages = extract_pages_from_pdf(uploaded_file, file_name=file_name)
        else:
            pages = extract_pages_from_pdf(str(uploaded_file), file_name=file_name)

        all_pages.extend(pages)
        files_summary.append({
            "file_name": file_name,
            "page_count": len(pages),
            "total_chars": sum(p.metadata.get("char_count", 0) for p in pages)
        })

    all_chunks = split_documents_into_chunks(
        all_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    stats = {
        "total_files": len(uploaded_files),
        "total_pages": len(all_pages),
        "total_chunks": len(all_chunks),
        "files_summary": files_summary,
        "avg_chunk_length": int(sum(len(c.page_content) for c in all_chunks) / max(len(all_chunks), 1))
    }

    return {
        "pages": all_pages,
        "chunks": all_chunks,
        "stats": stats
    }
