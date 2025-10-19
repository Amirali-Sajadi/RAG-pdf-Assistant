# # test_loader.py
# from pathlib import Path
# from data_loader import load_and_chunk_pdf
#
# pdf_path = Path(r"C:\Users\Test\Downloads\monopoly_instructions-1-2.pdf")
#
# print("Testing load_and_chunk_pdf with:", pdf_path)
# try:
#     assert pdf_path.exists(), f"File not found: {pdf_path}"
#     chunks = load_and_chunk_pdf(str(pdf_path))
#     print("✅ load_and_chunk_pdf returned", type(chunks), "with length:", len(chunks))
#     if len(chunks):
#         print("First chunk (200 chars):")
#         print(chunks[0][:200])
# except AssertionError as ae:
#     print("❌ AssertionError:", ae)
# except Exception as e:
#     import traceback
#     print("❌ Exception during load_and_chunk_pdf:")
#     print(e)
#     print(traceback.format_exc())

# test_load_step.py
from main import load_chunks_from_context, RAGChunkAndSrc

# Mock context that mimics Inngest's event data
class MockContext:
    class Event:
        def __init__(self, pdf_path):
            self.data = {"pdf_path": pdf_path, "source_id": pdf_path}
    def __init__(self, pdf_path):
        self.event = self.Event(pdf_path)

if __name__ == "__main__":
    pdf_path = r"C:\Users\Test\Downloads\Test.pdf"
    ctx = MockContext(pdf_path)

    print(f"Testing _load with path: {pdf_path}\n")
    try:
        result = load_chunks_from_context(ctx)
        print("✅ _load completed successfully.")
        print("Type:", type(result))
        print("Chunks:", len(result.chunks))
        print("Source ID:", result.source_id)
        print("First chunk (200 chars):")
        print(result.chunks[0][:200] if result.chunks else "(none)")
    except Exception as e:
        import traceback
        print("❌ Exception in _load:")
        print(e)
        print(traceback.format_exc())

