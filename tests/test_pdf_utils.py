"""Small unit tests for page-aware Week 3-compatible PDF chunking."""

import unittest

from pdf_utils import chunk_pdf_pages


class ChunkPDFPagesTests(unittest.TestCase):
    def test_default_chunking_uses_250_words_and_50_overlap(self):
        words = [f"word{i}" for i in range(451)]
        pages = [{"page_number": 1, "text": " ".join(words)}]

        chunks = chunk_pdf_pages(pages)

        self.assertEqual([len(chunk["text"].split()) for chunk in chunks], [250, 250, 51])
        first = chunks[0]["text"].split()
        second = chunks[1]["text"].split()
        self.assertEqual(first[-50:], second[:50])

    def test_chunk_keeps_page_numbers_when_spanning_pages(self):
        pages = [
            {"page_number": 1, "text": "one two three"},
            {"page_number": 2, "text": "four five six"},
        ]

        chunks = chunk_pdf_pages(pages, chunk_size=4, overlap=1)

        self.assertEqual(chunks[0]["page_numbers"], [1, 2])
        self.assertIn("four", chunks[0]["text"])

    def test_empty_pages_return_no_chunks(self):
        self.assertEqual(chunk_pdf_pages([]), [])
        self.assertEqual(chunk_pdf_pages([{"page_number": 1, "text": "  "}]), [])

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_pdf_pages([{"page_number": 1, "text": "words"}], chunk_size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
