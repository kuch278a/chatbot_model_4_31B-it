import os
import tempfile
import unittest
from src.rag.pipeline import RAGPipeline
from src.db.vector_store import SQLiteVectorStore

class MockEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 768

class TestRAGPipelineSync(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.embeddings = MockEmbeddings()
        self.vector_store = SQLiteVectorStore()
        self.vector_store.clear_all()
        self.pipeline = RAGPipeline(self.embeddings, vector_store=self.vector_store)

    def tearDown(self):
        self.vector_store.clear_all()
        self.temp_dir.cleanup()

    def test_strict_directory_synchronization(self):
        # Create 2 files in temp_dir
        file1 = os.path.join(self.temp_dir.name, "doc1.txt")
        file2 = os.path.join(self.temp_dir.name, "doc2.txt")
        
        with open(file1, "w", encoding="utf-8") as f:
            f.write("Content of doc 1")
        with open(file2, "w", encoding="utf-8") as f:
            f.write("Content of doc 2")

        # Index directory
        self.pipeline.index_directory(self.temp_dir.name)
        docs = set(self.vector_store.get_all_document_names())
        self.assertEqual(docs, {"doc1.txt", "doc2.txt"})

        # Remove file2 from directory
        os.remove(file2)

        # Re-index directory -> doc2.txt should be automatically purged from DB
        self.pipeline.index_directory(self.temp_dir.name)
        docs_after = set(self.vector_store.get_all_document_names())
        self.assertEqual(docs_after, {"doc1.txt"})

if __name__ == "__main__":
    unittest.main()
