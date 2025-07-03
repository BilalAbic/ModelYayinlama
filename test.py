# --- MERGED RAG & INTERACTION MODULE (CPU OPTIMIZED) ---

import os
import json
import torch
import pickle
import logging
import re
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

# Core transformers (always needed)
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional imports for GPU features (with fallbacks)
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# PDF Processing
import PyPDF2
import fitz  # PyMuPDF

# Sentence Transformers & Vector Store
from sentence_transformers import SentenceTransformer
import faiss

# NLTK for text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log availability of optional features
if not BITSANDBYTES_AVAILABLE:
    logger.warning("BitsAndBytesConfig not available - GPU quantization disabled")
if not PEFT_AVAILABLE:
    logger.warning("PEFT not available - LoRA adapters disabled")

# --- Configuration ---
@dataclass
class RAGConfig:
    """Central configuration for the RAG system - CPU optimized."""
    # RAG parameters - CPU optimized for low memory
    vector_store_path: str = "./fitness_rag_store_merged"
    chunk_size: int = 200  # Smaller chunks for CPU processing
    chunk_overlap_sentences: int = 1  # Less overlap to save memory
    retrieval_k: int = 3  # Fewer documents to process
    retrieval_score_threshold: float = 0.3  # Higher threshold for quality
    max_context_length: int = 820  # Conservative context for CPU (DialoGPT-medium)

    # Model parameters - CPU optimized
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight
    generator_model_name: str = "microsoft/DialoGPT-medium"  # CPU-friendly fallback
    peft_model_path: Optional[str] = None # Path to LoRA adapter

DEFAULT_SYSTEM_PROMPT = """Sen FitTürkAI, empatik ve profesyonel bir sağlık koçusun. Beslenme, egzersiz, uyku ve stres yönetimi konularında rehberlik yaparsın. Sağlık uzmanı değilsin, genel öneriler verirsin. Nazik, motive edici ve destekleyici yaklaşım sergilersin."""

# --- Data Structures ---
@dataclass
class Document:
    """Represents a document chunk with metadata."""
    content: str
    source: str
    doc_type: str  # 'pdf' or 'json'
    chunk_id: str
    metadata: Dict = field(default_factory=dict)

# --- Core RAG Components ---

class TurkishTextProcessor:
    """Handles advanced Turkish text preprocessing, cleaning, and chunking."""
    def __init__(self):
        self.turk_to_ascii_map = str.maketrans('ğüşıöçĞÜŞİÖÇ', 'gusiocGUSIOC')
        self.turkish_stopwords = {'ve', 'ile', 'bir', 'bu', 'da', 'de', 'için'}
        self._download_nltk_data() # Call the corrected downloader
        try:
            self.turkish_stopwords = set(stopwords.words('turkish'))
        except Exception:
            logger.warning("Could not load Turkish stopwords, using a basic set.")

    def _download_nltk_data(self):
      """
      Robustly downloads required NLTK data with proper error handling.
      Handles both old (punkt) and new (punkt_tab) NLTK versions.
      """
      logger.info("Checking/downloading NLTK data...")

      # List of packages to download with their alternatives
      packages_to_try = [
          ['punkt_tab', 'punkt'],  # Try new version first, then old
          ['stopwords']
      ]

      for package_group in packages_to_try:
          success = False

          if isinstance(package_group, list):
              # Try each package in the group until one succeeds
              for package in package_group:
                  try:
                      nltk.download(package, quiet=True)
                      logger.info(f"Successfully downloaded NLTK package: {package}")
                      success = True
                      break
                  except Exception as e:
                      logger.debug(f"Failed to download {package}: {e}")
                      continue
          else:
              # Single package
              try:
                  nltk.download(package_group, quiet=True)
                  logger.info(f"Successfully downloaded NLTK package: {package_group}")
                  success = True
              except Exception as e:
                  logger.debug(f"Failed to download {package_group}: {e}")

          if not success:
              package_name = package_group[0] if isinstance(package_group, list) else package_group
              logger.warning(f"Failed to download any variant of {package_name}")

      # Test if sentence tokenization works
      try:
          test_sentences = sent_tokenize("Bu bir test cümlesidir. Bu ikinci cümledir.", language='turkish')
          if len(test_sentences) >= 2:
              logger.info("NLTK sentence tokenization is working correctly.")
          else:
              logger.warning("NLTK sentence tokenization may not be working optimally.")
      except Exception as e:
          logger.warning(f"NLTK sentence tokenization test failed: {e}")
          logger.info("System will fall back to regex-based sentence splitting.")


    def turkish_lower(self, text: str) -> str:
        """Correctly lowercases Turkish text."""
        return text.replace('I', 'ı').replace('İ', 'i').lower()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.strip()
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ.,!?-]', '', text)
        return text

    def preprocess_for_embedding(self, text: str) -> str:
        """Prepares text for embedding."""
        text = self.clean_text(text)
        text = self.turkish_lower(text)
        return text

    def chunk_text(self, text: str, chunk_size: int, overlap_sentences: int) -> List[str]:
        """Split text into overlapping chunks based on sentences."""
        try:
            sentences = sent_tokenize(text, language='turkish')
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed ({e}), falling back to basic splitting.")
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences: return []

        chunks, current_chunk_words = [], []
        for i, sentence in enumerate(sentences):
            sentence_words = sentence.split()
            if len(current_chunk_words) + len(sentence_words) > chunk_size and current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                overlap_start_index = max(0, i - overlap_sentences)
                overlapped_sentences = sentences[overlap_start_index:i]
                current_chunk_words = " ".join(overlapped_sentences).split()
            current_chunk_words.extend(sentence_words)
        if current_chunk_words: chunks.append(" ".join(current_chunk_words))
        return chunks

class PDFProcessor:
    """Handles PDF document processing."""
    def __init__(self, text_processor: TurkishTextProcessor, config: RAGConfig):
        self.text_processor = text_processor
        self.config = config

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF using PyMuPDF with a fallback."""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                text = "".join(page.get_text() for page in doc)
            if text.strip(): return self.text_processor.clean_text(text)
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path}: {e}. Falling back.")
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            return self.text_processor.clean_text(text)
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""

    def process_directory(self, pdf_directory: str) -> List[Document]:
        """Process all PDFs in a directory."""
        documents = []
        pdf_files = list(Path(pdf_directory).rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in '{pdf_directory}'.")
        for pdf_path in pdf_files:
            text = self.extract_text_from_pdf(str(pdf_path))
            if not text: continue
            chunks = self.text_processor.chunk_text(text, self.config.chunk_size, self.config.chunk_overlap_sentences)
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:
                    documents.append(Document(
                        content=chunk, source=str(pdf_path), doc_type='pdf',
                        chunk_id=f"pdf_{pdf_path.stem}_{i}", metadata={'file_name': pdf_path.name}
                    ))
        return documents

class JSONProcessor:
    """Handles JSON/JSONL data processing."""
    def __init__(self, text_processor: TurkishTextProcessor, config: RAGConfig):
        self.text_processor = text_processor
        self.config = config

    def process_directory(self, json_directory: str) -> List[Document]:
        """Process all JSON/JSONL files in a directory."""
        all_docs = []
        json_files = list(Path(json_directory).rglob("*.json")) + list(Path(json_directory).rglob("*.jsonl"))
        logger.info(f"Found {len(json_files)} JSON files in '{json_directory}'.")
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f] if str(json_path).endswith('.jsonl') else json.load(f)
                if not isinstance(data, list): data = [data]
                for i, item in enumerate(data):
                    content = f"Soru: {item.get('soru', '')}\nCevap: {item.get('cevap', '')}" if 'soru' in item else item.get('text', '') or item.get('content', '') or ' '.join(str(v) for v in item.values() if isinstance(v, str))
                    if content and len(content.strip()) > 20:
                        all_docs.append(Document(
                            content=self.text_processor.clean_text(content), source=str(json_path),
                            doc_type='json', chunk_id=f"json_{Path(json_path).stem}_{i}",
                            metadata={'original_index': i}
                        ))
            except Exception as e:
                logger.error(f"Failed to process JSON file {json_path}: {e}")
        return all_docs

class VectorStore:
    """Manages document embeddings and FAISS-based similarity search."""
    def __init__(self, config: RAGConfig, text_processor: TurkishTextProcessor):
        self.config = config
        self.text_processor = text_processor
        self.model = None
        
        # Try loading embedding model with fallbacks
        models_to_try = [
            config.embedding_model_name,
            "all-MiniLM-L6-v2",  # Even smaller fallback
            "paraphrase-multilingual-MiniLM-L12-v2"  # Last resort
        ]
        
        for model_name in models_to_try:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
                if model_name != config.embedding_model_name:
                    logger.info(f"Using fallback model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load embedding model {model_name}: {e}")
                continue
        
        if not self.model:
            logger.error("Failed to load any embedding model! Vector store will be disabled.")
            
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None

    def build(self, documents: List[Document]):
        """Build the vector store from documents."""
        if not documents:
            logger.warning("No documents provided to build vector store.")
            return
        if not self.model:
            logger.warning("No embedding model available, cannot build vector store.")
            return
        self.documents = documents
        logger.info(f"Encoding {len(self.documents)} documents...")
        texts = [self.text_processor.preprocess_for_embedding(doc.content) for doc in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors.")

    def search(self, query: str) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if not self.index or not self.documents or not self.model: 
            return []
        processed_query = self.text_processor.preprocess_for_embedding(query)
        query_embedding = self.model.encode([processed_query], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype('float32'), self.config.retrieval_k)
        results = [(self.documents[idx], float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1 and score >= self.config.retrieval_score_threshold]
        return results

    def save(self):
        """Save the vector store to disk."""
        path = Path(self.config.vector_store_path)
        path.mkdir(parents=True, exist_ok=True)
        if self.index: faiss.write_index(self.index, str(path / 'faiss_index.bin'))
        with open(path / 'documents.pkl', 'wb') as f: pickle.dump(self.documents, f)
        logger.info(f"Vector store saved to {path}")

    def load(self) -> bool:
        """Load the vector store from disk."""
        path = Path(self.config.vector_store_path)
        if not (path / 'faiss_index.bin').exists() or not (path / 'documents.pkl').exists():
            return False
        self.index = faiss.read_index(str(path / 'faiss_index.bin'))
        with open(path / 'documents.pkl', 'rb') as f: self.documents = pickle.load(f)
        logger.info(f"Loaded vector store with {len(self.documents)} documents from {path}")
        return True

# --- Main Application Class ---

class FitnessRAG:
    """Orchestrates the entire RAG and generation process."""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_processor = TurkishTextProcessor()
        self.pdf_processor = PDFProcessor(self.text_processor, self.config)
        self.json_processor = JSONProcessor(self.text_processor, self.config)
        self.vector_store = VectorStore(self.config, self.text_processor)

        self.model, self.tokenizer = self._load_generator_model()

        if not self.vector_store.load():
            logger.info("No existing knowledge base found. Please build it.")

    def _load_generator_model(self):
        """Loads the causal language model and tokenizer with device-appropriate settings."""
        logger.info(f"Loading base model: {self.config.generator_model_name}")

        # Check if CUDA is available
        device_available = torch.cuda.is_available()
        
        try:
            if device_available and BITSANDBYTES_AVAILABLE:
                logger.info("CUDA detected - attempting 4-bit quantization for GPU inference")
                # Define quantization config for GPU
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                
                # Load model with quantization for GPU
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.generator_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                logger.info("No CUDA detected - using CPU inference without quantization")
                # Load model without quantization for CPU
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.generator_model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Optimize CPU memory usage
                )
        except Exception as e:
            logger.warning(f"Failed to load {self.config.generator_model_name}: {e}")
            logger.info("Falling back to smaller CPU-friendly model...")
            
            # Try DialoGPT-medium first, then DistilGPT-2 if that fails
            try:
                fallback_model = "microsoft/DialoGPT-medium"
                model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
                self.config.generator_model_name = fallback_model
            except Exception as e2:
                logger.warning(f"DialoGPT-medium also failed: {e2}")
                logger.info("Final fallback to DistilGPT-2...")
                fallback_model = "distilgpt2"
                model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
                self.config.generator_model_name = fallback_model

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.generator_model_name)
        
        # Handle missing pad token (common issue)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        # Load PEFT adapter if available
        if PEFT_AVAILABLE and self.config.peft_model_path and Path(self.config.peft_model_path).exists():
            logger.info(f"Loading PEFT adapter from: {self.config.peft_model_path}")
            
            try:
                # Load the LoRA adapter
                model = PeftModel.from_pretrained(
                    model,
                    self.config.peft_model_path,
                    is_trainable=False
                )
                
                # Merge adapter for CPU inference (more memory but faster)
                if not device_available:
                    logger.info("Merging adapter weights for CPU inference...")
                    model = model.merge_and_unload()
                
                logger.info("PEFT adapter loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PEFT adapter: {e}")
                logger.warning("Continuing with base model only")
        elif not PEFT_AVAILABLE:
            logger.warning("PEFT library not available. Using base model only.")
        else:
            logger.warning("PEFT adapter path not found or doesn't exist. Using base model only.")

        model.eval()
        return model, tokenizer

    def build_knowledge_base(self, pdf_dir: str = None, json_dir: str = None):
        """Builds the knowledge base from source files."""
        all_docs = []
        if pdf_dir and Path(pdf_dir).exists():
            all_docs.extend(self.pdf_processor.process_directory(pdf_dir))
        if json_dir and Path(json_dir).exists():
            all_docs.extend(self.json_processor.process_directory(json_dir))

        if not all_docs:
            logger.warning("No new documents found. Knowledge base not built.")
            return

        self.vector_store.build(all_docs)
        self.vector_store.save()

    def retrieve_context(self, query: str) -> str:
        """Retrieve and format context for a given query."""
        results = self.vector_store.search(query)
        if not results: return ""

        context_parts = []
        current_len = 0
        for doc, score in results:
            content = f"[Kaynak: {Path(doc.source).name}, Skor: {score:.2f}] {doc.content}"
            if current_len + len(content) > self.config.max_context_length: break
            context_parts.append(content)
            current_len += len(content)

        return "\n\n---\n\n".join(context_parts)

    def ask(self, user_query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Main method to ask a question and get a generated answer with smart token management."""
        start_time = time.time()
        context = self.retrieve_context(user_query)
        retrieval_time = time.time() - start_time
        context_length = len(context) if context else 0
        print(f"📚 DEBUG: Retrieved context: {context_length} characters")
        logger.info(f"Context retrieval took {retrieval_time:.2f}s.")

        # Build prompt - DialoGPT conversation style
        if context:
            prompt = f"Sistem: {system_prompt}\n\nBilgiler: {context}\n\nKullanıcı: {user_query}\nAsistan:"
        else:
            prompt = f"Sistem: {system_prompt}\n\nKullanıcı: {user_query}\nAsistan:"

        # Smart token management - ensure we don't exceed model limits
        max_input_tokens = 700  # Leave room for generation (1024 - 324 = 700)
        
        # Tokenize and check length
        temp_tokens = self.tokenizer.encode(prompt)
        prompt_length = len(temp_tokens)
        print(f"🔍 DEBUG: Initial prompt length: {prompt_length} tokens")
        logger.info(f"Initial prompt length: {prompt_length} tokens")
        
        # If too long, intelligently truncate
        if prompt_length > max_input_tokens:
            print(f"⚠️  DEBUG: Prompt too long ({prompt_length} tokens), truncating context...")
            logger.warning(f"Prompt too long ({prompt_length} tokens), truncating context...")
            
            # Truncate context first, keep system prompt and user query
            base_prompt = f"Sistem: {system_prompt}\n\nKullanıcı: {user_query}\nAsistan:"
            base_tokens = len(self.tokenizer.encode(base_prompt))
            
            if base_tokens >= max_input_tokens:
                # Even base prompt is too long, use minimal version
                minimal_prompt = f"Sistem: Sen sağlık koçu FitTürkAI'sın.\nKullanıcı: {user_query}\nAsistan:"
                prompt = minimal_prompt
                logger.warning("Using minimal prompt due to length constraints")
            else:
                # Gradually reduce context
                available_for_context = max_input_tokens - base_tokens - 50  # Safety margin
                if context and available_for_context > 100:
                    # Truncate context to fit
                    context_words = context.split()
                    while True:
                        test_context = " ".join(context_words)
                        test_prompt = f"Sistem: {system_prompt}\n\nBilgiler: {test_context}\n\nKullanıcı: {user_query}\nAsistan:"
                        if len(self.tokenizer.encode(test_prompt)) <= max_input_tokens:
                            prompt = test_prompt
                            break
                        context_words = context_words[:-10]  # Remove 10 words at a time
                        if len(context_words) < 20:  # Minimum context
                            prompt = base_prompt
                            break
                    logger.info(f"Context truncated to {len(context_words)} words")
                else:
                    prompt = base_prompt
                    logger.warning("No context used due to length constraints")

        # Final tokenization
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
        final_length = inputs['input_ids'].shape[1]
        print(f"✅ DEBUG: Final prompt length: {final_length} tokens (max: {max_input_tokens})")
        logger.info(f"Final prompt length: {final_length} tokens")
        
        inputs = inputs.to(self.model.device)

        print(f"🚀 DEBUG: Starting generation with {final_length} tokens input...")
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Conservative for CPU
                    min_new_tokens=10,   # Ensure some output
                    do_sample=True,
                    temperature=0.7,     # Lower for more focused responses
                    top_k=40,            # Slightly higher for diversity
                    top_p=0.85,          # Reduced for CPU  
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,         # No beam search for CPU
                    use_cache=True,
                    repetition_penalty=1.15,  # Stronger prevention of repetition
                    length_penalty=1.1,       # Encourage longer responses
                    no_repeat_ngram_size=3,   # Prevent 3-gram repetition
                )

            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response = response.strip()
            
            # Debug: Check if response is meaningful
            if len(response) < 10:
                print(f"⚠️  DEBUG: Short response ({len(response)} chars): '{response}'")
                logger.warning(f"Generated response is very short: '{response}'")
            
            print(f"✅ DEBUG: Generated response: {len(response)} characters")
            return response
            
        except Exception as e:
            print(f"❌ DEBUG: Generation failed: {e}")
            logger.error(f"Generation failed: {e}")
            print(f"💡 DEBUG: Using intelligent fallback for query: '{user_query}'")
            # Provide intelligent fallback based on query
            if "kilo" in user_query.lower():
                return "Merhaba! Kilo verme konusunda size yardımcı olmaya hazırım. Dengeli beslenme, düzenli hareket ve yeterli uyku en önemli faktörlerdir. Hangi konuda detaylı bilgi istersiniz?"
            elif "beslenme" in user_query.lower():
                return "Sağlıklı beslenme konusunda rehberlik edebilirim. Günlük öğünlerinizi düzenlemek ve dengeli beslenme alışkanlıkları kazanmak için size özel öneriler verebilirim."
            elif "egzersiz" in user_query.lower() or "spor" in user_query.lower():
                return "Egzersiz programları konusunda size yardımcı olabilirim. Seviyenize uygun, güvenli ve etkili hareketler önerebilirim. Hangi tür aktivitelerle başlamak istersiniz?"
            else:
                return "Merhaba! Ben FitTürkAI. Sağlıklı yaşam, beslenme, egzersiz ve uyku konularında size rehberlik edebilirim. Size nasıl yardımcı olabilirim?"

    def interactive_chat(self):
        """Starts an interactive chat session."""
        print("\n" + "="*60)
        print("🏋️  FitTürkAI RAG Sistemi - İnteraktif Sohbet Modu")
        print("="*60)
        if not self.vector_store.documents:
            print("❌ Bilgi tabanı boş. Lütfen önce `build_knowledge_base` çalıştırın.")
            return

        print("💡 Sorularınızı yazın (çıkmak için 'quit' veya 'q' yazın)")
        print("-" * 60)

        while True:
            try:
                user_query = input("\n🤔 Sorunuz: ").strip()
                if user_query.lower() in ['quit', 'exit', 'çık', 'q']:
                    print("👋 Görüşmek üzere!")
                    break
                if not user_query: continue

                print("\n⏳ Düşünüyorum ve kaynakları tarıyorum...")
                start_time = time.time()

                final_answer = self.ask(user_query)

                total_time = time.time() - start_time

                print("\n" + "-"*15 + f" FitTürkAI'nin Cevabı ({total_time:.2f}s) " + "-"*15)
                print(final_answer)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n\n👋 Program sonlandırıldı!")
                break
            except Exception as e:
                print(f"❌ Bir hata oluştu: {e}")

# --- Main Execution ---
def main():
    """Main function to run the RAG system."""
    import os
    import sys
    import subprocess
    
    # Get the current script directory (GitHub repo root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ---! UPDATED PATHS FOR GITHUB DEPLOYMENT !---
    # Paths relative to the GitHub repo root
    PDF_DATA_DIRECTORY = os.path.join(current_dir, "indirilen_pdfler")  # If you have PDFs
    JSON_DATA_DIRECTORY = os.path.join(current_dir, "DATA")           # If you have JSON files
    
    # Use the uploaded fine-tuned model from GitHub
    PEFT_ADAPTER_PATH = os.path.join(current_dir, "fine_tuned_FitTurkAI_QLoRA")
    VECTOR_STORE_PATH = os.path.join(current_dir, "fitness_rag_store_merged")
    
    # Check if the uploaded model exists, if not try to download
    if not os.path.exists(PEFT_ADAPTER_PATH):
        print(f"⚠️  Fine-tuned model not found at: {PEFT_ADAPTER_PATH}")
        print("🔄 Attempting to download model files...")
        
        try:
            download_script = os.path.join(current_dir, "download_models_gdown.py")
            if os.path.exists(download_script):
                subprocess.run([sys.executable, download_script], check=True)
            else:
                print("❌ Download script not found. Please download model files manually.")
                print("💡 You can run: python download_models_gdown.py")
        except subprocess.CalledProcessError:
            print("❌ Download failed. Using base model without fine-tuning.")
            PEFT_ADAPTER_PATH = None
        except Exception as e:
            print(f"❌ Download error: {e}")
            PEFT_ADAPTER_PATH = None
    
    # Final check for model existence
    if PEFT_ADAPTER_PATH and os.path.exists(PEFT_ADAPTER_PATH):
        print(f"✅ Fine-tuned model found at: {PEFT_ADAPTER_PATH}")
    else:
        print("⚠️  Will use base model without fine-tuning.")
        PEFT_ADAPTER_PATH = None
    
    # Check if vector store exists
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"✅ Vector store found at: {VECTOR_STORE_PATH}")
    else:
        print(f"⚠️  Vector store not found at: {VECTOR_STORE_PATH}")
        VECTOR_STORE_PATH = "./fitness_rag_store_merged"  # Will create new one
    
    # Update config with GitHub paths (temporarily disable PEFT for testing)
    config = RAGConfig(
        peft_model_path=None,  # Temporarily disabled for testing
        vector_store_path=VECTOR_STORE_PATH
    )
    
    # Debug: Print the actual paths being used
    print(f"🔧 DEBUG: Working directory: {current_dir}")
    print(f"🔧 DEBUG: PEFT path would be: {PEFT_ADAPTER_PATH}")
    print(f"🔧 DEBUG: Vector store path: {VECTOR_STORE_PATH}")
    print(f"🔧 DEBUG: PEFT adapter temporarily disabled for testing")

    # Initialize the entire system (including loading the LLM)
    print("🚀 FitTürkAI RAG Sistemi Başlatılıyor...")
    print("💻 CPU modunda çalışıyor - GPU optimizasyonları devre dışı")
    
    # Memory usage tip for CPU
    if not torch.cuda.is_available():
        print("💡 CPU için optimize edildi: daha küçük modeller ve parametreler kullanılıyor")
    
    rag_system = FitnessRAG(config)

    # Check if the knowledge base needs to be built
    vector_store_path = Path(config.vector_store_path)
    if not vector_store_path.exists():
        print(f"\n🔨 Bilgi tabanı '{vector_store_path}' bulunamadı, yeniden oluşturuluyor...")
        rag_system.build_knowledge_base(
            pdf_dir=PDF_DATA_DIRECTORY,
            json_dir=JSON_DATA_DIRECTORY
        )
    else:
        print(f"\n✅ Mevcut bilgi tabanı '{vector_store_path}' yüklendi.")
        rebuild = input("Bilgi tabanını yeniden oluşturmak ister misiniz? (y/N): ").strip().lower()
        if rebuild == 'y':
            import shutil
            shutil.rmtree(vector_store_path)
            print("🔄 Bilgi tabanı yeniden oluşturuluyor...")
            rag_system.build_knowledge_base(
                pdf_dir=PDF_DATA_DIRECTORY,
                json_dir=JSON_DATA_DIRECTORY
            )

    # Start interactive mode
    rag_system.interactive_chat()

if __name__ == "__main__":
    main()