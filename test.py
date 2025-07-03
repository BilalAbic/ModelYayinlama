# --- MERGED RAG & INTERACTION MODULE (CPU OPTIMIZED FOR LINUX) ---

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

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

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

# Transformers for Language Model Interaction
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class RAGConfig:
    """Central configuration for the RAG system."""
    # RAG parameters
    vector_store_path: str = "./fitness_rag_store_merged"
    chunk_size: int = 300  # Words per chunk
    chunk_overlap_sentences: int = 2  # Number of sentences to overlap
    retrieval_k: int = 5
    retrieval_score_threshold: float = 0.2
    max_context_length: int = 3000

    # Model parameters
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    generator_model_name: str = "ytu-ce-cosmos/Turkish-Llama-8b-v0.1"
    peft_model_path: Optional[str] = None # Path to LoRA adapter, e.g., "./fine_tuned_FitTurkAI_LoRA"

DEFAULT_SYSTEM_PROMPT = SISTEM_TALIMATI = """
[ROL]
Sen "FitTürkAI" adında, bütünsel yaklaşıma sahip, empatik ve proaktif bir kişisel sağlıklı yaşam koçusun. Görevin yalnızca beslenme önerileri vermek değil, aynı zamanda kullanıcının fiziksel, zihinsel ve yaşam tarzına dair tüm faktörleri dikkate alarak uyarlanabilir rehberler sunmaktır. Sağlık profesyoneli değilsin, tıbbi teşhis veya tedavi öneremezsin. Amacın kullanıcıya yol arkadaşlığı yapmak, rehberlik sağlamak ve davranış değişikliğini sürdürülebilir kılmaktır.

[GÖREV TANIMI]
Kullanıcının profil verilerini analiz ederek ona özel, bütünsel ve sürdürülebilir bir "Sağlıklı Yaşam Rehberi" oluştur. Bu rehber:
- Beslenme planı
- Egzersiz planı
- Uyku düzeni
- Stres yönetimi stratejileri
- Su tüketim hedefleri
bileşenlerini içermelidir. Rehberin sonunda kullanıcıyı küçük bir mikro hedef belirlemeye teşvik et.

[İLETİŞİM ADIMLARI – ZORUNLU AKIŞ]
1. *Tanıtım ve Uyarı:* Kendini "FitTürkAI" olarak tanıt, sağlık uzmanı olmadığını ve verdiğin bilgilerin sadece rehberlik amacı taşıdığını vurgula. Devam izni al.
2. *Profil Toplama:* Kullanıcıdan şu verileri iste:
   - Yaş, Cinsiyet, Kilo, Boy
   - Sağlık durumu (diyabet, obezite, hipertansiyon, vb.)
   - Beslenme tercihi/alerji (vejetaryen, glutensiz, vb.)
   - Hedef (kilo vermek, enerji kazanmak, vb.)
   - Fiziksel aktivite düzeyi
   - Uyku süresi, stres düzeyi
3. *Prensip Tanıtımı:* Kullanıcının durumuna özel 3–4 temel prensibi (örneğin: dengeli tabak, kan şekeri dengesi, stres ve uykunun etkisi) açıklayarak rehbere zemin hazırla.
4. *Kişiselleştirilmiş Sağlıklı Yaşam Rehberi Sun:*
   - *Beslenme*: Haftalık tablo veya örnek öğünler (tahmini kalori ve porsiyon bilgisiyle)
   - *Egzersiz*: Haftalık FITT prensibine dayalı plan
   - *Uyku & Stres*: Pratik iyileştirme önerileri
   - *Su*: Hedef ve içme taktikleri
5. *Mikro Hedef Belirleme:* Kullanıcıya küçük, uygulanabilir bir hedef seçtir ("Bu hafta neye odaklanalım?").
6. *Kapanış:* Rehberin sonunda doktor desteğinin önemini tekrar vurgula. Net ve cesaret verici bir mesajla bitir.

[KURALLAR VE KISITLAR]
- ❌ *Yasaklı Terimler:* "Tedavi", "reçete", "kesin sonuç", "garanti", "zayıflama diyeti"
- ✅ *İzinli Terimler:* "Öneri", "yaklaşık plan", "rehber", "eğitim amaçlı"
- 🔎 *Kalori ve Porsiyonlar:* Daima "tahmini" ya da "yaklaşık" gibi ifadelerle sun. Öğünler sade, dengeli ve kültürel olarak uygun olmalı.
- 🚫 *Teşhis/Tedavi:* Teşhis koyamazsın, ilaç öneremezsin.
- ✅ *Üslup:* Nazik, empatik, motive edici. Net ve profesyonel. Markdown ile netlik sağla (*kalın, *italik, tablolar).

[DİNAMİK ADAPTASYON VE PROAKTİFLİK]
- Alerji/tercih bildirildiğinde otomatik alternatif öner.
- Plandan sapıldığında kullanıcıyı motive et, çözüme odaklan, ardından planı revize et (örneğin: "gofret yedim" diyorsa → daha hafif akşam öner).
- Her zaman kriz anlarını büyütmeden yönet.

[EGZERSİZ PLANI – KURALLAR]
1. *Uyarı:* Egzersiz önerilerinin öncesinde doktor onayı gerektiğini açıkla.
2. *FITT Analizi:* Egzersizleri profile göre planla (Sıklık, Yoğunluk, Süre, Tür).
3. *Plan Formatı:* Haftalık tablo, güvenli hareketler, tekrar sayısı (örneğin: "formun bozulana kadar", ağırlıksız öneri).
4. *Gelişim Prensibi:* Kolaylaştıkça artırılabilecek yollar sun.

[EK YETENEKLER]
- Haftalık değerlendirme ("Geçen hafta nasıldı?")
- Tarif oluşturma
- Alışveriş listesi çıkarma
- "Neden bu yemek?" sorularını bilimsel ama sade cevaplama

[FEW-SHOT PROMPT – ÖRNEK]
*Kullanıcı:* Merhaba, kilo vermek istiyorum.
*FitTürkAI:* Merhaba! Ben FitTürkAI, yol arkadaşınız... [güvenlik uyarısı + devam onayı]
*Kullanıcı:* 35 yaş, erkek, obezite + hipertansiyon, memur, stresli, 5 saat uyuyor.
*FitTürkAI:* (Teşekkür + prensipler + beslenme tablosu + egzersiz planı + su + uyku + stres + mikro hedef + kapanış)

"""

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
        self.model = SentenceTransformer(config.embedding_model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None

    def build(self, documents: List[Document]):
        """Build the vector store from documents."""
        if not documents:
            logger.warning("No documents provided to build vector store.")
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
        if not self.index or not self.documents: return []
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
        """Loads the causal language model and tokenizer optimized for CPU."""
        logger.info(f"Loading base model for CPU: {self.config.generator_model_name}")

        # Force CPU usage and use float32 for better CPU compatibility
        device = torch.device("cpu")
        
        # Load the base model with CPU-optimized settings
        model = AutoModelForCausalLM.from_pretrained(
            self.config.generator_model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",           # Force CPU usage
            trust_remote_code=True,
            low_cpu_mem_usage=True,     # Enable memory optimization for CPU
        )

        tokenizer = AutoTokenizer.from_pretrained(self.config.generator_model_name)
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check for the PEFT adapter path and load the adapter
        if self.config.peft_model_path and Path(self.config.peft_model_path).exists():
            logger.info(f"Loading PEFT adapter from: {self.config.peft_model_path}")
            try:
                # Load the LoRA adapter onto the base model
                model = PeftModel.from_pretrained(
                    model,
                    self.config.peft_model_path,
                    is_trainable=False  # Important for inference
                )

                # Merge adapter for faster inference on CPU
                logger.info("Merging adapter weights into the base model for CPU optimization...")
                model = model.merge_and_unload()
            except Exception as e:
                logger.warning(f"Failed to load PEFT adapter: {e}. Using base model.")
        else:
            logger.warning("PEFT adapter path not found. Using the base model without fine-tuning.")

        # Ensure model is on CPU and in eval mode
        model = model.to(device)
        model.eval()
        
        # Optional: Enable CPU optimizations
        try:
            model = torch.jit.script(model)  # JIT compilation for CPU speedup
            logger.info("Model JIT compiled for CPU optimization.")
        except Exception as e:
            logger.info(f"JIT compilation not available or failed: {e}")
        
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
        """Main method to ask a question and get a generated answer."""
        start_time = time.time()
        context = self.retrieve_context(user_query)
        retrieval_time = time.time() - start_time
        logger.info(f"Context retrieval took {retrieval_time:.2f}s.")

        if context:
            prompt = f"{system_prompt}\n\n### BAĞLAMSAL BİLGİ KAYNAKLARI\n{context}\n\n### KULLANICI SORUSU\n\"{user_query}\"\n\n### CEVAP"
        else:
            prompt = f"{system_prompt}\n\n### KULLANICI SORUSU\n\"{user_query}\"\n\n### CEVAP"

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def interactive_chat(self):
        """Starts an interactive chat session."""
        print("\n" + "="*60)
        print("🏋️  FitTürkAI RAG Sistemi - İnteraktif Sohbet Modu (CPU)")
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
                logger.error(f"Error in interactive chat: {e}", exc_info=True)

# --- Main Execution ---
def main():
    """Main function to run the RAG system."""
    # ---! IMPORTANT !---
    # Set the correct paths for your data and models here.
    PDF_DATA_DIRECTORY = "./indirilen_pdfler"  # Folder with your PDF files
    JSON_DATA_DIRECTORY = "./DATA"           # Folder with your JSON/JSONL files
    # Set this to the path of your fine-tuned LoRA if you have one.
    # Otherwise, set it to None to use the base model.
    PEFT_ADAPTER_PATH = "./fine_tuned_FitTurkAI_QLoRA"

    config = RAGConfig(peft_model_path=PEFT_ADAPTER_PATH)

    # Initialize the entire system (including loading the LLM)
    print("🚀 FitTürkAI RAG Sistemi Başlatılıyor... (CPU Mode)")
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