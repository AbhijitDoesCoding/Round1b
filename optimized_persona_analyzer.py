#!/usr/bin/env python3
"""
Optimized Persona-Driven Document Intelligence Analyzer for Round 1B
Combines the best features from both implementations with performance optimizations.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
from datetime import datetime
import fitz  # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class OptimizedPersonaAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the analyzer with optimized settings"""
        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        print("Model loaded successfully.")

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with PDF artifact removal"""
        if not text or not text.strip():
            return ""
        
        # Remove common PDF artifacts
        text = re.sub(r'[\u2022\u25AA\u25CF\u25A0\u2013\u2014\u25B6\u25C6]', '', text)
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove standalone numbers and short fragments
        text = re.sub(r'\b\d+\b', '', text)
        
        return text.strip()

    def extract_section_title(self, text: str) -> str:
        """Enhanced section title extraction with keyword-based fallback"""
        if not text:
            return "Untitled Section"
        
        lines = text.split('\n')
        first_line = lines[0].strip()
        
        # Check if first line looks like a heading
        if len(first_line) < 120 and (
            first_line.isupper() or
            re.match(r'^\d+\.?\s+[A-Z]', first_line) or
            first_line.endswith(':') or
            re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', first_line)
        ):
            return first_line
        
        # Fallback: extract keywords for title
        words = word_tokenize(first_line.lower())
        keywords = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        if keywords:
            # Take top 4-6 most meaningful words
            title_words = keywords[:min(6, len(keywords))]
            title = ' '.join(title_words).title()
            return title + ('...' if len(first_line.split()) > 6 else '')
        
        # Last resort: truncate first line
        words = first_line.split()[:8]
        return ' '.join(words) + ('...' if len(first_line.split()) > 8 else '')

    def extract_text_with_structure(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text with improved structure detection"""
        sections = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if not text.strip():
                    continue
                
                # Clean the text
                cleaned_text = self.clean_text(text)
                if len(cleaned_text) < 30:  # Skip very short pages
                    continue
                
                # Split into logical paragraphs
                paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
                
                # If no clear paragraphs, split by sentences into chunks
                if len(paragraphs) <= 1:
                    sentences = sent_tokenize(cleaned_text)
                    chunk_size = 4  # Group sentences into chunks
                    paragraphs = []
                    for i in range(0, len(sentences), chunk_size):
                        chunk = ' '.join(sentences[i:i+chunk_size])
                        if len(chunk) > 50:
                            paragraphs.append(chunk)
                
                # Process each paragraph as a section
                for i, paragraph in enumerate(paragraphs):
                    if len(paragraph) > 50:  # Filter meaningful content
                        sections.append({
                            "document": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "section_id": f"page_{page_num + 1}_sec_{i + 1}",
                            "text": paragraph,
                            "section_title": self.extract_section_title(paragraph)
                        })
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
        
        return sections

    def calculate_enhanced_relevance(self, section_text: str, section_title: str, 
                                   persona: str, job: str) -> float:
        """Enhanced relevance calculation using multiple signals"""
        try:
            # Create comprehensive context
            query_context = f"Persona: {persona}. Task: {job}"
            
            # Combine title and text for richer representation
            section_context = f"Title: {section_title}. Content: {section_text[:500]}"  # Limit for efficiency
            
            # Generate embeddings
            query_embedding = self.model.encode(query_context)
            section_embedding = self.model.encode(section_context)
            
            # Calculate base similarity
            base_similarity = cosine_similarity(
                section_embedding.reshape(1, -1), 
                query_embedding.reshape(1, -1)
            )[0][0]
            
            # Apply keyword boosting
            persona_keywords = set(word.lower() for word in word_tokenize(persona) 
                                 if word.isalpha() and word not in self.stop_words)
            job_keywords = set(word.lower() for word in word_tokenize(job) 
                             if word.isalpha() and word not in self.stop_words)
            
            section_words = set(word.lower() for word in word_tokenize(section_text) 
                              if word.isalpha())
            
            # Keyword overlap bonus
            persona_overlap = len(persona_keywords.intersection(section_words)) / max(len(persona_keywords), 1)
            job_overlap = len(job_keywords.intersection(section_words)) / max(len(job_keywords), 1)
            
            # Combine signals with weights
            final_score = (
                base_similarity * 0.7 +  # Semantic similarity (primary)
                persona_overlap * 0.15 +  # Persona relevance
                job_overlap * 0.15        # Task relevance
            )
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return 0.0

    def extract_key_subsections(self, section_text: str, query_context: str, 
                              max_sentences: int = 3) -> str:
        """Extract most relevant sentences using extractive summarization"""
        if len(section_text.strip()) < 100:
            return section_text
        
        sentences = sent_tokenize(section_text)
        if len(sentences) <= max_sentences:
            return section_text
        
        try:
            # Encode query and sentences
            query_embedding = self.model.encode(query_context)
            sentence_embeddings = self.model.encode(sentences)
            
            # Calculate similarities
            similarities = cosine_similarity(sentence_embeddings, query_embedding.reshape(1, -1))
            
            # Get top sentences while maintaining order
            top_indices = sorted(np.argsort(similarities.flatten())[-max_sentences:])
            selected_sentences = [sentences[i] for i in top_indices]
            
            return ' '.join(selected_sentences)
            
        except Exception as e:
            print(f"Error in subsection extraction: {e}")
            return section_text[:500] + "..." if len(section_text) > 500 else section_text

    def analyze_documents(self, documents: List[str], persona: str, job: str) -> Dict[str, Any]:
        """Main analysis function with optimized processing"""
        print(f"Analyzing {len(documents)} documents...")
        
        all_sections = []
        
        # Extract sections from all documents
        for doc_path in documents:
            print(f"Processing: {os.path.basename(doc_path)}")
            doc_sections = self.extract_text_with_structure(doc_path)
            all_sections.extend(doc_sections)
        
        print(f"Extracted {len(all_sections)} sections total")
        
        # Calculate relevance scores
        query_context = f"{persona}. {job}"
        scored_sections = []
        
        for section in all_sections:
            score = self.calculate_enhanced_relevance(
                section["text"], 
                section["section_title"], 
                persona, 
                job
            )
            section["relevance_score"] = score
            scored_sections.append(section)
        
        # Sort by relevance and filter
        scored_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Dynamic threshold based on score distribution
        scores = [s["relevance_score"] for s in scored_sections]
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            threshold = max(0.1, mean_score - 0.5 * std_score)  # Adaptive threshold
        else:
            threshold = 0.1
        
        # Select top sections with minimum threshold
        top_sections = [s for s in scored_sections if s["relevance_score"] >= threshold][:25]
        
        print(f"Selected {len(top_sections)} relevant sections (threshold: {threshold:.3f})")
        
        # Prepare output
        extracted_sections = []
        subsection_analysis = []
        
        for i, section in enumerate(top_sections):
            # Main section info
            extracted_sections.append({
                "document": section["document"],
                "page_number": section["page"],
                "section_title": section["section_title"],
                "importance_rank": i + 1
            })
            
            # Detailed subsection analysis for top 10 sections
            if i < 10:
                refined_text = self.extract_key_subsections(
                    section["text"], 
                    query_context, 
                    max_sentences=3
                )
                
                subsection_analysis.append({
                    "document": section["document"],
                    "refined_text": refined_text,
                    "page_number": section["page"]
                })
        
        return {
            "metadata": {
                "input_documents": [os.path.basename(doc) for doc in documents],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_analyzed": len(all_sections),
                "relevant_sections_found": len(top_sections)
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }


def process_round1b_optimized(documents: List[str], persona: str, job: str, output_dir: str):
    """Process Round 1B with optimized analyzer"""
    analyzer = OptimizedPersonaAnalyzer()
    
    print(f"\n🚀 Starting optimized Round 1B processing...")
    print(f"📋 Persona: {persona}")
    print(f"🎯 Job: {job}")
    print(f"📚 Documents: {len(documents)}")
    
    # Analyze documents
    result = analyzer.analyze_documents(documents, persona, job)
    
    # Save output
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    output_file = output_path / "challenge1b_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis complete! Results saved to {output_file}")
    print(f"📊 Found {len(result['extracted_sections'])} relevant sections")
    print(f"🔍 Generated {len(result['subsection_analysis'])} detailed analyses")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 5:
        print("Usage: python optimized_persona_analyzer.py <input_dir> <persona> <job_to_be_done> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    persona = sys.argv[2]
    job = sys.argv[3]
    output_dir = sys.argv[4]
    
    # Find PDF files
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory '{input_dir}' does not exist!")
        sys.exit(1)
    
    documents = [str(p) for p in input_path.glob("*.pdf")]
    if not documents:
        print(f"❌ No PDF files found in '{input_dir}'!")
        sys.exit(1)
    
    process_round1b_optimized(documents, persona, job, output_dir)
