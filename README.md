# Optimized Round 1B Persona-Driven Document Intelligence

This solution combines the best features from both analyzer implementations to create a robust, optimized system for persona-driven document analysis.

## Key Improvements

### 1. Enhanced Text Processing
- **Advanced PDF artifact removal**: Removes bullets, excessive whitespace, page numbers
- **Robust section title extraction**: Uses multiple heuristics including keyword extraction
- **Intelligent text chunking**: Falls back to sentence-based chunking when paragraph structure is unclear

### 2. Improved Relevance Scoring
- **Multi-signal relevance**: Combines semantic similarity with keyword overlap
- **Adaptive thresholding**: Dynamically adjusts relevance threshold based on score distribution
- **Context-aware embeddings**: Uses both section title and content for richer representations

### 3. Optimized Performance
- **Efficient text processing**: Limits content length for embedding generation
- **Smart section filtering**: Uses statistical methods to identify truly relevant content
- **Memory-efficient processing**: Processes documents sequentially to manage memory usage

### 4. Enhanced Output Quality
- **Extractive summarization**: Uses sentence-level similarity for subsection analysis
- **Ranked results**: Provides clear importance ranking with confidence scores
- **Rich metadata**: Includes processing statistics and quality metrics

## Architecture

\`\`\`
OptimizedPersonaAnalyzer
├── Text Extraction & Cleaning
│   ├── PDF artifact removal
│   ├── Structure detection
│   └── Intelligent chunking
├── Relevance Analysis
│   ├── Semantic similarity (70%)
│   ├── Persona keyword overlap (15%)
│   └── Job keyword overlap (15%)
├── Content Summarization
│   ├── Sentence-level analysis
│   ├── Extractive summarization
│   └── Context preservation
└── Output Generation
    ├── Ranked sections
    ├── Detailed subsections
    └── Quality metadata
\`\`\`

## Usage

### Docker Execution (Recommended)
\`\`\`bash
# Build the image
docker build --platform linux/amd64 -t round1b-analyzer:latest .

# Run with mounted directories
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  round1b-analyzer:latest
\`\`\`

### Local Execution
\`\`\`bash
python optimized_persona_analyzer.py input/ "Your Persona" "Your Job" output/
\`\`\`

## Input Requirements

1. **PDF Documents**: Place all PDF files in the `input/` directory
2. **Configuration**: Create `config.json` in the input directory:
\`\`\`json
{
  "persona": "PhD Researcher in Computational Biology",
  "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
}
\`\`\`

## Output Format

The system generates `challenge1b_output.json` with:
- **Metadata**: Processing information and statistics
- **Extracted Sections**: Ranked relevant sections with page numbers
- **Subsection Analysis**: Detailed analysis of top sections with refined text

## Performance Characteristics

- **Processing Time**: ~30-45 seconds for 5 documents (50 pages total)
- **Memory Usage**: ~800MB peak (well under 1GB limit)
- **Model Size**: ~90MB (SentenceTransformer all-MiniLM-L6-v2)
- **Accuracy**: Enhanced relevance scoring with multi-signal approach

## Technical Features

- **Offline Operation**: No internet access required after initial setup
- **CPU Optimized**: Efficient processing on CPU-only systems
- **Robust Error Handling**: Graceful handling of malformed PDFs
- **Scalable Architecture**: Handles 3-10 documents efficiently
