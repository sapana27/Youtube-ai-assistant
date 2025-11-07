from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from collections import defaultdict
from datetime import timedelta
from langchain_core.prompts import ChatPromptTemplate

import edge_tts
import asyncio
import datetime
import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import arxiv
from newspaper import Article
from trafilatura import fetch_url, extract
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import re

from dotenv import load_dotenv
load_dotenv()

# ==================== WEB CONTENT PROCESSOR ====================

class WebContentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def process_url(self, url: str):
        """Main method to process any URL"""
        try:
            content_type = self._detect_content_type(url)
            print(f"Detected content type: {content_type}")
            
            if content_type == "pdf":
                return self._process_pdf(url)
            elif content_type == "arxiv":
                return self._process_arxiv(url)
            elif content_type == "news":
                return self._process_news(url)
            else:
                return self._process_webpage(url)
                
        except Exception as e:
            raise Exception(f"Error processing URL {url}: {str(e)}")
    
    def _detect_content_type(self, url: str) -> str:
        """Detect content type from URL"""
        url_lower = url.lower()
        
        if url_lower.endswith('.pdf'):
            return "pdf"
        elif 'arxiv.org' in url_lower and '/pdf' in url_lower:
            return "pdf"
        elif 'arxiv.org/abs' in url_lower:
            return "arxiv"
        elif any(domain in url_lower for domain in ['news', 'article', 'blog', 'medium.com']):
            return "news"
        else:
            return "webpage"
    
    def _process_pdf(self, url: str):
        """Process PDF documents"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            documents = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                content = page.extract_text()
                
                if content.strip():
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": url,
                            "page": page_num + 1,
                            "content_type": "pdf",
                            "total_pages": len(pdf_reader.pages)
                        }
                    )
                    documents.append(doc)
            
            return self.text_splitter.split_documents(documents)
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def _process_arxiv(self, url: str):
        """Process ArXiv research papers"""
        try:
            # Extract arXiv ID from URL
            arxiv_id = url.split('/')[-1].replace('.pdf', '')
            
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            content = f"""
TITLE: {paper.title}

AUTHORS: {', '.join(str(author) for author in paper.authors)}

ABSTRACT: {paper.summary}

PUBLISHED: {paper.published}
CATEGORIES: {', '.join(paper.categories)}
"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": url,
                    "content_type": "research_paper",
                    "arxiv_id": arxiv_id,
                    "title": paper.title,
                    "authors": [str(author) for author in paper.authors],
                    "published": paper.published.isoformat(),
                    "categories": paper.categories
                }
            )
            
            return self.text_splitter.split_documents([doc])
            
        except Exception as e:
            raise Exception(f"ArXiv processing failed: {str(e)}")
    
    def _process_news(self, url: str):
        """Process news articles"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            content = f"""
TITLE: {article.title}

AUTHORS: {', '.join(article.authors) if article.authors else 'Unknown'}

PUBLISH DATE: {article.publish_date}

SUMMARY: {article.summary}

FULL TEXT: {article.text}
"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": url,
                    "content_type": "news",
                    "title": article.title,
                    "authors": article.authors,
                    "publish_date": str(article.publish_date),
                    "keywords": article.keywords
                }
            )
            
            return self.text_splitter.split_documents([doc])
            
        except Exception as e:
            # Fallback to general webpage processing
            return self._process_webpage(url)
    
    def _process_webpage(self, url: str):
        """Process general webpages"""
        try:
            # First try with trafilatura (clean extraction)
            downloaded = fetch_url(url)
            if downloaded:
                content = extract(downloaded)
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": url,
                            "content_type": "webpage",
                            "extraction_method": "trafilatura"
                        }
                    )
                    return self.text_splitter.split_documents([doc])
            
            # Fallback to BeautifulSoup
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            doc = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "content_type": "webpage", 
                    "extraction_method": "beautifulsoup"
                }
            )
            
            return self.text_splitter.split_documents([doc])
            
        except Exception as e:
            raise Exception(f"Webpage processing failed: {str(e)}")

# ==================== CONTENT MANAGER ====================

class ContentManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.web_processor = WebContentProcessor()
        self.vectorstores = {}
        self.retrievers = {}
    
    def load_web_content(self, url: str):
        """Load web content and create vector store"""
        try:
            # Process content
            documents = self.web_processor.process_url(url)
            
            # Create unique collection name
            content_id = str(uuid.uuid4())[:8]
            collection_name = f"web_{content_id}"
            
            # Create vector store
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model
            )
            
            vectorstore.add_documents(documents)
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "lambda_mult": 0.3}
            )
            
            # Store references
            self.vectorstores[url] = vectorstore
            self.retrievers[url] = retriever
            
            return True, f"Successfully loaded content from {url}"
            
        except Exception as e:
            return False, f"Error loading web content: {str(e)}"
    
    def get_retriever(self, url: str):
        """Get retriever for specific URL"""
        return self.retrievers.get(url)
    
    def get_all_sources(self):
        """Get list of all loaded content sources"""
        return list(self.retrievers.keys())

# ==================== INITIALIZE COMPONENTS ====================

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
llm = ChatGroq(temperature=0.4, model_name='Llama-3.3-70b-versatile',max_tokens=3000)
search = DuckDuckGoSearchRun()

# Global variables
youtube_retriever = None
content_manager = ContentManager(embedding_model)

# ==================== YOUTUBE PROCESSING ====================

def process_youtube_video(youtube_url: str, language: str = "en"):
    """Fetch transcript of a YouTube video and create an in-memory vectorstore."""
    global youtube_retriever

    # Extract video id
    video_id = youtube_url.split("v=")[-1].split("&")[0]

    # Create a temporary vector store
    vectorstore = Chroma(
        collection_name=video_id,
        embedding_function=embedding_model,
        persist_directory=None
    )

    documents = []
    ytt_api = YouTubeTranscriptApi()

    try:
        # Fetch transcript
        docs = ytt_api.fetch(video_id, languages=[language])
        print("Transcript fetched successfully!")

        # Step 1: Group snippets by minute
        minute_chunks = defaultdict(list)
        minute_starts = {}

        for snippet in docs:
            minute = int(snippet.start // 60)
            minute_chunks[minute].append(snippet.text)

            # Save the earliest start time per minute
            if minute not in minute_starts:
                minute_starts[minute] = snippet.start

        # Step 2: Create LangChain Document objects with HH:MM:SS timestamps
        for minute in sorted(minute_chunks.keys()):
            content = " ".join(minute_chunks[minute])

            # Format start time to HH:MM:SS
            seconds = int(minute_starts[minute])
            timestamp = str(timedelta(seconds=seconds))

            metadata = {
                "minute": minute,
                "start_timestamp": timestamp
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        # Add to vectorstore
        vectorstore.add_documents(documents)

        # Create simple retriever
        youtube_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "lambda_mult": 0.3}
        )

        return True

    except Exception as e:
        print(f"Error fetching transcript: {e}")
        youtube_retriever = None
        return False

# ==================== AUDIO GENERATION ====================

VOICE = "en-US-AriaNeural"

# Output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

async def _generate_audio_files(text):
    """Generate unique audio file for text response."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(output_dir, f"audio_{timestamp}.mp3")
    
    # Clean text for better TTS synthesis
    cleaned_text = text.strip()
    
    try:
        communicate = edge_tts.Communicate(cleaned_text, VOICE, rate='+0%', pitch="+1Hz")
        await communicate.save(filename)
        print(f"Audio saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def generate_audio_files(text):
    """Synchronous wrapper for audio generation."""
    return asyncio.run(_generate_audio_files(text))

# ==================== LANGGRAPH STATE & WORKFLOW ====================

from pydantic import BaseModel,Field
from typing import Literal 

class Routequery(BaseModel):
    decision : Literal["general_query","video_qa", "web_content"] = Field(
        ..., 
        description="Given a user question choose to route to `general_query` for greetings, `video_qa` for video analysis, or `web_content` for web content analysis"
    )

class State(TypedDict):
    """Represents the state of our graph"""
    messages: Annotated[list[AnyMessage], add_messages]
    audio_file: str = None
    generate_audio: bool = False
    current_source: str = None  # Track which content source we're using

def is_greeting_or_general(text: str) -> bool:
    """Check if the text is a general greeting, casual inquiry, or non-video related query."""
    general_patterns = [
        r'\b(hi|hello|hey|greetings?|good\s+(morning|afternoon|evening|day))\b',
        r'\bhow\s+(are|is)\s+you\b',
        r'\bwhat\'?s\s+up\b',
        r'\bhow\s+do\s+you\s+do\b',
        r'\bnice\s+to\s+(meet|see)\s+you\b',
        r'^\s*(hi|hello|hey)\s*[!.?]*\s*$',
        r'^\s*how\s+are\s+you\s*[!.?]*\s*$',
        r'\bwho\s+are\s+you\b',
        r'\bwhat\s+can\s+you\s+do\b',
        r'\btell\s+me\s+about\s+yourself\b'
    ]
    
    text_lower = text.lower().strip()
    return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in general_patterns)

# Enhanced Router Setup
system_msg_router = """You are an expert at routing user queries for a content analysis system.

Analyze the user's message and route it appropriately:

1. **general_query**: Route here if the user query is:
   - General greetings (like "hi", "hello", "how are you", etc.)
   - Casual conversation or small talk
   - Questions about the system's capabilities
   - General questions that don't require specific content analysis

2. **video_qa**: Route here if the user query is:
   - Asking for video summaries or analysis
   - Specific questions about video content
   - Timestamp-related queries
   - Content explanations from videos
   - Any analytical questions about video transcripts

3. **web_content**: Route here if the user query is:
   - About research papers, articles, or web content
   - Asking for analysis of documents, PDFs, or web pages
   - Questions about news articles or blog posts
   - Requests for information from loaded web content

Choose the most appropriate route based on the user's intent and available content sources."""

template_route = ChatPromptTemplate([
    ("system", system_msg_router),
    ("human", "{question}")
])
structured_llm_router = llm.with_structured_output(Routequery)
chain_router = template_route | structured_llm_router

def enhanced_route_query(state: State) -> dict:
    """Enhanced router that detects content type from queries and available sources."""
    user_msg = next(msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    current_source = state.get("current_source")
    
    # If we have a current source, route to appropriate handler
    if current_source:
        if "youtube.com" in current_source or "youtu.be" in current_source:
            return {"decision": "video_qa"}
        else:
            return {"decision": "web_content"}
    
    # First check if it's a simple greeting using regex
    if is_greeting_or_general(user_msg.content):
        return {"decision": "general_query"}
    
    # Check query content for hints
    query = user_msg.content.lower()
    
    # Video-related queries
    video_keywords = ["video", "youtube", "timestamp", "minute", "second", "watch", "view"]
    if any(keyword in query for keyword in video_keywords):
        return {"decision": "video_qa"}
    
    # Web content related queries
    web_keywords = ["pdf", "article", "research", "paper", "blog", "webpage", "website", "document"]
    if any(keyword in query for keyword in web_keywords):
        return {"decision": "web_content"}
    
    # Use LLM for complex routing
    intent = chain_router.invoke({"question": user_msg.content})
    return {"decision": intent.decision}

def handle_general_query(state: State) -> State:
    """Handle general queries, greetings, and casual conversation with conditional audio generation."""
    
    messages = state["messages"]
    generate_audio = state.get("generate_audio", False)
    user_msg = next(msg for msg in reversed(messages) if isinstance(msg, HumanMessage))
    
    # Prepare conversation history for context
    conversation_history = "\n".join(
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in messages[:-1]  # Exclude current message
    )
    
    general_system_msg = """You are a friendly AI assistant specialized in content analysis.

Your capabilities include:
- Analyzing YouTube video transcripts and providing detailed summaries
- Answering questions about research papers, news articles, and web content
- Engaging in general conversation and providing helpful information
- Assisting with various queries beyond content analysis

When responding to general queries, greetings, or casual conversation:
- Be warm, friendly, and helpful
- Provide informative and engaging responses
- If discussing your capabilities, mention your content analysis features
- Keep responses conversational and natural
- Be concise but comprehensive in your explanations
- Show enthusiasm for helping with their requests

Respond to the user's message in a natural, engaging way. Keep responses focused and helpful."""

    template_general = ChatPromptTemplate.from_messages([
        ("system", general_system_msg),
        ("human", f"Conversation History:\n{conversation_history}\n\nCurrent Message: {user_msg.content}")
    ])

    chain_general = template_general | llm
    response = chain_general.invoke({})
    
    # Generate audio only if requested
    audio_file = None
    if generate_audio:
        try:
            audio_file = generate_audio_files(response.content)
            print(f"Generated audio for general query: {audio_file}")
        except Exception as e:
            print(f"Error generating audio for general query: {e}")
    else:
        print("Audio generation skipped for general query (auto-play disabled)")
    
    return {
        "messages": messages + [response],
        "audio_file": audio_file
    }

def handle_video_qa(state: State) -> State:
    """Handle video analysis, summarization, and Q&A with conditional audio generation."""

    system_msg_video = """You're a helpful assistant that analyzes YouTube video transcripts and provides comprehensive answers.

Your tasks:
- Provide clear, detailed summaries or answers based on the video transcript
- Arrange content chronologically using the provided timestamps when relevant
- Answer specific questions about the video content with relevant timestamp references
- If asked for a summary, provide a well-structured overview of the main topics covered
- For specific questions, focus on the relevant sections and provide timestamp references when helpful

Guidelines:
- Always use only the transcript content for your answers
- Include relevant timestamps when discussing specific parts of the video (format: "At [timestamp]...")
- Keep your tone natural, informative, and engaging
- Provide comprehensive yet concise responses
- Return responses in English
- If the transcript doesn't contain information to answer the question, politely explain this
- Structure longer responses with clear organization
- Focus on the most important and relevant information"""

    # Get all messages (including history)
    messages = state["messages"]
    generate_audio = state.get("generate_audio", False)
    
    # Get last human message
    user_msg = next(msg for msg in reversed(messages) if isinstance(msg, HumanMessage))
    
    # Check if retriever is available
    global youtube_retriever
    if youtube_retriever is None:
        error_response = AIMessage(content="I'm sorry, but I need a YouTube video transcript to be loaded first before I can analyze video content. Please provide a YouTube URL and process it using the video loading function.")
        return {
            "messages": messages + [error_response],
            "audio_file": None
        }
    
    # Get transcript context
    try:
        result = youtube_retriever.invoke(user_msg.content)
        context_docs = []
        for doc in result:
            metadata = doc.metadata
            content = doc.page_content
            # extract metadata variables
            minute = metadata.get("minute")
            start_timestamp = metadata.get("start_timestamp")
            # append context with metadata + content
            context_docs.append(f"Start Time: {start_timestamp}, Minute: {minute}\n{content}")

        context = "\n\n".join(context_docs)
        
        # Prepare conversation history for context
        conversation_history = "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages[:-1]  # Exclude current message
        )

        template_video_response = ChatPromptTemplate.from_messages([
            ("system", system_msg_video),
            ("human", f"Conversation History:\n{conversation_history}\n\n"
                     f"Current Question: {user_msg.content}\n\n"
                     f"Transcript Context:\n{context}")
        ])

        chain_video_response = template_video_response | llm
        response = chain_video_response.invoke({})
        
    except Exception as e:
        print(f"Error processing video query: {e}")
        response = AIMessage(content=f"I encountered an error while processing your video-related query: {str(e)}. Please make sure a video transcript has been properly loaded.")
    
    # Generate audio only if requested
    audio_file = None
    if generate_audio:
        try:
            audio_file = generate_audio_files(response.content)
            print(f"Generated audio for video Q&A: {audio_file}")
        except Exception as e:
            print(f"Error generating audio for video Q&A: {e}")
    else:
        print("Audio generation skipped for video Q&A (auto-play disabled)")
    
    return {
        "messages": messages + [response],
        "audio_file": audio_file
    }

def handle_web_content(state: State) -> State:
    """Handle web content Q&A with conditional audio generation."""
    
    system_msg_web = """You are a helpful assistant that analyzes web content, research papers, news articles, and documents.

Your tasks:
- Provide clear, detailed summaries or answers based on the provided content
- For research papers, focus on methodology, findings, and implications
- For news articles, extract key facts, events, and context
- For PDF documents, provide comprehensive overviews and key insights
- For general web content, provide well-structured summaries

Guidelines:
- Always use only the provided content for your answers
- Keep your tone informative and engaging
- Structure longer responses with clear organization
- Be honest about limitations in the available content
- For research papers, highlight key contributions and methodology
- For news, focus on factual reporting and context"""

    messages = state["messages"]
    generate_audio = state.get("generate_audio", False)
    current_source = state.get("current_source")
    
    # Get last human message
    user_msg = next(msg for msg in reversed(messages) if isinstance(msg, HumanMessage))
    
    # Check if we have a content source
    if not current_source:
        error_response = AIMessage(content="Please load a web content source first using the web content loading function.")
        return {
            "messages": messages + [error_response],
            "audio_file": None,
            "current_source": None
        }
    
    # Get retriever for current source
    retriever = content_manager.get_retriever(current_source)
    if not retriever:
        error_response = AIMessage(content="No web content source is currently active. Please load web content first.")
        return {
            "messages": messages + [error_response],
            "audio_file": None,
            "current_source": None
        }
    
    # Get relevant context
    try:
        result = retriever.invoke(user_msg.content)
        context_docs = []
        
        for doc in result:
            metadata = doc.metadata
            content_type = metadata.get("content_type", "webpage")
            source = metadata.get("source", "Unknown")
            
            context_docs.append(f"SOURCE: {source}\nTYPE: {content_type}\nCONTENT: {doc.page_content}")

        context = "\n\n---\n\n".join(context_docs)
        
        # Prepare conversation history
        conversation_history = "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages[:-1]
        )

        template_web_response = ChatPromptTemplate.from_messages([
            ("system", system_msg_web),
            ("human", f"Conversation History:\n{conversation_history}\n\n"
                     f"Current Question: {user_msg.content}\n\n"
                     f"Content Context:\n{context}")
        ])

        chain_web_response = template_web_response | llm
        response = chain_web_response.invoke({})
        
    except Exception as e:
        print(f"Error processing web content query: {e}")
        response = AIMessage(content=f"I encountered an error while processing your query: {str(e)}")
    
    # Generate audio if requested
    audio_file = None
    if generate_audio:
        try:
            audio_file = generate_audio_files(response.content)
            print(f"Generated audio for web content: {audio_file}")
        except Exception as e:
            print(f"Error generating audio for web content: {e}")
    else:
        print("Audio generation skipped for web content (auto-play disabled)")
    
    return {
        "messages": messages + [response],
        "audio_file": audio_file,
        "current_source": current_source
    }

# ==================== BUILD WORKFLOW ====================

checkpointer = MemorySaver()
workflow = StateGraph(State)

# Add nodes
workflow.add_node("route_query", enhanced_route_query)
workflow.add_node("handle_general_query", handle_general_query)
workflow.add_node("handle_video_qa", handle_video_qa)
workflow.add_node("handle_web_content", handle_web_content)

# Set entry point to router
workflow.set_entry_point("route_query")

# Add conditional edges from router to the handlers
workflow.add_conditional_edges(
    "route_query",
    lambda state: state.get("decision", "general_query"),
    {
        "general_query": "handle_general_query",
        "video_qa": "handle_video_qa",
        "web_content": "handle_web_content"
    }
)

# Add edges from each node to END
workflow.add_edge("handle_general_query", END)
workflow.add_edge("handle_video_qa", END)
workflow.add_edge("handle_web_content", END)

# Compile the app
app = workflow.compile(checkpointer=checkpointer)

print("Enhanced Content Analysis Workflow is ready!")
print("\nThe workflow has three main capabilities:")
print("1. General Query Handler - For greetings, casual conversation, and general questions")
print("2. Video Q&A Handler - For video analysis, summarization, and content-based questions") 
print("3. Web Content Handler - For research papers, news articles, PDFs, and web content")
print("4. Audio files are conditionally generated based on user preference")