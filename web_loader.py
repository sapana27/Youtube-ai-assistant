# web_loader.py
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import arxiv
from newspaper import Article
from trafilatura import fetch_url, extract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

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