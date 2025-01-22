from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from typing import List, Dict, Union, Any
import json
import os
import requests
from urllib.parse import urlparse
import logging
from logging_config import setup_logging
import pickle
from datetime import datetime
from bs4 import BeautifulSoup
import re

# Initialize logging
logger = setup_logging()

# Import the model caller using absolute import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_caller.modelCaller import OllamaClient

class TaskParser:
    def __init__(self, model_client: OllamaClient):
        self.model_client = model_client
        self.logger = logging.getLogger(__name__ + ".TaskParser")
        
    def parse_tasks(self, query: str) -> List[str]:
        """Parse user query into specific action items"""
        self.logger.info(f"Parsing query into tasks: {query}")
        system_prompt = """You are a task analyzer. Given a user query, break it down into specific actionable tasks.
        Return the tasks as a JSON array of strings. Each task should be clear and executable."""
        
        try:
            response = self.model_client.generate(
                prompt=query,
                system=system_prompt,
                temperature=0.3
            )
            self.logger.debug(f"Raw model response: {response}")
            
            tasks = json.loads(response)
            if isinstance(tasks, list):
                self.logger.info(f"Successfully parsed {len(tasks)} tasks")
                return tasks
            else:
                self.logger.warning("Model response was not a list, returning empty list")
                return []
        except Exception as e:
            self.logger.error(f"Error parsing tasks: {str(e)}")
            return []

class WebKnowledgeBase:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__ + ".WebKnowledgeBase")
        self.logger.info(f"Initialized WebKnowledgeBase with base URL: {self.base_url}")
        self.cache_dir = "knowledge_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{self._get_domain_name()}_cache.pkl")
        self.content_cache = self._load_cache()
        self.visited_urls = set()  # Track visited URLs to avoid cycles
        self.max_depth = 10  # Increased depth for better coverage
        
    def _get_domain_name(self) -> str:
        """Extract domain name from URL"""
        return urlparse(self.base_url).netloc.replace(".", "_")
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load cached content if it exists"""
        if os.path.exists(self.cache_file):
            self.logger.info(f"Loading cached content from {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Add visited_urls from cache if it exists
                    if 'visited_urls' in cache_data:
                        self.visited_urls = cache_data['visited_urls']
                    return cache_data.get('content_cache', {})
            except Exception as e:
                self.logger.error(f"Error loading cache: {str(e)}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save content and visited URLs to cache"""
        try:
            cache_data = {
                'content_cache': self.content_cache,
                'visited_urls': self.visited_urls
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            self.logger.info(f"Saved content to cache: {self.cache_file}")
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")

    def _clean_html_content(self, html_content: str, url: str) -> tuple[str, list[str]]:
        """Clean HTML content and extract meaningful text and links"""
        soup = BeautifulSoup(html_content, 'html.parser')
        base_url_parts = urlparse(self.base_url)
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Extract internal links before getting text
        internal_links = []
        for link in soup.find_all(['a', 'link'], href=True):
            href = link['href']
            # Skip anchor links and empty hrefs
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
                
            # Handle relative URLs
            if href.startswith('/'):
                href = f"{self.base_url}{href}"
            elif not href.startswith(('http://', 'https://')):
                href = f"{self.base_url}/{href}"
                
            # Only include links from the same domain
            if urlparse(href).netloc == base_url_parts.netloc:
                internal_links.append(href)
        
        # Extract text and clean it
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up excessive newlines and spaces
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        return text, list(set(internal_links))  # Remove duplicate links
    
    def _should_follow_link(self, url: str) -> bool:
        """Determine if a link should be followed"""
        if url in self.visited_urls:
            return False
            
        parsed_url = urlparse(url)
        base_url_parts = urlparse(self.base_url)
        
        # Only follow links within the same domain
        if parsed_url.netloc != base_url_parts.netloc:
            return False
            
        # Skip common non-content paths
        skip_patterns = [
            '/static/', '/assets/', '/images/', '/css/', '/js/',
            '.png', '.jpg', '.jpeg', '.gif', '.css', '.js', '.ico'
        ]
        
        return not any(pattern in parsed_url.path.lower() for pattern in skip_patterns)
    
    def _extract_urls_from_text(self, text: str) -> list[str]:
        """Extract both full URLs and relative paths from text"""
        urls = []
        base_url_parts = urlparse(self.base_url)
        
        # Match full URLs and relative paths
        patterns = [
            r'https?://[^\s<>"]+',  # Full URLs
            r'(?<=[\'"])/[^\s<>"\']+',  # Relative paths
            r'(?<=\s)/[a-zA-Z0-9-_/]+(?=[\s,.]|$)'  # Bare paths
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match.startswith('/'):
                    urls.append(f"{self.base_url}{match}")
                else:
                    # Only include URLs from the same domain
                    if urlparse(match).netloc == base_url_parts.netloc:
                        urls.append(match)
        
        return list(set(urls))  # Remove duplicates

    def fetch_content(self, path: str = "", current_depth: int = 0) -> Dict[str, Any]:
        """Fetch content from cache or website with related content"""
        url = f"{self.base_url}/{path.lstrip('/')}" if path else self.base_url
        
        # Check if we've hit the depth limit
        if current_depth > self.max_depth:
            return {'content': '', 'related_content': {}}
        
        # Check if we've already visited this URL
        if url in self.visited_urls:
            return {'content': self.content_cache.get(url, {}).get('content', ''),
                   'related_content': self.content_cache.get(url, {}).get('related_content', {})}
        
        # Mark URL as visited
        self.visited_urls.add(url)
        
        # Check cache first
        if url in self.content_cache:
            cache_entry = self.content_cache[url]
            age_hours = (datetime.now() - cache_entry['timestamp']).total_seconds() / 3600
            if age_hours < 24:  # Cache valid for 24 hours
                self.logger.info(f"Using cached content for {url}")
                return cache_entry
        
        self.logger.info(f"Fetching fresh content from: {url}")
        try:
            response = requests.get(url, timeout=10)  # Added timeout
            response.raise_for_status()
            content, internal_links = self._clean_html_content(response.text, url)
            
            # Fetch related content from internal links
            related_content = {}
            for link in internal_links:
                if self._should_follow_link(link):
                    self.logger.info(f"Following internal link: {link}")
                    link_path = urlparse(link).path
                    related_data = self.fetch_content(link_path, current_depth + 1)
                    if related_data['content']:
                        related_content[link] = related_data['content']
            
            # Cache the content and related content
            cache_entry = {
                'content': content,
                'related_content': related_content,
                'timestamp': datetime.now()
            }
            self.content_cache[url] = cache_entry
            self._save_cache()
            
            return cache_entry
            
        except Exception as e:
            self.logger.error(f"Error fetching content from {url}: {str(e)}")
            return {'content': '', 'related_content': {}}

    def fetch_page_content(self, page_url: str) -> str:
        """Fetch content from a specific documentation page with related content"""
        if not page_url.startswith('http'):
            # Convert relative URL to absolute URL
            page_url = f"{self.base_url}/{page_url.lstrip('/')}"
            
        self.logger.info(f"Fetching content from page: {page_url}")
        result = self.fetch_content(urlparse(page_url).path)
        
        # Combine main content with related content
        full_content = [f"Content from {page_url}:\n{result['content']}"]
        
        for url, related_content in result.get('related_content', {}).items():
            full_content.append(f"\nRelated information from {url}:\n{related_content}")
        
        return '\n'.join(full_content)

    def get_all_content(self) -> List[Dict[str, str]]:
        """Get all content from knowledge base"""
        self.logger.info("Getting all content from knowledge base")
        result = self.fetch_content()
        
        if not result['content']:
            self.logger.warning("No content retrieved from knowledge base")
            return []
        
        # Combine main content with all related content
        all_content = [{'content': result['content'], 'source': self.base_url}]
        
        for url, related_content in result.get('related_content', {}).items():
            all_content.append({'content': related_content, 'source': url})
        
        total_chars = sum(len(doc['content']) for doc in all_content)
        self.logger.info(f"Retrieved {total_chars} characters of content from {len(all_content)} pages")
        
        return all_content

class RAGAgent:
    def __init__(self, knowledge_source: Union[str, WebKnowledgeBase]):
        self.logger = logging.getLogger(__name__ + ".RAGAgent")
        self.logger.info("Initializing RAG Agent")
        
        self.embeddings = HuggingFaceEmbeddings()
        self.model_client = OllamaClient()
        self.task_parser = TaskParser(self.model_client)
        
        # Setup cache directories
        self.index_cache_dir = "faiss_cache"
        os.makedirs(self.index_cache_dir, exist_ok=True)
        
        if isinstance(knowledge_source, str):
            self.logger.info(f"Using local directory as knowledge source: {knowledge_source}")
            self.knowledge_source = None
            self.knowledge_dir = knowledge_source
            self.index_cache_file = os.path.join(self.index_cache_dir, "local_index")
        else:
            self.logger.info("Using web-based knowledge source")
            self.knowledge_source = knowledge_source
            self.knowledge_dir = "./web_knowledge"
            # Use domain name in cache file name
            domain = urlparse(knowledge_source.base_url).netloc.replace(".", "_")
            self.index_cache_file = os.path.join(self.index_cache_dir, f"{domain}_index")
            
        self.setup_vectorstore()
    
    def _should_rebuild_index(self) -> bool:
        """Check if we need to rebuild the index"""
        if not os.path.exists(self.index_cache_file):
            self.logger.info("No cached index found, will build new one")
            return True
            
        # Check index age
        index_time = datetime.fromtimestamp(os.path.getmtime(self.index_cache_file))
        age_hours = (datetime.now() - index_time).total_seconds() / 3600
        
        if age_hours > 24:  # Rebuild if older than 24 hours
            self.logger.info(f"Cached index is {age_hours:.1f} hours old, will rebuild")
            return True
            
        self.logger.info(f"Using cached index from {index_time}")
        return False
    
    def setup_vectorstore(self):
        """Initialize the vector store with documents"""
        self.logger.info("Setting up vector store")
        
        try:
            # Try to load cached index first
            if not self._should_rebuild_index():
                try:
                    self.vectorstore = FAISS.load_local(
                        self.index_cache_file, 
                        self.embeddings,
                        allow_dangerous_deserialization=True  # We trust our own cache
                    )
                    self.logger.info("Successfully loaded cached FAISS index")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to load cached index: {str(e)}, will rebuild")
            
            # If we get here, we need to build a new index
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            texts = []
            metadata = []
            
            if self.knowledge_source:
                self.logger.info("Fetching content from web knowledge base")
                documents = self.knowledge_source.get_all_content()
                for doc in documents:
                    chunks = text_splitter.split_text(doc['content'])
                    texts.extend(chunks)
                    metadata.extend([{'source': doc['source']} for _ in chunks])
                self.logger.info(f"Created {len(chunks)} chunks from web content")
            else:
                self.logger.info(f"Reading documents from {self.knowledge_dir}")
                for root, _, files in os.walk(self.knowledge_dir):
                    for file in files:
                        if file.endswith('.txt') or file.endswith('.md'):
                            filepath = os.path.join(root, file)
                            self.logger.debug(f"Processing file: {filepath}")
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                                chunks = text_splitter.split_text(content)
                                texts.extend(chunks)
                                metadata.extend([{'source': filepath} for _ in chunks])
                            self.logger.info(f"Created {len(chunks)} chunks from {filepath}")
            
            if texts:
                self.logger.info(f"Creating FAISS index with {len(texts)} text chunks")
                self.vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadata
                )
                # Save the index
                self.logger.info(f"Saving FAISS index to {self.index_cache_file}")
                self.vectorstore.save_local(self.index_cache_file)
            else:
                raise ValueError("No content found in knowledge source")
                
        except Exception as e:
            self.logger.error(f"Error setting up vector store: {str(e)}")
            raise
    
    def _search_internet(self, query: str) -> str:
        """Search internet for additional information"""
        self.logger.info(f"Searching internet for: {query}")
        try:
            # Use a search engine API or web scraping to get information
            search_url = f"https://www.google.com/search?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract relevant text from search results
            search_results = []
            for div in soup.find_all('div', class_=re.compile(r'g|result')):
                text = div.get_text()
                if text and len(text.strip()) > 50:  # Filter out short snippets
                    search_results.append(text.strip())
            
            return "\n\n".join(search_results[:3])  # Return top 3 results
        except Exception as e:
            self.logger.error(f"Error searching internet: {str(e)}")
            return ""

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context from vector store"""
        self.logger.info(f"Getting relevant context for query: {query}")
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            self.logger.info(f"Found {len(docs)} relevant documents")
            
            context_parts = []
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                context_parts.append(f"Source: {source}\n{doc.page_content}")
            
            context = "\n\n---\n\n".join(context_parts)
            self.logger.debug(f"Generated context of length {len(context)}")
            return context
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            return ""
    
    def generate_response(self, query: str, context: str = "", use_internet: bool = True) -> str:
        """Generate a response using the LLM"""
        self.logger.info("Generating response")
        
        # Extract URLs from context and query
        urls = self.knowledge_source._extract_urls_from_text(context)
        urls.extend(self.knowledge_source._extract_urls_from_text(query))
        
        # If we find documentation URLs, fetch their content
        additional_context = []
        for url in urls:
            if "docs.flytbase.com" in url:
                self.logger.info(f"Fetching additional content from: {url}")
                page_content = self.knowledge_source.fetch_page_content(url)
                if page_content:
                    additional_context.append(page_content)
        
        if additional_context:
            context = f"{context}\n\nAdditional Information:\n" + "\n\n".join(additional_context)
        
        if not context and use_internet:
            self.logger.info("No context found in knowledge base, searching internet")
            internet_context = self._search_internet(query)
            if internet_context:
                context = f"Additional information from internet search:\n{internet_context}"
        
        try:
            system_prompt = """You are a helpful AI assistant with expertise in FlytBase and drone technologies. 
            When responding:
            1. If using information from the context, cite it
            2. If using general knowledge, mention that
            3. Keep responses clear and concise
            4. If you're not sure about something, say so
            5. If you find references to documentation pages, make sure to include their content in your response
            """
            
            prompt = f"""Based on the following context, answer the user's query. If the context contains URLs, 
            I've already fetched their content for you. Please provide a detailed explanation using all available information.
            
            Context: {context}
            
            Query: {query}
            """
            
            response = self.model_client.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.7
            )
            
            self.logger.info("Successfully generated response")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def process_query(self, user_query: str) -> str:
        """Process user query and return response"""
        self.logger.info(f"Processing user query: {user_query}")
        
        try:
            # Get relevant context from our knowledge base
            context = self.get_relevant_context(user_query)
            
            # Generate response
            response = self.generate_response(user_query, context)
            self.logger.info("Successfully processed query")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return f"Error processing your query: {str(e)}"
