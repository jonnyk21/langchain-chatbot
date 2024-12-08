import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, PDFMinerLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import time
import requests
from bs4 import BeautifulSoup
from agent_graph import GraphChatbot

# Load environment variables
load_dotenv()

class Chatbot:
    def __init__(self, use_graph=False):
        load_dotenv()
        self.personalities = {
            "helpful": "You are a helpful AI assistant.",
            "professional": "You are a professional AI assistant focused on business and technical topics.",
            "creative": "You are a creative AI assistant that thinks outside the box.",
            "teacher": "You are a patient teacher that explains concepts clearly and thoroughly."
        }
        self.current_personality = "helpful"
        self.use_graph = use_graph
        self.graph_chatbot = GraphChatbot() if use_graph else None
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize OpenAI client with streaming
        self.chat = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME"),
            temperature=0.7,
            streaming=True
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize RAG system once
        if not self.use_graph:
            try:
                self.initialize_rag_system()
            except Exception as e:
                print(f"Error initializing RAG system: {str(e)}")
        
        # Load HNU knowledge and set system message
        self.hnu_knowledge = self.load_hnu_knowledge()
        self.messages = [SystemMessage(content=self.get_hnu_system_message())]
        
    def load_hnu_knowledge(self):
        """Load HNU-specific knowledge from scraped documents"""
        knowledge = {
            "programs": {
                "bachelor": [
                    "Business Studies (BA)",
                    "Business Studies in Healthcare Management (BA)",
                    "Data Science Management (BSc)",
                    "Digital Enterprise Management (BSc)",
                    "Digital Management in Medicine and Care (BSc)",
                    "Game Production and Management (BA)",
                    "Health Informatics (BSc) (part-time)",
                    "Information Management Automotive (BSc)",
                    "Information Management in Healthcare (BSc)",
                    "Information Management and Corporate Communications (BA)",
                    "Artificial Intelligence and Information Management (BSc)",
                    "Physician Assistant (BSc)",
                    "Systems Engineering (BEng)",
                    "Industrial Engineering (BEng)",
                    "Business Psychology (BSc)"
                ],
                "master": [
                    "Advanced Management (MSc)",
                    "Advanced Sales Management and Intelligence (MSc)",
                    "Artificial Intelligence and Data Analytics (MSc)",
                    "Communication & Design for Sustainability (MA)",
                    "Digital Healthcare Management (MA)",
                    "Digital Innovation Management (MSc)",
                    "Digital Transformation and Global Entrepreneurship (MSc)",
                    "International Corporate Communication and Media Management (MA)",
                    "International Entrepreneurship, Digitalization and Sustainability (MSc)",
                    "Social Entrepreneurship for Sustainable Development (MA)",
                    "Strategic Information Management (MSc)"
                ],
                "part_time": {
                    "bachelor": ["Management for Health and Nursing Professions (BA)"],
                    "master": [
                        "Digital Leadership and IT-Management (MBA)",
                        "Leadership and Management in Healthcare (MBA)",
                        "General Management (MBA)"
                    ]
                }
            }
        }
        return knowledge

    def get_hnu_system_message(self):
        return f"""You are a helpful AI assistant for Hochschule Neu-Ulm (HNU) University of Applied Sciences. 
        You help students, prospective students, and visitors with questions about the university.
        
        Here are the programs offered at HNU:
        
        Bachelor's Programs:
        {', '.join(self.hnu_knowledge['programs']['bachelor'])}
        
        Master's Programs:
        {', '.join(self.hnu_knowledge['programs']['master'])}
        
        Part-time Programs:
        - Bachelor: {', '.join(self.hnu_knowledge['programs']['part_time']['bachelor'])}
        - Master: {', '.join(self.hnu_knowledge['programs']['part_time']['master'])}
        
        You should:
        1. Provide accurate information about HNU's programs, admissions, and campus life
        2. Be friendly and helpful
        3. If you're not sure about something, say so and suggest where to find the information
        4. Respond in the same language as the question (English or German)
        """

    def initialize_rag_system(self):
        """Initialize the RAG system with document loading and embedding"""
        try:
            if not os.path.exists("docs"):
                os.makedirs("docs")
                
            # Load text documents
            txt_loader = DirectoryLoader(
                "docs",
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            
            # Load PDF documents
            pdf_loader = DirectoryLoader(
                "docs",
                glob="**/*.pdf",
                loader_cls=PDFMinerLoader
            )
            
            # Load HTML documents
            html_loader = DirectoryLoader(
                "docs",
                glob="**/*.html",
                loader_cls=BSHTMLLoader
            )
            
            # Combine documents from all loaders
            documents = txt_loader.load() + pdf_loader.load() + html_loader.load()
            
            if not documents:
                print("No documents found in the docs directory.")
                self.vectorstore = None
                self.qa_chain = None
                return
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Ensure db directory exists
            if not os.path.exists("db"):
                os.makedirs("db")
            
            # Create vector store
            collection_name = "hnu_docs"
            try:
                # Try to load existing collection first
                self.vectorstore = Chroma(
                    persist_directory="db",
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                
                # Only add documents if the collection is empty
                if len(self.vectorstore.get()['ids']) == 0:
                    self.vectorstore.add_documents(splits)
                    print(f"Added {len(splits)} documents to the vector store.")
                else:
                    print(f"Using existing vector store with {len(self.vectorstore.get()['ids'])} documents.")
                
            except Exception as e:
                print(f"Error with existing collection: {str(e)}")
                print("Creating new vector store...")
                # If loading fails, create a new one from documents
                self.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory="db",
                    collection_name=collection_name
                )
            
            # Create system prompt
            system_prompt = """You are a helpful AI assistant. Use information from the provided documents to answer questions accurately. Do not mention citations in your main response.

Given the following conversation and a question, create a final answer.

Chat History: {chat_history}
Human: {question}
Assistant: Let me help you with that.

{context}

Based on this information, here's my response:"""
            
            # Create retrieval chain with basic configuration
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.chat,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                verbose=False,
                chain_type="stuff",
                combine_docs_chain_kwargs={
                    "prompt": ChatPromptTemplate.from_template(system_prompt)
                }
            )
            
            print(f"RAG system initialized successfully with {len(documents)} documents!")
            
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            self.vectorstore = None
            self.qa_chain = None

    def change_personality(self, personality_type):
        if personality_type in self.personalities:
            self.current_personality = personality_type
            self.messages = [SystemMessage(content=self.personalities[personality_type])]
            return f"Personality changed to {personality_type}"
        return "Invalid personality type"
    
    def get_response(self, user_input):
        if user_input.lower() == 'quit':
            return "Goodbye!"
        
        if self.use_graph:
            return self.graph_chatbot.get_response(user_input)
        
        if self.qa_chain is not None:
            try:
                # Get response from RAG system using invoke
                result = self.qa_chain.invoke({
                    "question": user_input,
                    "chat_history": []
                })
                response = result["answer"]
                
                # Add source citations if documents were used
                if "source_documents" in result and result["source_documents"]:
                    # Use a set to store unique sources
                    sources = {
                        os.path.basename(doc.metadata.get("source", ""))
                        for doc in result["source_documents"]
                        if hasattr(doc, "metadata") and "source" in doc.metadata
                    }
                    # Remove empty strings
                    sources.discard("")
                    
                    if sources:
                        response += "\n\nSource" + ("s" if len(sources) > 1 else "") + ": "
                        response += ", ".join(f"[{source}]" for source in sorted(sources))
                
                # Add messages to history
                self.messages.append(HumanMessage(content=user_input))
                self.messages.append(AIMessage(content=response))
                
                return response
            except Exception as e:
                print(f"Error using RAG system: {str(e)}")
                # Fall back to regular chat if RAG fails
                pass
        
        # Regular chat without RAG
        try:
            # Add user message
            self.messages.append(HumanMessage(content=user_input))
            
            # Get response from chatbot
            response = self.chat.invoke(self.messages)
            
            # Add assistant message
            self.messages.append(AIMessage(content=response.content))
            
            return response.content
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_streaming_response(self, user_input):
        """Get streaming response from the chatbot"""
        if self.use_graph:
            return self.graph_chatbot.get_response(user_input)
            
        if self.qa_chain is not None:
            try:
                # Get response from RAG system using invoke
                result = self.qa_chain.invoke({
                    "question": user_input,
                    "chat_history": []
                })
                response = result["answer"]
                
                # Add source citations if documents were used
                if "source_documents" in result and result["source_documents"]:
                    # Use a set to store unique sources
                    sources = {
                        os.path.basename(doc.metadata.get("source", ""))
                        for doc in result["source_documents"]
                        if hasattr(doc, "metadata") and "source" in doc.metadata
                    }
                    # Remove empty strings
                    sources.discard("")
                    
                    if sources:
                        response += "\n\nSource" + ("s" if len(sources) > 1 else "") + ": "
                        response += ", ".join(f"[{source}]" for source in sorted(sources))
                
                # Add messages to history
                self.messages.append(HumanMessage(content=user_input))
                self.messages.append(AIMessage(content=response))
                
                # Yield response in chunks for streaming effect
                words = response.split()
                for i in range(0, len(words), 3):
                    chunk = " ".join(words[i:i+3]) + " "
                    yield chunk
                return
            except Exception as e:
                print(f"Error using RAG system: {str(e)}")
                # Fall back to regular chat if RAG fails
                pass
        
        # Regular chat without RAG
        try:
            # Add user message to history
            self.messages.append(HumanMessage(content=user_input))
            
            # Initialize an empty string to collect the full response
            full_response = ""
            
            # Get streaming response from chatbot
            for chunk in self.chat.stream(self.messages):
                if chunk.content is not None:
                    full_response += chunk.content
                    yield chunk.content
            
            # After streaming is done, add the full message to history
            self.messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            yield error_message
            self.messages.append(AIMessage(content=error_message))
    
    def scrape_website(self, url: str, use_selenium: bool = False) -> str:
        """Scrape content from a website and save it as a document"""
        try:
            filename = url.split('/')[-1]
            if not filename.endswith('.html'):
                filename = 'webpage_' + str(int(time.time())) + '.html'
            filepath = os.path.join('docs', filename)
            
            if use_selenium:
                try:
                    # Try importing required Selenium packages
                    from selenium import webdriver
                    from selenium.webdriver.chrome.service import Service
                    from selenium.webdriver.chrome.options import Options
                    from webdriver_manager.chrome import ChromeDriverManager
                except ImportError:
                    return "Selenium dependencies not found. Please install selenium and webdriver-manager packages."

                try:
                    # Set up Chrome options
                    options = Options()
                    options.add_argument('--headless')
                    options.add_argument('--no-sandbox')
                    options.add_argument('--disable-dev-shm-usage')
                    
                    try:
                        service = Service(ChromeDriverManager().install())
                        driver = webdriver.Chrome(service=service, options=options)
                    except Exception as e:
                        if "chrome not reachable" in str(e).lower():
                            return "Chrome browser not found. Please install Google Chrome or Chromium."
                        elif "chromedriver" in str(e).lower():
                            return "ChromeDriver not found. Please install chromium-chromedriver package."
                        else:
                            return f"Error setting up Chrome: {str(e)}"
                    
                    try:
                        driver.get(url)
                        time.sleep(2)  # Wait for JavaScript content to load
                        content = driver.page_source
                    finally:
                        driver.quit()
                except Exception as e:
                    return f"Error during Selenium scraping: {str(e)}"
            else:
                try:
                    # Try importing requests
                    import requests
                except ImportError:
                    return "Requests package not found. Please install the requests package."

                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    content = response.text
                except requests.exceptions.SSLError:
                    return "SSL verification failed. The website might be using an invalid certificate."
                except requests.exceptions.ConnectionError:
                    return "Failed to connect to the website. Please check your internet connection."
                except requests.exceptions.Timeout:
                    return "Request timed out. The website might be slow or unreachable."
                except requests.exceptions.RequestException as e:
                    return f"Error fetching the website: {str(e)}"
            
            # Save the content
            try:
                os.makedirs('docs', exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            except IOError as e:
                return f"Error saving content: {str(e)}"
            
            try:
                # Try importing BeautifulSoup
                from bs4 import BeautifulSoup
                
                # Parse and clean HTML content
                soup = BeautifulSoup(content, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                # Get text content
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Save cleaned text
                text_filepath = os.path.splitext(filepath)[0] + '.txt'
                with open(text_filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Load and process the new content
                loader = TextLoader(text_filepath)
                documents = loader.load()
            except ImportError:
                return "BeautifulSoup4 package not found. Please install beautifulsoup4 package."
            except Exception as e:
                return f"Error processing HTML content: {str(e)}"
            
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)
                
                if self.vectorstore is None:
                    self.vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=self.embeddings
                    )
                    self.qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=self.chat,
                        retriever=self.vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        ),
                        return_source_documents=True,
                        verbose=False
                    )
                else:
                    self.vectorstore.add_documents(splits)
                
                return f"Successfully scraped and added content from {url}"
            else:
                return f"No content could be extracted from {url}"
            
        except Exception as e:
            return f"Error scraping website: {str(e)}"
    
    def add_document(self, file_path: str) -> str:
        """Add a new document to the RAG system"""
        try:
            if not os.path.exists(file_path):
                return "File not found"
            
            # Choose appropriate loader based on file extension
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.pdf':
                loader = PDFMinerLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension == '.html':
                loader = BSHTMLLoader(file_path)
            else:
                return f"Unsupported file type: {file_extension}"
            
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            if self.vectorstore is not None:
                self.vectorstore.add_documents(splits)
                return f"Successfully added document: {file_path}"
            else:
                # Initialize vector store if it doesn't exist
                self.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings
                )
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.chat,
                    retriever=self.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    ),
                    return_source_documents=True,
                    verbose=False
                )
                return f"Initialized RAG system with document: {file_path}"
            
        except Exception as e:
            return f"Error adding document: {str(e)}"

def main():
    # Create chatbot instance
    chatbot = Chatbot(use_graph=True)
    
    print("Chatbot: Hello! I'm your AI assistant. How can I help you today?")
    print("Commands:")
    print("- Type 'quit' to exit")
    print("- Type 'personality <type>' to change personality")
    print("- Type 'reset' to clear conversation history")
    print("- Type 'add_doc <path>' to add a document to the knowledge base")
    print("- Type 'scrape <url>' to add web content (use 'scrape-js <url>' for JavaScript content)")
    print("Available personalities:", ", ".join(chatbot.personalities.keys()))
    print("\nSupported document types: .txt, .pdf, .html")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Process commands
            if user_input.lower() == 'quit':
                print("\nChatbot: Goodbye!")
                break
            elif user_input.lower() == 'reset':
                chatbot.messages = [SystemMessage(content=chatbot.personalities[chatbot.current_personality])]
                print("\nChatbot: Conversation history has been cleared.")
                continue
            elif user_input.lower().startswith('personality '):
                new_personality = user_input.split(' ')[1].lower()
                print("\nChatbot:", chatbot.change_personality(new_personality))
                continue
            elif user_input.lower().startswith('add_doc '):
                file_path = user_input[8:].strip()
                print("\nChatbot:", chatbot.add_document(file_path))
                continue
            elif user_input.lower().startswith('scrape-js '):
                url = user_input[9:].strip()
                print("\nChatbot: Scraping website with JavaScript support...")
                print(chatbot.scrape_website(url, use_selenium=True))
                continue
            elif user_input.lower().startswith('scrape '):
                url = user_input[7:].strip()
                print("\nChatbot: Scraping website...")
                print(chatbot.scrape_website(url))
                continue
            
            # Get and print chatbot response
            print("\nChatbot: ", end="", flush=True)
            response = chatbot.get_streaming_response(user_input)
            
            # Simulate typing effect
            for chunk in response:
                print(chunk, end="", flush=True)
                time.sleep(0.01)
            print()
            
        except KeyboardInterrupt:
            print("\nChatbot: Goodbye!")
            break
        except EOFError:
            print("\nChatbot: Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()
