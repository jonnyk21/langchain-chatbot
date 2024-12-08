from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from deep_translator import GoogleTranslator
import operator
from dotenv import load_dotenv
import os
import langdetect

# Load environment variables
load_dotenv()

# Define the state of our graph
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    next: str
    context: List[str]
    current_topic: str
    language: str
    agent_type: str

# Knowledge base for HNU FAQs (now bilingual)
HNU_FAQ_KNOWLEDGE = {
    "en": """
Hochschule Neu-Ulm (HNU) - Key Information:

Location and Contact:
- Address: Wileystraße 1, 89231 Neu-Ulm, Germany
- Phone: +49 731 9762-0
- Email: info@hnu.de

Programs:
1. Bachelor's Programs:
   - Business Administration
   - Information Management and Corporate Communications
   - Industrial Engineering
   - Information Management Automotive

2. Master's Programs:
   - Advanced Management
   - Digital Innovation Management
   - International Enterprise Information Management

Admission Requirements:
- Higher education entrance qualification
- For international students: German language proficiency (usually B2 level)
- Program-specific requirements vary

Important Dates:
- Winter semester: October to March
- Summer semester: March to September
- Application deadlines vary by program

Student Services:
- International Office for exchange students
- Career Center
- Library and Learning Resources
- IT Services
- Student Housing Support

Campus Facilities:
- Modern lecture halls
- Computer labs
- Library
- Cafeteria
- Sports facilities

Research Focus Areas:
- Digital Transformation
- Sustainable Business Practices
- Healthcare Management
- Information Systems
""",
    "de": """
Hochschule Neu-Ulm (HNU) - Wichtige Informationen:

Standort und Kontakt:
- Adresse: Wileystraße 1, 89231 Neu-Ulm, Deutschland
- Telefon: +49 731 9762-0
- E-Mail: info@hnu.de

Studiengänge:
1. Bachelor-Studiengänge:
   - Betriebswirtschaft
   - Informationsmanagement und Unternehmenskommunikation
   - Wirtschaftsingenieurwesen
   - Informationsmanagement Automotive

2. Master-Studiengänge:
   - Advanced Management
   - Digital Innovation Management
   - International Enterprise Information Management

Zulassungsvoraussetzungen:
- Hochschulzugangsberechtigung
- Für internationale Studierende: Deutsche Sprachkenntnisse (in der Regel B2-Niveau)
- Studiengangspezifische Anforderungen variieren

Wichtige Termine:
- Wintersemester: Oktober bis März
- Sommersemester: März bis September
- Bewerbungsfristen variieren je nach Studiengang

Studierendenservice:
- International Office für Austauschstudierende
- Career Center
- Bibliothek und Lernressourcen
- IT-Services
- Unterstützung bei der Wohnungssuche

Campus-Einrichtungen:
- Moderne Hörsäle
- Computerlabore
- Bibliothek
- Mensa
- Sportanlagen

Forschungsschwerpunkte:
- Digitale Transformation
- Nachhaltige Geschäftspraktiken
- Gesundheitsmanagement
- Informationssysteme
"""
}

# Knowledge base for specialized agents
ADMISSIONS_KNOWLEDGE = {
    "en": """
Admissions at HNU:

Application Process:
- Online application through the HNU portal
- Required documents: certificates, CV, language certificates
- Application deadlines: 
  * Winter semester: July 15
  * Summer semester: January 15

Requirements:
- General university entrance qualification
- Program-specific requirements
- Language requirements:
  * German programs: DSH-2 or TestDaF (level 4)
  * International programs: IELTS (6.5) or equivalent

Contact:
- Email: admissions@hnu.de
- Phone: +49 731 9762-2001
""",
    "de": """
Zulassung an der HNU:

Bewerbungsprozess:
- Online-Bewerbung über das HNU-Portal
- Erforderliche Unterlagen: Zeugnisse, Lebenslauf, Sprachzertifikate
- Bewerbungsfristen:
  * Wintersemester: 15. Juli
  * Sommersemester: 15. Januar

Voraussetzungen:
- Allgemeine Hochschulreife
- Studiengangspezifische Anforderungen
- Sprachanforderungen:
  * Deutsche Programme: DSH-2 oder TestDaF (Niveau 4)
  * Internationale Programme: IELTS (6.5) oder gleichwertig

Kontakt:
- E-Mail: zulassung@hnu.de
- Telefon: +49 731 9762-2001
"""
}

COURSE_REGISTRATION_KNOWLEDGE = {
    "en": """
Course Registration at HNU:

Registration Process:
- Log in to PRIMUSS Campus IT Portal
- Select courses during registration period
- Check prerequisites and capacity restrictions
- Confirm selection and save changes

Important Dates:
- Registration period: 2 weeks before semester
- Add/Drop period: First 2 weeks of semester
- Withdrawal deadline: 6 weeks before exams

Support:
- Email: registration@hnu.de
- Student Office hours: Mon-Fri 9:00-12:00
""",
    "de": """
Kursanmeldung an der HNU:

Anmeldeprozess:
- Einloggen in das PRIMUSS Campus IT Portal
- Kurswahl während der Anmeldephase
- Überprüfung der Voraussetzungen und Kapazitätsbeschränkungen
- Auswahl bestätigen und Änderungen speichern

Wichtige Termine:
- Anmeldezeitraum: 2 Wochen vor Semesterbeginn
- Änderungszeitraum: Erste 2 Wochen des Semesters
- Rücktrittsfrist: 6 Wochen vor Prüfungen

Unterstützung:
- E-Mail: studienbuero@hnu.de
- Öffnungszeiten Studienbüro: Mo-Fr 9:00-12:00
"""
}

def detect_language(text: str) -> str:
    """Detect the language of the input text"""
    try:
        lang = langdetect.detect(text)
        return "de" if lang == "de" else "en"
    except:
        return "en"  # Default to English if detection fails

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text between languages"""
    if source_lang == target_lang:
        return text
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return translator.translate(text)

def load_knowledge_base():
    """Load the knowledge base from scraped documents"""
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    knowledge_base = {"en": [], "de": []}
    
    # Read all files in the docs directory
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Determine language based on filename
            if "_en" in filename:
                knowledge_base["en"].append(content)
            elif "_de" in filename or any(word in filename for word in ["bibliothek", "studium", "bewerbung", "campus"]):
                knowledge_base["de"].append(content)
    
    # Join all texts for each language
    knowledge_base["en"] = "\n\n".join(knowledge_base["en"])
    knowledge_base["de"] = "\n\n".join(knowledge_base["de"])
    
    return knowledge_base

# Load the knowledge base
HNU_KNOWLEDGE_BASE = load_knowledge_base()

def create_retriever(lang: str = "en"):
    """Create a retriever for HNU knowledge in the specified language"""
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Use the appropriate knowledge base based on language
    if lang == "en":
        texts = text_splitter.split_text(HNU_KNOWLEDGE_BASE["en"])
    else:
        texts = text_splitter.split_text(HNU_KNOWLEDGE_BASE["de"])
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts,
        embeddings,
        collection_name=f"hnu_knowledge_{lang}"
    )
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

def create_specialized_agent(agent_type: str):
    """Create a specialized agent with specific knowledge and behavior."""
    
    # Define category-specific prompts
    category_prompts = {
        "admissions": """You are an admissions specialist at HNU University.
        Focus on providing accurate information about:
        - Application processes and requirements
        - Entry qualifications and deadlines
        - Required documents and language requirements
        - Application status and procedures
        Be specific and provide clear step-by-step guidance when needed.""",
        
        "registration": """You are a course registration expert at HNU University.
        Focus on providing accurate information about:
        - Course registration procedures
        - Study program details and requirements
        - Module selection and registration deadlines
        - Course schedules and prerequisites
        Guide students through the registration process step by step.""",
        
        "international": """You are an international student advisor at HNU University.
        Focus on providing accurate information about:
        - Exchange programs and partnerships
        - Visa requirements and procedures
        - Housing options for international students
        - Language courses and support services
        Be culturally sensitive and provide clear guidance for international students.""",
        
        "campus": """You are a campus life expert at HNU University.
        Focus on providing accurate information about:
        - Campus facilities and their locations
        - Library services and resources
        - IT services and student ID cards
        - Sports facilities and recreational activities
        Help students navigate campus life and available services.""",
        
        "academic": """You are an academic advisor at HNU University.
        Focus on providing accurate information about:
        - Academic calendar and important dates
        - Exam regulations and procedures
        - Study requirements and guidelines
        - Internship opportunities and thesis requirements
        Guide students through academic procedures and requirements."""
    }
    
    def agent(state: AgentState):
        # Get the current language
        lang = state["language"]
        
        # Create retriever for the current language
        retriever = create_retriever(lang)
        
        # Get the last message
        last_message = state["messages"][-1].content
        
        # Get relevant context from the knowledge base
        docs = retriever.get_relevant_documents(last_message)
        context = "\n".join(doc.page_content for doc in docs)
        
        # Create prompt template with category-specific instructions
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{category_prompts.get(agent_type, category_prompts['admissions'])}
            
            Use the following context to provide accurate information:
            
            Context: {{context}}"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create the chain
        model = ChatOpenAI(temperature=0.3)
        chain = prompt | model | StrOutputParser()
        
        # Generate response
        response = chain.invoke({
            "context": context,
            "messages": state["messages"]
        })
        
        # Update state
        return {
            **state,
            "messages": [*state["messages"], AIMessage(content=response)],
            "context": [context],
            "next": "end" if "goodbye" in last_message.lower() else "router"
        }
    
    return agent

def create_general_agent():
    """Create a general agent for handling various HNU queries."""
    def agent(state: AgentState):
        # Get the current language
        lang = state["language"]
        
        # Create retriever for the current language
        retriever = create_retriever(lang)
        
        # Get the last message
        last_message = state["messages"][-1].content
        
        # Get relevant context from the knowledge base
        docs = retriever.get_relevant_documents(last_message)
        context = "\n".join(doc.page_content for doc in docs)
        
        # Create prompt template with improved instructions
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly and knowledgeable assistant for Hochschule Neu-Ulm (HNU) University. 
            Use the following context to answer questions about HNU.
            
            Guidelines:
            1. Be concise but informative
            2. If information is not in the context, say so honestly
            3. For specific program details, suggest contacting the relevant department
            4. Maintain a helpful and professional tone
            
            Context: {context}"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create the chain with slightly higher temperature for more natural responses
        model = ChatOpenAI(temperature=0.3)
        chain = prompt | model | StrOutputParser()
        
        # Generate response
        response = chain.invoke({
            "context": context,
            "messages": state["messages"]
        })
        
        # Update state
        return {
            **state,
            "messages": [*state["messages"], AIMessage(content=response)],
            "context": [context],
            "next": "end" if "goodbye" in last_message.lower() else "router"
        }
    
    return agent

def should_route_to_specialized_agent(state: AgentState) -> str:
    """Determine if the query should be routed to a specialized agent."""
    # Get the last message
    last_message = state["messages"][-1].content.lower()
    
    # Create routing prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant for HNU University queries.
        Analyze the user's message and determine the most appropriate category from the following:

        1. admissions:
           - Application process and requirements
           - Entry requirements and qualifications
           - Application deadlines
           - Required documents
           - Language requirements
           Example questions:
           - "How do I apply to HNU?"
           - "What are the admission requirements for the Business Administration program?"
           - "When is the application deadline for winter semester?"
           - "Do I need TestDaF for applying?"

        2. registration:
           - Course registration process
           - Study program details
           - Module registration
           - Semester registration
           - Course schedules
           Example questions:
           - "How do I register for courses?"
           - "What courses are offered in Information Management?"
           - "When does course registration start?"
           - "Can I still add courses after the semester starts?"

        3. international:
           - Exchange programs
           - International student support
           - Visa information
           - Housing for international students
           - Language courses
           Example questions:
           - "What exchange programs does HNU offer?"
           - "How can I get a student visa?"
           - "Are there German language courses?"
           - "Where can international students live?"

        4. campus:
           - Campus facilities
           - Library services
           - IT services
           - Student life
           - Sports and recreation
           Example questions:
           - "Where is the library?"
           - "How do I get my student ID?"
           - "What sports facilities are available?"
           - "Is there a cafeteria on campus?"

        5. academic:
           - Academic calendar
           - Exam regulations
           - Study advice
           - Internships
           - Thesis guidelines
           Example questions:
           - "When do exams take place?"
           - "How long is a semester?"
           - "What are the thesis requirements?"
           - "Can I do an internship during my studies?"

        6. general:
           - All other queries
           - Basic information
           - Contact information
           - General FAQs
           Example questions:
           - "Where is HNU located?"
           - "How can I contact the university?"
           - "What is the history of HNU?"
           - "Is there parking on campus?"

        Respond with ONLY ONE of these exact words: admissions, registration, international, campus, academic, or general"""),
        ("human", "{message}")
    ])
    
    # Create the chain
    chain = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    
    # Get the routing decision
    decision = chain.invoke({"message": last_message})
    
    # Update state with the routing decision
    return {
        **state,
        "next": decision,
        "agent_type": decision
    }

def should_end(state: AgentState) -> bool:
    """Check if the conversation should end"""
    last_message = state["messages"][-1].content.lower()
    end_words = {
        "en": ["goodbye", "bye", "quit", "exit", "thank you", "thanks"],
        "de": ["tschüss", "auf wiedersehen", "danke", "beenden", "ende"]
    }
    return any(word in last_message for words in end_words.values() for word in words)

def create_chat_graph():
    """Create the chat graph with enhanced routing and knowledge base handling."""
    # Initialize agents
    general_agent = create_general_agent()
    admissions_agent = create_specialized_agent("admissions")
    registration_agent = create_specialized_agent("registration")
    international_agent = create_specialized_agent("international")
    campus_agent = create_specialized_agent("campus")
    academic_agent = create_specialized_agent("academic")
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("general", general_agent)
    workflow.add_node("admissions", admissions_agent)
    workflow.add_node("registration", registration_agent)
    workflow.add_node("international", international_agent)
    workflow.add_node("campus", campus_agent)
    workflow.add_node("academic", academic_agent)
    
    # Router function
    def router(state: AgentState) -> str:
        # Check if conversation should end
        if should_end(state):
            return "end"
        
        # Route to specialized agent if needed
        routing_result = should_route_to_specialized_agent(state)
        state.update(routing_result)
        return routing_result["next"]
    
    # Add router node
    workflow.add_node("router", router)
    
    # Set up edges
    workflow.add_edge("general", "router")
    workflow.add_edge("admissions", "router")
    workflow.add_edge("registration", "router")
    workflow.add_edge("international", "router")
    workflow.add_edge("campus", "router")
    workflow.add_edge("academic", "router")
    workflow.add_edge("router", "general")
    workflow.add_edge("router", "admissions")
    workflow.add_edge("router", "registration")
    workflow.add_edge("router", "international")
    workflow.add_edge("router", "campus")
    workflow.add_edge("router", "academic")
    workflow.add_edge("router", "end")
    
    # Set entry point
    workflow.set_entry_point("router")
    
    return workflow.compile()

def create_initial_state(message: str) -> AgentState:
    """Create the initial state for the chat graph with improved language handling."""
    # Detect language
    lang = detect_language(message)
    
    # Create initial state
    return {
        "messages": [HumanMessage(content=message)],
        "next": "router",
        "context": [],
        "current_topic": "general",
        "language": lang,
        "agent_type": "general"
    }

class GraphChatbot:
    """Enhanced chatbot for HNU using LangGraph."""
    
    def __init__(self):
        """Initialize the chatbot with the enhanced chat graph."""
        self.app = create_chat_graph()
    
    def get_response(self, user_input: str) -> str:
        """Get a response from the chatbot for the given user input."""
        # Create initial state
        state = create_initial_state(user_input)
        
        # Run the graph
        result = self.app.invoke(state)
        
        # Get the last message (the bot's response)
        if result["messages"]:
            return result["messages"][-1].content
        
        # Fallback response in case of error
        fallback = {
            "en": "I apologize, but I'm having trouble processing your request. Please try again.",
            "de": "Entschuldigung, aber ich habe Schwierigkeiten, Ihre Anfrage zu verarbeiten. Bitte versuchen Sie es erneut."
        }
        return fallback[state["language"]]
