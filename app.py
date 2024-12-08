import streamlit as st
from chatbot import Chatbot
import os
from PIL import Image
import base64
from dotenv import load_dotenv

# Function to load and encode the logo
def get_base64_logo():
    """Load and encode the HNU logo as base64"""
    logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'hnu_logo.svg')
    with open(logo_path, "rb") as f:
        logo_data = f.read()
    return base64.b64encode(logo_data).decode()

# Custom CSS for the header
def set_header_style():
    st.markdown("""
        <style>
        /* Fixed header container */
        .header-wrapper {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 999;
            background-color: rgb(17, 19, 23);
            padding: 1rem 1rem 0 1rem;
        }
        
        /* Main content padding to prevent overlap with fixed header */
        .main-content {
            margin-top: 160px;
            padding: 1rem;
        }
        
        /* Compact tab styling */
        .stTabs {
            background-color: transparent;
        }
        .stTabs > div > div {
            padding-top: 0.5rem;
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 2rem;
            padding: 0 1rem;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding: 0.5rem 0;
        }
        
        /* Reduce spacing in markdown */
        .stMarkdown p {
            margin-bottom: 0.5rem;
        }
        
        /* Compact file uploader */
        .stFileUploader > div {
            padding: 0.5rem;
        }
        .stFileUploader > div > div {
            padding: 0.5rem;
        }
        
        /* Compact text input */
        .stTextInput > div {
            padding-bottom: 0.5rem;
        }
        
        .header-container {
            display: flex;
            align-items: center;
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin-bottom: 1rem;
            min-height: 120px;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }
        .logo-container {
            flex: 0 0 300px;
            margin-right: 2rem;
        }
        .logo-container img {
            width: 100%;
            height: auto;
        }
        .info-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100%;
        }
        .header-title {
            color: #1e3c72;
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }
        .header-subtitle {
            color: #4a4a4a;
            font-size: 1rem;
            line-height: 1.4;
            white-space: pre-line;
        }
        
        /* Hide Streamlit's default header */
        header {
            visibility: hidden;
        }
        
        /* Adjust footer position */
        footer {
            visibility: hidden;
        }
        </style>
    """, unsafe_allow_html=True)

# Categories in both languages
CATEGORIES = {
    "en": ["General Questions", "Admissions Questions", "Course Registration Questions", 
           "International Student Questions", "Campus Life Questions", "Academic Questions"],
    "de": ["Allgemeine Fragen", "Zulassungsfragen", "Kursanmeldungsfragen", 
           "Internationale Studierende", "Campus-Leben", "Akademische Fragen"]
}

# Example questions in both English and German
EXAMPLE_QUESTIONS = {
    "General Questions": {
        "name": {"en": "General Questions", "de": "Allgemeine Fragen"},
        "en": [
            "Where is HNU located?",
            "What is the history of HNU?",
            "How can I contact the student services?",
            "Is there parking available on campus?",
            "What public transport options are there to reach HNU?"
        ],
        "de": [
            "Wo befindet sich die HNU?",
            "Was ist die Geschichte der HNU?",
            "Wie kann ich den Studierendenservice kontaktieren?",
            "Gibt es Parkplätze auf dem Campus?",
            "Welche öffentlichen Verkehrsmittel fahren zur HNU?"
        ]
    },
    "Admissions Questions": {
        "name": {"en": "Admissions Questions", "de": "Zulassungsfragen"},
        "en": [
            "What are the admission requirements for international students?",
            "When is the application deadline for summer semester?",
            "Do I need to take a language test?",
            "How do I submit my application documents?",
            "Can I apply for multiple programs at once?"
        ],
        "de": [
            "Was sind die Zulassungsvoraussetzungen für internationale Studierende?",
            "Wann ist die Bewerbungsfrist für das Sommersemester?",
            "Muss ich einen Sprachtest ablegen?",
            "Wie reiche ich meine Bewerbungsunterlagen ein?",
            "Kann ich mich für mehrere Studiengänge gleichzeitig bewerben?"
        ]
    },
    "Course Registration Questions": {
        "name": {"en": "Course Registration Questions", "de": "Kursanmeldungsfragen"},
        "en": [
            "How do I register for exams?",
            "Can I change courses after registration?",
            "What is the maximum number of courses per semester?",
            "How do I view my course schedule?",
            "What happens if I miss the registration deadline?"
        ],
        "de": [
            "Wie melde ich mich für Prüfungen an?",
            "Kann ich nach der Anmeldung Kurse wechseln?",
            "Wie viele Kurse kann ich maximal pro Semester belegen?",
            "Wie kann ich meinen Stundenplan einsehen?",
            "Was passiert, wenn ich die Anmeldefrist verpasse?"
        ]
    },
    "International Student Questions": {
        "name": {"en": "International Student Questions", "de": "Internationale Studierende"},
        "en": [
            "What support services are available for international students?",
            "How can I find accommodation in Neu-Ulm?",
            "Do you offer German language courses?",
            "How do I extend my student visa?",
            "Are there international student organizations?"
        ],
        "de": [
            "Welche Unterstützung gibt es für internationale Studierende?",
            "Wie finde ich eine Unterkunft in Neu-Ulm?",
            "Bietet ihr Deutschkurse an?",
            "Wie verlängere ich mein Studentenvisum?",
            "Gibt es internationale Studierendenorganisationen?"
        ]
    },
    "Campus Life Questions": {
        "name": {"en": "Campus Life Questions", "de": "Campus-Leben"},
        "en": [
            "What sports facilities are available?",
            "How do I get a library card?",
            "Where is the student cafeteria?",
            "Are there study rooms I can use?",
            "What student clubs can I join?"
        ],
        "de": [
            "Welche Sportanlagen gibt es?",
            "Wie bekomme ich einen Bibliotheksausweis?",
            "Wo ist die Mensa?",
            "Gibt es Lernräume, die ich nutzen kann?",
            "Welchen Studierendengruppen kann ich beitreten?"
        ]
    },
    "Academic Questions": {
        "name": {"en": "Academic Questions", "de": "Akademische Fragen"},
        "en": [
            "When do exams take place?",
            "How can I find an internship?",
            "What are the thesis requirements?",
            "How do I choose a thesis supervisor?",
            "Can I study abroad for a semester?"
        ],
        "de": [
            "Wann finden die Prüfungen statt?",
            "Wie finde ich ein Praktikum?",
            "Was sind die Anforderungen für die Abschlussarbeit?",
            "Wie wähle ich einen Betreuer für die Abschlussarbeit?",
            "Kann ich ein Semester im Ausland studieren?"
        ]
    }
}

# UI text translations
UI_TEXT = {
    "en": {
        "welcome": "Welcome to the HNU AI Assistant!",
        "header_text": """Your digital guide for: • Admissions & Applications • Study Programs • Campus Life • Student Services
Ask me anything in English or German 🇬🇧🇩🇪""",
        "chat_placeholder": "Type your message here...",
        "clear_chat": "Clear Chat History",
        "scrape_button": "Scrape HNU Website",
        "scrape_status": "Scraping Status:",
        "scrape_progress": "Scraping in progress...",
        "scrape_complete": "Scraping completed!",
        "scrape_error": "Error during scraping:",
        "clear_scrape": "Clear Scraped Data",
        "scraped_cleared": "Scraped data cleared successfully!",
        "title": "HNU AI Assistant",
        "settings": "Chatbot Settings",
        "select_language": "Select Language",
        "example_questions": "Example Questions",
        "select_category": "Select Category",
        "select_question": "Select a Question",
        "ask_button": "Ask Selected Question",
        "select_personality": "Select Personality",
        "change_personality": "Change Personality",
        "personality_changed": "Personality changed to",
        "upload_document": "Upload Document",
        "web_scraping": "Web Scraping",
        "enter_url": "Enter URL to scrape",
        "scrape": "Scrape",
        "clear": "Clear",
        "clear_chat": "Clear Chat",
        "scraped_cleared": "Scraped data cleared"
    },
    "de": {
        "welcome": "Willkommen beim HNU AI Assistenten!",
        "header_text": """Ihr digitaler Begleiter für: • Bewerbung & Zulassung • Studiengänge • Campusleben • Studierendenservice
Fragen Sie auf Deutsch oder Englisch 🇩🇪🇬🇧""",
        "chat_placeholder": "Geben Sie Ihre Nachricht ein...",
        "clear_chat": "Chat-Verlauf löschen",
        "scrape_button": "HNU Website durchsuchen",
        "scrape_status": "Scraping-Status:",
        "scrape_progress": "Scraping läuft...",
        "scrape_complete": "Scraping abgeschlossen!",
        "scrape_error": "Fehler beim Scraping:",
        "clear_scrape": "Gescrapte Daten löschen",
        "scraped_cleared": "Gescrapte Daten erfolgreich gelöscht!",
        "title": "HNU KI-Assistent",
        "settings": "Chatbot-Einstellungen",
        "select_language": "Sprache auswählen",
        "example_questions": "Beispielfragen",
        "select_category": "Kategorie auswählen",
        "select_question": "Frage auswählen",
        "ask_button": "Ausgewählte Frage stellen",
        "select_personality": "Persönlichkeit auswählen",
        "change_personality": "Persönlichkeit ändern",
        "personality_changed": "Persönlichkeit geändert zu",
        "upload_document": "Dokument hochladen",
        "web_scraping": "Web Scraping",
        "enter_url": "URL zum Scrapen eingeben",
        "scrape": "Scrapen",
        "clear": "Löschen",
        "clear_chat": "Chat löschen",
        "scraped_cleared": "Gescrapte Daten gelöscht"
    }
}

# Add personality translations
PERSONALITIES = {
    "en": {
        "helpful": "Helpful",
        "professional": "Professional",
        "creative": "Creative",
        "teacher": "Teacher"
    },
    "de": {
        "helpful": "Hilfsbereit",
        "professional": "Professionell",
        "creative": "Kreativ",
        "teacher": "Lehrer"
    }
}

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize or reload chatbot
if "chatbot" not in st.session_state or not hasattr(st.session_state.chatbot, 'get_streaming_response'):
    st.session_state.chatbot = Chatbot()

def main():
    # Set page config
    st.set_page_config(
        layout="wide"
    )
    
    # Initialize language selection in session state if not present
    if 'lang_code' not in st.session_state:
        st.session_state.lang_code = 'en'
    
    # Sidebar organization
    with st.sidebar:
        # Section 1: Example Questions
        st.markdown("### 💬 Chat")
        # Category selector with translated names
        category_display_names = [EXAMPLE_QUESTIONS[cat]["name"][st.session_state.lang_code] for cat in EXAMPLE_QUESTIONS.keys()]
        category_index = st.selectbox(
            UI_TEXT[st.session_state.lang_code]["select_category"],
            range(len(category_display_names)),
            format_func=lambda x: category_display_names[x]
        )
        
        # Get the category key from the index
        category_key = list(EXAMPLE_QUESTIONS.keys())[category_index]
        
        # Question selector
        selected_question = st.selectbox(
            UI_TEXT[st.session_state.lang_code]["select_question"],
            EXAMPLE_QUESTIONS[category_key][st.session_state.lang_code]
        )
        
        if st.button(UI_TEXT[st.session_state.lang_code]["ask_button"], key="ask_example"):
            # Add user message to chat history and trigger rerun to show in main chat
            st.session_state.messages.append({"role": "user", "content": selected_question})
            st.rerun()
        
        st.divider()
        
        # Section 2: Settings
        st.markdown("### ⚙️ Settings")
        
        # Language settings
        language = st.selectbox(
            "Language/Sprache",
            ["English", "Deutsch"],
            index=0 if st.session_state.lang_code == "en" else 1,
            key="language_selector"
        )
        new_lang_code = "en" if language == "English" else "de"
        
        # If language changed, update session state and rerun
        if new_lang_code != st.session_state.lang_code:
            st.session_state.lang_code = new_lang_code
            st.rerun()
        
        # Personality settings
        personality_options = ["helpful", "professional", "creative", "teacher"]
        personality = st.selectbox(
            UI_TEXT[st.session_state.lang_code]["select_personality"],
            personality_options,
            format_func=lambda x: PERSONALITIES[st.session_state.lang_code][x],
            index=personality_options.index(st.session_state.chatbot.current_personality) if hasattr(st.session_state.chatbot, 'current_personality') else 0
        )
        
        if st.button(UI_TEXT[st.session_state.lang_code]["change_personality"], key="change_personality"):
            st.session_state.chatbot.change_personality(personality)
            st.success(f"{UI_TEXT[st.session_state.lang_code]['personality_changed']} {PERSONALITIES[st.session_state.lang_code][personality]}")
            
        st.divider()
        
        # Section 3: Knowledge Base
        st.markdown("### 📚 Knowledge Base")
        tab1, tab2 = st.tabs(["📄 Documents", "🌐 Web"])
        
        with tab1:
            uploaded_file = st.file_uploader(UI_TEXT[st.session_state.lang_code]["upload_document"], 
                                           type=["txt", "pdf", "html"],
                                           label_visibility="collapsed")
            if uploaded_file:
                file_path = os.path.join("docs", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                result = st.session_state.chatbot.process_document(file_path)
                st.success(result)
        
        with tab2:
            url = st.text_input(UI_TEXT[st.session_state.lang_code]["enter_url"],
                              label_visibility="collapsed",
                              placeholder=UI_TEXT[st.session_state.lang_code]["enter_url"])
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(UI_TEXT[st.session_state.lang_code]["scrape"], 
                           key="scrape_url", 
                           use_container_width=True):
                    result = st.session_state.chatbot.scrape_url(url)
                    st.success(result)
            with col2:
                if st.button(UI_TEXT[st.session_state.lang_code]["clear"], 
                           key="clear_scrape",
                           use_container_width=True):
                    st.session_state.chatbot.clear_scraped_data()
                    st.success(UI_TEXT[st.session_state.lang_code]["scraped_cleared"])
        
        # Add some space before the clear chat button
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        st.divider()
        
        # Clear chat button at the bottom
        if st.button(UI_TEXT[st.session_state.lang_code]["clear_chat"], type="primary"):
            st.session_state.messages = []
            st.rerun()
            
    # Set header style
    set_header_style()
    
    # Create header with logo and information
    st.markdown("""
        <div class="header-wrapper">
            <div class="header-container">
                <div class="logo-container">
                    <img src="data:image/svg+xml;base64,{}" alt="HNU Logo">
                </div>
                <div class="info-container">
                    <div class="header-title">{}</div>
                    <div class="header-subtitle">
                        {}
                    </div>
                </div>
            </div>
        </div>
        <div class="main-content">
    """.format(
        get_base64_logo(),
        UI_TEXT[st.session_state.lang_code]['welcome'],
        UI_TEXT[st.session_state.lang_code]['header_text']
    ), unsafe_allow_html=True)
    
    # Main chat interface

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new messages (both from chat input and example questions)
    if prompt := st.chat_input(UI_TEXT[st.session_state.lang_code]["chat_placeholder"]):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in st.session_state.chatbot.get_streaming_response(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            # Remove the cursor and display final response
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Handle streaming response for example questions
    elif len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in st.session_state.chatbot.get_streaming_response(st.session_state.messages[-1]["content"]):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            # Remove the cursor and display final response
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Close the main-content div
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
