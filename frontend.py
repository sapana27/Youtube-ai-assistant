import streamlit as st
from backend import process_youtube_video, app, content_manager
from streamlit_mic_recorder import speech_to_text
from langchain_core.messages import HumanMessage, AIMessage
import os
import uuid
# Initialize session state
if "message_history" not in st.session_state:
    st.session_state['message_history'] = []

if "content_loaded" not in st.session_state:
    st.session_state['content_loaded'] = False

if "current_source_url" not in st.session_state:
    st.session_state['current_source_url'] = ""

if "current_source_type" not in st.session_state:
    st.session_state['current_source_type'] = ""

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

if "waiting_for_response" not in st.session_state:
    st.session_state['waiting_for_response'] = False

# Language options
LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi": "hi", 
    "French": "fr",
    "Spanish": "es",
    "German": "de"
}

# --- Sidebar for Content Loading ---
st.sidebar.header("📥 Content Sources")

# Content type selection
content_type = st.sidebar.radio(
    "Select Content Type",
    ["YouTube Video", "Web Content"],
    help="Choose the type of content to analyze"
)

if content_type == "YouTube Video":
    youtube_url = st.sidebar.text_input(
        "Enter YouTube URL",
        value=st.session_state.get('current_source_url', ''),
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    selected_language_name = st.sidebar.selectbox(
        "Select Transcript Language", 
        list(LANGUAGE_OPTIONS.keys())
    )
    language = LANGUAGE_OPTIONS[selected_language_name]
    
    load_button = st.sidebar.button("🎥 Load YouTube Video", type="primary")
    
    if load_button and youtube_url:
        with st.spinner("Loading YouTube video transcript..."):
            try:
                process_youtube_video(youtube_url, language)
                st.session_state['content_loaded'] = True
                st.session_state['current_source_url'] = youtube_url
                st.session_state['current_source_type'] = "youtube"
                st.sidebar.success("✅ YouTube video loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

else:  # Web Content
    web_url = st.sidebar.text_input(
        "Enter Web URL",
        value=st.session_state.get('current_source_url', ''),
        placeholder="https://example.com/article.pdf or https://arxiv.org/..."
    )
    
    load_web_button = st.sidebar.button("🌐 Load Web Content", type="primary")
    
    if load_web_button and web_url:
        with st.spinner("Loading and processing web content..."):
            try:
                success, message = content_manager.load_web_content(web_url)
                if success:
                    st.session_state['content_loaded'] = True
                    st.session_state['current_source_url'] = web_url
                    st.session_state['current_source_type'] = "web"
                    st.sidebar.success(f"✅ {message}")
                else:
                    st.sidebar.error(f"❌ {message}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

# Show current content status
if st.session_state['content_loaded']:
    source_type = "YouTube Video" if st.session_state['current_source_type'] == "youtube" else "Web Content"
    st.sidebar.info(f"✅ {source_type} Loaded")
    st.sidebar.caption(f"Source: {st.session_state['current_source_url']}")
else:
    st.sidebar.warning("⚠️ No content loaded")

# --- Audio Settings ---
st.sidebar.markdown("---")
st.sidebar.header("🔊 Audio Settings")
auto_play_audio = st.sidebar.checkbox(
    "Auto-play response audio", 
    value=True,
    help="Generate and auto-play audio responses"
)

# --- Controls ---
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Controls")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state['message_history'] = []
        st.session_state['waiting_for_response'] = False
        st.rerun()

with col2:
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state['message_history'] = []
        st.session_state['content_loaded'] = False
        st.session_state['current_source_url'] = ""
        st.session_state['current_source_type'] = ""
        st.session_state['waiting_for_response'] = False
        st.session_state['thread_id'] = str(uuid.uuid4())
        st.rerun()

# --- Main Chat Section ---
st.title("🤖 Content Analysis AI")
st.markdown("Analyze YouTube videos, research papers, news articles, and web content!")

# Create a container for the chat messages
chat_container = st.container()

# Display chat history
with chat_container:
    for i, message in enumerate(st.session_state['message_history']):
        with st.chat_message(message['role']):
            st.write(message['content'])
            
            # Add audio player for assistant messages
            if (message['role'] == 'assistant' and 
                'audio_file' in message and 
                message['audio_file'] and
                auto_play_audio):
                if os.path.exists(message['audio_file']):
                    st.audio(message['audio_file'], format='audio/mp3', autoplay=True)

# Input section at the bottom
st.markdown("---")

# Create columns for different input methods
input_col1, input_col2 = st.columns([3, 7])

with input_col1:
    speech_text = speech_to_text(
        language="en", 
        start_prompt="🎤 Record", 
        stop_prompt="⏹️ Stop", 
        key="speech_input",
        just_once=True,
        use_container_width=True
    )

with input_col2:
    user_input = st.chat_input(
        "Type your message here..." if not st.session_state['waiting_for_response'] else "Please wait for the current response...",
        disabled=st.session_state['waiting_for_response']
    )

# Process input
final_input = None

if user_input and not st.session_state['waiting_for_response']:
    final_input = user_input
elif speech_text and not st.session_state['waiting_for_response']:
    final_input = speech_text
    st.success(f"🎤 Voice input: \"{speech_text}\"")

if final_input:
    # Set waiting state
    st.session_state['waiting_for_response'] = True
    
    # Add user message to history
    st.session_state['message_history'].append({
        "role": "user", 
        "content": final_input,
        "audio_file": None
    })
    
    # Display user message immediately
    with chat_container:
        with st.chat_message("user"):
            st.write(final_input)

    # Show assistant "thinking" placeholder
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            audio_placeholder = st.empty()
            message_placeholder.write("🤔 Processing...")

            try:
                # Prepare request with current source info
                config = {"configurable": {"thread_id": st.session_state['thread_id']}}
                
                request_data = {
                    "messages": [HumanMessage(content=final_input)],
                    "generate_audio": auto_play_audio,
                    "current_source": st.session_state['current_source_url'] if st.session_state['content_loaded'] else None
                }
                
                response = app.invoke(request_data, config=config)
                
                # Extract response
                ai_response = response['messages'][-1].content
                audio_file = response.get('audio_file')

                # Update message placeholder with final response
                message_placeholder.write(ai_response)

                # Play audio if available
                if auto_play_audio and audio_file and os.path.exists(audio_file):
                    audio_placeholder.audio(audio_file, format='audio/mp3', autoplay=True)
                    
                # Save assistant response to history
                st.session_state['message_history'].append({
                    "role": "assistant", 
                    "content": ai_response,
                    "audio_file": audio_file
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.write(error_msg)
                st.session_state['message_history'].append({
                    "role": "assistant", 
                    "content": error_msg,
                    "audio_file": None
                })
            
            finally:
                # Reset waiting state
                st.session_state['waiting_for_response'] = False
                st.rerun()

# Auto-scroll to bottom
st.markdown(
    """
    <script>
    var chatContainer = window.parent.document.querySelector('section.main');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    </script>
    """,
    unsafe_allow_html=True
)
