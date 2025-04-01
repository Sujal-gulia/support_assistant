import streamlit as st
from transformers import pipeline, BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load lightweight model (1B parameters distilled version)
@st.cache_resource
def load_model():
    model_name = "facebook/blenderbot-1B-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input handling
prompt = st.chat_input("Type your question here...")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("Thinking..."):
        inputs = tokenizer(
            " ".join([m["content"] for m in st.session_state.messages[-3:]]),
            return_tensors="pt",
            truncation=True
        )
        response_ids = model.generate(**inputs, max_length=1000)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to display new messages
    st.rerun()
