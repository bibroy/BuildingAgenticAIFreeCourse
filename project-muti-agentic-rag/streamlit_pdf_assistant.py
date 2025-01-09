import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.pgvector import PgVector
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embedder
embedder = SentenceTransformerEmbedder(dimensions=384)

# Set up database URL
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Set up assistant storage
storage = PgAgentStorage(table_name="pdf_assistant", db_url=db_url)

# Streamlit UI
st.title("PDF Assistant with PDFKnowledgeBase")
st.markdown("Upload a PDF document, and interact with the assistant to answer your queries based on the document content.")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save the uploaded file to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Create a knowledge base using PDFKnowledgeBase
        pdf_knowledge_base = PDFKnowledgeBase(
            path=temp_dir,  # Directory where the uploaded PDF is saved
            vector_db=PgVector(
                table_name="pdf_documents",  # Table for storing vectorized data
                db_url=db_url  # Database URL
            ),
            reader=PDFReader(chunk=True),  # Use PDFReader for reading and chunking
        )

        with st.spinner("Loading the knowledge base..."):
            pdf_knowledge_base.load()

        # Run the assistant
        st.markdown("### Assistant Interaction")
        user_query = st.text_input("Enter your question:")

        if user_query:
            assistant = Agent(
                model=Groq(id="llama-3.3-70b-versatile", embedder=embedder),
                run_id=None,
                user_id="user",
                knowledge_base=pdf_knowledge_base,
                storage=storage,
                show_tool_calls=True,
                search_knowledge=True,
                read_chat_history=True,
                add_history_to_messages=True,
            )

            with st.spinner("Processing your query..."):
                response = st.infoassistant.run(user_query)
                st.markdown("**Assistant Response:**")
                st.write(response.content)

            # Display conversation history
            if assistant.memory:
                st.markdown("### Conversation History")
                for role, content in assistant.memory:
                    st.markdown(f"**{role}:** {content}")

else:
    st.info("Please upload a PDF to get started.")

# Footer
st.sidebar.write("### About")
st.sidebar.info(
    "This PDF Assistant uses the Groq model and a custom knowledge base to answer questions from uploaded PDFs. "
    "Built with ❤️ using Streamlit and Phi."
)

# <function=search_knowledge_base{"query": "Saatvik skills and college name"}</function>