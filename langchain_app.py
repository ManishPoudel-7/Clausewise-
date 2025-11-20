# type: ignore[unused-ignore]
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import TypedDict, Annotated
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from mongodb_helper import MongoDBHelper
from dotenv import load_dotenv
import os
import time
import tempfile
import requests
import base64
import io
import wave
from google.genai import Client
from google.genai import types

load_dotenv()

# Initialize MongoDB
mongo_uri = os.getenv('MONGO_URI')
db = MongoDBHelper(mongo_uri)
user_id = db.get_or_create_user_id()


# Cache the embedding model to prevent reloading
@st.cache_resource
def get_embedding_model():
    """Get or create the embedding model (cached)"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def generate_speech_data_url(text, voice='Kore'):
    """Generate speech from text and return as data URL for HTML5 audio"""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("Please set GOOGLE_API_KEY in your .env file")
            return None
            
        client = Client(api_key=api_key)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",  # Use the correct model name
            contents=f"Read this summary clearly: {text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                ),
            )
        )
        
        # Get the audio data
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)  # 24kHz
            wf.writeframes(audio_data)
        
        # Convert to base64 for data URL
        wav_data = buffer.getvalue()
        b64_data = base64.b64encode(wav_data).decode()
        return f"data:audio/wav;base64,{b64_data}"
        
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None
    
def run_langchain_app():
    apiKey = os.getenv('GROQ_API_KEY')
    if not apiKey:
        st.error("Please set GROQ_API_KEY in your .env file")
        st.stop()

    model = ChatGroq(
        api_key=apiKey,
        model_name="llama-3.1-8b-instant"
    )

    st.title("ClauseWise")
    st.write("Welcome! You are logged in.")

    # Initialize session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'summary_data' not in st.session_state:
        st.session_state.summary_data = None
    if 'current_doc_id' not in st.session_state:
        st.session_state.current_doc_id = None
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'vectorStore' not in st.session_state:
        st.session_state.vectorStore = None
    if 'skip_upload' not in st.session_state:
        st.session_state.skip_upload = False

    uploadedFile = st.file_uploader("Choose a File", type=['pdf', 'txt', 'docx', 'doc'])

    # Check if this file was already uploaded before
    file_already_exists = False
    existing_doc = None
    if uploadedFile is not None and not st.session_state.skip_upload:
        # Check if a document with this filename already exists for this user
        user_docs = db.get_user_documents(user_id)
        for doc in user_docs:
            if doc['filename'] == uploadedFile.name:
                file_already_exists = True
                existing_doc = doc
                break
        
        if file_already_exists:
            st.info(f"📄 This document '{uploadedFile.name}' was already uploaded. Loading existing analysis...")
            
            # Load the existing document
            st.session_state.current_doc_id = existing_doc['doc_id']
            st.session_state.current_file_name = existing_doc['filename']
            st.session_state.summary_data = existing_doc['summary_data']
            st.session_state.analysis_complete = True
            st.session_state.vectorStore = None  # Will be loaded below
            
            # Get sessions for this document
            sessions = db.get_document_sessions(existing_doc['doc_id'])
            if sessions:
                st.session_state.current_session_id = sessions[0]['session_id']
                messages = db.get_session_messages(sessions[0]['session_id'])
                st.session_state.chat_messages = messages
            else:
                session_id = db.create_session(user_id, existing_doc['doc_id'], f"Chat - {existing_doc['filename']}")
                st.session_state.current_session_id = session_id
                st.session_state.chat_messages = []
            
            # Set flag and rerun
            st.session_state.skip_upload = True
            st.rerun()

    # Reset skip_upload flag after rerun
    if st.session_state.skip_upload:
        st.session_state.skip_upload = False

    # MAIN LOGIC: Handle both new uploads and loaded documents
    if uploadedFile is not None or st.session_state.current_doc_id is not None:
        
        # Handle NEW FILE UPLOAD
        if uploadedFile is not None and (
            st.session_state.current_file_name != uploadedFile.name or
            not st.session_state.analysis_complete
        ):
            # Check if we need to reprocess or just add vector store
            needs_full_processing = True
            if file_already_exists and existing_doc:
                # Check if vector store exists
                faiss_file = db.fs.find_one({'filename': f"{existing_doc['doc_id']}_index.faiss"})
                pkl_file = db.fs.find_one({'filename': f"{existing_doc['doc_id']}_index.pkl"})
                
                if not faiss_file or not pkl_file:
                    st.info("📦 Vector store missing. Creating it now...")
                    needs_full_processing = False  # We already have the analysis
                    
                    # Just create the vector store
                    temp_filename = "temp_uploaded.pdf"
                    try:
                        with open(temp_filename, "wb") as file:
                            file.write(uploadedFile.read())

                        # FIXED: Use file extension instead of MIME type for reliability
                        extension = os.path.splitext(uploadedFile.name)[1].lower()
                        if extension == '.pdf':
                            loader = PyPDFLoader(temp_filename)
                        elif extension == '.txt':
                            loader = TextLoader(temp_filename)
                        elif extension in ['.docx', '.doc']:
                            loader = Docx2txtLoader(temp_filename)
                        else:
                            st.error(f"Unsupported file type: {extension}")
                            return
                        
                        docs = loader.load()
                        
                        # Create embeddings and vector store
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=400,
                            chunk_overlap=100
                        )
                        chunks = splitter.split_documents(docs)

                        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        
                        vectorStore = FAISS.from_documents(
                            embedding=embedding,
                            documents=chunks
                        )
                        st.session_state.vectorStore = vectorStore
                        st.success(f"✅ Vector store created in memory! Vectors: {vectorStore.index.ntotal}")
                        
                        # Save vector store to MongoDB GridFS
                        with st.spinner("Saving vector store to cloud..."):
                            vector_path = db.save_vector_store(vectorStore, existing_doc['doc_id'])
                            st.success(f"✅ Vector store created and saved!")
                        
                        # Use existing data
                        st.session_state.current_doc_id = existing_doc['doc_id']
                        st.session_state.current_file_name = existing_doc['filename']
                        st.session_state.summary_data = existing_doc['summary_data']
                        st.session_state.analysis_complete = True
                        
                        needs_full_processing = False
                        
                    except Exception as e:
                        st.error(f"Error creating vector store: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                    finally:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                else:
                    st.info("✅ Document fully loaded from database!")
                    needs_full_processing = False
            
            if not needs_full_processing:
                # Skip to display
                pass
            elif needs_full_processing:
                st.session_state.chat_messages = []
                st.session_state.current_file_name = uploadedFile.name
                temp_filename = "temp_uploaded.pdf"

                try:
                    with open(temp_filename, "wb") as file:
                        file.write(uploadedFile.read())

                    # FIXED: Use file extension instead of MIME type for reliability
                    extension = os.path.splitext(uploadedFile.name)[1].lower()
                    if extension == '.pdf':
                        loader = PyPDFLoader(temp_filename)
                    elif extension == '.txt':
                        loader = TextLoader(temp_filename)
                    elif extension in ['.docx', '.doc']:
                        loader = Docx2txtLoader(temp_filename)
                    else:
                        st.error(f"Unsupported file type: {extension}")
                        return
                    
                    # Load docs for ALL file types
                    docs = loader.load()

                    str_parser = StrOutputParser()
                    
                    # Use simple JSON parsing instead of structured output
                    import json
                    import re

                    text = "\n".join([doc.page_content for doc in docs])

                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                    status_text.info("📝 **Step 1/5:** Generating document summary...")
                    progress_bar.progress(10)

                    summaryPrompt = PromptTemplate(
                        template="""Read the following text carefully and create a detailed summary. 
                        Highlight the main points, key arguments, important clauses, and any potential risks or concerns. 
                        Use clear and simple language, but be thorough. Output the summary in well-structured paragraphs.
                        Text: {text}""",
                        input_variables=['text']
                    )

                    summaryChain = summaryPrompt | model | str_parser
                    summary = summaryChain.invoke({'text': text})

                    progress_bar.progress(25)
                    status_text.success("✅ **Step 1 Complete:** Document summary generated!")
                    time.sleep(0.5)

                    status_text.info("⚖️ **Step 2/5:** Evaluating legal risks...")
                    progress_bar.progress(35)

                    legalRiskPrompt = PromptTemplate(
                        template="""Based on the summary below, identify any clauses or points that could pose legal risks to the user.
                        Respond with ONLY a valid JSON object (no markdown, no extra text) with exactly these keys:
                        {{"summary": "your explanation here", "score": 7}}
                        
                        The score must be an integer from 1 (low risk) to 10 (high risk).

                        Summary: {summary}""",
                        input_variables=['summary']
                    )

                    legalRiskChain = legalRiskPrompt | model | str_parser
                    try:
                        legal_risk_text = legalRiskChain.invoke({'summary': summary})
                        # Extract JSON from response
                        legal_risk_text = legal_risk_text.strip()
                        # Remove markdown code blocks if present
                        legal_risk_text = re.sub(r'```json\s*|\s*```', '', legal_risk_text)
                        legal_risk_result = json.loads(legal_risk_text)
                        
                        # Validate the result
                        if not isinstance(legal_risk_result, dict):
                            legal_risk_result = {'summary': 'Could not analyze legal risks', 'score': 5}
                        if 'score' not in legal_risk_result or 'summary' not in legal_risk_result:
                            legal_risk_result = {'summary': legal_risk_result.get('summary', 'Could not analyze legal risks'), 'score': 5}
                        # Ensure score is an integer
                        legal_risk_result['score'] = int(legal_risk_result['score'])
                    except Exception as e:
                        st.warning(f"Issue with legal risk analysis: {e}")
                        legal_risk_result = {'summary': 'Could not analyze legal risks', 'score': 5}

                    progress_bar.progress(45)
                    status_text.success("✅ **Step 2 Complete:** Legal risk analysis finished!")
                    time.sleep(0.5)

                    status_text.info("🛡️ **Step 3/5:** Checking for missing rights and protections...")
                    progress_bar.progress(55)

                    missingRightsPrompt = PromptTemplate(
                        template="""From the summary, identify any important rights or protections that are missing for the user.
                        Respond with ONLY a valid JSON object (no markdown, no extra text) with exactly these keys:
                        {{"summary": "your explanation here", "score": 7}}
                        
                        The score must be an integer from 1 (few missing rights) to 10 (many missing rights).

                        Summary: {summary}""",
                        input_variables=['summary']
                    )

                    missingRightsChain = missingRightsPrompt | model | str_parser
                    try:
                        missing_rights_text = missingRightsChain.invoke({'summary': summary})
                        # Extract JSON from response
                        missing_rights_text = missing_rights_text.strip()
                        # Remove markdown code blocks if present
                        missing_rights_text = re.sub(r'```json\s*|\s*```', '', missing_rights_text)
                        missing_rights_result = json.loads(missing_rights_text)
                        
                        # Validate the result
                        if not isinstance(missing_rights_result, dict):
                            missing_rights_result = {'summary': 'Could not analyze missing rights', 'score': 5}
                        if 'score' not in missing_rights_result or 'summary' not in missing_rights_result:
                            missing_rights_result = {'summary': missing_rights_result.get('summary', 'Could not analyze missing rights'), 'score': 5}
                        # Ensure score is an integer
                        missing_rights_result['score'] = int(missing_rights_result['score'])
                    except Exception as e:
                        st.warning(f"Issue with missing rights analysis: {e}")
                        missing_rights_result = {'summary': 'Could not analyze missing rights', 'score': 5}

                    progress_bar.progress(65)
                    status_text.success("✅ **Step 3 Complete:** Missing rights analysis completed!")
                    time.sleep(0.5)

                    status_text.info("🔍 **Step 4/5:** Analyzing document clarity and comprehension...")
                    progress_bar.progress(75)

                    clarityPrompt = PromptTemplate(
                        template="""Evaluate how clear and understandable the summary is for a regular user.
                        Respond with ONLY a valid JSON object (no markdown, no extra text) with exactly these keys:
                        {{"summary": "your explanation here", "score": 7}}
                        
                        The score must be an integer from 1 (very unclear) to 10 (very clear).
                        
                        Summary: {summary}""",
                        input_variables=['summary']
                    )

                    clarityChain = clarityPrompt | model | str_parser
                    try:
                        clarity_text = clarityChain.invoke({'summary': summary})
                        # Extract JSON from response
                        clarity_text = clarity_text.strip()
                        # Remove markdown code blocks if present
                        clarity_text = re.sub(r'```json\s*|\s*```', '', clarity_text)
                        clarity_result = json.loads(clarity_text)
                        
                        # Validate the result
                        if not isinstance(clarity_result, dict):
                            clarity_result = {'summary': 'Could not analyze clarity', 'score': 5}
                        if 'score' not in clarity_result or 'summary' not in clarity_result:
                            clarity_result = {'summary': clarity_result.get('summary', 'Could not analyze clarity'), 'score': 5}
                        # Ensure score is an integer
                        clarity_result['score'] = int(clarity_result['score'])
                    except Exception as e:
                        st.warning(f"Issue with clarity analysis: {e}")
                        clarity_result = {'summary': 'Could not analyze clarity', 'score': 5}

                    progress_bar.progress(85)
                    status_text.success("✅ **Step 4 Complete:** Clarity analysis finished!")

                    results = {
                        'legalRisk': legal_risk_result,
                        'missingRights': missing_rights_result,
                        'clarity': clarity_result
                    }

                    # Safely extract scores with defaults
                    try:
                        legalRisk_score = results['legalRisk'].get('score', 5)
                        if not isinstance(legalRisk_score, (int, float)):
                            legalRisk_score = 5
                    except:
                        legalRisk_score = 5
                    
                    try:
                        missingRights_score = results['missingRights'].get('score', 5)
                        if not isinstance(missingRights_score, (int, float)):
                            missingRights_score = 5
                    except:
                        missingRights_score = 5
                    
                    try:
                        clarity_score = results['clarity'].get('score', 5)
                        if not isinstance(clarity_score, (int, float)):
                            clarity_score = 5
                    except:
                        clarity_score = 5

                    final_score = (legalRisk_score + missingRights_score + clarity_score) / 3

                    time.sleep(0.5)

                    status_text.info("📋 **Step 5/5:** Creating your personalized analysis...")
                    progress_bar.progress(90)

                    finalSummary = PromptTemplate(
                        template="""Create a user-friendly summary that regular people can understand easily.
                        Main Summary: {summary}
                        Legal Risk Summary: {legalRiskSummary}  
                        Missing Rights Summary: {missingRightsSummary}
                        Clarity Summary: {claritySummary}
                        Final Score: {finalScore}/10
                        Recommendation: {recommendation}

                        Write in simple language and include:
                        - **What This Means for You**: Plain English explanation
                        - **Main Concerns**: Top 3 things to worry about  
                        - **What's Missing**: Important protections you don't have
                        - **Bottom Line**: Should you sign this? Why or why not?

                        Use bullet points and avoid legal jargon.
                        """,
                        input_variables=['summary', 'legalRiskSummary', 'missingRightsSummary', 'claritySummary', 'finalScore', 'recommendation']
                    )

                    finalSummaryChain = finalSummary | model

                    if final_score >= 7:
                        recommendation = "✅ **GENERALLY SAFE TO ACCEPT** - This document has reasonable terms with some standard limitations."
                        color = "green"
                    elif final_score >= 5:
                        recommendation = "⚠️ **REVIEW CAREFULLY** - This document has some concerning terms that need attention."
                        color = "orange"
                    else:
                        recommendation = "❌ **HIGH RISK** - This document has significant issues. Consider legal advice."
                        color = "red"

                    summary_input = {
                        'summary': summary,
                        'legalRiskSummary': results['legalRisk']['summary'],
                        'missingRightsSummary': results['missingRights']['summary'],
                        'claritySummary': results['clarity']['summary'],
                        'finalScore': final_score,
                        'recommendation': recommendation
                    }

                    progress_bar.progress(95)
                    
                    # Generate final summary without displaying it yet
                    streamed_content = ""
                    for chunk in finalSummaryChain.stream(summary_input):
                        streamed_content += chunk.content

                    progress_bar.progress(100)
                    status_text.success("✅ **All Steps Complete!** Your document analysis is ready.")
                    time.sleep(1)
                    progress_container.empty()

                    st.session_state.analysis_complete = True
                    st.session_state.summary_data = {
                        'summary': summary,
                        'results': results,
                        'final_score': final_score,
                        'recommendation': recommendation,
                        'color': color,
                        'final_summary': streamed_content
                    }

                    # Create embeddings and vector store
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=400,
                        chunk_overlap=100
                    )
                    chunks = splitter.split_documents(docs)

                    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    st.info("✅ Embedding Created Successfully")

                    vectorStore = FAISS.from_documents(
                        embedding=embedding,
                        documents=chunks
                    )
                    st.info("✅ Vector Store Created Successfully")
                    
                    # Create doc_id FIRST before storing anything
                    import uuid
                    doc_id = str(uuid.uuid4())[:16]
                    st.session_state.current_doc_id = doc_id
                    
                    # Store in session state IMMEDIATELY
                    st.session_state.vectorStore = vectorStore
                    st.success(f"✅ Vector store stored in memory! Vectors: {vectorStore.index.ntotal}")
                    st.info(f"✅ Doc ID assigned: {doc_id}")

                    # Save vector store to MongoDB GridFS
                    with st.spinner("Saving to cloud database..."):
                        try:
                            st.info(f"Starting save for doc_id: {doc_id}")
                            vector_path = db.save_vector_store(vectorStore, doc_id)
                            st.success(f"✅ Vector store saved to cloud! Path: {vector_path}")
                            
                            # Verify it was saved by trying to list the files
                            faiss_file = db.fs.find_one({'filename': f"{doc_id}_index.faiss"})
                            pkl_file = db.fs.find_one({'filename': f"{doc_id}_index.pkl"})
                            
                            if faiss_file and pkl_file:
                                st.success(f"✅ Verified: Both files found in GridFS")
                            else:
                                st.error(f"⚠️ Files not found! faiss: {faiss_file is not None}, pkl: {pkl_file is not None}")
                            
                            # Save document metadata
                            db.save_document(
                                user_id=user_id,
                                filename=uploadedFile.name,
                                vector_store_path=vector_path,
                                summary_data=st.session_state.summary_data,
                                doc_id=doc_id
                            )

                            # Create session
                            session_id = db.create_session(user_id, doc_id, f"Chat - {uploadedFile.name}")
                            st.session_state.current_session_id = session_id
                            
                            st.success(f"✅ Document saved! Doc ID: {doc_id}")
                            
                            # DON'T RERUN - Just display the summary below
                            
                        except Exception as e:
                            st.error(f"Error saving to database: {e}")
                            import traceback
                            st.error(traceback.format_exc())

                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    import traceback
                    st.error(traceback.format_exc())

                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)

        # DISPLAY SUMMARY AND CHAT (for both new and loaded documents)
        if st.session_state.analysis_complete and st.session_state.summary_data:
            # Try to load vector store from MongoDB if not in session state
            vector_store_loaded = False
            
            # CRITICAL FIX: Check if vector store exists in session state first
            if st.session_state.vectorStore is not None:
                vector_store_loaded = True
                st.success("✅ Vector store ready in memory!")
            elif st.session_state.current_doc_id:
                # Only try to load from MongoDB if not in memory
                faiss_file = db.fs.find_one({'filename': f"{st.session_state.current_doc_id}_index.faiss"})
                pkl_file = db.fs.find_one({'filename': f"{st.session_state.current_doc_id}_index.pkl"})
                
                if faiss_file and pkl_file:
                    with st.spinner("Loading document from cloud..."):
                        try:
                            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                            st.session_state.vectorStore = db.load_vector_store(
                                st.session_state.current_doc_id,
                                embedding
                            )
                            st.success("✅ Document loaded from cloud!")
                            vector_store_loaded = True
                        except Exception as e:
                            st.error(f"⚠️ Error loading vector store: {e}")
                            import traceback
                            st.error(traceback.format_exc())
                            vector_store_loaded = False
                else:
                    st.warning("⚠️ Vector store files not found in database.")
                    st.info("📝 This document was uploaded before cloud storage was configured. To ask questions, please re-upload the document.")
                    vector_store_loaded = False
            
            # Debug info - REMOVE THIS AFTER TESTING
            with st.expander("🔍 Debug Info"):
                st.write(f"**Vector store in memory:** {st.session_state.vectorStore is not None}")
                st.write(f"**Current doc_id:** {st.session_state.current_doc_id}")
                st.write(f"**Vector store loaded:** {vector_store_loaded}")
                st.write(f"**Analysis complete:** {st.session_state.analysis_complete}")
                if st.session_state.current_doc_id:
                    faiss_exists = db.fs.find_one({'filename': f"{st.session_state.current_doc_id}_index.faiss"}) is not None
                    pkl_exists = db.fs.find_one({'filename': f"{st.session_state.current_doc_id}_index.pkl"}) is not None
                    st.write(f"**FAISS file in DB:** {faiss_exists}")
                    st.write(f"**PKL file in DB:** {pkl_exists}")
                
                # Show all session state keys related to document
                st.write("**Session State Keys:**")
                st.write(f"- current_file_name: {st.session_state.get('current_file_name', 'None')}")
                st.write(f"- skip_upload: {st.session_state.get('skip_upload', 'None')}")
                
                if st.session_state.vectorStore is not None:
                    st.write(f"**Vector Store Details:**")
                    st.write(f"- Type: {type(st.session_state.vectorStore)}")
                    st.write(f"- Total vectors: {st.session_state.vectorStore.index.ntotal}")

            # Show the final summary
            st.markdown(st.session_state.summary_data.get('final_summary', ''))
            
            # Show audio button
            if st.session_state.summary_data.get('final_summary'):
                if st.button("🔊 Listen to Summary"):
                    with st.spinner("Generating audio..."):
                        audio_url = generate_speech_data_url(st.session_state.summary_data['final_summary'])
                        if audio_url:
                            st.session_state.audio_url = audio_url
                            st.session_state.show_audio = True

                # Show audio player if generated
                if getattr(st.session_state, 'show_audio', False):
                    audio_html = f"""
                    <audio controls style="width: 100%; margin: 10px 0;">
                        <source src="{st.session_state.audio_url}" type="audio/wav">
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
            
            # Show scores
            st.markdown("---")
            st.markdown("### 📊 Detailed Scores")
            
            results = st.session_state.summary_data.get('results', {})
            final_score = st.session_state.summary_data.get('final_score', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Score", f"{final_score:.1f}/10")
            with col2:
                legalRisk_score = results.get('legalRisk', {}).get('score', 0)
                st.metric("Legal Risk", f"{legalRisk_score}/10", help="Higher = More Risky")
            with col3:
                missingRights_score = results.get('missingRights', {}).get('score', 0)
                st.metric("Missing Rights", f"{missingRights_score}/10", help="Higher = More Missing")
            with col4:
                clarity_score = results.get('clarity', {}).get('score', 0)
                st.metric("Clarity", f"{clarity_score}/10", help="Higher = Clearer")
            
            # Show recommendation box
            recommendation = st.session_state.summary_data.get('recommendation', '')
            color = st.session_state.summary_data.get('color', 'green')
            
            st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 15px; border-radius: 10px; margin: 20px 0; background-color: {'#36454F' if color=='green' else '#A39E96' if color=='orange' else '#FF2A00'}">
            <h3>{recommendation}</h3>
            <p><strong>Overall Score: {final_score:.1f}/10</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # CHAT INTERFACE - ALWAYS SHOW (even if vector store not loaded)
            st.markdown("---")
            st.markdown("### 💬 Chat with Your Document")
            
            # Show chat input only if vector store is loaded
            if vector_store_loaded and st.session_state.vectorStore:
                retriever = st.session_state.vectorStore.as_retriever(
                    search_kwargs={"k": 6, "lambda_mult": 0},
                    search_type="mmr"
                )
                
                st.markdown("""
                    <style>
                    .stButton button {
                        height: 70px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown("#### 💭 Ask a New Question")
                col1, col2 = st.columns([90, 10])
                with col1:  
                    userQuery = st.text_area("Enter your query", placeholder="Write your query here....", height=100, key="user_query_input")
                with col2:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    mic = st.button("🎤", use_container_width=True)
                
                submit_query = st.button("Send", type="primary", use_container_width=True)

                if submit_query and userQuery:
                    # Add to session state
                    st.session_state.chat_messages.append({
                        "role": "human",
                        "content": userQuery
                    })

                    # Save to MongoDB
                    if st.session_state.current_session_id:
                        db.save_message(st.session_state.current_session_id, "human", userQuery)

                    result = retriever.invoke(userQuery)
                    content = "\n".join([doc.page_content for doc in result])
                    
                    for i, doc in enumerate(result):
                        st.write(f"Chunk {i} Score:", doc.metadata.get("score", "N/A"))

                    context_text = ""
                    if len(st.session_state.chat_messages) > 1:
                        context_text = "\n\nPrevious conversation:\n"
                        for msg in st.session_state.chat_messages[:-1]:
                            role = "User" if msg["role"] == "human" else "Assistant"
                            context_text += f"{role}: {msg['content']}\n"

                    prompt = PromptTemplate(
                        template="""Based on the following document content and previous conversation context, answer the user's question clearly 
                        and concisely. If the information is not available in the document, say so.
                        Document Content: {result}
                        {context}

                        Current User Question: {userQuery}

                        Answer:""",
                        input_variables=['result', 'userQuery', 'context']
                    )
                    promptOutput = prompt.invoke({
                        'result': content,
                        'userQuery': userQuery,
                        'context': context_text
                    })

                    with st.spinner("Generating answer..."):
                        finalOutput = model.invoke(promptOutput)
                        ai_response = finalOutput.content

                        # Add to session state
                        st.session_state.chat_messages.append({
                            "role": "ai",
                            "content": ai_response
                        })

                        # Save to MongoDB
                        if st.session_state.current_session_id:
                            db.save_message(st.session_state.current_session_id, "ai", ai_response)
                        
                        # Show the answer immediately
                        st.success("✅ Answer generated!")
                        st.markdown("**Answer:**")
                        st.markdown(ai_response)
                        
                        # Add a button to clear input for next question
                        if st.button("Ask Another Question"):
                            st.rerun()
            else:
                st.warning("⚠️ Cannot ask new questions - vector store not available.")
                
                # Offer to re-upload just the vector store
                st.info("💡 **Solution:** Re-upload this document to restore chat functionality.")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("**Option 1:** Upload the document again at the top")
                with col2:
                    if st.button("🔄 Or Click Here to Re-upload", type="primary", use_container_width=True):
                        # Clear the current doc to force re-upload
                        st.session_state.current_doc_id = None
                        st.session_state.current_file_name = None
                        st.session_state.analysis_complete = False
                        st.session_state.vectorStore = None
                        st.rerun()
                
                # Add a helpful expandable section
                with st.expander("ℹ️ Why can't I ask questions?"):
                    st.write("""
                    This document was saved before the cloud storage system was fully configured. 
                    The document analysis and chat history are saved, but the document content needed to answer questions is missing.
                    
                    **To continue chatting with this document:**
                    1. Click the "🔄 Or Click Here to Re-upload" button above, OR
                    2. Upload the same document again using the file uploader at the top
                    3. The system will detect it's the same file and just update the missing parts
                    4. You'll be able to ask questions again
                    
                    Your previous analysis scores and chat history will not be lost!
                    """)

        # SIDEBAR: Chat History
        with st.sidebar:
            st.header("💬 Chat History")
            
            if st.session_state.chat_messages:
                st.write(f"**Total messages:** {len(st.session_state.chat_messages)}")
                
                for i, msg in enumerate(st.session_state.chat_messages):
                    role_name = "You" if msg["role"] == "human" else "AI"
                    icon = "👤" if msg["role"] == "human" else "🤖"
                    
                    with st.expander(f"{icon} {role_name}: {msg['content'][:50]}..."):
                        st.write(msg['content'])

                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chat_messages = []
                    # Clear from MongoDB
                    if st.session_state.current_session_id:
                        db.clear_session_messages(st.session_state.current_session_id)
                    st.rerun()
            else:
                st.info("No chat history yet. Ask a question about your document!")

    else:
        # NO FILE UPLOADED - Show previous documents
        st.info("Please upload a file to get started.")
        
        # Show user's previous documents
        st.markdown("### 📚 Your Previous Documents")
        user_docs = db.get_user_documents(user_id)
        
        if user_docs:
            for doc in user_docs:
                with st.expander(f"📄 {doc['filename']} - {doc['upload_date'].strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"**Score:** {doc['summary_data']['final_score']:.1f}/10")
                    st.write(f"**Recommendation:** {doc['summary_data']['recommendation']}")
                    
                    col_btn, col_del = st.columns([3, 1])
                    with col_btn:
                        if st.button(f"📂 Load Document", key=f"load_{doc['doc_id']}", use_container_width=True):
                            # Load this document - set all state first
                            st.session_state.current_doc_id = doc['doc_id']
                            st.session_state.current_file_name = doc['filename']
                            st.session_state.summary_data = doc['summary_data']
                            st.session_state.analysis_complete = True
                            st.session_state.vectorStore = None  # Reset to None, will load in display section
                            
                            # Get or create a session for this document
                            sessions = db.get_document_sessions(doc['doc_id'])
                            if sessions:
                                # Use the most recent session
                                st.session_state.current_session_id = sessions[0]['session_id']
                                # Load chat messages from that session
                                messages = db.get_session_messages(sessions[0]['session_id'])
                                st.session_state.chat_messages = messages
                            else:
                                # Create a new session
                                session_id = db.create_session(user_id, doc['doc_id'], f"Chat - {doc['filename']}")
                                st.session_state.current_session_id = session_id
                                st.session_state.chat_messages = []
                            
                            # Force rerun to display the document
                            st.rerun()
                    
                    with col_del:
                        if st.button("🗑️", key=f"del_{doc['doc_id']}", help="Delete document", use_container_width=True):
                            try:
                                db.delete_vector_store(doc['doc_id'])
                            except:
                                pass  # Vector store might not exist
                            db.delete_document(doc['doc_id'])
                            
                            # Clear session state if this was the current document
                            if st.session_state.current_doc_id == doc['doc_id']:
                                st.session_state.current_doc_id = None
                                st.session_state.current_file_name = None
                                st.session_state.summary_data = None
                                st.session_state.analysis_complete = False
                                st.session_state.vectorStore = None
                                st.session_state.chat_messages = []
                                st.session_state.current_session_id = None
                            
                            st.success("Document deleted!")
                            st.rerun()
        else:
            st.info("No previous documents found. Upload a document to get started!")


if __name__ == "__main__":
    run_langchain_app()