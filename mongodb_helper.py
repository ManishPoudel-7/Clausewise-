from pymongo import MongoClient
from datetime import datetime
import uuid
import os
import json
import gridfs
import pickle

class MongoDBHelper:
    def __init__(self, connection_string):
        """Initialize MongoDB connection"""
        self.client = MongoClient(connection_string)
        self.db = self.client['clausewise_db']
        
        # Collections
        self.users = self.db['users']
        self.documents = self.db['documents']
        self.chat_sessions = self.db['chat_sessions']
        self.messages = self.db['messages']
        
        # GridFS for storing vector files
        self.fs = gridfs.GridFS(self.db)
    
    # ============ USER MANAGEMENT ============
    
    def get_or_create_user_id(self):
        """Get user ID from local file or create new one"""
        user_file = 'user_config.json'
        
        if os.path.exists(user_file):
            with open(user_file, 'r') as f:
                data = json.load(f)
                return data.get('user_id')
        else:
            # Generate new user ID
            user_id = str(uuid.uuid4())[:16]
            
            # Save locally
            with open(user_file, 'w') as f:
                json.dump({'user_id': user_id, 'created_at': datetime.now().isoformat()}, f)
            
            # Save to MongoDB
            self.users.insert_one({
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_login': datetime.now()
            })
            
            return user_id
    
    def update_user_login(self, user_id):
        """Update last login time"""
        self.users.update_one(
            {'user_id': user_id},
            {'$set': {'last_login': datetime.now()}}
        )
    
    # ============ DOCUMENT MANAGEMENT ============
    
    def save_document(self, user_id, filename, vector_store_path, summary_data, doc_id=None):
        """Save document metadata to MongoDB"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:16]
        
        doc_data = {
            'doc_id': doc_id,
            'user_id': user_id,
            'filename': filename,
            'upload_date': datetime.now(),
            'vector_store_path': vector_store_path,
            'summary_data': summary_data
        }
        
        self.documents.insert_one(doc_data)
        return doc_id
    
    def get_user_documents(self, user_id):
        """Get all documents for a user"""
        docs = self.documents.find(
            {'user_id': user_id}
        ).sort('upload_date', -1)
        
        return list(docs)
    
    def get_document(self, doc_id):
        """Get specific document by ID"""
        return self.documents.find_one({'doc_id': doc_id})
    
    def delete_document(self, doc_id):
        """Delete document and its sessions"""
        # Delete all sessions for this document
        sessions = self.chat_sessions.find({'doc_id': doc_id})
        for session in sessions:
            self.delete_session(session['session_id'])
        
        # Delete document
        self.documents.delete_one({'doc_id': doc_id})
    
    # ============ SESSION MANAGEMENT ============
    
    def create_session(self, user_id, doc_id, session_name=None):
        """Create new chat session"""
        session_id = str(uuid.uuid4())[:16]
        
        if not session_name:
            session_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'doc_id': doc_id,
            'session_name': session_name,
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
        
        self.chat_sessions.insert_one(session_data)
        return session_id
    
    def get_document_sessions(self, doc_id):
        """Get all sessions for a document"""
        sessions = self.chat_sessions.find(
            {'doc_id': doc_id}
        ).sort('last_updated', -1)
        
        return list(sessions)
    
    def update_session_name(self, session_id, new_name):
        """Update session name"""
        self.chat_sessions.update_one(
            {'session_id': session_id},
            {'$set': {
                'session_name': new_name,
                'last_updated': datetime.now()
            }}
        )
    
    def delete_session(self, session_id):
        """Delete session and all its messages"""
        # Delete all messages in this session
        self.messages.delete_many({'session_id': session_id})
        
        # Delete session
        self.chat_sessions.delete_one({'session_id': session_id})
    
    # ============ MESSAGE MANAGEMENT ============
    
    def save_message(self, session_id, role, content):
        """Save a message to MongoDB"""
        message_data = {
            'session_id': session_id,
            'role': role,  # 'human' or 'ai'
            'content': content,
            'timestamp': datetime.now()
        }
        
        # Add message
        self.messages.insert_one(message_data)
        
        # Update session last_updated time
        self.chat_sessions.update_one(
            {'session_id': session_id},
            {'$set': {'last_updated': datetime.now()}}
        )
    
    def get_session_messages(self, session_id):
        """Get all messages for a session"""
        messages = self.messages.find(
            {'session_id': session_id}
        ).sort('timestamp', 1)
        
        message_list = []
        for msg in messages:
            message_list.append({
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg.get('timestamp')
            })
        
        return message_list
    
    def clear_session_messages(self, session_id):
        """Clear all messages from a session"""
        self.messages.delete_many({'session_id': session_id})
    
    # ============ VECTOR STORE MANAGEMENT (GridFS) ============
    
    def save_vector_store(self, vector_store, doc_id):
        """Save FAISS vector store to GridFS"""
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, doc_id)
        
        try:
            # Save FAISS to temp location
            vector_store.save_local(temp_path)
            
            # Upload index.faiss
            faiss_file = os.path.join(temp_path, 'index.faiss')
            with open(faiss_file, 'rb') as f:
                self.fs.put(f, filename=f"{doc_id}_index.faiss", doc_id=doc_id)
            
            # Upload index.pkl
            pkl_file = os.path.join(temp_path, 'index.pkl')
            with open(pkl_file, 'rb') as f:
                self.fs.put(f, filename=f"{doc_id}_index.pkl", doc_id=doc_id)
            
            return f"gridfs://{doc_id}"
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)
    
    def load_vector_store(self, doc_id, embeddings):
        """Load FAISS vector store from GridFS"""
        from langchain_community.vectorstores import FAISS
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, doc_id)
        os.makedirs(temp_path, exist_ok=True)
        
        try:
            # Download index.faiss
            faiss_data = self.fs.find_one({'filename': f"{doc_id}_index.faiss"})
            if not faiss_data:
                raise FileNotFoundError(f"Vector store not found for doc_id: {doc_id}")
            
            with open(os.path.join(temp_path, 'index.faiss'), 'wb') as f:
                f.write(faiss_data.read())
            
            # Download index.pkl
            pkl_data = self.fs.find_one({'filename': f"{doc_id}_index.pkl"})
            with open(os.path.join(temp_path, 'index.pkl'), 'wb') as f:
                f.write(pkl_data.read())
            
            # Load FAISS
            vector_store = FAISS.load_local(temp_path, embeddings, allow_dangerous_deserialization=True)
            
            return vector_store
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)
    
    def delete_vector_store(self, doc_id):
        """Delete vector store from GridFS"""
        self.fs.delete(self.fs.find_one({'filename': f"{doc_id}_index.faiss"})._id)
        self.fs.delete(self.fs.find_one({'filename': f"{doc_id}_index.pkl"})._id)
    
    # ============ UTILITY FUNCTIONS ============
    
    def get_user_stats(self, user_id):
        """Get user statistics"""
        total_documents = self.documents.count_documents({'user_id': user_id})
        total_sessions = self.chat_sessions.count_documents({'user_id': user_id})
        
        # Count total messages
        sessions = self.chat_sessions.find({'user_id': user_id})
        total_messages = 0
        for session in sessions:
            total_messages += self.messages.count_documents({'session_id': session['session_id']})
        
        return {
            'total_documents': total_documents,
            'total_sessions': total_sessions,
            'total_messages': total_messages
        }