"""
Collaboration Tools

Collaborative analytics tools for real-time analytics dashboard with:
- Multi-user analytics workspace
- Real-time collaboration
- Shared dashboards
- Comments and annotations
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any


class CollaborationTools:
    """
    Collaboration tools component for real-time analytics.
    """
    
    def __init__(self):
        """Initialize the collaboration tools."""
        self.users = {}
        self.shared_dashboards = {}
        self.comments = {}
        self.annotations = {}
        self.collaboration_sessions = {}
    
    def add_user(self, user_id: str, name: str, role: str = "viewer") -> bool:
        """Add a user to the collaboration system."""
        user = {
            'id': user_id,
            'name': name,
            'role': role,
            'joined_at': datetime.now().isoformat(),
            'last_active': datetime.now().isoformat(),
            'permissions': self._get_default_permissions(role)
        }
        
        self.users[user_id] = user
        return True
    
    def _get_default_permissions(self, role: str) -> List[str]:
        """Get default permissions for a role."""
        if role == "admin":
            return ["read", "write", "delete", "share", "manage_users"]
        elif role == "editor":
            return ["read", "write", "share"]
        elif role == "viewer":
            return ["read"]
        else:
            return ["read"]
    
    def share_dashboard(self, dashboard_id: str, owner_id: str, 
                       shared_with: List[str], permissions: List[str]) -> bool:
        """Share a dashboard with other users."""
        if owner_id not in self.users:
            return False
        
        shared_dashboard = {
            'dashboard_id': dashboard_id,
            'owner_id': owner_id,
            'shared_with': shared_with,
            'permissions': permissions,
            'shared_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat()
        }
        
        self.shared_dashboards[dashboard_id] = shared_dashboard
        return True
    
    def add_comment(self, user_id: str, target_type: str, target_id: str, 
                   comment: str, parent_comment_id: Optional[str] = None) -> str:
        """Add a comment to a dashboard element."""
        if user_id not in self.users:
            return None
        
        comment_id = f"comment_{len(self.comments) + 1}"
        
        comment_obj = {
            'id': comment_id,
            'user_id': user_id,
            'user_name': self.users[user_id]['name'],
            'target_type': target_type,  # 'dashboard', 'chart', 'metric'
            'target_id': target_id,
            'comment': comment,
            'parent_comment_id': parent_comment_id,
            'created_at': datetime.now().isoformat(),
            'replies': []
        }
        
        self.comments[comment_id] = comment_obj
        
        # Add to parent comment if it's a reply
        if parent_comment_id and parent_comment_id in self.comments:
            self.comments[parent_comment_id]['replies'].append(comment_id)
        
        return comment_id
    
    def add_annotation(self, user_id: str, target_type: str, target_id: str,
                      annotation_type: str, content: str, position: Dict[str, Any]) -> str:
        """Add an annotation to a dashboard element."""
        if user_id not in self.users:
            return None
        
        annotation_id = f"annotation_{len(self.annotations) + 1}"
        
        annotation = {
            'id': annotation_id,
            'user_id': user_id,
            'user_name': self.users[user_id]['name'],
            'target_type': target_type,
            'target_id': target_id,
            'annotation_type': annotation_type,  # 'highlight', 'note', 'marker'
            'content': content,
            'position': position,
            'created_at': datetime.now().isoformat()
        }
        
        self.annotations[annotation_id] = annotation
        return annotation_id
    
    def start_collaboration_session(self, session_name: str, host_id: str, 
                                   participants: List[str]) -> str:
        """Start a real-time collaboration session."""
        if host_id not in self.users:
            return None
        
        session_id = f"session_{len(self.collaboration_sessions) + 1}"
        
        session = {
            'id': session_id,
            'name': session_name,
            'host_id': host_id,
            'participants': participants,
            'started_at': datetime.now().isoformat(),
            'status': 'active',
            'shared_elements': [],
            'chat_messages': []
        }
        
        self.collaboration_sessions[session_id] = session
        return session_id
    
    def add_chat_message(self, session_id: str, user_id: str, message: str) -> bool:
        """Add a chat message to a collaboration session."""
        if session_id not in self.collaboration_sessions:
            return False
        
        if user_id not in self.users:
            return False
        
        chat_message = {
            'user_id': user_id,
            'user_name': self.users[user_id]['name'],
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.collaboration_sessions[session_id]['chat_messages'].append(chat_message)
        return True
    
    def get_comments_for_target(self, target_type: str, target_id: str) -> List[Dict[str, Any]]:
        """Get all comments for a specific target."""
        comments = []
        for comment in self.comments.values():
            if comment['target_type'] == target_type and comment['target_id'] == target_id:
                comments.append(comment)
        return sorted(comments, key=lambda x: x['created_at'])
    
    def get_annotations_for_target(self, target_type: str, target_id: str) -> List[Dict[str, Any]]:
        """Get all annotations for a specific target."""
        annotations = []
        for annotation in self.annotations.values():
            if annotation['target_type'] == target_type and annotation['target_id'] == target_id:
                annotations.append(annotation)
        return sorted(annotations, key=lambda x: x['created_at'])
    
    def get_active_users(self) -> List[Dict[str, Any]]:
        """Get all active users."""
        return list(self.users.values())
    
    def get_shared_dashboards_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all dashboards shared with a user."""
        shared = []
        for dashboard in self.shared_dashboards.values():
            if user_id in dashboard['shared_with']:
                shared.append(dashboard)
        return shared
    
    def render_collaboration_panel(self):
        """Render the collaboration panel."""
        st.markdown("## 👥 Collaboration Tools")
        
        # User management
        with st.expander("User Management", expanded=False):
            self._render_user_management()
        
        # Shared dashboards
        st.markdown("### 📊 Shared Dashboards")
        self._render_shared_dashboards()
        
        # Comments and annotations
        st.markdown("### 💬 Comments & Annotations")
        self._render_comments_annotations()
        
        # Collaboration sessions
        st.markdown("### 🎯 Collaboration Sessions")
        self._render_collaboration_sessions()
        
        # Real-time chat
        st.markdown("### 💭 Real-Time Chat")
        self._render_chat()
    
    def _render_user_management(self):
        """Render user management interface."""
        st.markdown("#### Add User")
        
        with st.form("add_user"):
            user_id = st.text_input("User ID")
            user_name = st.text_input("User Name")
            role = st.selectbox("Role", ["viewer", "editor", "admin"])
            
            if st.form_submit_button("Add User"):
                if user_id and user_name:
                    success = self.add_user(user_id, user_name, role)
                    if success:
                        st.success(f"User {user_name} added successfully")
                    else:
                        st.error("Failed to add user")
                else:
                    st.error("Please enter user ID and name")
        
        # User list
        st.markdown("#### Active Users")
        users = self.get_active_users()
        
        if not users:
            st.info("No users added yet")
            return
        
        for user in users:
            st.write(f"**{user['name']}** ({user['role']}) - Joined: {user['joined_at'][:10]}")
    
    def _render_shared_dashboards(self):
        """Render shared dashboards interface."""
        st.markdown("#### Share Dashboard")
        
        with st.form("share_dashboard"):
            dashboard_id = st.text_input("Dashboard ID")
            owner_id = st.selectbox(
                "Owner",
                [user['id'] for user in self.users.values()],
                format_func=lambda x: self.users[x]['name']
            )
            shared_with = st.multiselect(
                "Share with",
                [user['id'] for user in self.users.values()],
                format_func=lambda x: self.users[x]['name']
            )
            permissions = st.multiselect(
                "Permissions",
                ["read", "write", "comment", "share"]
            )
            
            if st.form_submit_button("Share Dashboard"):
                if dashboard_id and owner_id and shared_with:
                    success = self.share_dashboard(dashboard_id, owner_id, shared_with, permissions)
                    if success:
                        st.success("Dashboard shared successfully")
                    else:
                        st.error("Failed to share dashboard")
                else:
                    st.error("Please fill in all required fields")
        
        # Shared dashboards list
        st.markdown("#### Shared Dashboards")
        if not self.shared_dashboards:
            st.info("No shared dashboards")
            return
        
        for dashboard_id, dashboard in self.shared_dashboards.items():
            owner_name = self.users.get(dashboard['owner_id'], {}).get('name', 'Unknown')
            st.write(f"**Dashboard {dashboard_id}** - Owner: {owner_name} - Shared: {dashboard['shared_at'][:10]}")
    
    def _render_comments_annotations(self):
        """Render comments and annotations interface."""
        st.markdown("#### Add Comment")
        
        with st.form("add_comment"):
            user_id = st.selectbox(
                "User",
                [user['id'] for user in self.users.values()],
                format_func=lambda x: self.users[x]['name']
            )
            target_type = st.selectbox("Target Type", ["dashboard", "chart", "metric"])
            target_id = st.text_input("Target ID")
            comment = st.text_area("Comment")
            
            if st.form_submit_button("Add Comment"):
                if user_id and target_id and comment:
                    comment_id = self.add_comment(user_id, target_type, target_id, comment)
                    if comment_id:
                        st.success("Comment added successfully")
                    else:
                        st.error("Failed to add comment")
                else:
                    st.error("Please fill in all required fields")
        
        # Comments list
        st.markdown("#### Recent Comments")
        if not self.comments:
            st.info("No comments yet")
            return
        
        for comment in list(self.comments.values())[-5:]:  # Show last 5 comments
            st.write(f"**{comment['user_name']}** on {comment['target_type']} {comment['target_id']}: {comment['comment'][:50]}...")
    
    def _render_collaboration_sessions(self):
        """Render collaboration sessions interface."""
        st.markdown("#### Start Session")
        
        with st.form("start_session"):
            session_name = st.text_input("Session Name")
            host_id = st.selectbox(
                "Host",
                [user['id'] for user in self.users.values()],
                format_func=lambda x: self.users[x]['name']
            )
            participants = st.multiselect(
                "Participants",
                [user['id'] for user in self.users.values()],
                format_func=lambda x: self.users[x]['name']
            )
            
            if st.form_submit_button("Start Session"):
                if session_name and host_id:
                    session_id = self.start_collaboration_session(session_name, host_id, participants)
                    if session_id:
                        st.success(f"Session started: {session_id}")
                    else:
                        st.error("Failed to start session")
                else:
                    st.error("Please enter session name and select host")
        
        # Active sessions
        st.markdown("#### Active Sessions")
        active_sessions = [s for s in self.collaboration_sessions.values() if s['status'] == 'active']
        
        if not active_sessions:
            st.info("No active sessions")
            return
        
        for session in active_sessions:
            host_name = self.users.get(session['host_id'], {}).get('name', 'Unknown')
            st.write(f"**{session['name']}** - Host: {host_name} - Started: {session['started_at'][:16]}")
    
    def _render_chat(self):
        """Render real-time chat interface."""
        st.markdown("#### Chat")
        
        # Select session
        active_sessions = [s for s in self.collaboration_sessions.values() if s['status'] == 'active']
        
        if not active_sessions:
            st.info("No active collaboration sessions")
            return
        
        session_id = st.selectbox(
            "Select Session",
            [s['id'] for s in active_sessions],
            format_func=lambda x: self.collaboration_sessions[x]['name']
        )
        
        if session_id:
            session = self.collaboration_sessions[session_id]
            
            # Chat messages
            st.markdown("**Chat Messages:**")
            for message in session['chat_messages'][-10:]:  # Show last 10 messages
                st.write(f"**{message['user_name']}** ({message['timestamp'][:16]}): {message['message']}")
            
            # Add message
            with st.form("add_chat_message"):
                user_id = st.selectbox(
                    "User",
                    [user['id'] for user in self.users.values()],
                    format_func=lambda x: self.users[x]['name']
                )
                message = st.text_input("Message")
                
                if st.form_submit_button("Send"):
                    if user_id and message:
                        success = self.add_chat_message(session_id, user_id, message)
                        if success:
                            st.success("Message sent")
                            st.rerun()
                        else:
                            st.error("Failed to send message")
                    else:
                        st.error("Please select user and enter message")


# Factory function for creating collaboration tools
def create_collaboration_tools() -> CollaborationTools:
    """Create collaboration tools component."""
    return CollaborationTools()
