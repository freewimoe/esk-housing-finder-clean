#!/usr/bin/env python3
"""
GitHub Copilot Session Logger
Automated session tracking and documentation system
"""

import os
import datetime
import shutil
from pathlib import Path

class CopilotSessionLogger:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.docs_dir = self.project_root / "docs" / "copilot-sessions"
        self.template_file = self.docs_dir / "session-template.md"
        self.index_file = self.docs_dir / "index.md"
        
    def create_new_session(self, title, session_type="🚀 Feature"):
        """Create a new session file from template"""
        timestamp = datetime.datetime.now()
        session_id = timestamp.strftime(f"%Y-%m-%d_%H-%M_{title}")
        session_file = self.docs_dir / f"{session_id}.md"
        
        # Read template
        if self.template_file.exists():
            template_content = self.template_file.read_text(encoding='utf-8')
        else:
            template_content = self._get_default_template()
        
        # Replace placeholders
        content = template_content.replace("[SESSION_TITLE]", title)
        content = content.replace("[DATE]", timestamp.strftime("%Y-%m-%d"))
        content = content.replace("[START_TIME]", timestamp.strftime("%H:%M"))
        content = content.replace("[SESSION_TYPE]", session_type)
        content = content.replace("[YYYY-MM-DD_HH-MM_TITLE]", session_id)
        
        # Write session file
        session_file.write_text(content, encoding='utf-8')
        print(f"✅ Created new session: {session_file}")
        
        return session_file
    
    def update_index(self, session_id, title, session_type, status="🔄 In Progress", features=""):
        """Update the session index with new entry"""
        if not self.index_file.exists():
            return
        
        content = self.index_file.read_text(encoding='utf-8')
        
        # Find the table and add new row
        table_marker = "| Session ID | Date | Type | Title | Status | Key Features |"
        if table_marker in content:
            lines = content.split('\n')
            table_index = next(i for i, line in enumerate(lines) if table_marker in line)
            
            # Create new row
            date = session_id.split('_')[0]
            new_row = f"| {session_id} | {date} | {session_type} | {title} | {status} | {features} |"
            
            # Insert after header row (skip separator row)
            lines.insert(table_index + 2, new_row)
            
            # Update statistics
            total_sessions = len([line for line in lines if line.startswith('| 2')]) - 1
            content = '\n'.join(lines)
            content = content.replace("**Total Sessions:** 1", f"**Total Sessions:** {total_sessions}")
            
            self.index_file.write_text(content, encoding='utf-8')
            print(f"✅ Updated index with session: {session_id}")
    
    def finalize_session(self, session_file, status="✅ Complete", features=""):
        """Mark session as complete and update metadata"""
        if isinstance(session_file, str):
            session_file = Path(session_file)
        
        # Update session file with end time
        content = session_file.read_text(encoding='utf-8')
        end_time = datetime.datetime.now().strftime("%H:%M")
        content = content.replace("[END_TIME]", end_time)
        content = content.replace("**Status:** [✅ Complete | 🔄 Partial | ❌ Blocked]", f"**Status:** {status}")
        
        session_file.write_text(content, encoding='utf-8')
        
        # Update index
        session_id = session_file.stem
        title = self._extract_title_from_session(session_file)
        session_type = self._extract_type_from_session(session_file)
        self.update_index(session_id, title, session_type, status, features)
        
        print(f"✅ Finalized session: {session_file}")
    
    def auto_git_commit(self, message=None):
        """Automatically commit session changes to git"""
        if not message:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            message = f"📝 Update Copilot session logs - {timestamp}"
        
        os.system(f'git add {self.docs_dir}')
        os.system(f'git commit -m "{message}"')
        print(f"✅ Committed session logs to git")
    
    def _get_default_template(self):
        """Return default template if file doesn't exist"""
        return """# GitHub Copilot Session - [SESSION_TITLE]

**Session Date:** [DATE]  
**Session Time:** [START_TIME] - [END_TIME]  
**Project:** Smart Real Estate Predictor  
**Session Type:** [SESSION_TYPE]  
**Session ID:** [YYYY-MM-DD_HH-MM_TITLE]  

---

## 🎯 Session Objective
[Add session objectives here]

## 📋 Session Timeline
[Document development process]

## 🔧 Technical Implementation
[Include code changes and technical details]

## 🚨 Issues & Resolutions
[Document problems and solutions]

## ✅ Achievements
[List completed tasks and features]

## 🔄 Next Steps
[Plan for future development]

---

**Session Completed:** [END_TIME]  
**Status:** [✅ Complete | 🔄 Partial | ❌ Blocked]  
"""
    
    def _extract_title_from_session(self, session_file):
        """Extract title from session file"""
        try:
            content = session_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            title_line = next(line for line in lines if line.startswith('# GitHub Copilot Session'))
            return title_line.replace('# GitHub Copilot Session - ', '').replace('[SESSION_TITLE]', 'Unknown')
        except:
            return "Unknown"
    
    def _extract_type_from_session(self, session_file):
        """Extract session type from session file"""
        try:
            content = session_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            type_line = next(line for line in lines if '**Session Type:**' in line)
            return type_line.split('**Session Type:** ')[1].split('  ')[0]
        except:
            return "🚀 Feature"

# CLI Interface
if __name__ == "__main__":
    import sys
    
    logger = CopilotSessionLogger()
    
    if len(sys.argv) < 2:
        print("Usage: python session_logger.py <command> [args]")
        print("Commands:")
        print("  new <title> [type] - Create new session")
        print("  finalize <session_file> [status] - Finalize session")
        print("  commit [message] - Commit session changes")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "new":
        title = sys.argv[2] if len(sys.argv) > 2 else "New-Session"
        session_type = sys.argv[3] if len(sys.argv) > 3 else "🚀 Feature"
        session_file = logger.create_new_session(title, session_type)
        logger.update_index(session_file.stem, title, session_type)
        
    elif command == "finalize":
        session_file = sys.argv[2] if len(sys.argv) > 2 else None
        status = sys.argv[3] if len(sys.argv) > 3 else "✅ Complete"
        if session_file:
            logger.finalize_session(session_file, status)
        
    elif command == "commit":
        message = sys.argv[2] if len(sys.argv) > 2 else None
        logger.auto_git_commit(message)
        
    else:
        print(f"Unknown command: {command}")
