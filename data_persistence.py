"""
Data Persistence and Reporting Module
Handles saving discussion sessions and generating comprehensive reports
"""

import json
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pickle
from dataclasses import asdict
import sqlite3

class SessionDatabase:
    """SQLite database for storing discussion sessions"""
    
    def __init__(self, db_path: str = "discussion_sessions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,
                start_time REAL,
                end_time REAL,
                duration REAL,
                participant_count INTEGER,
                total_speaking_time REAL,
                quality_score REAL,
                created_at TEXT,
                data_path TEXT
            )
        """)
        
        # Participants table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS participants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                person_id INTEGER,
                total_speaking_time REAL,
                speaking_percentage REAL,
                speaking_turns INTEGER,
                interruptions_made INTEGER,
                times_interrupted INTEGER,
                participation_level TEXT,
                average_intensity REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        # Events table (for timeline analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp REAL,
                event_type TEXT,
                person_id INTEGER,
                description TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_session(self, session_data: Dict, session_name: Optional[str] = None) -> int:
        """Save a complete session to the database"""
        if session_name is None:
            session_name = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Extract session metadata
            session_info = session_data.get('session_info', {})
            metrics = session_data.get('metrics', {})
            
            # Insert session record
            cursor.execute("""
                INSERT INTO sessions (
                    session_name, start_time, end_time, duration, 
                    participant_count, total_speaking_time, quality_score, 
                    created_at, data_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_name,
                session_info.get('start_time', 0),
                session_info.get('export_time', 0),
                metrics.get('discussion_duration', 0),
                metrics.get('total_participants', 0),
                metrics.get('total_speaking_time', 0),
                metrics.get('discussion_quality_score', 0),
                datetime.now().isoformat(),
                f"{session_name}.json"
            ))
            
            session_id = cursor.lastrowid
            
            # Insert participant records
            participants_data = session_data.get('participants', {})
            activity_matrix = session_data.get('activity_matrix', {})
            
            for person_id, participant_data in participants_data.items():
                activity_stats = activity_matrix.get(int(person_id), {})
                
                cursor.execute("""
                    INSERT INTO participants (
                        session_id, person_id, total_speaking_time, 
                        speaking_percentage, speaking_turns, interruptions_made, 
                        times_interrupted, participation_level, average_intensity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    int(person_id),
                    participant_data.get('total_speaking_time', 0),
                    activity_stats.get('speaking_percentage', 0),
                    participant_data.get('speaking_turns', 0),
                    participant_data.get('interruptions_made', 0),
                    participant_data.get('times_interrupted', 0),
                    participant_data.get('participation_level', 'unknown'),
                    participant_data.get('average_speaking_intensity', 0)
                ))
            
            conn.commit()
            return session_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_session_list(self) -> List[Dict]:
        """Get list of all stored sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, session_name, start_time, duration, 
                   participant_count, quality_score, created_at
            FROM sessions
            ORDER BY created_at DESC
        """)
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row[0],
                'name': row[1],
                'start_time': datetime.fromtimestamp(row[2]) if row[2] else None,
                'duration': timedelta(seconds=row[3]) if row[3] else None,
                'participant_count': row[4],
                'quality_score': row[5],
                'created_at': row[6]
            })
        
        conn.close()
        return sessions
    
    def get_session_details(self, session_id: int) -> Optional[Dict]:
        """Get detailed information about a specific session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        session_row = cursor.fetchone()
        
        if not session_row:
            conn.close()
            return None
        
        # Get participants
        cursor.execute("SELECT * FROM participants WHERE session_id = ?", (session_id,))
        participant_rows = cursor.fetchall()
        
        # Get events
        cursor.execute("SELECT * FROM events WHERE session_id = ?", (session_id,))
        event_rows = cursor.fetchall()
        
        conn.close()
        
        return {
            'session': dict(zip([desc[0] for desc in cursor.description], session_row)),
            'participants': [dict(zip([desc[0] for desc in cursor.description], row)) 
                           for row in participant_rows],
            'events': [dict(zip([desc[0] for desc in cursor.description], row)) 
                      for row in event_rows]
        }

class ReportGenerator:
    """Generate various types of reports from discussion data"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_summary_report(self, session_data: Dict, output_format: str = "json") -> str:
        """Generate a summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract key metrics
        metrics = session_data.get('metrics', {})
        participants = session_data.get('participants', {})
        activity_matrix = session_data.get('activity_matrix', {})
        analytics = session_data.get('analytics', {})
        
        # Create summary
        summary = {
            'report_generated': datetime.now().isoformat(),
            'session_overview': {
                'duration_minutes': metrics.get('discussion_duration', 0) / 60,
                'total_participants': metrics.get('total_participants', 0),
                'total_speaking_time_minutes': metrics.get('total_speaking_time', 0) / 60,
                'silence_time_minutes': metrics.get('silence_time', 0) / 60,
                'quality_score': metrics.get('discussion_quality_score', 0)
            },
            'participation_analysis': self._analyze_participation(activity_matrix),
            'discussion_quality': analytics.get('conversation_quality', {}),
            'key_insights': self._extract_key_insights(session_data),
            'recommendations': analytics.get('conversation_quality', {}).get('recommendations', [])
        }
        
        # Save report
        if output_format.lower() == "json":
            filename = os.path.join(self.output_dir, f"summary_report_{timestamp}.json")
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        elif output_format.lower() == "csv":
            filename = os.path.join(self.output_dir, f"summary_report_{timestamp}.csv")
            self._save_summary_as_csv(summary, filename)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return filename
    
    def generate_participant_report(self, session_data: Dict) -> str:
        """Generate detailed participant analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"participant_report_{timestamp}.csv")
        
        activity_matrix = session_data.get('activity_matrix', {})
        participants_data = session_data.get('participants', {})
        
        # Prepare participant data for CSV
        participant_rows = []
        for person_id, activity_stats in activity_matrix.items():
            participant_data = participants_data.get(str(person_id), {})
            
            row = {
                'participant_id': person_id,
                'speaking_time_seconds': activity_stats.get('speaking_time', 0),
                'speaking_percentage': activity_stats.get('speaking_percentage', 0),
                'speaking_turns': activity_stats.get('speaking_turns', 0),
                'interruptions_made': activity_stats.get('interruptions_made', 0),
                'times_interrupted': activity_stats.get('times_interrupted', 0),
                'average_intensity': activity_stats.get('average_intensity', 0),
                'participation_level': activity_stats.get('participation_level', 'unknown'),
                'currently_active': activity_stats.get('is_active_now', False)
            }
            participant_rows.append(row)
        
        # Sort by speaking time
        participant_rows.sort(key=lambda x: x['speaking_time_seconds'], reverse=True)
        
        # Save to CSV
        if participant_rows:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = participant_rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(participant_rows)
        
        return filename
    
    def generate_timeline_report(self, session_data: Dict) -> str:
        """Generate timeline analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"timeline_report_{timestamp}.json")
        
        analytics = session_data.get('analytics', {})
        discussion_map = analytics.get('discussion_map', {})
        
        timeline_data = {
            'report_generated': datetime.now().isoformat(),
            'timeline_events': discussion_map.get('timeline', []),
            'flow_diagram': discussion_map.get('flow_diagram', []),
            'engagement_analysis': analytics.get('engagement', {}),
            'turn_taking_analysis': analytics.get('turn_taking', {}),
            'interaction_patterns': analytics.get('interactions', {})
        }
        
        with open(filename, 'w') as f:
            json.dump(timeline_data, f, indent=2, default=str)
        
        return filename
    
    def _analyze_participation(self, activity_matrix: Dict) -> Dict:
        """Analyze participation patterns"""
        if not activity_matrix:
            return {}
        
        speaking_percentages = [stats['speaking_percentage'] for stats in activity_matrix.values()]
        
        return {
            'most_active_participant': max(activity_matrix.items(), 
                                         key=lambda x: x[1]['speaking_percentage'])[0],
            'least_active_participant': min(activity_matrix.items(), 
                                          key=lambda x: x[1]['speaking_percentage'])[0],
            'average_speaking_percentage': sum(speaking_percentages) / len(speaking_percentages),
            'participation_distribution': {
                'dominant': len([p for p in activity_matrix.values() 
                               if p['participation_level'] == 'dominant']),
                'balanced': len([p for p in activity_matrix.values() 
                               if p['participation_level'] == 'balanced']),
                'quiet': len([p for p in activity_matrix.values() 
                            if p['participation_level'] == 'quiet']),
                'silent': len([p for p in activity_matrix.values() 
                             if p['participation_level'] == 'silent'])
            }
        }
    
    def _extract_key_insights(self, session_data: Dict) -> List[str]:
        """Extract key insights from session data"""
        insights = []
        
        metrics = session_data.get('metrics', {})
        activity_matrix = session_data.get('activity_matrix', {})
        analytics = session_data.get('analytics', {})
        
        # Duration insights
        duration_minutes = metrics.get('discussion_duration', 0) / 60
        if duration_minutes > 60:
            insights.append(f"Long discussion: {duration_minutes:.1f} minutes")
        elif duration_minutes < 5:
            insights.append(f"Brief discussion: {duration_minutes:.1f} minutes")
        
        # Participation insights
        if activity_matrix:
            speaking_percentages = [stats['speaking_percentage'] for stats in activity_matrix.values()]
            max_speaking = max(speaking_percentages)
            min_speaking = min(speaking_percentages)
            
            if max_speaking > 50:
                insights.append("Discussion dominated by one participant")
            elif max_speaking - min_speaking < 10:
                insights.append("Very balanced participation")
        
        # Quality insights
        quality_score = metrics.get('discussion_quality_score', 0)
        if quality_score > 0.8:
            insights.append("High-quality discussion with good dynamics")
        elif quality_score < 0.4:
            insights.append("Discussion quality could be improved")
        
        # Turn-taking insights
        turn_taking = analytics.get('turn_taking', {})
        if turn_taking.get('smooth_transitions', 0) > 0.7:
            insights.append("Excellent turn-taking patterns")
        elif turn_taking.get('overlaps', 0) > 0.3:
            insights.append("High amount of overlapping speech")
        
        return insights
    
    def _save_summary_as_csv(self, summary: Dict, filename: str):
        """Save summary data as CSV"""
        # Flatten the summary data for CSV format
        flattened_data = []
        
        # Session overview
        overview = summary.get('session_overview', {})
        for key, value in overview.items():
            flattened_data.append({'category': 'session_overview', 'metric': key, 'value': value})
        
        # Participation analysis
        participation = summary.get('participation_analysis', {})
        for key, value in participation.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened_data.append({
                        'category': f'participation_{key}', 
                        'metric': sub_key, 
                        'value': sub_value
                    })
            else:
                flattened_data.append({'category': 'participation', 'metric': key, 'value': value})
        
        # Discussion quality
        quality = summary.get('discussion_quality', {})
        for key, value in quality.items():
            if not isinstance(value, (list, dict)):
                flattened_data.append({'category': 'quality', 'metric': key, 'value': value})
        
        # Save to CSV
        if flattened_data:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['category', 'metric', 'value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)

class DataExporter:
    """Export discussion data in various formats"""
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_raw_data(self, session_data: Dict, format_type: str = "json") -> str:
        """Export raw session data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type.lower() == "json":
            filename = os.path.join(self.output_dir, f"raw_data_{timestamp}.json")
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
        elif format_type.lower() == "pickle":
            filename = os.path.join(self.output_dir, f"raw_data_{timestamp}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(session_data, f)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return filename
    
    def export_for_analysis(self, session_data: Dict) -> str:
        """Export data in format suitable for external analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"analysis_data_{timestamp}.csv")
        
        # Prepare data for analysis tools
        activity_matrix = session_data.get('activity_matrix', {})
        
        analysis_data = []
        for person_id, stats in activity_matrix.items():
            analysis_data.append({
                'participant_id': person_id,
                'speaking_time': stats.get('speaking_time', 0),
                'speaking_percentage': stats.get('speaking_percentage', 0),
                'speaking_turns': stats.get('speaking_turns', 0),
                'interruptions_made': stats.get('interruptions_made', 0),
                'times_interrupted': stats.get('times_interrupted', 0),
                'average_intensity': stats.get('average_intensity', 0),
                'participation_level_dominant': 1 if stats.get('participation_level') == 'dominant' else 0,
                'participation_level_balanced': 1 if stats.get('participation_level') == 'balanced' else 0,
                'participation_level_quiet': 1 if stats.get('participation_level') == 'quiet' else 0,
                'participation_level_silent': 1 if stats.get('participation_level') == 'silent' else 0
            })
        
        if analysis_data:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = analysis_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(analysis_data)
        
        return filename