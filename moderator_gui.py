"""
Real-time GUI Interface for AI Moderator
Provides live visualization of discussion dynamics and participant activity
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import threading
import time
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class ModeratorGUI:
    """Main GUI application for the AI Moderator"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Discussion Moderator")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize variables
        self.is_running = False
        self.camera_thread = None
        self.update_thread = None
        
        # Video display variables
        self.video_label = None
        self.current_frame = None
        
        # Data variables
        self.participant_data = {}
        self.discussion_metrics = {}
        self.activity_history = []
        
        # Video processing
        self.video_frame_callback = None  # Will be set by main app
        
        # Setup GUI components
        self.setup_gui()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        
        # Main container
        main_container = tk.Frame(self.root, bg='#2b2b2b')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - Controls
        self.setup_control_panel(main_container)
        
        # Middle section - Video and Live Stats
        middle_frame = tk.Frame(main_container, bg='#2b2b2b')
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left side - Video feed
        self.setup_video_panel(middle_frame)
        
        # Right side - Live statistics
        self.setup_stats_panel(middle_frame)
        
        # Bottom section - Analytics and Matrix
        self.setup_analytics_panel(main_container)
    
    def setup_control_panel(self, parent):
        """Setup control panel with start/stop buttons and session info"""
        control_frame = tk.Frame(parent, bg='#3c3c3c', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = tk.Label(control_frame, text="AI Discussion Moderator", 
                              font=('Arial', 18, 'bold'), fg='white', bg='#3c3c3c')
        title_label.pack(pady=10)
        
        # Control buttons frame
        button_frame = tk.Frame(control_frame, bg='#3c3c3c')
        button_frame.pack(pady=10)
        
        # Start/Stop button
        self.start_button = tk.Button(button_frame, text="Start Monitoring", 
                                     command=self.toggle_monitoring,
                                     font=('Arial', 12, 'bold'),
                                     bg='#4CAF50', fg='white',
                                     width=15, height=2)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Reset session button
        reset_button = tk.Button(button_frame, text="Reset Session",
                                command=self.reset_session,
                                font=('Arial', 12),
                                bg='#f44336', fg='white',
                                width=15, height=2)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # Export button
        export_button = tk.Button(button_frame, text="Export Data",
                                 command=self.export_session,
                                 font=('Arial', 12),
                                 bg='#2196F3', fg='white',
                                 width=15, height=2)
        export_button.pack(side=tk.LEFT, padx=5)
        
        # Session info frame
        info_frame = tk.Frame(control_frame, bg='#3c3c3c')
        info_frame.pack(pady=5)
        
        self.session_time_label = tk.Label(info_frame, text="Session Time: 00:00:00",
                                          font=('Arial', 10), fg='white', bg='#3c3c3c')
        self.session_time_label.pack(side=tk.LEFT, padx=20)
        
        self.participant_count_label = tk.Label(info_frame, text="Participants: 0",
                                               font=('Arial', 10), fg='white', bg='#3c3c3c')
        self.participant_count_label.pack(side=tk.LEFT, padx=20)
        
        self.status_label = tk.Label(info_frame, text="Status: Ready",
                                    font=('Arial', 10), fg='#4CAF50', bg='#3c3c3c')
        self.status_label.pack(side=tk.LEFT, padx=20)
    
    def setup_video_panel(self, parent):
        """Setup video display panel"""
        video_frame = tk.LabelFrame(parent, text="Live Video Feed", 
                                   font=('Arial', 12, 'bold'),
                                   fg='white', bg='#3c3c3c',
                                   relief=tk.RAISED, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video display
        self.video_label = tk.Label(video_frame, bg='black', 
                                   text="Camera feed will appear here\nClick 'Start Monitoring' to begin",
                                   fg='white', font=('Arial', 14))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video controls
        video_controls = tk.Frame(video_frame, bg='#3c3c3c')
        video_controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Detection settings
        detection_frame = tk.LabelFrame(video_controls, text="Detection Settings",
                                       fg='white', bg='#3c3c3c')
        detection_frame.pack(fill=tk.X, pady=5)
        
        # Show landmarks checkbox
        self.show_landmarks = tk.BooleanVar(value=True)
        landmarks_check = tk.Checkbutton(detection_frame, text="Show Face Detection",
                                        variable=self.show_landmarks,
                                        fg='white', bg='#3c3c3c',
                                        selectcolor='#3c3c3c')
        landmarks_check.pack(side=tk.LEFT, padx=5)
        
        # Show activity bars checkbox
        self.show_activity = tk.BooleanVar(value=True)
        activity_check = tk.Checkbutton(detection_frame, text="Show Activity Info",
                                       variable=self.show_activity,
                                       fg='white', bg='#3c3c3c',
                                       selectcolor='#3c3c3c')
        activity_check.pack(side=tk.LEFT, padx=5)
    
    def setup_stats_panel(self, parent):
        """Setup live statistics panel"""
        stats_frame = tk.LabelFrame(parent, text="Live Statistics", 
                                   font=('Arial', 12, 'bold'),
                                   fg='white', bg='#3c3c3c',
                                   relief=tk.RAISED, bd=2)
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create notebook for different stat views
        self.stats_notebook = ttk.Notebook(stats_frame)
        self.stats_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Participants tab
        self.setup_participants_tab()
        
        # Activity tab
        self.setup_activity_tab()
        
        # Insights tab
        self.setup_insights_tab()
    
    def setup_participants_tab(self):
        """Setup participants statistics tab"""
        participants_frame = tk.Frame(self.stats_notebook, bg='#2b2b2b')
        self.stats_notebook.add(participants_frame, text="Participants")
        
        # Header
        header_frame = tk.Frame(participants_frame, bg='#2b2b2b')
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(header_frame, text="ID", font=('Arial', 10, 'bold'),
                fg='white', bg='#2b2b2b', width=5).pack(side=tk.LEFT)
        tk.Label(header_frame, text="Status", font=('Arial', 10, 'bold'),
                fg='white', bg='#2b2b2b', width=10).pack(side=tk.LEFT)
        tk.Label(header_frame, text="Speaking %", font=('Arial', 10, 'bold'),
                fg='white', bg='#2b2b2b', width=12).pack(side=tk.LEFT)
        tk.Label(header_frame, text="Level", font=('Arial', 10, 'bold'),
                fg='white', bg='#2b2b2b', width=10).pack(side=tk.LEFT)
        
        # Scrollable frame for participants
        canvas = tk.Canvas(participants_frame, bg='#2b2b2b')
        scrollbar = ttk.Scrollbar(participants_frame, orient="vertical", command=canvas.yview)
        self.participants_scrollable = tk.Frame(canvas, bg='#2b2b2b')
        
        self.participants_scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.participants_scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
    
    def setup_activity_tab(self):
        """Setup activity chart tab"""
        activity_frame = tk.Frame(self.stats_notebook, bg='#2b2b2b')
        self.stats_notebook.add(activity_frame, text="Activity")
        
        # Create matplotlib figure for activity chart
        self.activity_fig = Figure(figsize=(6, 4), dpi=100, facecolor='#2b2b2b')
        self.activity_ax = self.activity_fig.add_subplot(111)
        self.activity_ax.set_facecolor('#2b2b2b')
        self.activity_ax.tick_params(colors='white')
        self.activity_ax.set_xlabel('Time', color='white')
        self.activity_ax.set_ylabel('Active Speakers', color='white')
        self.activity_ax.set_title('Discussion Activity Over Time', color='white')
        
        self.activity_canvas = FigureCanvasTkAgg(self.activity_fig, activity_frame)
        self.activity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize empty plot
        self.activity_history = []
        self.activity_ax.plot([], [], 'g-', linewidth=2)
        self.activity_canvas.draw()
    
    def setup_insights_tab(self):
        """Setup insights and recommendations tab"""
        insights_frame = tk.Frame(self.stats_notebook, bg='#2b2b2b')
        self.stats_notebook.add(insights_frame, text="Insights")
        
        # Scrollable text widget for insights
        self.insights_text = tk.Text(insights_frame, bg='#3c3c3c', fg='white',
                                    font=('Arial', 10), wrap=tk.WORD)
        insights_scrollbar = ttk.Scrollbar(insights_frame, command=self.insights_text.yview)
        self.insights_text.configure(yscrollcommand=insights_scrollbar.set)
        
        self.insights_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        insights_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial insights text
        self.insights_text.insert(tk.END, "🎯 Discussion Insights\n\n")
        self.insights_text.insert(tk.END, "• Start monitoring to see live insights\n")
        self.insights_text.insert(tk.END, "• Participation balance recommendations\n")
        self.insights_text.insert(tk.END, "• Speaking pattern analysis\n")
        self.insights_text.config(state=tk.DISABLED)
    
    def setup_analytics_panel(self, parent):
        """Setup analytics panel"""
        analytics_frame = tk.LabelFrame(parent, text="Session Analytics", 
                                       font=('Arial', 12, 'bold'),
                                       fg='white', bg='#3c3c3c',
                                       relief=tk.RAISED, bd=2)
        analytics_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Summary metrics
        metrics_frame = tk.Frame(analytics_frame, bg='#3c3c3c')
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create metric displays
        self.total_time_label = tk.Label(metrics_frame, text="Total Time: 00:00",
                                        font=('Arial', 10), fg='white', bg='#3c3c3c')
        self.total_time_label.pack(side=tk.LEFT, padx=20)
        
        self.speaking_time_label = tk.Label(metrics_frame, text="Speaking Time: 00:00",
                                           font=('Arial', 10), fg='white', bg='#3c3c3c')
        self.speaking_time_label.pack(side=tk.LEFT, padx=20)
        
        self.silence_time_label = tk.Label(metrics_frame, text="Silence Time: 00:00",
                                          font=('Arial', 10), fg='white', bg='#3c3c3c')
        self.silence_time_label.pack(side=tk.LEFT, padx=20)
    
    def toggle_monitoring(self):
        """Toggle monitoring on/off"""
        if not self.is_running:
            self.start_monitoring()
        else:
            self.stop_monitoring()
    
    def start_monitoring(self):
        """Start the monitoring session"""
        self.is_running = True
        self.start_button.config(text="Stop Monitoring", bg='#f44336')
        self.status_label.config(text="Status: Monitoring", fg='#4CAF50')
        
        # Start update loop
        self.update_thread = threading.Thread(target=self.update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Camera processing will be handled by the main app
        self.camera_thread = None
        
        print("GUI: Monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring session"""
        self.is_running = False
        self.start_button.config(text="Start Monitoring", bg='#4CAF50')
        self.status_label.config(text="Status: Stopped", fg='#f44336')
        
        print("GUI: Monitoring stopped")
    
    def reset_session(self):
        """Reset the current session"""
        if messagebox.askyesno("Reset Session", "Are you sure you want to reset the current session?"):
            self.stop_monitoring()
            # Reset data
            self.participant_data = {}
            self.discussion_metrics = {}
            self.activity_history = []
            
            # Clear displays
            self.update_participant_display()
            self.update_activity_chart()
            
            print("Session reset")
    
    def export_session(self):
        """Export session data"""
        # This would implement data export functionality
        messagebox.showinfo("Export", "Export functionality would be implemented here")
    
    def update_loop(self):
        """GUI update loop - optimized for performance"""
        start_time = time.time()
        update_counter = 0
        
        while self.is_running:
            try:
                update_counter += 1
                
                # Update session time every second
                elapsed = time.time() - start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                self.root.after(0, lambda: self.session_time_label.config(text=f"Session Time: {time_str}"))
                
                # Update participant count every second
                if hasattr(self, 'get_participant_data') and callable(self.get_participant_data):
                    participant_data = self.get_participant_data()
                    count = len(participant_data) if participant_data else 0
                    self.root.after(0, lambda: self.participant_count_label.config(text=f"Participants: {count}"))
                
                # Update participant display every 2 seconds to reduce overhead
                if update_counter % 2 == 0:
                    self.root.after(0, self.update_participant_display)
                
                # Update activity chart every 3 seconds
                if update_counter % 3 == 0:
                    self.root.after(0, self.update_activity_chart)
                
                # Update analytics every 5 seconds
                if update_counter % 5 == 0:
                    self.root.after(0, self.update_analytics)
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                print(f"Update error: {e}")
                break
    
    def update_video_frame(self, frame):
        """Update the video display with a new frame - optimized for performance"""
        if frame is not None:
            try:
                # Convert frame from BGR to RGB for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to fit the display area - optimized size
                height, width = frame_rgb.shape[:2]
                display_width = 480  # Reduced from 640 for better performance
                display_height = 360  # Reduced from 480 for better performance
                
                # Calculate scaling to maintain aspect ratio
                scale = min(display_width/width, display_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Use faster interpolation
                frame_resized = cv2.resize(frame_rgb, (new_width, new_height), 
                                         interpolation=cv2.INTER_LINEAR)
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update the video label
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo  # Keep a reference
                
            except Exception as e:
                print(f"Error updating video frame: {e}")
    
    def update_participant_display(self):
        """Update participant list display"""
        # Clear existing participant widgets
        for widget in self.participants_scrollable.winfo_children()[1:]:  # Skip header
            widget.destroy()
        
        # Get real participant data from the data callback
        if hasattr(self, 'get_participant_data') and callable(self.get_participant_data):
            participant_data = self.get_participant_data()
        else:
            participant_data = {}
        
        if participant_data:
            # Sort participants by speaking time
            sorted_participants = sorted(participant_data.items(), 
                                       key=lambda x: x[1]['speaking_percentage'], 
                                       reverse=True)
            
            for person_id, stats in sorted_participants:
                participant_frame = tk.Frame(self.participants_scrollable, bg='#2b2b2b')
                participant_frame.pack(fill=tk.X, padx=5, pady=2)
                
                # Participant ID
                tk.Label(participant_frame, text=f"P{person_id}", font=('Arial', 10),
                        fg='white', bg='#2b2b2b', width=5).pack(side=tk.LEFT)
                
                # Speaking status
                is_speaking = stats.get('is_active_now', False)
                status = "Speaking" if is_speaking else "Quiet"
                color = '#4CAF50' if is_speaking else '#757575'
                tk.Label(participant_frame, text=status, font=('Arial', 10),
                        fg=color, bg='#2b2b2b', width=10).pack(side=tk.LEFT)
                
                # Speaking percentage
                percentage = f"{stats['speaking_percentage']:.1f}%"
                tk.Label(participant_frame, text=percentage, font=('Arial', 10),
                        fg='white', bg='#2b2b2b', width=12).pack(side=tk.LEFT)
                
                # Participation level
                level = stats['participation_level'].title()
                level_color = {
                    'Dominant': '#f44336',
                    'Balanced': '#4CAF50', 
                    'Quiet': '#FF9800',
                    'Silent': '#757575'
                }.get(level, '#757575')
                
                tk.Label(participant_frame, text=level, font=('Arial', 10),
                        fg=level_color, bg='#2b2b2b', width=10).pack(side=tk.LEFT)
        else:
            # Show "No participants" message
            no_data_frame = tk.Frame(self.participants_scrollable, bg='#2b2b2b')
            no_data_frame.pack(fill=tk.X, padx=5, pady=20)
            
            tk.Label(no_data_frame, text="No participants detected yet", 
                    font=('Arial', 12), fg='#757575', bg='#2b2b2b').pack()
    
    def update_activity_chart(self):
        """Update activity chart"""
        # Get participant data for activity tracking
        if hasattr(self, 'get_participant_data') and callable(self.get_participant_data):
            participant_data = self.get_participant_data()
            active_count = sum(1 for stats in participant_data.values() if stats.get('is_active_now', False))
        else:
            active_count = 0
        
        # Add to history
        self.activity_history.append(active_count)
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)
        
        # Update plot
        self.activity_ax.clear()
        self.activity_ax.set_facecolor('#2b2b2b')
        self.activity_ax.tick_params(colors='white')
        self.activity_ax.set_xlabel('Time', color='white')
        self.activity_ax.set_ylabel('Active Speakers', color='white')
        self.activity_ax.set_title('Discussion Activity Over Time', color='white')
        
        if self.activity_history:
            x_data = list(range(len(self.activity_history)))
            self.activity_ax.plot(x_data, self.activity_history, 'g-', linewidth=2)
            self.activity_ax.fill_between(x_data, self.activity_history, alpha=0.3, color='green')
        
        self.activity_canvas.draw()
    
    def update_analytics(self):
        """Update analytics and insights"""
        if hasattr(self, 'get_participant_data') and callable(self.get_participant_data):
            participant_data = self.get_participant_data()
            
            if participant_data:
                # Update insights
                self.insights_text.config(state=tk.NORMAL)
                self.insights_text.delete(1.0, tk.END)
                
                self.insights_text.insert(tk.END, "🎯 Live Discussion Insights\n\n")
                
                # Participation analysis
                dominant_count = sum(1 for stats in participant_data.values() if stats['participation_level'] == 'dominant')
                quiet_count = sum(1 for stats in participant_data.values() if stats['participation_level'] == 'quiet')
                silent_count = sum(1 for stats in participant_data.values() if stats['participation_level'] == 'silent')
                
                self.insights_text.insert(tk.END, "📊 Participation Analysis:\n")
                if dominant_count > 0:
                    self.insights_text.insert(tk.END, f"• {dominant_count} participant(s) dominating\n")
                if quiet_count > 0:
                    self.insights_text.insert(tk.END, f"• {quiet_count} participant(s) are quiet\n")
                if silent_count > 0:
                    self.insights_text.insert(tk.END, f"• {silent_count} participant(s) haven't spoken\n")
                
                self.insights_text.insert(tk.END, "\n💡 Recommendations:\n")
                if dominant_count > 1:
                    self.insights_text.insert(tk.END, "• Multiple people dominating - encourage turn-taking\n")
                elif dominant_count == 1:
                    self.insights_text.insert(tk.END, "• One person dominating - engage others\n")
                
                if quiet_count > 0:
                    self.insights_text.insert(tk.END, "• Ask quiet participants for their thoughts\n")
                
                if silent_count > 0:
                    self.insights_text.insert(tk.END, "• Check if silent participants need support\n")
                
                if dominant_count == 0 and quiet_count <= 1:
                    self.insights_text.insert(tk.END, "• Great discussion balance! 🎉\n")
                
                self.insights_text.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_monitoring()
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ModeratorGUI()
    app.run()