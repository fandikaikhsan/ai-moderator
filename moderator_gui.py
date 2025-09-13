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
        self._last_frame_update = 0
        self._video_configured = False
        self._frame_skip_counter = 0
        
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
        
        # Center - Transcription panel  
        self.setup_transcription_panel(middle_frame)
        
        # Right side - Live statistics
        self.setup_stats_panel(middle_frame)
        
        # Bottom section - Analytics and Matrix
        self.setup_analytics_panel(main_container)
        
        # Initialize microphone list
        self.current_microphone_index = None
        self.microphone_callback = None
        self.refresh_microphones()
    
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
    
    def setup_transcription_panel(self, parent):
        """Setup real-time transcription panel"""
        transcription_frame = tk.LabelFrame(parent, text="Live Transcription", 
                                          font=('Arial', 12, 'bold'),
                                          fg='white', bg='#3c3c3c',
                                          relief=tk.RAISED, bd=2)
        transcription_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Transcription controls
        controls_frame = tk.Frame(transcription_frame, bg='#3c3c3c')
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Start/Stop transcription button
        self.transcription_button = tk.Button(controls_frame, text="Start Transcription",
                                            command=self.toggle_transcription,
                                            font=('Arial', 10, 'bold'),
                                            bg='#4CAF50', fg='white',
                                            width=15)
        self.transcription_button.pack(side=tk.LEFT, padx=5)
        
        # Generate Summary button
        self.summary_button = tk.Button(controls_frame, text="Generate Summary",
                                       command=self.generate_summary,
                                       font=('Arial', 10, 'bold'),
                                       bg='#FF9800', fg='white',
                                       width=15, state=tk.DISABLED)
        self.summary_button.pack(side=tk.LEFT, padx=5)
        
        # Clear transcription button
        clear_button = tk.Button(controls_frame, text="Clear",
                               command=self.clear_transcription,
                               font=('Arial', 10),
                               bg='#f44336', fg='white',
                               width=10)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Microphone selection
        mic_frame = tk.Frame(controls_frame, bg='#3c3c3c')
        mic_frame.pack(side=tk.RIGHT, padx=5)
        
        tk.Label(mic_frame, text="Microphone:", 
                font=('Arial', 10), fg='white', bg='#3c3c3c').pack(side=tk.LEFT)
        
        self.microphone_var = tk.StringVar()
        self.microphone_dropdown = ttk.Combobox(mic_frame, textvariable=self.microphone_var,
                                               width=20, state="readonly")
        self.microphone_dropdown.pack(side=tk.LEFT, padx=(5, 0))
        self.microphone_dropdown.bind('<<ComboboxSelected>>', self.on_microphone_changed)
        
        # Refresh microphones button
        refresh_button = tk.Button(mic_frame, text="🔄",
                                 command=self.refresh_microphones,
                                 font=('Arial', 8),
                                 bg='#607D8B', fg='white',
                                 width=3)
        refresh_button.pack(side=tk.LEFT, padx=(2, 0))
        
        # Create notebook for transcription tabs
        self.transcription_notebook = ttk.Notebook(transcription_frame)
        self.transcription_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Live transcript tab
        self.setup_live_transcript_tab()
        
        # Summary tab
        self.setup_summary_tab()
        
        # Transcription variables
        self.is_transcribing = False
        self.transcription_data = {}
    
    def setup_live_transcript_tab(self):
        """Setup live transcript display tab"""
        transcript_frame = tk.Frame(self.transcription_notebook, bg='#2b2b2b')
        self.transcription_notebook.add(transcript_frame, text="Live Transcript")
        
        # Scrollable text area for transcript
        transcript_scroll_frame = tk.Frame(transcript_frame, bg='#2b2b2b')
        transcript_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        transcript_scrollbar = tk.Scrollbar(transcript_scroll_frame)
        transcript_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget for transcript
        self.transcript_text = tk.Text(transcript_scroll_frame, 
                                     bg='#1e1e1e', fg='white',
                                     font=('Courier', 10),
                                     wrap=tk.WORD,
                                     yscrollcommand=transcript_scrollbar.set,
                                     state=tk.DISABLED)
        self.transcript_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        transcript_scrollbar.config(command=self.transcript_text.yview)
        
        # Configure text tags for different participants
        self.transcript_text.tag_configure("participant1", foreground="#4CAF50")
        self.transcript_text.tag_configure("participant2", foreground="#2196F3") 
        self.transcript_text.tag_configure("participant3", foreground="#FF9800")
        self.transcript_text.tag_configure("participant4", foreground="#9C27B0")
        self.transcript_text.tag_configure("timestamp", foreground="#757575")
        
        # Status frame at bottom
        status_frame = tk.Frame(transcript_frame, bg='#2b2b2b')
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.transcription_status = tk.Label(status_frame, 
                                           text="Transcription stopped",
                                           font=('Arial', 9),
                                           fg='#757575', bg='#2b2b2b')
        self.transcription_status.pack(side=tk.LEFT)
        
        self.word_count_label = tk.Label(status_frame,
                                       text="Words: 0",
                                       font=('Arial', 9),
                                       fg='#757575', bg='#2b2b2b')
        self.word_count_label.pack(side=tk.RIGHT)
    
    def setup_summary_tab(self):
        """Setup discussion summary display tab"""
        summary_frame = tk.Frame(self.transcription_notebook, bg='#2b2b2b')
        self.transcription_notebook.add(summary_frame, text="Summary")
        
        # Summary display with scrollbar
        summary_scroll_frame = tk.Frame(summary_frame, bg='#2b2b2b')
        summary_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        summary_scrollbar = tk.Scrollbar(summary_scroll_frame)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.summary_text = tk.Text(summary_scroll_frame,
                                  bg='#1e1e1e', fg='white',
                                  font=('Arial', 11),
                                  wrap=tk.WORD,
                                  yscrollcommand=summary_scrollbar.set,
                                  state=tk.DISABLED)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scrollbar.config(command=self.summary_text.yview)
        
        # Configure text tags for summary formatting
        self.summary_text.tag_configure("title", font=('Arial', 12, 'bold'), foreground="#4CAF50")
        self.summary_text.tag_configure("section", font=('Arial', 11, 'bold'), foreground="#2196F3")
        self.summary_text.tag_configure("highlight", foreground="#FF9800")
        self.summary_text.tag_configure("normal", foreground="white")
        
        # Summary controls at bottom
        summary_controls = tk.Frame(summary_frame, bg='#2b2b2b')
        summary_controls.pack(fill=tk.X, padx=5, pady=2)
        
        export_summary_btn = tk.Button(summary_controls, text="Export Summary",
                                     command=self.export_summary,
                                     font=('Arial', 9),
                                     bg='#2196F3', fg='white')
        export_summary_btn.pack(side=tk.LEFT)
        
        self.summary_status = tk.Label(summary_controls,
                                     text="No summary generated",
                                     font=('Arial', 9),
                                     fg='#757575', bg='#2b2b2b')
        self.summary_status.pack(side=tk.RIGHT)
    
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
                
                # Update participant display every 3 seconds to reduce overhead
                if update_counter % 3 == 0:
                    self.root.after(0, self.update_participant_display)
                
                # Update activity chart every 5 seconds
                if update_counter % 5 == 0:
                    self.root.after(0, self.update_activity_chart)
                
                # Update analytics every 10 seconds
                if update_counter % 10 == 0:
                    self.root.after(0, self.update_analytics)
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                print(f"Update error: {e}")
                break
    
    def update_video_frame(self, frame):
        """Update the video display with a new frame - highly optimized for smooth display"""
        if frame is not None and hasattr(self, 'video_label') and self.video_label:
            try:
                # Skip frame processing if update is too frequent (reduce flickering and CPU load)
                current_time = time.time()
                if hasattr(self, '_last_frame_update'):
                    if current_time - self._last_frame_update < 0.04:  # 25 FPS max for GUI
                        return
                self._last_frame_update = current_time
                
                # Skip every other frame for GUI display to improve performance
                self._frame_skip_counter = getattr(self, '_frame_skip_counter', 0) + 1
                if self._frame_skip_counter % 2 != 0:
                    return
                
                # Convert frame from BGR to RGB for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to fit the display area - smaller for better performance
                height, width = frame_rgb.shape[:2]
                display_width = 320  # Reduced size for better performance
                display_height = 240  # Maintain 4:3 aspect ratio
                
                # Calculate scaling to maintain aspect ratio
                scale = min(display_width/width, display_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Use fastest interpolation for real-time display
                frame_resized = cv2.resize(frame_rgb, (new_width, new_height), 
                                         interpolation=cv2.INTER_NEAREST)
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update the video label with less frequent configure calls
                if not hasattr(self, '_video_configured') or not self._video_configured:
                    self.video_label.configure(image=photo, text="")
                    self._video_configured = True
                else:
                    self.video_label.configure(image=photo)
                
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
    
    # Transcription control methods
    def toggle_transcription(self):
        """Toggle speech transcription on/off"""
        if self.is_transcribing:
            self.stop_transcription()
        else:
            self.start_transcription()
    
    def start_transcription(self):
        """Start speech transcription"""
        self.is_transcribing = True
        self.transcription_button.config(text="Stop Transcription", bg='#f44336')
        self.summary_button.config(state=tk.NORMAL)
        self.transcription_status.config(text="Transcription active", fg='#4CAF50')
        
        # Notify parent app if callback is set
        if hasattr(self, 'transcription_callback') and self.transcription_callback:
            self.transcription_callback('start')
    
    def stop_transcription(self):
        """Stop speech transcription"""
        self.is_transcribing = False
        self.transcription_button.config(text="Start Transcription", bg='#4CAF50')
        self.transcription_status.config(text="Transcription stopped", fg='#757575')
        
        # Notify parent app if callback is set
        if hasattr(self, 'transcription_callback') and self.transcription_callback:
            self.transcription_callback('stop')
    
    def update_transcription(self, participant_id: int, text: str):
        """Update the live transcription display"""
        if not text.strip():
            return
        
        timestamp = time.strftime("%H:%M:%S")
        participant_name = f"Participant {participant_id}"
        
        # Enable text widget for editing
        self.transcript_text.config(state=tk.NORMAL)
        
        # Add timestamp
        self.transcript_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        
        # Add participant name with color
        tag_name = f"participant{min(participant_id, 4)}"  # Limit to 4 colors
        self.transcript_text.insert(tk.END, f"{participant_name}: ", tag_name)
        
        # Add the transcribed text
        self.transcript_text.insert(tk.END, f"{text}\n", "normal")
        
        # Scroll to bottom
        self.transcript_text.see(tk.END)
        
        # Disable editing
        self.transcript_text.config(state=tk.DISABLED)
        
        # Update word count
        content = self.transcript_text.get(1.0, tk.END)
        word_count = len(content.split())
        self.word_count_label.config(text=f"Words: {word_count}")
    
    def clear_transcription(self):
        """Clear all transcription data"""
        self.transcript_text.config(state=tk.NORMAL)
        self.transcript_text.delete(1.0, tk.END)
        self.transcript_text.config(state=tk.DISABLED)
        
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.config(state=tk.DISABLED)
        
        self.word_count_label.config(text="Words: 0")
        self.summary_status.config(text="No summary generated")
        
        # Notify parent app if callback is set
        if hasattr(self, 'transcription_callback') and self.transcription_callback:
            self.transcription_callback('clear')
    
    def generate_summary(self):
        """Generate discussion summary"""
        self.summary_status.config(text="Generating summary...", fg='#FF9800')
        
        # Notify parent app if callback is set
        if hasattr(self, 'summary_callback') and self.summary_callback:
            self.summary_callback()
    
    def refresh_microphones(self):
        """Refresh the list of available microphones"""
        try:
            # Import here to avoid circular imports
            from pyaudio_speech_transcriber import PyAudioSpeechTranscriber
            
            microphones = PyAudioSpeechTranscriber.get_available_microphones()
            
            # Update dropdown values
            mic_names = []
            self.microphone_map = {}  # Map display names to indices
            
            for mic in microphones:
                display_name = f"{mic['name']}"
                if mic.get('is_default', False):
                    display_name += " (Default)"
                
                mic_names.append(display_name)
                self.microphone_map[display_name] = mic['index']
            
            self.microphone_dropdown['values'] = mic_names
            
            # Set current selection
            if hasattr(self, 'current_microphone_index'):
                for name, index in self.microphone_map.items():
                    if index == self.current_microphone_index:
                        self.microphone_var.set(name)
                        break
            elif mic_names:
                # Select default microphone
                for name in mic_names:
                    if "(Default)" in name:
                        self.microphone_var.set(name)
                        break
                else:
                    self.microphone_var.set(mic_names[0])
            
        except Exception as e:
            print(f"Error refreshing microphones: {e}")
            self.microphone_dropdown['values'] = ["No microphones found"]
            self.microphone_var.set("No microphones found")
    
    def on_microphone_changed(self, event=None):
        """Handle microphone selection change"""
        selected_name = self.microphone_var.get()
        
        if selected_name in self.microphone_map:
            microphone_index = self.microphone_map[selected_name]
            self.current_microphone_index = microphone_index
            
            # Notify parent app if callback is set
            if hasattr(self, 'microphone_callback') and self.microphone_callback:
                self.microphone_callback(microphone_index)
            
            print(f"Selected microphone: {selected_name} (Index: {microphone_index})")
    
    def set_microphone_callback(self, callback):
        """Set callback for microphone changes"""
        self.microphone_callback = callback
    
    def display_summary(self, summary_data: dict):
        """Display the generated summary"""
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        
        # Title
        self.summary_text.insert(tk.END, "Discussion Summary\n", "title")
        self.summary_text.insert(tk.END, "=" * 50 + "\n\n", "normal")
        
        # Main summary
        if 'summary' in summary_data:
            self.summary_text.insert(tk.END, "Overview:\n", "section")
            self.summary_text.insert(tk.END, f"{summary_data['summary']}\n\n", "normal")
        
        # Key points
        if 'key_points' in summary_data and summary_data['key_points']:
            self.summary_text.insert(tk.END, "Key Points:\n", "section")
            for point in summary_data['key_points']:
                self.summary_text.insert(tk.END, f"• {point}\n", "normal")
            self.summary_text.insert(tk.END, "\n", "normal")
        
        # Participant contributions
        if 'participants_contribution' in summary_data:
            self.summary_text.insert(tk.END, "Participant Analysis:\n", "section")
            for participant, contribution in summary_data['participants_contribution'].items():
                self.summary_text.insert(tk.END, f"• {participant}: ", "highlight")
                self.summary_text.insert(tk.END, f"{contribution}\n", "normal")
            self.summary_text.insert(tk.END, "\n", "normal")
        
        # Action items
        if 'action_items' in summary_data and summary_data['action_items']:
            self.summary_text.insert(tk.END, "Action Items:\n", "section")
            for item in summary_data['action_items']:
                self.summary_text.insert(tk.END, f"• {item}\n", "normal")
            self.summary_text.insert(tk.END, "\n", "normal")
        
        # Discussion stats
        if 'duration_minutes' in summary_data:
            self.summary_text.insert(tk.END, "Discussion Stats:\n", "section")
            self.summary_text.insert(tk.END, f"Duration: {summary_data['duration_minutes']} minutes\n", "normal")
            
            if 'sentiment' in summary_data:
                self.summary_text.insert(tk.END, f"Overall Sentiment: {summary_data['sentiment']}\n", "normal")
        
        self.summary_text.config(state=tk.DISABLED)
        self.summary_status.config(text="Summary ready", fg='#4CAF50')
        
        # Switch to summary tab
        self.transcription_notebook.select(1)
    
    def export_summary(self):
        """Export summary to file"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Summary"
            )
            
            if filename:
                content = self.summary_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                messagebox.showinfo("Export Complete", f"Summary exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export summary: {e}")
    
    def set_transcription_callback(self, callback):
        """Set callback for transcription control"""
        self.transcription_callback = callback
    
    def set_summary_callback(self, callback):
        """Set callback for summary generation"""
        self.summary_callback = callback
    
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