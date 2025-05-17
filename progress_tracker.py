import os
import time
from typing import Dict, Any
import threading

class ProgressTracker:
    """
    A class to track and store progress information for various training processes.
    This class provides a centralized way to track progress that can be accessed
    by both the training processes and the UI.
    """

    def __init__(self, progress_dir: str = "progress"):
        """
        Initialize the progress tracker.

        Args:
            progress_dir: Directory to store progress files
        """
        self.progress_dir = progress_dir
        self.lock = threading.Lock()

        # In-memory progress data
        self.forecaster_progress = {}
        self.rl_agent_progress = {}
        self.overall_progress = {}

        # Create progress directory if it doesn't exist
        os.makedirs(progress_dir, exist_ok=True)

        # Initialize progress data
        self._init_progress_data()

    def _init_progress_data(self):
        """Initialize the progress data with default values."""
        # Forecaster progress
        self.forecaster_progress = {
            "status": "idle",
            "message": "Forecaster not started",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_trial": 0,
            "total_trials": 0,
            "best_score": None,
            "start_time": None,
            "end_time": None,
            "elapsed_time": 0,
            "estimated_time_remaining": None,
            "error": None
        }

        # RL agent progress
        self.rl_agent_progress = {
            "status": "idle",
            "message": "RL agent not started",
            "progress": 0.0,
            "current_episode": 0,
            "total_episodes": 0,
            "current_step": 0,
            "total_steps": 0,
            "current_eval": 0,
            "total_evals": 0,
            "best_score": None,
            "current_score": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "start_time": None,
            "end_time": None,
            "elapsed_time": 0,
            "estimated_time_remaining": None,
            "error": None
        }

        # Overall progress
        self.overall_progress = {
            "status": "idle",
            "message": "Training not started",
            "progress": 0.0,
            "current_phase": None,
            "total_phases": 2,  # Forecaster and RL agent
            "start_time": None,
            "end_time": None,
            "elapsed_time": 0,
            "estimated_time_remaining": None,
            "error": None
        }

    def _get_default_progress(self, progress_type: str) -> Dict[str, Any]:
        """
        Get default progress data for a given type.

        Args:
            progress_type: Type of progress (forecaster, rl_agent, overall)

        Returns:
            Default progress data
        """
        if progress_type == "forecaster":
            return {
                "status": "idle",
                "message": "Forecaster not started",
                "progress": 0.0,
                "current_step": 0,
                "total_steps": 0,
                "current_epoch": 0,
                "total_epochs": 0,
                "current_trial": 0,
                "total_trials": 0,
                "best_score": None,
                "elapsed_time": 0,
            }
        elif progress_type == "rl_agent":
            return {
                "status": "idle",
                "message": "RL agent not started",
                "progress": 0.0,
                "current_episode": 0,
                "total_episodes": 0,
                "current_score": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
                "elapsed_time": 0,
            }
        else:  # overall
            return {
                "status": "idle",
                "message": "Training not started",
                "progress": 0.0,
                "current_phase": None,
                "total_phases": 2,
                "elapsed_time": 0,
            }

    def _save_progress(self, progress_type: str, progress_data: Dict[str, Any]):
        """
        Save progress data to memory.

        Args:
            progress_type: Type of progress (forecaster, rl_agent, overall)
            progress_data: Progress data to save
        """
        with self.lock:
            if progress_type == "forecaster":
                self.forecaster_progress = progress_data
            elif progress_type == "rl_agent":
                self.rl_agent_progress = progress_data
            else:  # overall
                self.overall_progress = progress_data

    def _load_progress(self, progress_type: str) -> Dict[str, Any]:
        """
        Load progress data from memory.

        Args:
            progress_type: Type of progress (forecaster, rl_agent, overall)

        Returns:
            Progress data
        """
        with self.lock:
            try:
                if progress_type == "forecaster":
                    return self.forecaster_progress
                elif progress_type == "rl_agent":
                    return self.rl_agent_progress
                else:  # overall
                    return self.overall_progress
            except Exception:
                # Return default values if there's any issue
                return self._get_default_progress(progress_type)

    def update_forecaster_progress(self, **kwargs):
        """
        Update forecaster progress.

        Args:
            **kwargs: Progress data to update
        """
        progress_data = self._load_progress("forecaster")

        # Update progress data
        for key, value in kwargs.items():
            progress_data[key] = value

        # Update elapsed time if start_time is set
        if progress_data.get("start_time") and not progress_data.get("end_time"):
            progress_data["elapsed_time"] = time.time() - progress_data["start_time"]

        # Save updated progress
        self._save_progress("forecaster", progress_data)

        # Update overall progress
        self._update_overall_progress()

    def update_rl_agent_progress(self, **kwargs):
        """
        Update RL agent progress.

        Args:
            **kwargs: Progress data to update
        """
        progress_data = self._load_progress("rl_agent")

        # Update progress data
        for key, value in kwargs.items():
            progress_data[key] = value

        # Update elapsed time if start_time is set
        if progress_data.get("start_time") and not progress_data.get("end_time"):
            progress_data["elapsed_time"] = time.time() - progress_data["start_time"]

        # Save updated progress
        self._save_progress("rl_agent", progress_data)

        # Update overall progress
        self._update_overall_progress()

    def update_overall_progress(self, **kwargs):
        """
        Update overall progress directly.

        Args:
            **kwargs: Progress data to update
        """
        progress_data = self._load_progress("overall")

        # Update progress data
        for key, value in kwargs.items():
            progress_data[key] = value

        # Update elapsed time if start_time is set
        if progress_data.get("start_time") and not progress_data.get("end_time"):
            progress_data["elapsed_time"] = time.time() - progress_data["start_time"]

        # Save updated progress
        self._save_progress("overall", progress_data)

    def _update_overall_progress(self):
        """Update overall progress based on forecaster and RL agent progress."""
        forecaster_progress = self._load_progress("forecaster")
        rl_agent_progress = self._load_progress("rl_agent")
        overall_progress = self._load_progress("overall")

        # Determine current phase
        if forecaster_progress.get("status") == "running":
            overall_progress["current_phase"] = 1
            overall_progress["message"] = "Training forecasting model..."
        elif rl_agent_progress.get("status") == "running":
            overall_progress["current_phase"] = 2
            overall_progress["message"] = "Training RL agent..."
        elif forecaster_progress.get("status") == "completed" and rl_agent_progress.get("status") == "completed":
            overall_progress["current_phase"] = 2
            overall_progress["status"] = "completed"
            overall_progress["message"] = "Training completed successfully!"
            overall_progress["progress"] = 1.0
            overall_progress["end_time"] = time.time()

        # Calculate overall progress
        if overall_progress["current_phase"] == 1:
            # In forecaster phase, overall progress is forecaster progress / 2
            overall_progress["progress"] = forecaster_progress.get("progress", 0.0) / 2
        elif overall_progress["current_phase"] == 2:
            # In RL agent phase, overall progress is 0.5 + RL agent progress / 2
            overall_progress["progress"] = 0.5 + rl_agent_progress.get("progress", 0.0) / 2

        # Update start time if not set
        if not overall_progress.get("start_time") and (
            forecaster_progress.get("start_time") or rl_agent_progress.get("start_time")
        ):
            overall_progress["start_time"] = min(
                x for x in [
                    forecaster_progress.get("start_time", float('inf')),
                    rl_agent_progress.get("start_time", float('inf'))
                ] if x is not None
            )

        # Update elapsed time if start_time is set
        if overall_progress.get("start_time") and not overall_progress.get("end_time"):
            overall_progress["elapsed_time"] = time.time() - overall_progress["start_time"]

        # Save updated overall progress
        self._save_progress("overall", overall_progress)

    def get_forecaster_progress(self) -> Dict[str, Any]:
        """Get forecaster progress."""
        return self._load_progress("forecaster")

    def get_rl_agent_progress(self) -> Dict[str, Any]:
        """Get RL agent progress."""
        return self._load_progress("rl_agent")

    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress."""
        return self._load_progress("overall")

    def reset(self):
        """Reset all progress."""
        self._init_progress_data()

# Create a global instance of the progress tracker
progress_tracker = ProgressTracker()
