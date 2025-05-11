import numpy as np
import torch
from collections import deque
from io import BytesIO
from typing import Tuple
import zlib
import warnings
import gc

class CompressedExperience:
    """Efficient experience compression and storage"""
    def __init__(self, state, action, reward, next_state, done):
        self.compressed_data = None
        self.compress((state, action, reward, next_state, done))
    
    def compress(self, experience: Tuple) -> None:
        """Compress experience data using zlib"""
        state, action, reward, next_state, done = experience
        
        # Convert tensors to numpy and quantize floating point data
        def process_tensor(t):
            if isinstance(t, torch.Tensor):
                # Move to CPU and convert to numpy
                t = t.detach().cpu().numpy()
                # Quantize float32 to float16 for compression
                if t.dtype == np.float32:
                    t = t.astype(np.float16)
            return t
        
        # Process all components
        state = process_tensor(state)
        next_state = process_tensor(next_state)
        
        # Handle tuple actions specially
        if isinstance(action, tuple):
            action_discrete = process_tensor(action[0])
            action_allocation = process_tensor(action[1])
            is_tuple_action = True
        else:
            action_discrete = process_tensor(action)
            action_allocation = None
            is_tuple_action = False
        
        reward = float(reward) if isinstance(reward, (torch.Tensor, np.ndarray)) else reward
        done = bool(done)
        
        # Serialize and compress using BytesIO
        buffer = BytesIO()
        np.savez_compressed(
            buffer,
            state=state,
            action_discrete=action_discrete,
            action_allocation=action_allocation,
            is_tuple_action=is_tuple_action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        buffer.seek(0)
        self.compressed_data = zlib.compress(buffer.getvalue(), level=9)
    
    def decompress(self) -> Tuple:
        """Decompress experience data"""
        # Decompress data
        buffer = BytesIO(zlib.decompress(self.compressed_data))
        
        # Load numpy array from buffer
        with np.load(buffer, allow_pickle=True) as data:
            # Convert back to tensors
            state = torch.from_numpy(data['state'].astype(np.float32))
            
            # Handle tuple actions
            is_tuple_action = bool(data['is_tuple_action'])
            if is_tuple_action:
                action = (
                    torch.from_numpy(data['action_discrete'].astype(np.float32)),
                    torch.from_numpy(data['action_allocation'].astype(np.float32))
                )  # Close the tuple
            else:
                action = torch.from_numpy(data['action_discrete'].astype(np.float32))
            
            reward = float(data['reward'])
            next_state = torch.from_numpy(data['next_state'].astype(np.float32))
            done = bool(data['done'])
            
            return state, action, reward, next_state, done
class PrioritizedReplayBuffer:
    def __init__(self, capacity=50_000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
        # Keep everything on CPU by default
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add experience hashing to prevent duplicates
        self.experience_hash_set = set()
        
        # Add TD error history for adaptive priorities
        self.td_error_history = deque(maxlen=1000)
        self.priority_variance = 1.0
        
        # Memory optimization
        self._enable_compression = True
        self._compression_threshold = 1000  # Start compressing after this many experiences
        self._batch_decompression = True
        self._prefetch_size = 64  # Number of experiences to decompress in advance
        
        # Cached decompressed experiences
        self._decompressed_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Memory monitoring
        self._total_memory_usage = 0
        self._monitor_memory_usage()
        
        print(f"Initialized PrioritizedReplayBuffer:")
        print(f"Capacity: {capacity}")
        print(f"Device: {self.device}")
        print(f"Compression enabled: {self._enable_compression}")
        print(f"Compression threshold: {self._compression_threshold}")
        print(f"Prefetch size: {self._prefetch_size}")
    
    def _monitor_memory_usage(self):
        """Monitor and log memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self._total_memory_usage = memory_info.rss / 1024 / 1024  # Convert to MB
            
            if self._total_memory_usage > 1000:  # If using more than 1GB
                warnings.warn(f"High memory usage in replay buffer: {self._total_memory_usage:.2f} MB")
                
                # Clear cache if memory usage is too high
                if len(self._decompressed_cache) > 100:
                    self._decompressed_cache.clear()
                    gc.collect()
        except Exception as e:
            print(f"Error monitoring memory usage: {str(e)}")
            self._total_memory_usage = 0  # Set default value on error
    
    def _hash_experience(self, state, action, next_state) -> int:
        """Create a unique hash for an experience to detect duplicates"""
        def tensor_hash(t):
            if isinstance(t, torch.Tensor):
                return hash(t.cpu().numpy().tobytes())
            elif isinstance(t, tuple):
                return hash(tuple(tensor_hash(x) for x in t))
            return hash(t)
        
        combined_hash = tensor_hash(state) ^ tensor_hash(action) ^ tensor_hash(next_state)
        return combined_hash
    
    def add(self, state, action, reward, next_state, done):
        """Store experience in buffer with duplicate detection and compression"""
        # Check for duplicates
        exp_hash = self._hash_experience(state, action, next_state)
        if exp_hash in self.experience_hash_set:
            return
        
        self.experience_hash_set.add(exp_hash)
        
        # Compress if enabled and buffer is large enough
        if self._enable_compression and len(self.buffer) >= self._compression_threshold:
            experience = CompressedExperience(state, action, reward, next_state, done)
        else:
            # Store uncompressed for small buffers
            experience = (state, action, reward, next_state, done)
        
        self.buffer.append(experience)
        
        # Use adaptive priority based on TD error history
        if len(self.td_error_history) > 0:
            priority = max(abs(np.mean(self.td_error_history)), self.priority_variance)
        else:
            priority = self.max_priority
        
        self.priorities.append(priority)
        
        # Monitor memory periodically
        if len(self.buffer) % 1000 == 0:
            self._monitor_memory_usage()
    
    def _prefetch_experiences(self, indices):
        """Prefetch and decompress experiences for batch sampling"""
        if not self._batch_decompression:
            return
        
        for idx in indices:
            if idx not in self._decompressed_cache:
                experience = self.buffer[idx]
                if isinstance(experience, CompressedExperience):
                    self._decompressed_cache[idx] = experience.decompress()
                    self._cache_misses += 1
            else:
                self._cache_hits += 1
    
    def sample(self, batch_size, device=None):
        """Sample a batch of experiences with memory efficiency"""
        if len(self.buffer) < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices and calculate importance weights
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Prefetch experiences
        self._prefetch_experiences(indices)
        
        # Get samples with decompression if needed
        samples = []
        for idx in indices:
            if idx in self._decompressed_cache:
                samples.append(self._decompressed_cache[idx])
            else:
                experience = self.buffer[idx]
                if isinstance(experience, CompressedExperience):
                    decompressed = experience.decompress()
                    if len(self._decompressed_cache) < self._prefetch_size:
                        self._decompressed_cache[idx] = decompressed
                    samples.append(decompressed)
                else:
                    samples.append(experience)
        
        # Unpack samples
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Convert to tensors efficiently
        device = device or self.device
        
        # Batch convert to tensors
        states = torch.stack([torch.as_tensor(s, device=device) for s in states])
        next_states = torch.stack([torch.as_tensor(s, device=device) for s in next_states])
        
        if isinstance(actions[0], tuple):
            actions = (
                torch.stack([torch.as_tensor(a[0], device=device) for a in actions]),
                torch.stack([torch.as_tensor(a[1], device=device) for a in actions])
            )
        else:
            actions = torch.stack([torch.as_tensor(a, device=device) for a in actions])
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        # Clear cache if it's too large
        if len(self._decompressed_cache) > self._prefetch_size:
            self._decompressed_cache.clear()
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, td_errors):
        """Update priorities with TD error tracking"""
        for idx, error in zip(indices, td_errors):
            error_val = abs(error.item())
            self.priorities[idx] = error_val + 1e-6
            self.max_priority = max(self.max_priority, error_val)
            
            # Update TD error history
            self.td_error_history.append(error_val)
            
            # Update priority variance
            if len(self.td_error_history) > 1:
                self.priority_variance = np.var(self.td_error_history)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        return {
            'total_memory_mb': self._total_memory_usage,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_ratio': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            'compressed_experiences': sum(1 for x in self.buffer if isinstance(x, CompressedExperience)),
            'cache_size': len(self._decompressed_cache)
        }