import torch
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F

class FinancialDataset(Dataset):
    def __init__(self, env, mode='train', sequence_length=15, batch_size=32, shuffle_within_sequences=False):
        self.env = env
        self.mode = mode
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.shuffle_within_sequences = shuffle_within_sequences

        # Select appropriate dates based on mode
        if mode == 'train':
            self.dates = env.pure_train_dates
        elif mode == 'rollout':
            self.dates = env.rollout_dates
        elif mode == 'final_eval':
            self.dates = env.final_eval_dates
        else:  # val/eval mode
            self.dates = env.val_dates

        # Pre-calculate expected feature dimensions
        self.num_features_per_ticker = env.num_features_per_ticker
        self.num_tickers = env.num_tickers
        self.total_features = env.total_features

        # Create sequence indices that respect temporal ordering
        self._create_sequence_indices()

        # Initialize cache for efficient data loading
        self.observation_cache = {}
        self.cache_size = min(2000, len(self.dates))  # Increased cache size for longer sequences

        print(f"Dataset initialized with:")
        print(f"Mode: {mode}")
        print(f"Number of dates: {len(self.dates)}")
        print(f"Features per ticker: {self.num_features_per_ticker}")
        print(f"Number of tickers: {self.num_tickers}")
        print(f"Total features: {self.total_features}")
        print(f"Sequence length: {sequence_length}")
        print(f"Batch size: {batch_size}")
        print(f"Cache size: {self.cache_size}")

    def _create_sequence_indices(self):
        """Create sequence indices that respect temporal ordering"""
        # Calculate number of complete sequences
        num_sequences = (len(self.dates) - self.sequence_length) // self.batch_size

        # Create sequence start indices
        self.sequence_indices = []
        for i in range(num_sequences):
            start_idx = i * self.batch_size
            if start_idx + self.sequence_length <= len(self.dates):
                self.sequence_indices.append(start_idx)

        # Shuffle sequence start indices if in training mode
        if self.mode == 'train' and self.shuffle_within_sequences:
            random.shuffle(self.sequence_indices)

    def _get_cached_observation(self, date_idx):
        """Get observation from cache or compute and cache it"""
        date = self.dates[date_idx]

        if date not in self.observation_cache:
            # Remove oldest items if cache is full
            if len(self.observation_cache) >= self.cache_size:
                oldest_date = min(self.observation_cache.keys())
                del self.observation_cache[oldest_date]

            # Get observation based on mode
            if self.mode == 'train':
                obs = self.env.get_train_observation(date_idx)
            elif self.mode == 'rollout':
                obs = self.env.get_rollout_observation(date_idx)
            elif self.mode == 'final_eval':
                # Use the same method as rollout but with final_eval dates
                obs = self.env.get_val_observation(date_idx)
            else:  # val/eval mode
                obs = self.env.get_val_observation(date_idx)

            # Cache the observation
            self.observation_cache[date] = obs

        return self.observation_cache[date]

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        """Get a sequence of observations with proper temporal ordering"""
        try:
            # Get sequence start index
            start_idx = self.sequence_indices[idx]

            # Get sequence of observations
            sequence = []
            next_sequence = []
            temporal_info = []

            for i in range(self.sequence_length):
                date_idx = start_idx + i
                next_date_idx = date_idx + 1

                # Get current and next observations
                state = self._get_cached_observation(date_idx)
                next_state = self._get_cached_observation(next_date_idx)

                # Ensure proper tensor type and device
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32)
                if not isinstance(next_state, torch.Tensor):
                    next_state = torch.tensor(next_state, dtype=torch.float32)

                # Verify and fix feature dimensions
                expected_features = self.num_tickers * self.num_features_per_ticker + 2

                if state.size(-1) != expected_features:
                    if state.size(-1) > expected_features:
                        state = state[:expected_features]
                        next_state = next_state[:expected_features]
                    else:
                        pad_size = expected_features - state.size(-1)
                        state = F.pad(state, (0, pad_size))
                        next_state = F.pad(next_state, (0, pad_size))

                sequence.append(state)
                next_sequence.append(next_state)

                # Add temporal information
                temporal_info.append({
                    'date_idx': int(date_idx),
                    'is_first_in_sequence': bool(i == 0)
                })

            # Stack sequences
            sequence = torch.stack(sequence)  # [sequence_length, features]
            next_sequence = torch.stack(next_sequence)  # [sequence_length, features]

            return sequence, next_sequence, temporal_info

        except Exception as e:
            print(f"Error loading sequence at index {idx}: {e}")
            # Return zero tensors as fallback with proper temporal info
            sequence = torch.zeros(self.sequence_length, self.total_features)
            next_sequence = torch.zeros(self.sequence_length, self.total_features)
            temporal_info = [
                {'date_idx': -1, 'is_first_in_sequence': (i == 0)}
                for i in range(self.sequence_length)
            ]
            return sequence, next_sequence, temporal_info

class TemporalDataLoader:
    """Custom data loader that respects temporal ordering and prevents look-ahead bias"""
    def __init__(self, dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=4, pin_memory=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle and dataset.mode == 'train'  # Only shuffle in training mode
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()  # Only pin if CUDA is available

        # Calculate number of batches
        self.num_batches = len(dataset) // batch_size
        if not drop_last and len(dataset) % batch_size != 0:
            self.num_batches += 1

        # Create batch indices
        self._create_batch_indices()

        # Create PyTorch DataLoader for parallel loading
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # We handle shuffling ourselves
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between iterations
            prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
        )

    def _collate_fn(self, batch):
        """Custom collate function to handle sequence batching"""
        # Separate sequences, next_sequences, and temporal_info
        sequences, next_sequences, temporal_infos = zip(*batch)

        # Stack sequences and next_sequences
        sequences = torch.stack(sequences)
        next_sequences = torch.stack(next_sequences)

        return sequences, next_sequences, list(temporal_infos)

    def _create_batch_indices(self):
        """Create batch indices that respect temporal ordering"""
        self.batch_indices = list(range(len(self.dataset)))
        if self.shuffle:
            # Only shuffle within temporal constraints
            chunks = [self.batch_indices[i:i + self.batch_size]
                     for i in range(0, len(self.batch_indices), self.batch_size)]
            random.shuffle(chunks)
            self.batch_indices = [idx for chunk in chunks for idx in chunk]

    def __iter__(self):
        self._create_batch_indices()  # Recreate indices for each epoch
        self.data_iter = iter(self.data_loader)
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
            return batch
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return self.num_batches