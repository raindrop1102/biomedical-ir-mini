"""What one indexing run measured."""
import time
import tracemalloc
from dataclasses import dataclass, field


@dataclass
class Stats:
    method: str
    setting: str = ""
    documents: int = 0
    blocks: int = 0
    pairs: int = 0                    # (termID, docID) pairs generated — BSBI only
    postings: int = 0                 # postings in the final index
    terms: int = 0                    # vocabulary size
    peak_items: int = 0               # most pairs (BSBI) or postings (SPIMI) held at once
    intermediate_files: int = 0
    intermediate_bytes: int = 0
    final: dict = field(default_factory=dict)
    seconds: float = 0.0
    phases: dict = field(default_factory=dict)        # phase → seconds
    blocks_info: list = field(default_factory=list)
    memory: dict = field(default_factory=dict)        # phase → tracemalloc peak bytes, when tracing

    def add_time(self, phase: str, since: float) -> None:
        self.phases[phase] = self.phases.get(phase, 0.0) + time.perf_counter() - since

    def mark_memory(self, phase: str) -> None:
        """Peak traced memory since the last mark, under `phase` (only while tracemalloc runs)."""
        if tracemalloc.is_tracing():
            self.memory[phase] = tracemalloc.get_traced_memory()[1]
            tracemalloc.reset_peak()
