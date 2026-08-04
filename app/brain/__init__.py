"""Brain — hierarchical trade decision intelligence.

Architecture:
  Layer 1: PerceptionEngine    — market state classification
  Layer 2: ThesisValidator     — entry condition tracking
  Layer 3: OpportunityScanner  — cross-symbol opportunity cost
  Layer 4: DigitalTwin         — Monte Carlo simulation
  Layer 5: MetaFusion          — 5-model ensemble decision
  Layer 6: ReflectionEngine    — post-trade learning

All layers share a single FeatureCache for memory efficiency.
"""
from app.brain.feature_cache import FeatureCache
from app.brain.perception import PerceptionEngine
from app.brain.thesis_validator import ThesisValidator
from app.brain.opportunity import OpportunityScanner
from app.brain.digital_twin import DigitalTwin
from app.brain.meta_fusion import MetaFusion
from app.brain.reflection import ReflectionEngine
from app.brain.orchestrator import BrainOrchestrator
