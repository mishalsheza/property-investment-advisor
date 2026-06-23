"""Test RAG retrieval for property addresses."""

import sys
import os
from pathlib import Path

# Get the project root (where src/ is located)
project_root = Path(__file__).parent.parent  # Go up from tests/ to root
src_path = project_root / "src"

# Add src to Python path
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import without "src." prefix
from property_advisor.agents.rag_agent import rag_agent
from property_advisor.state import PropertyState

def test_rag():
    """Test RAG retrieval."""
    print("Testing RAG...")
    
    # Create state
    state = PropertyState(
        property_address="Whitefield, Bangalore, 560066",
        budget=9500000,
        investment_horizon_years=5,
        investment_strategy="rental"
    )
    
    # Run RAG agent
    result = rag_agent(state)
    rag_context = result.get('rag_context', [])
    
    print(f"Found {len(rag_context)} documents")
    for doc in rag_context[:3]:
        print(f"  - {doc.get('source', 'Unknown')}")

if __name__ == "__main__":
    test_rag()