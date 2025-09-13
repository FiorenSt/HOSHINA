#!/usr/bin/env python3
"""
Setup script for ATLAS transient dataset
Creates appropriate classes for BOGUS vs REALS classification
"""

import sys
from pathlib import Path

# Add the backend to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db import get_session, ClassDef, init_db
from sqlmodel import select

def setup_default_classes():
    """Set up default classes for general image classification"""
    init_db()
    session = get_session()
    
    # Clear existing classes
    existing = session.exec(select(ClassDef)).all()
    for cls in existing:
        session.delete(cls)
    
    # Add default classes - you can customize these later
    classes = [
        ClassDef(name="Class_1", key="1", order=0),
        ClassDef(name="Class_2", key="2", order=1),
        ClassDef(name="Class_3", key="3", order=2),
    ]
    
    for cls in classes:
        session.add(cls)
    
    session.commit()
    print("✅ Default classes configured:")
    print("  1 - Class_1")
    print("  2 - Class_2") 
    print("  3 - Class_3")
    print("  0 - Unsure")
    print("  X - Skip")
    print("\n💡 You can edit these classes in the UI by clicking 'Classes' button")

def main():
    print("🔧 Setting up default classes...")
    setup_default_classes()
    print("✅ Setup complete!")

if __name__ == "__main__":
    main()
