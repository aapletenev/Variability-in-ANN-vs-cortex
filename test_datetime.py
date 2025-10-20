from datetime import datetime
print("Testing datetime import...")
print(f"Current time: {datetime.now()}")

# Test importing the visualizations module
try:
    import visualizations
    print("Successfully imported visualizations module")
    
    # Test the datetime import in the module
    print(f"datetime import in visualizations: {visualizations.datetime}")
    
except Exception as e:
    print(f"Error importing visualizations: {e}")
    import traceback
    traceback.print_exc()
